"""
AgentOrchestrator —— EchoGuide 编排器（决策闭环：Intent → Planner → Harness）。

架构（v5 收口：Task-scoped SubAgent，中心化轻量 Multi-Agent Harness）：
  - 不再有 QA/EXECUTOR 职责 Agent 类，也没有 Role→Agent Pool。真正的
    Agent 单位 = 围绕一个 Task 的一次独立 TaskAgent Run：每个 Task 拥有
    独立 goal / message / domain / action / depends_on 与协作上下文。
  - QA/EXECUTOR 降级为 Execution Policy（roles.write_policy_for）：
      非 REQUEST 动作 → READ_ONLY（只读工具面）；
      REQUEST 动作 → WRITE_ALLOWED（可暴露满足策略的写工具）。
    Action 只影响工具可见性、写权限与行为指引，不构成 Agent 身份。
  - 领域（IntentDomain）只提供领域人格（DOMAIN_PERSONA）语境与 Skills
    挂载键，不决定工具可见性、不选执行实体 —— "顾问"而非"门卫"。
  - Fast/Deep 是 Execution Profile（profiles.py）：编排器按 profile 池
    选实例（在线表现路由），不是两个不同 Agent。
  - 工具可见性 = 公共工具层：所有 agent_exposed=True 的工具对任何请求
    可见，门禁两层：注册级 agent_exposed + Run 级写策略（READ_ONLY 缺省）
    + Action 级读写策略（QUERY/GREETING 拒写），写工具集合由工具自身声明
    （Tool.write → tool_manager.write_tools()），不再手工维护黑名单。
  - 决策闭环：
      Intent（domain/action/needs_knowledge，不做复杂度判定）
        → Planner（统一输出 ExecutionPlan / Task DAG，single/parallel/
          dependent 由最终 DAG 自动推导；本地 Fast Path 零 LLM，拿不准才
          升级 LLM 规划）
        → Harness（TaskExecutor 按 depends_on 分波并行执行，失败传播：
          依赖失败 → 下游 BLOCKED；无依赖任务并行；依赖任务只读取声明的
          前序结果注入上下文）
        → Synthesizer 合并（LLM 失败降级拼接）。
    协作统一由 Harness 层负责，不实现 Agent-to-Agent 自由通信。
  - Task 级 Trace：execution meta 记录每个 task 的 domain/action/goal/
    depends_on/status/tools/latency/token，供前端过程可视化与 Monitor。

Monitor 有限反馈：Fast profile 在线表现不健康时，Orchestrator 临时把
本应走 Fast 的请求升级 Deep（不引入 RL/Bandit/在线学习）。
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.domains import IntentAction, IntentDomain
from core.intent_recognizer import IntentCategory, IntentRecognizer
from memory.layered_store import LayeredStore
from runtime.policy import ExecutionPolicy
from runtime.runtime import AgentRuntime
from runtime.state import RunState

from agents.persona import action_allows_tool
from agents.profiles import ExecutionProfile, ProfileName, select_profile_name
from agents.roles import (
    AgentResponse, BaseAgent, TaskAgent, WritePolicy, write_policy_for,
)
from agents.verifier import ResponseVerifier
from agents.workflow import (
    ExecutionPlan, SharedState, Synthesizer, Task, TaskExecutor, TaskPlanner,
)

logger = logging.getLogger(__name__)


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    intent:      Optional[IntentCategory] = None        # 兼容字段
    domain:      Optional[IntentDomain]   = None        # 领域语境（人格挂载键）
    action:      Optional[IntentAction]   = None        # 动作（Run 执行策略依据）
    goal:        str = ""        # Task goal（Task-scoped 指令，注入 system prompt）
    task_id:     str = ""        # 所属 Task（Agent Run 边界标识）
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    confidence: float = 0.0
    classifier_stage: str = "preset"
    profile: Optional[ProfileName] = None
    complexity_mode: str = "single"          # 由 Planner 产出的 ExecutionPlan.mode 回填
    complexity_reason: str = "单领域请求"
    benchmark_strategy: str = "adaptive"
    state: Optional[RunState] = None   # Runtime 运行状态（编排器 run() 创建后回填）
    state_query: Optional[Dict[str, Any]] = None  # 查询理解产出（needs_knowledge，Verifier 消费）


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  str                       # 展示标签：任务领域值（task.domain.value）
    intent:      Optional[IntentCategory]
    domain:      Optional[IntentDomain] = None
    action:      Optional[IntentAction] = None
    latency_ms:  float = 0.0
    tools_used:  List[str] = field(default_factory=list)  # 本次调用的工具（RAG 等）
    tool_evidence: List[Dict[str, Any]] = field(default_factory=list)
    execution: Dict[str, Any] = field(default_factory=dict)


class AgentOrchestrator:
    """
    西电校园智慧助手的编排器：决策闭环（Intent → Planner → Harness）。

    - 真正的 Agent 单位是 Task：每个 Task 由一次独立 TaskAgent Run 执行
      （Task-scoped context：goal + 领域人格 + 依赖结果 + 执行策略），
      Fast/Deep 只是 Execution Profile；领域只做人格/Skills 挂载键；
    - Planner 统一输出 ExecutionPlan：本地 Fast Path（单任务/规则链/并行）
      零 LLM；"拿不准"才升级 LLM 规划，复杂度模式由 DAG 自动推导；
    - Harness 分波并行 + 失败传播（依赖失败 → 下游 BLOCKED），依赖任务
      只读取声明的 depends_on 结果注入上下文；Synthesizer 合并；
    - 写权限执行策略（write_policy_for）：仅 REQUEST 允许写，READ_ONLY 缺省；
    - Monitor 有限反馈：Fast 不健康时临时升级 Deep。
    """

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        tool_manager: Optional[Any] = None,
        fast_api_key: Optional[str] = None,
        fast_base_url: Optional[str] = None,
        fast_model: Optional[str] = None,
        deep_api_key: Optional[str] = None,
        deep_base_url: Optional[str] = None,
        deep_model: Optional[str] = None,
        memory_store: Optional[LayeredStore] = None,
        runtime: Optional[AgentRuntime] = None,      # Agent Runtime（缺省自建：默认策略 + 默认中间件）
        policy: Optional[ExecutionPolicy] = None,    # 执行预算（缺省读 ECHOGUIDE_RUNTIME_* 环境变量）
    ):
        def make_client(key: str, url: Optional[str]) -> AsyncAnthropic:
            kwargs: Dict[str, Any] = {"api_key": key}
            if url:
                kwargs["base_url"] = url
            return AsyncAnthropic(**kwargs)

        fast_key = fast_api_key or api_key
        fast_url = fast_base_url if fast_base_url is not None else base_url
        deep_key = deep_api_key or api_key
        deep_url = deep_base_url if deep_base_url is not None else base_url
        fast_name = fast_model or model
        deep_name = deep_model or model
        fast_client = make_client(fast_key, fast_url)
        deep_client = make_client(deep_key, deep_url)
        self._profiles = {
            ProfileName.FAST: ExecutionProfile(ProfileName.FAST, fast_name, 768, False, 3, False, False),
            ProfileName.DEEP: ExecutionProfile(ProfileName.DEEP, deep_name, 1536, True, 5, True, True),
        }

        self._client = deep_client
        self._model = deep_name
        self._runtime = runtime or AgentRuntime(policy=policy)
        # 统一模型调用入口：意图识别 / 规划 / 工具循环 / 合成 / 出口校验全部经
        # ModelGateway 进出（模型调用计数、token 统计、预算、Trace 口径一致）。
        self._gateway = self._runtime.model_gateway
        self._intent_recognizer = IntentRecognizer(
            api_key=fast_key, base_url=fast_url, model=fast_name,
            gateway=self._gateway,
        )
        # Planner（统一输出 ExecutionPlan）：本地 Fast Path + LLM 规划升级
        self._planner = TaskPlanner(
            client=fast_client, model=fast_name, gateway=self._gateway,
            max_tasks=self._runtime.policy.max_tasks,
            max_agents=self._runtime.policy.max_agents,
        )
        # 轻量 Multi-Agent Harness：TaskExecutor（分波执行，失败传播）
        # → SharedState → Synthesizer（合并）。协作只经 Harness，无 Agent 间通信。
        self._executor = TaskExecutor(self._run_task)
        self._synthesizer = Synthesizer(
            deep_client, deep_name, max_tokens=self._runtime.policy.synth_max_tokens,
            gateway=self._gateway,
        )
        # 出口校验（Verifier/Grounding）：规则校验全量；LLM 判定按策略开关，
        # 走廉价 Fast 模型，仅 DEEP/执行路径启用（Fast 路径不付这笔成本）。
        self._verifier = ResponseVerifier(
            client=fast_client, model=fast_name,
            llm_enabled=self._runtime.policy.verifier_llm_enabled,
            gateway=self._gateway,
        )
        self._verification_flags: Dict[str, int] = {}
        self._skill_manager = skill_manager
        self._tool_manager  = tool_manager
        self._memory_store  = memory_store  # 上下文卸载落盘（refs 表），与 MemoryManager 共享

        # TaskAgent × Fast/Deep 双实例（Execution Profile 池）：Agent 类是唯一
        # 的 TaskAgent，实例按 profile 区分；每个 Task 以独立 Run 执行（每次
        # 注入独立 goal/domain/action/上下文），领域不参与实例选择。
        def agents(cls):
            return [
                cls(fast_client, fast_name, skill_manager, tool_manager, self._profiles[ProfileName.FAST]),
                cls(deep_client, deep_name, skill_manager, tool_manager, self._profiles[ProfileName.DEEP]),
            ]

        self._pool: Dict[ProfileName, List[BaseAgent]] = {
            ProfileName.FAST: agents(TaskAgent),
            ProfileName.DEEP: agents(TaskAgent),
        }
        # Runtime 广播到所有 Agent 实例（工具/模型边界钩子），与 set_* 注入同一模式
        for agent_list in self._pool.values():
            for agent in agent_list:
                agent._runtime = self._runtime

        # Monitor 有限反馈：Fast profile 在线表现不健康 → 临时升级 Deep
        self._fast_unhealthy = False

    @property
    def runtime(self) -> AgentRuntime:
        """对外暴露 Agent Runtime（策略与中间件链），供观测/扩展使用。"""
        return self._runtime

    @property
    def intent_recognizer(self) -> IntentRecognizer:
        """对外暴露意图识别器，供评测器等复用（避免重复实例导致缓存/统计分家）。"""
        return self._intent_recognizer

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    def set_tool_manager(self, tool_manager: Optional[Any]) -> None:
        """更新工具管理器引用（Agentic RAG：让 Agent 自主调用工具）。"""
        self._tool_manager = tool_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._tool_manager = tool_manager

    def set_memory_store(self, memory_store: Optional[LayeredStore]) -> None:
        """注入/更新分层记忆存储（上下文卸载落盘），与 MemoryManager 共享实例。"""
        self._memory_store = memory_store
        for agents in self._pool.values():
            for agent in agents:
                agent._memory_store = memory_store

    def expose_external_tools(self, tool_names: List[str]) -> None:
        """把外部 MCP 工具显式加入公共工具层（任何请求可见，仍受 Action 门禁）。"""
        names = set(tool_names)
        if not names or self._tool_manager is None:
            return
        for tool in self._tool_manager._tools.values():
            if tool.name in names:
                tool.agent_exposed = True
        logger.info("外部 MCP 工具已进入公共工具层: %s", sorted(names))

    # ── Monitor 有限反馈 ─────────────────────────────────────────────────────

    def set_fast_health(self, healthy: bool) -> None:
        """Monitor 反馈：Fast profile 是否健康（不健康 → 临时升级 Deep）。"""
        self._fast_unhealthy = not healthy

    @property
    def fast_unhealthy(self) -> bool:
        return self._fast_unhealthy

    # ── 主入口 ────────────────────────────────────────────────────────────────

    # 兼容映射：旧调用方只传 IntentCategory 时，推导出 domain / action。
    _CATEGORY_TO_DOMAIN = {
        IntentCategory.ACADEMIC:    IntentDomain.ACADEMIC,
        IntentCategory.CAMPUS_LIFE: IntentDomain.CAMPUS_LIFE,
        IntentCategory.AFFAIRS:     IntentDomain.AFFAIRS,
        IntentCategory.IT_HELP:     IntentDomain.IT_HELP,
        IntentCategory.PERSONAL:    IntentDomain.PERSONAL,
    }
    _CATEGORY_TO_ACTION = {
        IntentCategory.QUERY:      IntentAction.QUERY,
        IntentCategory.REQUEST:    IntentAction.REQUEST,
        IntentCategory.GREETING:   IntentAction.GREETING,
        IntentCategory.COMPLAINT:  IntentAction.COMPLAINT,
        IntentCategory.FEEDBACK:   IntentAction.FEEDBACK,
    }

    async def run(self, req: Request, on_event: Optional[Any] = None) -> OrchestratorResult:
        """
        处理一次请求的完整流程（Agent Runtime 入口）：

        创建 RunState 挂到 req.state，业务核心 _run_single 作为 core 在 Runtime
        中间件链内执行（before_run → core → before_finish → after_run）。
        Guard 拦截 / 预算超限时 core 不执行，返回带拒绝文案的结果。
        """
        t0 = time.monotonic()
        state = RunState(
            request_id=req.request_id,
            user_id=req.user_id,
            conv_id=req.conv_id,
            message=req.message,
            policy=self._runtime.policy,
        )
        req.state = state

        async def core(ctx):
            return await self._run_single(req, on_event)

        result = await self._runtime.run(
            state, core, on_event=on_event, services={"req": req},
        )
        if result is None:
            reason = state.meta.get("reject_message", "请求被安全策略拦截")
            return OrchestratorResult(
                request_id=req.request_id,
                response=f"抱歉，{reason}。",
                agent_type="task_agent",
                intent=req.intent,
                domain=req.domain,
                action=req.action,
                latency_ms=(time.monotonic() - t0) * 1000,
                execution={
                    **self._execution_meta(req, mode="blocked", agents=[], responses=[]),
                    "guard_rejected": True,
                    "reject_message": reason,
                },
            )
        return result

    async def _run_single(self, req: Request, on_event: Optional[Any] = None) -> OrchestratorResult:
        """
        单次请求的业务核心（在 Runtime 中间件链内执行）：
          意图识别（领域×动作）→ Planner（ExecutionPlan）→ Harness 执行 → 出口校验

        on_event: 可选异步回调，透传给 Agent（SSE 流式输出 / 工具调用可视化）。
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）—— 只理解用户想做什么，不做复杂度判定
        intent_result = None
        if req.domain is None:
            if req.intent is not None:
                # 兼容：只有旧版单维 intent 时，推导 domain/action，避免重复调用 LLM
                req.domain = self._CATEGORY_TO_DOMAIN.get(req.intent, IntentDomain.OTHER)
                req.action = self._CATEGORY_TO_ACTION.get(req.intent, IntentAction.OTHER)
            else:
                intent_result = await self._intent_recognizer.recognize(
                    req.message,
                    history=req.history,
                    force_llm="always_llm" in req.benchmark_strategy,
                    state=req.state,
                )
                req.domain  = intent_result.domain
                req.action  = intent_result.action
                req.intent  = intent_result.intent
                req.confidence = intent_result.confidence
                req.classifier_stage = intent_result.classifier_stage
                # 查询理解产出：needs_knowledge 由 Verifier 消费（判定需要但无证据 → 标记）
                req.state_query = {"needs_knowledge": intent_result.needs_knowledge}
                if on_event is not None:
                    await on_event({
                        "type": "meta",
                        "domain": req.domain.value if req.domain else "other",
                        "action": req.action.value if req.action else "other",
                        "confidence": intent_result.confidence,
                        "classifier_stage": intent_result.classifier_stage,
                        "needs_knowledge": intent_result.needs_knowledge,
                    })

        # 2. Planner：统一输出 ExecutionPlan（Task DAG）；mode 由 DAG 自动推导
        plan = await self._planner.plan(req, req.domain or IntentDomain.OTHER, req.action or IntentAction.OTHER)
        # Benchmark 单 Agent 基线：强制压回单任务（只影响执行形态，不影响意图）
        if req.benchmark_strategy == "single_agent" and plan.mode != "single":
            single_task = plan.tasks[0]
            plan = ExecutionPlan([single_task], reason="Benchmark 单 Agent 基线")
        req.complexity_mode = plan.mode
        req.complexity_reason = plan.reason
        req.profile = (
            ProfileName.DEEP
            if "always_deep" in req.benchmark_strategy
            else self._select_profile(req, plan.mode)
        )
        if req.state is not None:
            req.state.complexity_mode = plan.mode
            req.state.profile = req.profile.value if req.profile else ""
        if on_event is not None:
            await on_event({
                "type": "meta",
                "mode": plan.mode,
                "profile": req.profile.value,
                "complexity_reason": plan.reason,
            })

        # 3. 执行：single → 单任务直执行（1 次 TaskAgent Run）；
        #    parallel/dependent → Harness 分波协作（每个 Task 独立 Run）
        if plan.mode == "single":
            task = plan.tasks[0]
            response = await self._execute_single_task(req, task, on_event)
            agents = [task.domain.value]
        else:
            return await self.run_parallel(req, plan, on_event)

        # 4. 出口校验（Grounding）：规则全量 + 可选 LLM 判定；只标注不阻断
        verification = await self._verify(req, response)

        execution = self._execution_meta(
            req,
            mode="single",
            agents=agents,
            responses=[response],
        )
        execution["verification"] = verification
        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.label,
            intent=req.intent,
            domain=req.domain,
            action=req.action,
            latency_ms=(time.monotonic() - t0) * 1000,
            tools_used=response.tools_used,
            tool_evidence=response.tool_evidence,
            execution=execution,
        )

    async def _execute_single_task(
        self,
        req: Request,
        task: Task,
        on_event: Optional[Any] = None,
    ) -> AgentResponse:
        """单任务执行：构造 Task-scoped 上下文（goal/domain/action），执行一次 TaskAgent Run。"""
        task_req = self._task_request(req, task)
        if on_event is not None:
            await on_event({"type": "meta", "agent": task.domain.value})
        response = await self._execute(task_req, on_event)
        response.task_id = task.task_id
        response.agent_type = task.domain.value
        return response

    async def run_parallel(
        self,
        req: Request,
        plan: ExecutionPlan,
        on_event: Optional[Any] = None,
    ) -> OrchestratorResult:
        """
        多任务协作执行（parallel / dependent）：

          Harness（TaskExecutor）按 depends_on 分波并行执行（失败传播：
          依赖失败 → BLOCKED），结果写入 SharedState，依赖任务只读取自己
          声明的 depends_on 结果注入协作上下文
          → Synthesizer 读取 SharedState 合并为最终回复（LLM 失败降级拼接）

        plan 来自 Planner 的 ExecutionPlan（LLM 任务链已校验或本地规则生成）。
        """
        t0 = time.monotonic()

        req.profile = ProfileName.DEEP
        # Harness：分波执行（失败传播），产出 SharedState
        shared = await self._executor.execute(
            req, plan.tasks, on_event, max_tasks=self._runtime.policy.max_tasks,
        )

        # Synthesizer：合并最终回复
        responses = list(shared.all_results().values())
        final_text = await self._synthesizer.synthesize(req, responses)

        tools_used = [tool for r in responses for tool in r.tools_used]
        tool_evidence = [e for r in responses for e in r.tool_evidence]

        # 出口校验：对合成后的最终回复做 Grounding（证据 = 各任务证据汇总）
        synthesized = AgentResponse(
            content=final_text,
            success=True,
            tools_used=tools_used,
            tool_evidence=tool_evidence,
            profile=req.profile.value if req.profile else "deep",
            agent_type=plan.tasks[0].domain.value if plan.tasks else "task_agent",
        )
        verification = await self._verify(req, synthesized)

        execution = self._execution_meta(
            req,
            mode=plan.mode,
            agents=list(dict.fromkeys(t.domain.value for t in plan.tasks)),
            responses=responses,
            tasks=shared.task_meta(),
        )
        execution["verification"] = verification
        return OrchestratorResult(
            request_id=req.request_id,
            response=synthesized.content,
            agent_type=synthesized.label,
            intent=req.intent,
            domain=req.domain,
            action=req.action,
            latency_ms=(time.monotonic() - t0) * 1000,
            tools_used=tools_used,
            tool_evidence=tool_evidence,
            execution=execution,
        )

    # ── 协作任务执行 ──────────────────────────────────────────────────────────

    def _task_request(self, req: Request, task: Task) -> Request:
        """Task-scoped 上下文构造：用户必要上下文 + Task goal + 领域 + 执行策略。

        不把完整主会话复制给每个 Task（自包含 message 已携带目标与请求）。
        """
        task_req = Request(
            message=task.message,
            user_id=req.user_id,
            conv_id=req.conv_id,
            context=req.context,
            history=req.history,
            intent=req.intent,
            domain=task.domain,   # 任务领域 → 挂载键（人格/Skills）
            action=task.action,   # 任务自己的 action（不继承原始请求，决定 Run 执行策略）
            goal=task.goal,       # Task goal（注入 system prompt [任务目标]）
            task_id=task.task_id, # Agent Run 边界标识
            request_id=req.request_id,
            state=req.state,      # 协作任务继承运行状态（中间件钩子/预算计数不断链）
        )
        task_req.profile = req.profile
        task_req.complexity_mode = req.complexity_mode
        task_req.complexity_reason = req.complexity_reason
        task_req.classifier_stage = req.classifier_stage
        task_req.confidence = req.confidence
        return task_req

    async def _run_task(
        self,
        req: Request,
        task: Task,
        shared: SharedState,
        on_event: Optional[Any] = None,
    ) -> AgentResponse:
        """
        执行单个协作任务：自包含 message + 依赖上下文（只读取声明的 depends_on）。
        执行策略由 task.action 决定（write_policy_for：非 REQUEST 一律 READ_ONLY）。
        """
        task_req = self._task_request(req, task)
        snapshot = shared.snapshot_for(task)
        if snapshot:
            # 注入协作上下文：只含本任务声明的依赖结果（避免上下文膨胀），
            # 让本任务知道前序子任务已给出什么（避免重复检索/重复回答）
            task_req.context = f"{task_req.context}\n\n[协作上下文]\n{snapshot}".strip()
        response = await self._execute(task_req, on_event)
        # 展示/合成标签回填为任务领域：协作结果按任务区分（Synthesizer 分节
        # 标签、execution.tasks 可观测），领域不构成 Agent 身份。
        response.task_id = task.task_id
        response.agent_type = task.domain.value

        # 依赖 DAG 的终点是一次真实写操作；模型只给出建议而忘记调用工具时，
        # 按任务的 required_tool 后置条件补执行，避免出现"任务 success 但
        # 待办未创建"。由 Planner 声明，不硬编码具体任务标识。
        if task.required_tool and task.required_tool not in response.tools_used:
            write_tools = self._tool_manager.write_tools() if self._tool_manager else frozenset()
            if task.action is not None and not action_allows_tool(task.action, task.required_tool, write_tools):
                # Action 层策略（如 QUERY 只读）：补执行写操作被禁止，跳过（策略内不执行即符合预期）
                logger.warning(
                    f"协作任务补执行被 Action 策略拒绝: {task.required_tool} (action={task.action.value})"
                )
                return response
            if on_event is not None:
                await on_event({
                    "type": "tool", "name": task.required_tool,
                    "status": "start", "input": task.required_tool_args,
                })
            result = await self._tool_manager.call(
                task.required_tool,
                task.required_tool_args,
                context={"agent_type": task.domain.value, "user_id": req.user_id},
                use_rewrite=False,
            ) if self._tool_manager is not None else None
            if result is not None and result.success:
                response.tools_used.append(task.required_tool)
                content = str(task.required_tool_args.get("content", "待办事项"))
                response.content = f"{response.content.rstrip()}\n\n已按协作计划记录待办：{content}。"
                if on_event is not None:
                    await on_event({"type": "tool", "name": task.required_tool, "status": "done", "titles": []})
            else:
                response.success = False
                error = getattr(result, "error", "工具管理器不可用")
                response.content = f"{response.content.rstrip()}\n\n待办写入失败：{error}"
        return response

    # ── Execution Profile / 执行策略 ────────────────────────────────────────

    def _select_profile(self, req: Request, mode: str) -> ProfileName:
        """Profile 决策：纯逻辑在 profiles.py；叠加 Monitor 有限反馈。"""
        selected = select_profile_name(
            complexity_mode=mode,
            message=req.message,
            classifier_stage=req.classifier_stage,
            confidence=req.confidence,
        )
        # Monitor 反馈：Fast 在线表现不健康 → 临时升级 Deep（有限反馈，不引入在线学习）
        if selected == ProfileName.FAST and self._fast_unhealthy:
            logger.warning("Fast profile 不健康（Monitor 反馈），临时升级 Deep")
            return ProfileName.DEEP
        return selected

    @staticmethod
    def _execution_meta(
        req: Request,
        *,
        mode: str,
        agents: List[str],
        responses: List[AgentResponse],
        tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "mode": mode,
            "profile": req.profile.value if req.profile else "fast",
            "domain": req.domain.value if req.domain else None,
            "classifier_stage": req.classifier_stage,
            "complexity_reason": req.complexity_reason,
            "agents": list(dict.fromkeys(agents)),
            "tools": list(dict.fromkeys(tool for resp in responses for tool in resp.tools_used)),
            "tasks": tasks or [],
            "model": next((resp.model for resp in responses if resp.model), ""),
            "input_tokens": sum(resp.input_tokens for resp in responses),
            "output_tokens": sum(resp.output_tokens for resp in responses),
            "offloaded_chars": sum(resp.offloaded_chars for resp in responses),
            "saved_tokens": sum(resp.saved_tokens for resp in responses),
        }
        # Runtime 执行摘要（step/tool/retry 计数与 trace_id），纯增量字段
        if req.state is not None:
            meta["runtime"] = req.state.summary()
        # 查询理解产出：needs_knowledge（观测与 Verifier 消费）
        if req.state_query:
            meta["query_understanding"] = req.state_query
        return meta

    def _best_agent(self, profile: Optional[ProfileName] = None) -> Optional[BaseAgent]:
        """
        TaskAgent 实例选择：从该 Profile 池（Fast/Deep 各双实例）中按在线表现
        选实例（成功率高、延迟低、避开正在处理的实例）；缺省 Fast。

        领域不参与实例选择 —— 领域只做人格/Skills 挂载键，没有领域 Agent 实体。
        """
        agents = self._pool.get(profile or ProfileName.FAST, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.effective_routing_score())

    async def _execute(
        self,
        req: Request,
        on_event: Optional[Any] = None,
    ) -> AgentResponse:
        """
        执行一次 TaskAgent Run：按 Execution Profile 选实例（写策略由
        req.action 决定，write_policy_for 在 Agent 内生效）；Fast 失败时
        同任务 Deep 重试。
        """
        agent = self._best_agent(req.profile)
        if agent is None:
            return AgentResponse(
                content="助手暂时不可用，请稍后重试，或直接联系辅导员/教务老师。",
                success=False,
                task_id=req.task_id,
            )

        response = await self._handle_with_runtime(req, agent, on_event)

        # Fast 失败 → Deep 重试（同任务 Run，受 policy.max_retries 约束）
        if not response.success and req.profile == ProfileName.FAST:
            max_retries = 1
            if req.state is not None and req.state.policy is not None:
                max_retries = req.state.policy.max_retries
            if req.state is not None and req.state.retry_count >= max_retries:
                logger.warning(
                    f"Fast 执行失败，已达降级次数上限 {max_retries}，不再重试"
                )
                return response
            logger.warning("Fast 执行失败，降级重试 Deep")
            if req.state is not None:
                req.state.retry_count += 1
            fallback = self._best_agent(ProfileName.DEEP)
            if fallback:
                response = await self._handle_with_runtime(req, fallback, on_event)

        return response

    async def _handle_with_runtime(
        self,
        req: Request,
        agent: BaseAgent,
        on_event: Optional[Any] = None,
    ) -> AgentResponse:
        """在 Runtime 中间件边界内执行一个 Agent（handle 本身在链内运行）。

        模型级钩子（before_model/after_model）不再在此触发——由 ModelGateway
        在每次真实模型调用时触发（step_count = 模型调用次数而非 handle 次数，
        token 逐次落 RunState）。无 state 时 gateway 钩子全部跳过，行为与旧版一致。
        """
        return await agent.handle(req, on_event=on_event)

    # ── 出口校验（Verifier / Grounding）──────────────────────────────────────

    async def _verify(self, req: Request, response: AgentResponse) -> Dict[str, Any]:
        """出口校验：规则校验全量，LLM 判定按策略/路径启用。

        只标注不阻断（honest-by-design）：flags 进 execution meta 与
        verification_stats 计数；LLM 判定未通过时给回答追加免责声明。
        """
        write_tools = self._tool_manager.write_tools() if self._tool_manager else frozenset()
        needs_knowledge = bool((req.state_query or {}).get("needs_knowledge"))
        result = await self._verifier.verify(
            req=req,
            content=response.content,
            tools_used=response.tools_used,
            tool_evidence=response.tool_evidence,
            profile=response.profile,
            write_tools=write_tools,
            needs_knowledge=needs_knowledge,
        )
        for flag in result.flags:
            self._verification_flags[flag] = self._verification_flags.get(flag, 0) + 1
        if result.disclaimer:
            response.content = f"{response.content.rstrip()}\n\n{result.disclaimer}"
        return result.summary()

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for profile, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{profile.value}.{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                    "effective_routing_score": round(agent.stats.effective_routing_score(), 3),
                    "in_flight": agent.stats.in_flight,
                    "profile": agent.profile.name.value,
                    "model": agent.profile.model,
                }
        return result

    def verification_stats(self) -> Dict[str, int]:
        """出口校验 flag 计数（health 端点与 Monitor 面板可见，面试可报数）。"""
        return dict(self._verification_flags)

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项（实例级）。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 fast.0 / deep.1
        （兼容 legacy 下划线写法 fast_0）。Profile 级反馈走 set_fast_health
        （Fast 不健康 → 临时升级 Deep）。
        """
        for profile, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{profile.value}.{i}"
                legacy_key = f"{profile.value}_{i}"
                penalty = penalties.get(key, penalties.get(legacy_key, 0.0))
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
