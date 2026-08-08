"""
亮点：多 Agent 路由与编排（领域路由 + 轻量协作流水线）

核心问题：多 Agent 情况下如何做 Routing 与协作？

路由策略（三层决策）：
  1. 领域路由 —— 根据 IntentDomain（学业/生活/校务/IT）直接映射到专属 Agent。
     意图体系为「领域 domain × 动作 action」二维，路由只看领域，
     修复旧版"请求句式（帮我/我要）被标成通用 REQUEST 后丢失领域"的缺陷。
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 AcademicAgent（校园通用接待）

轻量多 Agent 协作（复杂请求，非 Agent 间聊天）：
  Planner 拆分任务 DAG（自包含任务，支持跨任务依赖）
    → Executor 按 depends_on 分波并行执行，结果写入 SharedState
    → 依赖任务执行时注入协作上下文（使用前序 Agent 结果）
    → Synthesizer 合并为最终回复（LLM 失败降级拼接）。
  工具按 Agent 类型做最小权限隔离（AGENT_TOOL_ALLOWLIST），
  职责外工具不暴露、调用直接拒绝，避免误调与重复执行。

升级机制：
  - 动作维度为转人工 / 置信度低于阈值 / 紧急度 CRITICAL → 自动升级
"""
import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from anthropic import AsyncAnthropic

from core.domains import DOMAIN_KEYWORDS, IntentAction, IntentDomain, keyword_hit
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    ACADEMIC   = "academic"   # 学业支持：选课/课表/考试/成绩/绩点/重修
    CAMPUS_LIFE = "campus_life"  # 校园生活：宿舍/食堂/校车/校园卡/快递
    AFFAIRS    = "affairs"    # 校务咨询：校历/请假/奖学金/办事流程/注册
    IT_HELP    = "it_help"    # IT 助手：教务系统/校园网/VPN/邮箱/统一身份认证
    PERSONAL   = "personal"   # 个人助理：我的课表/待办/考试安排/日程提醒
    ESCALATION = "escalation"  # 转人工（辅导员/教务老师）


# 工具权限边界：每个 Agent 类型只暴露职责内的工具（最小权限原则）。
# 与 Tool.agent_exposed 取交集：agent_exposed=False 的工具对所有 Agent 都不可见。
# 目的：避免 Agent 职责模糊、工具选择空间过大导致错误调用，多 Agent 协作时不重复执行。
AGENT_TOOL_ALLOWLIST: Dict[AgentType, Set[str]] = {
    AgentType.ACADEMIC:    {"knowledge_search"},
    AgentType.CAMPUS_LIFE: {"knowledge_search", "query_campus_info", "get_weather"},
    AgentType.AFFAIRS:     {"knowledge_search"},
    AgentType.IT_HELP:     {"knowledge_search"},
    AgentType.PERSONAL:    {"knowledge_search", "query_schedule", "query_todo",
                            "add_todo", "complete_todo", "query_ddl"},
    AgentType.ESCALATION:  set(),
}


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def routing_score(self) -> float:
        """路由评分：成功率高、延迟低的 Agent 得分高。"""
        latency_score = 1.0 / (1.0 + self.avg_ms / 1000)
        base_score = self.success_rate * 0.7 + latency_score * 0.3
        return base_score * max(0.0, 1.0 - self.monitor_penalty)


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    escalate:    bool  = False   # 是否需要升级
    tools_used:  List[str] = field(default_factory=list)  # 本次调用的工具（供过程可视化）


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    intent:      Optional[IntentCategory] = None        # 兼容字段
    domain:      Optional[IntentDomain]   = None        # 领域（路由依据）
    action:      Optional[IntentAction]   = None        # 动作（行为依据）
    urgency:     Optional[UrgencyLevel]   = None
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    domain:      Optional[IntentDomain] = None
    action:      Optional[IntentAction] = None
    escalated:   bool  = False
    latency_ms:  float = 0.0
    tools_used:  List[str] = field(default_factory=list)  # 本次调用的工具（RAG 等）


# ── 轻量多 Agent 协作（Task Planner / Shared State / Synthesizer）─────────────

@dataclass
class Task:
    """多 Agent 协作中的最小执行单元（自包含：不依赖原始对话上下文）。"""
    task_id:     str
    agent_type:  AgentType
    goal:        str                    # 领域化任务目标（给 Agent 的指令）
    message:     str                    # 自包含请求内容
    depends_on:  List[str] = field(default_factory=list)  # 依赖的其他 task_id


class SharedState:
    """协作共享状态：记录每个 Task 的结果，供依赖任务（如汇总）读取。"""

    def __init__(self) -> None:
        self._results: Dict[str, AgentResponse] = {}

    def set_result(self, task_id: str, resp: AgentResponse) -> None:
        self._results[task_id] = resp

    def get_result(self, task_id: str) -> Optional[AgentResponse]:
        return self._results.get(task_id)

    def done(self, task_id: str) -> bool:
        return task_id in self._results

    def all_results(self) -> Dict[str, AgentResponse]:
        return dict(self._results)

    def snapshot(self) -> str:
        """把已完成任务的结果序列化，注入依赖任务作为协作上下文。"""
        if not self._results:
            return ""
        return "\n\n".join(
            f"[{task_id}]\n{resp.content}"
            for task_id, resp in self._results.items()
        )


class TaskPlanner:
    """
    规则式 Task Planner：复杂请求 → 自包含任务链（带依赖），最后汇总。

    两级策略：
      1. 内置复合规则（RULES）：命中特定复合意图时，生成**真实的依赖任务链**，
         后续任务 depends_on 前序任务，执行时从 SharedState 读取前序结果
         （注入协作上下文），例如：
            t1 查课表(PERSONAL) ──┐
                                  ├──→ t3 创建待办(PERSONAL, depends_on=[t1,t2])
            t2 查办理信息(CAMPUS_LIFE)┘                │
                                                       ↓
                                                  Synthesizer 汇总
      2. 通用降级：未命中任何规则时，每个领域一个自包含任务并行执行，
         末尾一个汇总任务（depends_on 全部领域任务）。
    """

    GOAL_TEMPLATES: Dict[AgentType, str] = {
        AgentType.ACADEMIC:    "从学业支持角度回答用户的请求（选课/课表/考试/成绩等）",
        AgentType.CAMPUS_LIFE: "从校园生活角度回答用户的请求（宿舍/食堂/校车/天气等）",
        AgentType.AFFAIRS:     "从校务办事角度回答用户的请求（校历/请假/奖学金/证明等）",
        AgentType.IT_HELP:     "从 IT 支持角度回答用户的请求（教务系统/校园网/VPN/邮箱等）",
        AgentType.PERSONAL:    "从个人助理角度回答用户的请求（我的课表/待办/考试安排等）",
    }

    # ── 内置复合规则 ──────────────────────────────────────────────────────────

    @staticmethod
    def _plan_schedule_errand(req: Request) -> Optional[List[Task]]:
        """
        规则：个人日程 + 线下办事 + 记待办 的复合请求。

        例："我明天下午有空，想去办校园卡，帮我记个待办"
          t1 查课表（PERSONAL）→ t2 查办理信息（CAMPUS_LIFE）
          → t3 创建待办（PERSONAL，depends_on=[t1,t2]，读取 SharedState 结果）
        """
        msg = req.message
        has_schedule = any(keyword_hit(kw, msg) for kw in ("课表", "课程", "空闲", "上课", "没课", "有空"))
        has_errand   = any(keyword_hit(kw, msg) for kw in ("校园卡", "办理", "材料", "缴费", "办证"))
        has_todo     = any(keyword_hit(kw, msg) for kw in ("待办", "提醒", "记一下", "安排上"))
        if not (has_schedule and has_errand and has_todo):
            return None
        return [
            Task(
                task_id="t1",
                agent_type=AgentType.PERSONAL,
                goal="查询课程空闲时间",
                message=f"查询用户课程/空闲时间（如明天下午是否有课）。用户请求: {msg}",
            ),
            Task(
                task_id="t2",
                agent_type=AgentType.CAMPUS_LIFE,
                goal="查询校园卡办理信息",
                message=f"查询校园卡办理地点和所需材料。用户请求: {msg}",
            ),
            Task(
                task_id="t3",
                agent_type=AgentType.PERSONAL,
                goal="创建校园卡办理待办",
                message=(
                    "根据协作上下文中的课程空闲时间和校园卡办理信息，"
                    "为用户创建一个合适的办理待办/提醒（时间安排在空闲时段）。"
                    f"用户请求: {msg}"
                ),
                depends_on=["t1", "t2"],
            ),
        ]

    RULES = [_plan_schedule_errand]

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def plan(self, req: Request, agent_types: List[AgentType]) -> List[Task]:
        """生成任务 DAG：规则命中 → 依赖任务链；否则领域并行任务。"""
        for rule in self.RULES:
            tasks = rule(req)
            if tasks:
                return tasks

        # 通用降级：每个领域一个自包含任务
        tasks = []
        for i, at in enumerate(agent_types):
            goal = self.GOAL_TEMPLATES.get(at, "回答用户的请求")
            tasks.append(Task(
                task_id=f"t{i}",
                agent_type=at,
                goal=goal,
                message=f"{goal}。\n用户请求: {req.message}",
            ))
        return tasks


class TaskExecutor:
    """
    按依赖 DAG 分波执行任务：wave = 依赖全部完成的任务并行执行，
    结果写入 SharedState；后续任务的 context 注入协作上下文快照
    （真正"使用前序 Agent 结果"）。
    """

    def __init__(self, run_task):
        """
        run_task: async (req, task, shared, on_event) -> AgentResponse
        （由编排器提供，负责把任务分发给对应领域 Agent）。
        """
        self._run_task = run_task

    async def execute(
        self,
        req: Request,
        tasks: List[Task],
        on_event: Optional[Any] = None,
    ) -> SharedState:
        shared = SharedState()
        pending = {t.task_id: t for t in tasks}

        while pending:
            # 当前波：所有依赖已完成的任务
            wave = [t for t in pending.values()
                    if all(shared.done(dep) for dep in t.depends_on)]
            if not wave:
                wave = list(pending.values())  # 依赖无法满足（防御）：剩余任务直接执行
            for t in wave:
                del pending[t.task_id]

            results = await asyncio.gather(
                *[self._run_task(req, t, shared, on_event) for t in wave],
                return_exceptions=True,
            )
            for t, r in zip(wave, results):
                if isinstance(r, AgentResponse):
                    shared.set_result(t.task_id, r)
                else:
                    logger.warning(f"任务 {t.task_id} 执行失败: {r}")
                    shared.set_result(t.task_id, AgentResponse(
                        agent_type=t.agent_type, content="（该领域助手处理失败）", success=False,
                    ))

        return shared


class Synthesizer:
    """
    协作合成器：一次 LLM 调用把多个任务结果合并为连贯的最终回复。

    职责独立于业务任务（不是 Task，也不是 Specialist Agent）：
    只读 SharedState 的最终结果做合并。LLM 失败时降级为规则拼接。
    """

    def __init__(self, client: AsyncAnthropic, model: str):
        self._client = client
        self._model  = model

    async def synthesize(
        self,
        req: Request,
        results: List[AgentResponse],
    ) -> str:
        parts = [
            (r.agent_type, r.content) for r in results
            if r.success and r.content and r.content != "（该领域助手处理失败）"
        ]
        if not parts:
            return "抱歉，多个助手模块暂时都没能处理成功，请稍后重试。"
        if len(parts) == 1:
            return parts[0][1]

        system = (
            "你是 EchoGuide 多 Agent 协作的合成器。把多个领域助手的回答合并成一段给用户的连贯回复："
            "去除重复内容，保留各自的有效信息与 [n] 引用标注，不要编造新的信息。"
            "如果某个领域回答是失败占位（如「处理失败」），直接忽略它。"
        )
        content = "\n\n".join(f"[{at.value}]\n{text}" for at, text in parts)
        from core.tracing import span

        try:
            async with span("synthesize", agents=",".join(at.value for at, _ in parts)):
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=system,
                    messages=[{
                        "role": "user",
                        "content": f"用户请求: {req.message}\n\n各领域助手回答:\n{content}",
                    }],
                )
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            if text:
                return text
        except Exception as ex:
            logger.warning(f"合成器调用失败，降级为规则拼接: {ex}")

        return self._merge_parts(parts)

    @staticmethod
    def _merge_parts(parts: List[Tuple[AgentType, str]]) -> str:
        """规则拼接（Synthesizer LLM 不可用时的兜底）。"""
        return "\n\n".join(f"[{at.value}]\n{text}" for at, text in parts)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用和统计。"""

    agent_type: AgentType
    system_prompt: str

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        tool_manager: Optional[Any] = None,
    ):
        self._client = client
        self._model  = model
        self._skill_manager = skill_manager
        self._tool_manager  = tool_manager
        self.stats   = AgentStats()

    async def handle(self, req: Request, on_event: Optional[Any] = None) -> AgentResponse:
        """
        处理一次请求：LLM 工具调用循环（Agentic RAG）。

        on_event: 可选异步回调，接收过程事件（meta/tool/delta），供 SSE 流式输出使用。
        """
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            from core.tracing import span

            async with span("agent_handle", agent=self.agent_type.value):
                content, tools_used = await self._call_llm(req, on_event=on_event)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            escalate = self._needs_escalation(content)
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                escalate=escalate,
                tools_used=tools_used,
            )
        except Exception as ex:
            ms = (time.monotonic() - t0) * 1000
            self.stats.total_ms += ms
            logger.error(f"{self.agent_type.value} 处理失败: {ex}")
            return AgentResponse(
                agent_type=self.agent_type,
                content="抱歉，处理你的请求时出现了问题，请稍后重试，或换个方式描述一下。",
                success=False,
                latency_ms=ms,
            )

    # ── Agentic RAG：LLM 工具调用循环 ────────────────────────────────────────

    MAX_TOOL_ROUNDS = 2  # 工具调用循环上限，防止死循环与成本失控

    def _build_tools(self) -> List[Dict[str, Any]]:
        """
        把 MCPToolManager 中注册的工具暴露给 LLM（function calling）。

        权限边界：只暴露本 Agent 职责内的工具（默认取 AGENT_TOOL_ALLOWLIST，
        实例可设置 _tool_allowlist 覆盖，供测试或定制场景使用），
        避免 Agent 拿满屏无关工具造成误调/重复执行。
        """
        if self._tool_manager is None:
            return []
        allowed = getattr(self, "_tool_allowlist", None)
        if allowed is None:
            allowed = AGENT_TOOL_ALLOWLIST.get(self.agent_type, set())
        tools = []
        for name, tool in self._tool_manager._tools.items():
            if not getattr(tool, "agent_exposed", True):
                continue
            if name not in allowed:
                continue  # 最小权限：职责外工具不暴露
            tools.append({
                "name": name,
                "description": tool.description,
                "input_schema": tool.schema,
            })
        return tools

    async def _call_llm(self, req: Request, on_event: Optional[Any] = None) -> tuple[str, List[str]]:
        """
        工具调用循环：
          1. 首次调用带 tools（function calling），让 Agent 自主决定是否检索知识库
          2. 返回 tool_use → 执行工具 → 把 tool_result 回填 → 再次调用
          3. 无工具请求或达到轮次上限 → 返回最终文本

        兼容性：若上游（如部分 DeepSeek 兼容端点）不支持 tools 参数，
        自动降级为普通调用，保证主链路可用。
        """
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        tools = self._build_tools()
        system = self._build_system_prompt(req)
        if tools:
            system = (
                f"{system}\n\n[工具使用]\n"
                "你可以调用 knowledge_search 工具检索校园知识库来回答事实性问题。"
                "检索到相关资料时，回答末尾用 [1][2] 标注引用来源；检索不到时如实说明，不要编造。"
            )

        tools_used: List[str] = []
        hit_tool_use_at_limit = False
        for _round in range(self.MAX_TOOL_ROUNDS):
            try:
                if on_event is not None:
                    resp = await self._stream_llm(system, messages, tools, on_event)
                else:
                    resp = await self._client.messages.create(
                        model=self._model,
                        max_tokens=1024,
                        system=system,
                        messages=messages,
                        tools=tools or None,
                    )
            except Exception as ex:
                # 上游不支持 tools → 降级为普通调用（不再带工具重试）
                if tools and _round == 0:
                    logger.warning(f"工具调用模式失败，降级为普通调用: {ex}")
                    tools = []
                    system = self._build_system_prompt(req)
                    if on_event is not None:
                        resp = await self._stream_llm(system, messages, [], on_event)
                    else:
                        resp = await self._client.messages.create(
                            model=self._model, max_tokens=1024,
                            system=system, messages=messages,
                        )
                else:
                    raise

            stop_reason = getattr(resp, "stop_reason", None)
            if stop_reason == "tool_use":
                tool_calls = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
                if not tool_calls:
                    break
                # 把 assistant 的 tool_use 消息回填，供下一轮携带
                messages.append({"role": "assistant", "content": resp.content})
                # 协议约束：一条 assistant 消息里的所有 tool_use，必须在同一条
                # 下一条 user 消息中全部回填 tool_result（逐条分开会 400）
                tool_results = []
                for block in tool_calls:
                    name = getattr(block, "name", "")
                    tool_input = getattr(block, "input", {}) or {}
                    tools_used.append(name)
                    if on_event is not None:
                        await on_event({"type": "tool", "name": name, "status": "start", "input": tool_input})
                    data, error = await self._execute_tool(name, tool_input, req)
                    if on_event is not None:
                        await on_event({
                            "type": "tool", "name": name, "status": "done",
                            "titles": self._tool_result_titles(data),
                        })
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": getattr(block, "id", ""),
                        "content": self._clean_text(data) if data is not None else f"工具执行失败: {error}",
                    })
                messages.append({"role": "user", "content": tool_results})
                hit_tool_use_at_limit = _round == self.MAX_TOOL_ROUNDS - 1
                continue

            # 正常结束：提取文本
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            return text, tools_used

        # 达到工具轮次上限仍有工具请求：用普通调用收尾，保证一定有最终答复
        if hit_tool_use_at_limit:
            logger.warning(f"{self.agent_type.value} 工具调用达到轮次上限，普通调用收尾")
            # 最后一条 assistant 消息可能带未回填的 tool_use：按协议补占位
            # tool_result（同一条 user 消息），否则 Anthropic 兼容端点会 400
            if messages and messages[-1].get("role") == "assistant":
                last_content = messages[-1].get("content")
                if isinstance(last_content, list):
                    pending = [b for b in last_content if getattr(b, "type", "") == "tool_use"]
                    if pending:
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "tool_result",
                                 "tool_use_id": getattr(b, "id", ""),
                                 "content": "工具调用轮次已达上限，未执行。"}
                                for b in pending
                            ],
                        })
            system = self._build_system_prompt(req)
            if on_event is not None:
                # 收尾调用同样逐 token 推送，避免"先蹦一句、再整段出现"的割裂体验
                resp = await self._stream_llm(system, messages, [], on_event)
            else:
                resp = await self._client.messages.create(
                    model=self._model, max_tokens=1024,
                    system=system, messages=messages,
                )
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            if text:
                return text, tools_used

        return "抱歉，处理超时或模型未返回有效内容，请稍后重试。", tools_used

    async def _stream_llm(self, system: str, messages: List[Dict], tools: List[Dict], on_event: Any):
        """
        流式调用：逐 token 通过 on_event({"type": "delta", "text": ...}) 推送，
        同时返回最终 Message 供工具循环判断 stop_reason。

        SDK 兼容说明：anthropic 的流式 API 在不同版本有差异，统一按两种形态适配：
          - 0.40（项目锁定版）：messages.stream() 同步返回 manager，__aenter__ 产出
            AsyncMessageStream，迭代得到 RawMessageStreamEvent（文本在
            content_block_delta.delta.text）。
          - 0.5x+：messages.stream() 为 async 方法（需 await），事件结构相同。
        """
        stream_cm = self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=tools or None,
        )
        if inspect.isawaitable(stream_cm):
            stream_cm = await stream_cm  # 新版 SDK：stream() 是 async 方法
        async with stream_cm as stream:
            async for chunk in stream:
                if getattr(chunk, "type", "") == "content_block_delta":
                    delta = getattr(chunk, "delta", None)
                    text = getattr(delta, "text", None)
                    if text:
                        await on_event({"type": "delta", "text": text})
            final = await stream.get_final_message()
        return final

    async def _execute_tool(self, name: str, params: Dict, req: Request) -> tuple[Any, Optional[str]]:
        """执行工具，返回 (结构化数据, 错误信息)。"""
        if self._tool_manager is None:
            return None, "工具管理器不可用"
        # 权限边界（防御纵深）：即使 LLM 声明了职责外工具，也直接拒绝执行
        allowed = getattr(self, "_tool_allowlist", None)
        if allowed is None:
            allowed = AGENT_TOOL_ALLOWLIST.get(self.agent_type, set())
        if name not in allowed:
            logger.warning(f"{self.agent_type.value} 尝试调用权限外工具 {name}，已拒绝")
            return None, f"工具 {name} 不在 {self.agent_type.value} Agent 权限范围内"
        from core.tracing import span

        async with span("tool_call", tool=name, query=str(params.get("query", ""))[:80]):
            result = await self._tool_manager.call(
                name,
                params,
                context={"agent_type": self.agent_type.value, "user_id": req.user_id},
                rerank_top_k=0,
            )
        if not result.success:
            return None, result.error or "工具执行失败"
        return result.data, None

    @staticmethod
    def _tool_result_titles(data: Any) -> List[str]:
        """从工具结果中提取标题（用于前端过程可视化）。"""
        if isinstance(data, list):
            return [str(item.get("title", "")) for item in data if isinstance(item, dict) and item.get("title")]
        return []

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        if not isinstance(value, str):
            import json as _json
            try:
                value = _json.dumps(value, ensure_ascii=False)
            except Exception:
                value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    def _build_system_prompt(self, req: Request) -> str:
        """
        组装 system prompt：
          1. Agent 静态角色定义
          2. 动态 Skills（业务 SOP，随请求热加载）
          3. 时间上下文（当前日期/星期/第几节/第几周）—— 所有 Agent 统一注入，
             是"今天有什么课""现在第几节"类问答的前提。
        """
        prompt = self.system_prompt
        if self._skill_manager is not None:
            skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value, req.history)
            if skill_prompt:
                prompt = f"{prompt}\n\n[动态 Skills]\n{skill_prompt}"

        from personal.time_context import build_time_context
        prompt = f"{prompt}\n\n{build_time_context()}"
        return prompt

    def _needs_escalation(self, content: str) -> bool:
        """检测 Agent 是否建议升级（简单关键词检测）。"""
        keywords = ["转人工", "找辅导员", "教务老师", "escalate", "无法处理", "建议联系"]
        return any(kw in content for kw in keywords)


class AcademicAgent(BaseAgent):
    """学业支持：选课、课表、考试、成绩、绩点、重修、转专业、保研。"""
    agent_type    = AgentType.ACADEMIC
    system_prompt = (
        "你是西电校园智慧助手（EchoGuide）的学业支持顾问。"
        "友好、简洁地回答西安电子科技大学学生的学业问题，包括选课、课表、考试安排、成绩与绩点、重修、转专业、保研等。"
        "回答基于西电教务规则和公开常识，步骤清晰、用语克制。"
        "不要编造具体分数、排名或个人成绩数据；涉及具体成绩或学籍操作时，提示学生前往教务系统或学院教务老师处确认。"
    )


class CampusLifeAgent(BaseAgent):
    """校园生活：宿舍、食堂、校车、校园卡、快递、水电、社团、运动。"""
    agent_type    = AgentType.CAMPUS_LIFE
    system_prompt = (
        "你是西电校园智慧助手（EchoGuide）的校园生活向导，熟悉西安电子科技大学南、北校区的日常生活信息。"
        "覆盖宿舍、食堂、校园穿梭车、校园卡充值与挂失、快递、水电、社团、运动场馆等问题。"
        "回答尽量给出位置（校区/楼栋）和时段，步骤清晰。"
        "不要编造精确的电话、价格或人员信息；涉及报修、补办等需现场办理的事项，指引用户到对应服务网点。"
    )


class AffairsAgent(BaseAgent):
    """校务咨询：校历、请假、奖学金、助学金、证明开具、办事流程、学费注册。"""
    agent_type    = AgentType.AFFAIRS
    system_prompt = (
        "你是西电校园智慧助手（EchoGuide）的校务咨询顾问，负责西安电子科技大学的校务办事指引。"
        "覆盖校历、请假流程、奖学金与助学金评定、各类证明开具、学籍注册、学费缴纳等事项。"
        "回答以办事流程、所需材料、办理地点和系统入口为主，清晰可执行。"
        "不要编造具体的截止日期、金额或审批结果；涉及实际审批的事项，提示以学院或学生处最新通知为准。"
    )


class ITHelpAgent(BaseAgent):
    """IT 助手：教务系统、校园网、VPN、邮箱、统一身份认证排障。"""
    agent_type    = AgentType.IT_HELP
    system_prompt = (
        "你是西电校园智慧助手（EchoGuide）的 IT 支持助手，帮助西安电子科技大学学生解决校园信息系统使用问题。"
        "覆盖教务系统、校园网、VPN、学校邮箱、统一身份认证的故障排查与配置指引。"
        "提供清晰的步骤化解决方案。遇到需要后台操作或账号重置的问题，说明需联系信息化建设处或网络中心处理。"
    )


class PersonalAgent(BaseAgent):
    """个人助理：我的课表、待办、考试/DDL、日程提醒（数据来自用户导入的个人数据中心）。"""
    agent_type    = AgentType.PERSONAL
    system_prompt = (
        "你是西电校园智慧助手（EchoGuide）的个人助理，帮助用户管理自己的课表、待办、考试与 DDL 安排。"
        "查询前先调用工具获取用户个人数据（query_schedule / query_todo / query_ddl / add_todo 等），"
        "不要凭记忆编造课程或待办。"
        "用户未导入课表时，引导其通过「我的课表」上传 .ics 文件或 JSON 课表。"
        "回答按时间组织，带上课时间与地点；涉及考试/DDL 时给出剩余天数。"
    )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    西电校园智慧助手的多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射（学业/生活/校务/IT 四大领域）
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 AcademicAgent（校园最高频诉求，承担通用接待）
    """

    # 领域 → Agent 类型的静态映射（路由表）。路由只看领域，动作只影响行为。
    _INTENT_ROUTING: Dict[IntentDomain, AgentType] = {
        IntentDomain.ACADEMIC:    AgentType.ACADEMIC,
        IntentDomain.CAMPUS_LIFE: AgentType.CAMPUS_LIFE,
        IntentDomain.AFFAIRS:     AgentType.AFFAIRS,
        IntentDomain.IT_HELP:     AgentType.IT_HELP,
        IntentDomain.PERSONAL:    AgentType.PERSONAL,
        # 领域 OTHER（问候/闲聊/无法判断）→ ACADEMIC（兜底接待）
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
        tool_manager: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._client = client      # 供合成器（Synthesizer）等直接调用 LLM
        self._model  = model
        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        # 轻量多 Agent 协作链：Planner（拆分）→ Executor（分波执行）→ SharedState → Synthesizer（合并）
        self._executor = TaskExecutor(self._run_task)
        self._synthesizer = Synthesizer(client, model)
        self._skill_manager = skill_manager
        self._tool_manager  = tool_manager

        # Agent 池：每种类型保持多个实例（水平扩展）。
        # 每类型 2 个实例让"性能路由 + 监控惩罚"闭环真实生效：
        # 实例 A 失败率高 → Monitor 施加惩罚 → _best_agent 切换到实例 B 接管。
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.ACADEMIC:    [AcademicAgent(client, model, skill_manager, tool_manager)
                                    for _ in range(2)],
            AgentType.CAMPUS_LIFE: [CampusLifeAgent(client, model, skill_manager, tool_manager)
                                    for _ in range(2)],
            AgentType.AFFAIRS:     [AffairsAgent(client, model, skill_manager, tool_manager)
                                    for _ in range(2)],
            AgentType.IT_HELP:     [ITHelpAgent(client, model, skill_manager, tool_manager)
                                    for _ in range(2)],
            AgentType.PERSONAL:    [PersonalAgent(client, model, skill_manager, tool_manager)
                                    for _ in range(2)],
        }

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
        IntentCategory.ESCALATION: IntentAction.ESCALATION,
    }

    async def run(self, req: Request, on_event: Optional[Any] = None) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别（领域×动作）→ 路由选 Agent → 执行 → 检查升级 → 返回结果

        on_event: 可选异步回调，透传给 Agent（SSE 流式输出 / 工具调用可视化）。
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.domain is None:
            if req.intent is not None:
                # 兼容：只有旧版单维 intent 时，推导 domain/action，避免重复调用 LLM
                req.domain = self._CATEGORY_TO_DOMAIN.get(req.intent, IntentDomain.OTHER)
                req.action = self._CATEGORY_TO_ACTION.get(req.intent, IntentAction.OTHER)
            else:
                intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
                req.domain  = intent_result.domain
                req.action  = intent_result.action
                req.intent  = intent_result.intent
                req.urgency = intent_result.urgency
                if on_event is not None:
                    await on_event({
                        "type": "meta",
                        "domain": req.domain.value if req.domain else "other",
                        "action": req.action.value if req.action else "other",
                        "confidence": intent_result.confidence,
                    })

        # 复杂问题自动并行协作，例如同一句同时涉及教务系统故障和选课问题。
        collaboration = self._collaboration_targets(req)
        if len(collaboration) > 1:
            return await self.run_parallel(req, collaboration, on_event)

        # 2. 路由：按领域选择 Agent 类型
        agent_type = self._route(req.domain, req.urgency)
        if on_event is not None:
            await on_event({"type": "meta", "agent": agent_type.value})

        # 3. 执行（含降级）
        response = await self._execute(req, agent_type, on_event)

        # 4. 升级检查（动作维度转人工 / 紧急度 CRITICAL / 响应建议升级）
        escalated = False
        if (
            response.escalate
            or req.urgency == UrgencyLevel.CRITICAL
            or req.action == IntentAction.ESCALATION
        ):
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: domain={req.domain} action={req.action} urgency={req.urgency}")
            # 生产环境：此处创建工单、通知辅导员/教务老师

        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            domain=req.domain,
            action=req.action,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            tools_used=response.tools_used,
        )

    async def run_parallel(
        self,
        req: Request,
        agent_types: List[AgentType],
        on_event: Optional[Any] = None,
    ) -> OrchestratorResult:
        """
        轻量多 Agent 协作（非 Agent 间聊天），职责链清晰：

          Planner 拆分任务 DAG（自包含任务，支持跨任务依赖）
            → Executor 按 depends_on 分波并行执行，结果写入 SharedState，
              依赖任务执行时注入协作上下文（能看到前序 Agent 结果）
            → Synthesizer 读取 SharedState 合并为最终回复（LLM 失败降级拼接）

        OrchestratorResult 字段保持兼容。
        """
        t0 = time.monotonic()

        # 1. Planner：拆分为任务 DAG（规则命中 → 依赖链；否则领域并行）
        tasks = TaskPlanner().plan(req, agent_types)

        # 2. Executor：分波执行，产出 SharedState
        shared = await self._executor.execute(req, tasks, on_event)

        # 3. Synthesizer：合并最终回复
        final_text = await self._synthesizer.synthesize(
            req, list(shared.all_results().values()),
        )

        escalated = any(r.escalate for r in shared.all_results().values())
        tools_used = [
            tool
            for r in shared.all_results().values()
            for tool in r.tools_used
        ]

        return OrchestratorResult(
            request_id=req.request_id,
            response=final_text,
            agent_type=agent_types[0],
            intent=req.intent,
            domain=req.domain,
            action=req.action,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            tools_used=tools_used,
        )

    # ── 协作任务执行 ──────────────────────────────────────────────────────────

    async def _run_task(
        self,
        req: Request,
        task: Task,
        shared: SharedState,
        on_event: Optional[Any] = None,
    ) -> AgentResponse:
        """
        执行单个协作任务：自包含 message + 协作上下文（SharedState 快照）。
        依赖任务能看到前序 Agent 已给出的结果（真正使用 SharedState）。
        """
        task_req = Request(
            message=task.message,
            user_id=req.user_id,
            conv_id=req.conv_id,
            context=req.context,
            history=req.history,
            intent=req.intent,
            domain=req.domain,
            action=req.action,
            urgency=req.urgency,
            request_id=req.request_id,
        )
        snapshot = shared.snapshot()
        if snapshot:
            # 注入协作上下文：让本任务知道其他 Agent 已给出什么（避免重复检索/重复回答）
            task_req.context = f"{task_req.context}\n\n[协作上下文]\n{snapshot}".strip()
        return await self._execute(task_req, task.agent_type, on_event)

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, domain: Optional[IntentDomain], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 领域映射（学业/生活/校务/IT 四大领域）—— 路由只看领域，动作不参与
          2. 紧急度覆盖（CRITICAL 直接升级到转人工）
          3. 默认 ACADEMIC（校园最高频诉求，承担通用接待）
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if domain and domain in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[domain]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.ACADEMIC

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        领域关键词统一来自 core.domains.DOMAIN_KEYWORDS（单一事实来源，
        与意图识别器、API 层共用，消除旧版三处重复维护的漂移问题）。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        def hit(domain: IntentDomain) -> bool:
            return any(keyword_hit(kw, msg) for kw in DOMAIN_KEYWORDS.get(domain, []))

        if req.domain == IntentDomain.ACADEMIC or hit(IntentDomain.ACADEMIC):
            targets.append(AgentType.ACADEMIC)
        if req.domain == IntentDomain.CAMPUS_LIFE or hit(IntentDomain.CAMPUS_LIFE):
            targets.append(AgentType.CAMPUS_LIFE)
        if req.domain == IntentDomain.AFFAIRS or hit(IntentDomain.AFFAIRS):
            targets.append(AgentType.AFFAIRS)
        if req.domain == IntentDomain.IT_HELP or hit(IntentDomain.IT_HELP):
            targets.append(AgentType.IT_HELP)
        if req.domain == IntentDomain.PERSONAL or hit(IntentDomain.PERSONAL):
            targets.append(AgentType.PERSONAL)

        # personal（"我的"日程）语义比 academic（教务规则）更具体：
        # "考试安排/课表"类词会同时命中两个领域，此时只保留 personal，
        # 避免"查我的考试"被并行拆给两个 Agent 造成回答分裂。
        if AgentType.ACADEMIC in targets and AgentType.PERSONAL in targets:
            targets.remove(AgentType.ACADEMIC)

        # 保持顺序去重，并只返回当前有实例的 Agent 类型。
        deduped = list(dict.fromkeys(targets))
        return [agent_type for agent_type in deduped if self._pool.get(agent_type)]

    def _best_agent(self, agent_type: AgentType) -> Optional[BaseAgent]:
        """
        性能路由：从同类 Agent 中选 routing_score() 最高的。
        这是"基于在线表现动态调整路由"的核心。
        """
        agents = self._pool.get(agent_type, [])
        if not agents:
            return None
        return max(agents, key=lambda a: a.stats.routing_score())

    async def _execute(
        self,
        req: Request,
        agent_type: AgentType,
        on_event: Optional[Any] = None,
    ) -> AgentResponse:
        """执行 Agent，失败时降级到 AcademicAgent（校园通用接待）。"""
        agent = self._best_agent(agent_type)
        if agent is None:
            agent = self._best_agent(AgentType.ACADEMIC)
        if agent is None:
            return AgentResponse(
                agent_type=AgentType.ACADEMIC,
                content="助手暂时不可用，请稍后重试，或直接联系辅导员/教务老师。",
                success=False,
            )

        response = await agent.handle(req, on_event=on_event)

        # 专属 Agent 失败时降级到 AcademicAgent
        if not response.success and agent_type != AgentType.ACADEMIC:
            logger.warning(f"{agent_type.value} 失败，降级到 AcademicAgent")
            fallback = self._best_agent(AgentType.ACADEMIC)
            if fallback:
                response = await fallback.handle(req, on_event=on_event)

        return response

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                result[key] = {
                    "total":        agent.stats.total,
                    "success_rate": round(agent.stats.success_rate, 3),
                    "avg_ms":       round(agent.stats.avg_ms, 1),
                    "monitor_penalty": round(agent.stats.monitor_penalty, 3),
                    "routing_score": round(agent.stats.routing_score(), 3),
                }
        return result

    def update_routing_penalties(self, penalties: Dict[str, float]) -> None:
        """
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 technical_0。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, 0.0)
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
