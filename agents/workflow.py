"""
轻量多任务协作（复杂任务编排）—— ExecutionPlan / Task / Planner / Executor / Synthesizer。

决策闭环：复杂度判定从 IntentRecognizer 移入 Planner——
Intent 只负责理解用户想做什么（domain/action），Planner 统一输出
ExecutionPlan（Task DAG），single/parallel/dependent 由最终 DAG 自动推导：

    1 个 task            → single
    多个无依赖 task       → parallel
    存在 depends_on       → dependent

Planner 两条路径，但都输出统一 ExecutionPlan：
  - Fast Path（本地规则，零 LLM）：明显单任务直接生成 1 个 Task；命中复合
    规则生成依赖链；多领域 + 连接词生成并行任务（每个任务用 _task_action
    推导自己的 action，不继承顶层 action）。
  - LLM 规划（升级路径）：本地判单任务但"拿不准"（多从句/长句/多领域无
    连接词/复合形态含写操作词即 QUERY+REQUEST 混合）→ 一次轻量 LLM 调用
    输出任务链（含每个任务的 action 与 tools），硬校验后采用，非法回落本地。

每个 Task 携带自己的 action（QUERY→READ_ONLY / REQUEST→WRITE_ALLOWED，
见 roles.write_policy_for），不再继承原始请求的 action —— 复合请求拆分后
t1/t2 可能是 QUERY、t3 才是 REQUEST。REQUEST 任务还带 allowed_write_tools
（任务级写工具白名单，最小权限；读工具不受限）：规则链显式声明、写操作词族
提示兜底、LLM 规划可输出，执行侧与 Action 策略、注册级声明取交集。

Task 才是真正的 Agent 边界（Task-scoped SubAgent）：每个 Task 由一次独立
TaskAgent Run 执行，拥有独立 goal / message / domain / action / depends_on
/ allowed_write_tools 与协作上下文；领域（domain）只做人格挂载键，
不参与 Agent 类选择，也不过滤 Skill（Skill 平级发现）。

确定性规则链（_plan_schedule_errand 等）只是**高频业务场景 Fast Path**：
只有"高频、稳定、确定性强、用规则明显比 LLM 更可靠"才加规则；通用复杂
请求（含 QUERY+REQUEST 混合、多个副作用任务、子任务权限不同）由 LLM 规划
显式生成各 Task 的 action / tools，输出后硬校验，不能完全相信 LLM。

DAG 失败传播：任务状态 SUCCESS / FAILED / BLOCKED / SKIPPED。
依赖任务 FAILED/BLOCKED → 下游任务 BLOCKED（不执行、不注入上下文），
不能因为前置"执行完成（但失败）"就继续执行依赖任务。

SharedState 只保存 Task result/status/meta 与必要的依赖快照；后续 Task
只能读取自己声明的 depends_on 结果，不默认注入全部历史任务内容
（避免上下文膨胀与不必要耦合）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from anthropic import AsyncAnthropic

from core.domains import (
    ACTION_KEYWORDS, DOMAIN_KEYWORDS, IntentAction, IntentDomain, keyword_hit,
)
from agents.roles import AgentResponse

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request

logger = logging.getLogger(__name__)

# Task 执行状态（DAG 失败传播）
TASK_SUCCESS = "success"
TASK_FAILED = "failed"
TASK_BLOCKED = "blocked"
TASK_SKIPPED = "skipped"


@dataclass
class Task:
    """多 Agent 协作中的最小执行单元（自包含：不依赖原始对话上下文）。

    domain 是领域挂载键（人格语境；Skill 平级发现，不按领域过滤），不参与 Agent 类选择 ——
    真正的 Agent 单位是围绕本 Task 的一次 TaskAgent Run。
    """
    task_id:     str
    domain:      IntentDomain             # 任务领域（人格/上下文挂载键）
    goal:        str                      # 领域化任务目标（给 Agent 的指令）
    message:     str                      # 自包含请求内容
    action:      IntentAction = IntentAction.QUERY  # 任务自己的动作（决定 Run 执行策略）
    depends_on:  List[str] = field(default_factory=list)  # 依赖的其他 task_id
    # 任务级写工具能力（最小权限）：None = 不做任务级写限制（仍受 Action 策略约束）；
    # 非空 = 本任务只能调用列表内的写工具（读工具不受限，与 Action 策略取交集）。
    # 由 Planner 声明（规则链显式声明、写操作词族提示兜底、LLM 规划可输出）。
    allowed_write_tools: Optional[List[str]] = None
    # 后置条件：本任务应落地的写操作（模型忘记调用工具时由执行器补执行）。
    # 由 Planner 声明，避免执行器硬编码任务标识。
    required_tool: Optional[str] = None
    required_tool_args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Planner 统一输出：任务 DAG + 推导出的复杂度模式。"""
    tasks: List[Task]
    reason: str = ""

    @property
    def mode(self) -> str:
        """复杂度模式由最终 DAG 自动推导（不再由 Intent/LLM 单独分类）。"""
        if len(self.tasks) <= 1:
            return "single"
        if any(t.depends_on for t in self.tasks):
            return "dependent"
        return "parallel"


class SharedState:
    """协作共享状态：记录每个 Task 的结果与执行状态，供依赖任务读取。"""

    def __init__(self) -> None:
        self._results: Dict[str, AgentResponse] = {}
        self._status: Dict[str, str] = {}
        self._task_meta: Dict[str, Dict[str, Any]] = {}

    def set_result(self, task_id: str, resp: AgentResponse, status: str = TASK_SUCCESS) -> None:
        self._results[task_id] = resp
        self._status[task_id] = status

    def get_result(self, task_id: str) -> Optional[AgentResponse]:
        return self._results.get(task_id)

    def status(self, task_id: str) -> str:
        """任务状态：success / failed / blocked / skipped / 未执行（pending）。"""
        return self._status.get(task_id, "pending")

    def done(self, task_id: str) -> bool:
        """依赖是否可继续：只有 SUCCESS 才视为依赖满足（失败/阻塞不算）。"""
        return self._status.get(task_id) == TASK_SUCCESS

    def all_results(self) -> Dict[str, AgentResponse]:
        return dict(self._results)

    def set_task_meta(
        self,
        task: Task,
        status: str,
        duration_ms: float = 0.0,
        response: Optional[AgentResponse] = None,
    ) -> None:
        """Task 级 Trace：DAG、状态、执行配置、工具与耗时（前端过程可视化/Monitor）。"""
        self._task_meta[task.task_id] = {
            "id": task.task_id,
            "domain": task.domain.value,
            "action": task.action.value,
            "goal": task.goal,
            "depends_on": list(task.depends_on),
            "status": status,
            "duration_ms": round(duration_ms, 1),
            "profile": response.profile if response else "",
            "tools": list(response.tools_used) if response else [],
            "input_tokens": response.input_tokens if response else 0,
            "output_tokens": response.output_tokens if response else 0,
        }

    def task_meta(self) -> List[Dict[str, Any]]:
        return list(self._task_meta.values())

    def snapshot_for(self, task: Task) -> str:
        """只把本任务声明的依赖（depends_on）中 SUCCESS 的结果序列化，注入协作上下文。

        不默认注入全部历史任务内容（避免上下文膨胀与不必要耦合）；
        失败/阻塞任务不注入 —— 依赖任务不能把失败结果当成有效上下文。
        """
        deps = [
            dep for dep in task.depends_on
            if self._status.get(dep) == TASK_SUCCESS and dep in self._results
        ]
        if not deps:
            return ""
        return "\n\n".join(f"[{dep}]\n{self._results[dep].content}" for dep in deps)


class TaskPlanner:
    """
    统一 Task Planner：任何请求都输出 ExecutionPlan（Task DAG）。

    Fast Path（本地规则，零 LLM）：
      1. 单任务：直接生成 1 个 Task（domain/action 来自意图识别）；
      2. 复合规则（RULES）：命中则生成依赖任务链（后续任务 depends_on 前序，
         执行时从 SharedState 读取前序结果）；
      3. 多领域 + 显式连接词：每个领域一个并行任务。

    LLM 规划（升级路径）：
      本地判 single 但"拿不准"（多从句/长句/多领域无连接词）→ 一次轻量
      LLM 调用输出任务链（含每个任务的 action），硬校验后采用；LLM 不可用
      或输出非法 → 回落本地 Fast Path 结果（行为不比现状差）。
    """

    GOAL_TEMPLATES: Dict[IntentDomain, str] = {
        IntentDomain.ACADEMIC:    "从学业支持角度回答用户的请求（选课/课表/考试/成绩等）",
        IntentDomain.CAMPUS_LIFE: "从校园生活角度回答用户的请求（宿舍/食堂/校车/天气等）",
        IntentDomain.AFFAIRS:     "从校务办事角度回答用户的请求（校历/请假/奖学金/证明等）",
        IntentDomain.IT_HELP:     "从 IT 支持角度回答用户的请求（教务系统/校园网/VPN/邮箱等）",
        IntentDomain.PERSONAL:    "从个人助理角度回答用户的请求（我的课表/待办/考试安排等）",
    }

    # 写工具提示（Fast Path 最小权限声明）：REQUEST 任务按写操作词族声明必要
    # 工具，避免"因为 REQUEST 就拿到所有写工具"。多词族命中取并集；未命中
    # （语义不明）不做任务级限制，由 Action 策略兜底。
    _WRITE_TOOL_HINTS: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
        (("添加", "新增", "记一下", "记个", "提醒我", "帮我记", "创建", "设个提醒", "定个提醒", "设置提醒"), ("add_todo",)),
        (("标记完成", "标为完成", "设为完成", "勾选完成", "标成完成"), ("complete_todo",)),
    )

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "",
        gateway: Optional[Any] = None,
        max_tasks: int = 6,
        max_agents: int = 3,
        tool_names: Optional[Set[str]] = None,
    ):
        # LLM 规划能力（可选注入）：client/model/gateway 由编排器传入；
        # 不注入时 Planner 只走 Fast Path（单任务/规则链/并行）。
        self._client = client
        self._model = model
        self._gateway = gateway
        self._max_tasks = max_tasks
        self._max_agents = max_agents
        # 注册工具名集合（LLM 规划的 tools 字段硬校验；None = 不做名称校验）
        self._tool_names = tool_names

    def set_tool_names(self, tool_names: Optional[Set[str]]) -> None:
        """注入注册工具名集合（编排器 set_tool_manager 时同步更新）。"""
        self._tool_names = tool_names

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def plan(self, req: "Request", domain: IntentDomain, action: IntentAction) -> ExecutionPlan:
        """
        生成 ExecutionPlan：Fast Path 先判；判 single 但"拿不准"时 LLM 规划升级。
        返回前对 REQUEST 任务做最小权限兜底（写工具提示，见 _apply_write_hints）。
        """
        fast = self._fast_plan(req, domain, action)
        if fast.mode != "single" or not self._needs_llm_planning(req):
            return self._apply_write_hints(fast)
        llm_plan = await self._llm_plan(req)
        # 单任务 LLM 规划只能细化目标，不能推翻已完成的顶层意图识别。
        # 否则规划器偶发的领域漂移会让执行器选错领域人格和结构化工具，出现
        # "顶层识别为校务、实际按校园生活执行" 这类不可观测的执行偏差。
        # 多任务计划允许各子任务拥有不同领域，仍由其 DAG 结构表达协作关系。
        if (
            llm_plan is not None
            and (
                llm_plan.mode != "single"
                or (
                    llm_plan.tasks[0].domain == domain
                    and llm_plan.tasks[0].action == action
                )
            )
        ):
            return self._apply_write_hints(llm_plan)
        if llm_plan is not None:
            logger.warning(
                "LLM 单任务规划与顶层意图不一致，回落 Fast Path: "
                "root=%s/%s planned=%s/%s",
                domain.value,
                action.value,
                llm_plan.tasks[0].domain.value,
                llm_plan.tasks[0].action.value,
            )
        return self._apply_write_hints(fast)  # LLM 不可用/输出非法：回落本地（行为不比现状差）

    # ── Fast Path（本地规则）────────────────────────────────────────────────

    def _fast_plan(self, req: "Request", domain: IntentDomain, action: IntentAction) -> ExecutionPlan:
        """本地规则生成：规则链 → 并行 → 单任务。零 LLM 调用。"""
        # 1. 复合规则命中 → 依赖任务链
        rule_tasks = self._apply_rules(req)
        if rule_tasks is not None:
            return ExecutionPlan(rule_tasks, reason="命中复合规则（存在前后依赖）")

        # 2. 多领域 + 显式连接词 → 并行任务（每个任务自己的 action，
        #    不继承顶层 action：写操作词 + 个人领域才是 REQUEST，见 _task_action）
        targets = self._collaboration_targets(req, domain)
        connectors = ("同时", "还要", "并且", "另外", "以及", "顺便", "然后")
        if len(targets) >= 2 and any(word in req.message for word in connectors):
            tasks = [
                Task(
                    task_id=f"t{i}",
                    domain=at,
                    goal=self.GOAL_TEMPLATES.get(at, "回答用户的请求"),
                    message=f"{self.GOAL_TEMPLATES.get(at, '回答用户的请求')}。\n用户请求: {req.message}",
                    action=self._task_action(req.message, at),
                )
                for i, at in enumerate(targets[: self._max_agents])
            ]
            return ExecutionPlan(tasks, reason="显式复合语义涉及多个校园领域")

        # 3. 单任务（绝大多数请求）
        goal = self.GOAL_TEMPLATES.get(domain, "回答用户的请求")
        return ExecutionPlan(
            [Task(task_id="t0", domain=domain, goal=goal, message=req.message, action=action)],
            reason="单领域请求",
        )

    # ── 复合规则 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _task_action(message: str, domain: IntentDomain) -> IntentAction:
        """复合请求中单个任务的 action（确定性判定，不继承顶层 action）。

        Fast Path 兜底口径：写操作词 + 个人领域 → REQUEST（本系统写工具均为
        个人数据操作）；其余一律 QUERY。拿不准的混合请求优先升级 LLM 规划
        （_needs_llm_planning），这里只保证回落路径不把查询任务误开写权限
        （fail-closed：非 REQUEST 一律只读）。
        """
        msg = (message or "").lower()
        if domain == IntentDomain.PERSONAL and any(
            keyword_hit(kw, msg) for kw in ACTION_KEYWORDS.get(IntentAction.REQUEST, [])
        ):
            return IntentAction.REQUEST
        return IntentAction.QUERY

    @classmethod
    def _write_tool_hint(cls, message: str) -> Optional[List[str]]:
        """REQUEST 任务的最小权限声明：消息命中哪个写操作词族就只给哪个写工具。

        多词族命中取并集（如"添加待办并标记完成"→ add_todo + complete_todo）；
        无命中返回 None（语义不明，不做任务级限制，Action 策略兜底）。
        """
        msg = (message or "").lower()
        tools: List[str] = []
        for kws, names in cls._WRITE_TOOL_HINTS:
            if any(keyword_hit(kw, msg) for kw in kws):
                tools.extend(names)
        return list(dict.fromkeys(tools)) or None

    def _apply_write_hints(self, plan: ExecutionPlan) -> ExecutionPlan:
        """REQUEST 任务最小权限兜底：Planner 未显式声明写能力时按写操作词族声明。

        规则链/LLM 规划已显式声明 allowed_write_tools 的任务不覆盖；
        例如"帮我添加一个待办"→ 只给 add_todo，不因 REQUEST 获得全部写工具。
        """
        for task in plan.tasks:
            if task.action == IntentAction.REQUEST and not task.allowed_write_tools:
                hinted = self._write_tool_hint(task.message)
                if hinted:
                    task.allowed_write_tools = hinted
        return plan

    @staticmethod
    def _extract_errand_content(msg: str) -> str:
        """从用户请求中提取办事内容（"办校园卡" → "校园卡"），避免硬编码业务。"""
        m = re.search(r"(?:办|办理|补办|申请|搞)([\u4e00-\u9fff]{2,8})", msg)
        return m.group(1) if m else "校园事务"

    @classmethod
    def _plan_schedule_errand(cls, req: "Request") -> Optional[List[Task]]:
        """
        高频业务场景 Fast Path（确定性规则，非 DAG 能力的主要证明）：
        个人日程 + 线下办事 + 记待办 的复合请求。

        例："我明天下午有空，想去办校园卡，帮我记个待办"
          t1 查课表（personal, QUERY）→ t2 查办理信息（affairs, QUERY）
          → t3 创建待办（personal, REQUEST，depends_on=[t1,t2]）
        每个任务有自己的 action：t1/t2 是 QUERY（READ_ONLY），
        t3 是 REQUEST（WRITE_ALLOWED）且只声明 add_todo 能力（最小权限）。
        只有"高频、稳定、确定性强、用规则明显比 LLM 更可靠"的场景才加这类
        规则；通用复杂请求由 Planner/LLM 规划负责（输出后仍有硬校验）。
        """
        msg = req.message
        has_schedule = any(keyword_hit(kw, msg) for kw in ("课表", "课程", "空闲", "上课", "没课", "有空"))
        has_errand   = any(keyword_hit(kw, msg) for kw in ("校园卡", "办理", "材料", "缴费", "办证"))
        has_todo     = any(keyword_hit(kw, msg) for kw in ("待办", "提醒", "记一下", "安排上", "安排"))
        if not (has_schedule and has_errand and has_todo):
            return None
        content = cls._extract_errand_content(msg)
        return [
            Task(
                task_id="t1",
                domain=IntentDomain.PERSONAL,
                action=IntentAction.QUERY,
                goal="查询课程空闲时间",
                message=f"查询用户课程/空闲时间（如明天下午是否有课）。用户请求: {msg}",
            ),
            Task(
                task_id="t2",
                domain=IntentDomain.AFFAIRS,
                action=IntentAction.QUERY,
                goal="查询校园卡办理信息",
                message=f"查询校园卡办理地点和所需材料。用户请求: {msg}",
            ),
            Task(
                task_id="t3",
                domain=IntentDomain.PERSONAL,
                action=IntentAction.REQUEST,
                goal="创建校园卡办理待办",
                message=(
                    "根据协作上下文中的课程空闲时间和校园卡办理信息，"
                    "为用户创建一个合适的办理待办/提醒（时间安排在空闲时段）。"
                    f"用户请求: {msg}"
                ),
                depends_on=["t1", "t2"],
                # 任务级最小权限：本任务只允许调用 add_todo 这个写工具
                # （读工具不受限），不因 REQUEST 获得其他写工具。
                allowed_write_tools=["add_todo"],
                required_tool="add_todo",
                required_tool_args={"content": f"办理{content}", "kind": "todo"},
            ),
        ]

    RULES = ["_plan_schedule_errand"]

    def _apply_rules(self, req: "Request") -> Optional[List[Task]]:
        for rule_name in self.RULES:
            rule = getattr(self, rule_name)
            tasks = rule(req)
            if tasks:
                return tasks
        return None

    # ── 并行目标（多领域判定）───────────────────────────────────────────────

    def _collaboration_targets(self, req: "Request", domain: IntentDomain) -> List[IntentDomain]:
        """多领域目标判定：领域关键词统一来自 core.domains.DOMAIN_KEYWORDS。"""
        msg = req.message.lower()
        targets: List[IntentDomain] = []

        def hit(d: IntentDomain) -> bool:
            return any(keyword_hit(kw, msg) for kw in DOMAIN_KEYWORDS.get(d, []))

        if domain == IntentDomain.ACADEMIC or hit(IntentDomain.ACADEMIC):
            targets.append(IntentDomain.ACADEMIC)
        if domain == IntentDomain.CAMPUS_LIFE or hit(IntentDomain.CAMPUS_LIFE):
            targets.append(IntentDomain.CAMPUS_LIFE)
        if domain == IntentDomain.AFFAIRS or hit(IntentDomain.AFFAIRS):
            targets.append(IntentDomain.AFFAIRS)
        if domain == IntentDomain.IT_HELP or hit(IntentDomain.IT_HELP):
            targets.append(IntentDomain.IT_HELP)
        if domain == IntentDomain.PERSONAL or hit(IntentDomain.PERSONAL):
            targets.append(IntentDomain.PERSONAL)

        # personal（"我的"日程）语义比 academic（教务规则）更具体
        if IntentDomain.ACADEMIC in targets and IntentDomain.PERSONAL in targets:
            targets.remove(IntentDomain.ACADEMIC)

        return list(dict.fromkeys(targets))

    # ── LLM 规划升级（"拿不准"时）──────────────────────────────────────────

    # 升级预筛的从句切分器
    _UPGRADE_CLAUSE_RE = re.compile(
        r"[，。；、,;]+|(?:同时|另外|顺便|然后|再|还要|以及|并且)"
    )

    def _needs_llm_planning(self, req: "Request") -> bool:
        """
        升级预筛：本地判 single 后，判断这条请求是否值得升级 LLM 规划。

        注意前提：本函数只在 fast.mode == "single" 时被调用（多领域 + 显式
        连接词时 Fast Path 已生成并行任务，直接返回，不升级）。任一信号命中即升级：
          1. 消息被切出 ≥3 个从句（信息量大，可能复合）；
          2. 长消息（>24 字）且 ≥2 个从句；
          3. 领域关键词命中 ≥2 个领域（无显式连接词时 Fast Path 只能判
             single，即"隐式复合"）；
          4. 恰好 2 个从句（≥3 已被第 1 条覆盖）且含写操作词 —— QUERY+
             REQUEST 混合 / 多个副作用任务的信号：子任务 action / 写工具
             权限可能不同，让 LLM 显式生成各任务的 action 与 tools；
             确定性兜底（_task_action）只保证回落路径不把查询任务误开写权限。
        """
        msg = req.message
        clauses = [c for c in self._UPGRADE_CLAUSE_RE.split(msg) if c.strip()]
        if len(clauses) >= 3:
            return True
        if len(msg) > 24 and len(clauses) >= 2:
            return True
        lowered = msg.lower()
        hit_domains = {
            domain for domain, kws in DOMAIN_KEYWORDS.items()
            if any(keyword_hit(kw, lowered) for kw in kws)
        }
        if len(hit_domains) >= 2:
            return True
        if len(clauses) >= 2 and any(
            keyword_hit(kw, lowered) for kw in ACTION_KEYWORDS.get(IntentAction.REQUEST, [])
        ):
            return True
        return False

    async def _llm_plan(self, req: "Request") -> Optional[ExecutionPlan]:
        """一次轻量 LLM 调用输出任务链（含每个任务的 action），硬校验后采用。

        失败/非法 → None（调用方回落本地 Fast Path）。
        """
        if self._client is None or not self._model:
            return None
        prompt = (
            "你是 EchoGuide 的任务规划器。判断用户请求是否需要拆分为多个子任务执行，"
            "并输出任务链。普通单领域问题输出 1 个任务；涉及多个诉求/条件/依赖时"
            "拆分为多个任务，有先后依赖的任务用 depends_on 表达。\n"
            "任务字段：\n"
            '  {"id": "t1", "domain": "<领域值>", "action": "<query/request>", '
            '"goal": "<任务目标>", "message": "<给该子任务的独立请求>", '
            '"depends_on": ["<前置任务id>"], "tools": ["<允许调用的工具名>"]}\n'
            "- domain 可选值: academic, campus_life, affairs, it_help, personal\n"
            "- action: query=查询咨询；request=需要系统写数据/产生副作用（创建待办等）\n"
            "- 只有明确需要写操作的任务才是 request，查询类任务一律 query\n"
            "- tools 可选：本任务允许调用的工具名数组（最小权限，如 [\"add_todo\"]）；"
            "查询类任务不写 tools 即可\n"
            "- depends_on 引用前面已定义任务的 id；无前置依赖省略或为空数组\n"
            f"用户消息: {req.message!r}\n\n"
            "返回格式（仅 JSON，不要其他文字）:\n"
            '{"tasks": [<任务>...], "reason": "<一句话规划理由>"}'
        )
        try:
            if self._gateway is not None:
                result = await self._gateway.call(
                    client=self._client,
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    state=req.state,
                    span_name="planner_llm",
                    max_tokens=512,
                    temperature=0.1,
                    thinking={"type": "disabled"},
                )
                resp = result.response
            else:
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=512,
                    temperature=0.1,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}],
                )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            tasks = self._tasks_from_llm(data.get("tasks"), req)
            if tasks is None:
                return None
            reason = str(data.get("reason") or "")[:120] or "LLM 规划任务链"
            return ExecutionPlan(tasks, reason=reason)
        except Exception as ex:
            logger.warning(f"LLM 规划失败，回落本地规则: {ex}")
            return None

    def _tasks_from_llm(
        self,
        raw_tasks: Optional[Any],
        req: "Request",
    ) -> Optional[List[Task]]:
        """
        LLM 任务链硬校验（任一不满足 → None 整链作废）：

          - 1~max_tasks 个任务；id 非空且唯一
          - domain 必须是已知领域（不含 OTHER）
          - action 必须是合法动作（默认 QUERY）
          - depends_on 引用的 id 必须存在，且无环（拓扑检查）
          - tools（可选）：必须是字符串数组；注入过工具名集合时，引用未注册
            工具 → 整链作废（fail-closed，不能完全相信 LLM）
          - message 缺失时回落自包含格式（含原始用户请求）
          - required_tool 一律不采用：后置条件是关键词规则链的保险带
        """
        if not isinstance(raw_tasks, list) or not (1 <= len(raw_tasks) <= self._max_tasks):
            return None
        tasks: List[Task] = []
        seen_ids: Set[str] = set()
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                return None
            task_id = str(raw.get("id") or "").strip()
            if not task_id or task_id in seen_ids:
                return None
            try:
                domain = IntentDomain(str(raw.get("domain") or ""))
            except ValueError:
                return None
            if domain == IntentDomain.OTHER:
                return None
            try:
                action = IntentAction(str(raw.get("action") or "query"))
            except ValueError:
                return None
            # 当前系统的写工具全部作用于用户个人数据（待办/日程）。LLM 若把
            # "创建提醒"这类 REQUEST 归到校务等知识领域，运行期会因能力边界
            # 拒绝写入；在规划边界统一归正到 PERSONAL，而不是按某个业务词补丁。
            if action == IntentAction.REQUEST:
                domain = IntentDomain.PERSONAL
            goal = str(raw.get("goal") or "").strip() or self._task_goal_fallback(domain)
            message = str(raw.get("message") or "").strip()
            if not message:
                message = f"{goal}。用户请求: {req.message}"
            depends = raw.get("depends_on") or []
            if isinstance(depends, str):
                depends = [depends]
            if not isinstance(depends, list) or not all(
                isinstance(d, str) and d.strip() for d in depends
            ):
                return None
            deps = list(dict.fromkeys(d.strip() for d in depends))
            # 任务级写工具能力（最小权限）：LLM 声明了 tools 就必须合法；
            # 名称集合不可用时不校验名称（运行期 Agent 门禁仍会按实际注册表拦截）。
            # 语义为"允许的写工具白名单"：读工具不受限，非写工具名在其中无副作用。
            raw_tools = raw.get("tools")
            allowed_write_tools: Optional[List[str]] = None
            if raw_tools is not None:
                if not isinstance(raw_tools, list) or not all(
                    isinstance(t, str) and t.strip() for t in raw_tools
                ):
                    return None
                if self._tool_names is not None and any(
                    t not in self._tool_names for t in raw_tools
                ):
                    return None  # 引用了未注册工具 → 整链作废（fail-closed）
                allowed_write_tools = list(dict.fromkeys(t.strip() for t in raw_tools))
            tasks.append(Task(
                task_id=task_id,
                domain=domain,
                action=action,
                goal=goal,
                message=message,
                depends_on=deps,
                allowed_write_tools=allowed_write_tools,
            ))
            seen_ids.add(task_id)
        # 依赖引用与无环校验
        for task in tasks:
            if any(dep not in seen_ids for dep in task.depends_on):
                return None
        if not self._is_acyclic(tasks):
            return None
        return tasks

    @staticmethod
    def _is_acyclic(tasks: List[Task]) -> bool:
        """Kahn 拓扑排序判环（任务量 ≤6，O(n²) 足够）。"""
        incoming = {t.task_id: set(t.depends_on) for t in tasks}
        ready = [t.task_id for t in tasks if not incoming[t.task_id]]
        done = 0
        while ready:
            tid = ready.pop()
            done += 1
            for t in tasks:
                if tid in incoming[t.task_id]:
                    incoming[t.task_id].discard(tid)
                    if not incoming[t.task_id]:
                        ready.append(t.task_id)
        return done == len(tasks)

    @staticmethod
    def _task_goal_fallback(domain: IntentDomain) -> str:
        """LLM 未给 goal 时的兜底（与 GOAL_TEMPLATES 同源）。"""
        return TaskPlanner.GOAL_TEMPLATES.get(domain, "回答用户的请求")


class TaskExecutor:
    """
    按依赖 DAG 分波执行任务，带失败传播：

      - wave = 依赖全部 SUCCESS 的任务并行执行；
      - 依赖中存在 FAILED/BLOCKED → 本任务 BLOCKED（不执行、不注入上下文）；
      - 任务执行失败 → FAILED，其下游依赖任务连锁 BLOCKED。
    """

    def __init__(self, run_task):
        """
        run_task: async (req, task, shared, on_event) -> AgentResponse
        （由编排器提供：构造 Task-scoped 上下文执行一次 TaskAgent Run ——
        领域只提供人格语境，执行策略由 task.action 决定）。
        """
        self._run_task = run_task

    async def execute(
        self,
        req: "Request",
        tasks: List[Task],
        on_event: Optional[Any] = None,
        max_tasks: int = 6,  # 任务 DAG 上限（默认 6，可由 ExecutionPolicy 覆盖）
    ) -> SharedState:
        shared = SharedState()
        pending = {t.task_id: t for t in tasks}

        if len(tasks) > max_tasks:
            raise ValueError(f"协作任务数量超过上限 {max_tasks}")

        while pending:
            # 1. 失败传播：依赖中存在 FAILED/BLOCKED 的任务 → 本任务 BLOCKED（不执行）
            blocked = [
                t for t in pending.values()
                if any(shared.status(dep) in (TASK_FAILED, TASK_BLOCKED) for dep in t.depends_on)
            ]
            for t in blocked:
                logger.warning(f"任务 {t.task_id} 依赖失败，标记 BLOCKED")
                shared.set_result(t.task_id, AgentResponse(
                    content="（该任务因依赖失败已跳过）", success=False,
                    agent_type=t.domain.value, task_id=t.task_id,
                ), status=TASK_BLOCKED)
                shared.set_task_meta(t, TASK_BLOCKED)
                del pending[t.task_id]

            # 2. 当前波：依赖全部 SUCCESS 的任务
            wave = [t for t in pending.values()
                    if all(shared.done(dep) for dep in t.depends_on)]
            if not wave:
                if not pending:
                    break  # 全部任务已结束（成功/失败/阻塞）
                blocked = ",".join(sorted(pending))
                raise ValueError(f"任务依赖无法满足，可能存在循环或缺失依赖: {blocked}")
            for t in wave:
                del pending[t.task_id]

            async def run_one(t: Task):
                # 每个 Task 独立计时：同 wave 并行任务的耗时互不影响，
                # task_meta.duration_ms 表示该 Task 自身真实执行耗时
                t0 = time.monotonic()
                try:
                    r = await self._run_task(req, t, shared, on_event)
                except Exception as ex:  # noqa: BLE001 —— 与 gather(return_exceptions) 语义一致
                    r = ex
                return t, r, (time.monotonic() - t0) * 1000

            results = await asyncio.gather(
                *[run_one(t) for t in wave],
            )
            for t, r, duration_ms in results:
                if isinstance(r, AgentResponse):
                    status = TASK_SUCCESS if r.success else TASK_FAILED
                    shared.set_result(t.task_id, r, status=status)
                    shared.set_task_meta(t, status, duration_ms, response=r)
                else:
                    logger.warning(f"任务 {t.task_id} 执行失败: {r}")
                    shared.set_result(t.task_id, AgentResponse(
                        content="（该子任务处理失败）", success=False,
                        agent_type=t.domain.value, task_id=t.task_id,
                    ), status=TASK_FAILED)
                    shared.set_task_meta(t, TASK_FAILED, duration_ms)

        return shared


class Synthesizer:
    """
    协作合成器：一次 LLM 调用把多个 Task 的结果合并为连贯的最终回复。

    职责独立于业务任务（不是 Task，也不是 Specialist Agent）：
    只读 SharedState 的最终结果做合并。LLM 失败时降级为规则拼接。
    """

    def __init__(self, client: AsyncAnthropic, model: str, max_tokens: int = 1024,
                 gateway: Optional[Any] = None):
        self._client = client
        self._model  = model
        self._max_tokens = max_tokens  # 合成预算（默认 1024，可由 ExecutionPolicy 覆盖）
        self._gateway = gateway        # 统一模型调用入口（编排器注入；None 时直接调用）

    async def synthesize(
        self,
        req: "Request",
        results: List[AgentResponse],
    ) -> str:
        parts = [
            (r.label, r.content) for r in results
            if r.success and r.content and r.content != "（该子任务处理失败）"
            and "（该任务因依赖失败已跳过）" not in r.content
        ]
        if not parts:
            return "抱歉，多个助手模块暂时都没能处理成功，请稍后重试。"
        if len(parts) == 1:
            return parts[0][1]

        system = (
            "你是 EchoGuide 多 Agent 协作的合成器。把多个子任务的回答合并成一段给用户的连贯回复："
            "去除重复内容，保留各自的有效信息与 [n] 引用标注，不要编造新的信息。"
            "如果某个领域回答是失败占位（如「处理失败」），直接忽略它。"
        )
        content = "\n\n".join(f"[{label}]\n{text}" for label, text in parts)
        from core.tracing import span

        try:
            async with span("synthesize", agents=",".join(label for label, _ in parts)):
                kwargs = {
                    "model": self._model,
                    "max_tokens": self._max_tokens,
                    "system": system,
                    "messages": [{
                        "role": "user",
                        "content": f"用户请求: {req.message}\n\n各子任务回答:\n{content}",
                    }],
                }
                if self._gateway is not None:
                    result = await self._gateway.call(
                        client=self._client,
                        state=req.state,
                        span_name="synthesize",
                        **kwargs,
                    )
                    resp = result.response
                else:
                    resp = await self._client.messages.create(**kwargs)
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            if text:
                return text
        except Exception as ex:
            logger.warning(f"合成器调用失败，降级为规则拼接: {ex}")

        return self._merge_parts(parts)

    @staticmethod
    def _merge_parts(parts: List[Tuple[str, str]]) -> str:
        """规则拼接（Synthesizer LLM 不可用时的兜底）。"""
        return "\n\n".join(f"[{label}]\n{text}" for label, text in parts)
