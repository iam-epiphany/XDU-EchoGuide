"""
轻量多 Agent 协作（复杂任务编排）—— ExecutionPlan / Task / Planner / Executor / Synthesizer。

决策闭环（v4 收口）：复杂度判定从 IntentRecognizer 移入 Planner——
Intent 只负责理解用户想做什么（domain/action），Planner 统一输出
ExecutionPlan（Task DAG），single/parallel/dependent 由最终 DAG 自动推导：

    1 个 task            → single
    多个无依赖 task       → parallel
    存在 depends_on       → dependent

Planner 两条路径，但都输出统一 ExecutionPlan：
  - Fast Path（本地规则）：明显单任务直接生成 1 个 Task；命中复合规则生成
    依赖链；多领域 + 连接词生成并行任务。零额外 LLM 调用。
  - LLM 规划（升级路径）：本地判单任务但"拿不准"（多从句/长句/多领域无
    连接词）→ 一次轻量 LLM 调用输出任务链，硬校验后采用，非法回落本地。

每个 Task 携带自己的 action（QUERY→QA / REQUEST→Executor），不再继承
原始请求的 action —— 复合请求拆分后 t1/t2 可能是 QUERY、t3 才是 REQUEST。

DAG 失败传播：任务状态 SUCCESS / FAILED / BLOCKED / SKIPPED。
依赖任务 FAILED/BLOCKED → 下游任务 BLOCKED（不执行、不注入上下文），
不能因为前置"执行完成（但失败）"就继续执行依赖任务。

任务角色标签直接用 IntentDomain（领域值只做人格/Skills 挂载键，
执行实体是 QA/EXECUTOR 职责角色，见 roles.py）。
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

from core.domains import DOMAIN_KEYWORDS, IntentAction, IntentDomain, keyword_hit
from agents.roles import AgentResponse, Role

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
    """多 Agent 协作中的最小执行单元（自包含：不依赖原始对话上下文）。"""
    task_id:     str
    agent_type:  IntentDomain             # 任务角色标签（领域值，只做挂载键）
    goal:        str                      # 领域化任务目标（给 Agent 的指令）
    message:     str                      # 自包含请求内容
    action:      IntentAction = IntentAction.QUERY  # 任务自己的动作（决定执行角色）
    depends_on:  List[str] = field(default_factory=list)  # 依赖的其他 task_id
    # 后置条件：本任务应落地的写操作（模型忘记调用工具时由 Executor 补执行）。
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

    def set_task_meta(self, task: Task, status: str, duration_ms: float = 0.0) -> None:
        self._task_meta[task.task_id] = {
            "id": task.task_id,
            "agent": task.agent_type.value,
            "action": task.action.value,
            "depends_on": list(task.depends_on),
            "status": status,
            "duration_ms": round(duration_ms, 1),
        }

    def task_meta(self) -> List[Dict[str, Any]]:
        return list(self._task_meta.values())

    def snapshot(self) -> str:
        """把已完成（成功）任务的结果序列化，注入依赖任务作为协作上下文。

        失败/阻塞任务不注入 —— 依赖任务不能把失败结果当成有效上下文。
        """
        if not self._results:
            return ""
        return "\n\n".join(
            f"[{task_id}]\n{resp.content}"
            for task_id, resp in self._results.items()
            if self._status.get(task_id) == TASK_SUCCESS
        )


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

    def __init__(
        self,
        client: Optional[Any] = None,
        model: str = "",
        gateway: Optional[Any] = None,
        max_tasks: int = 6,
        max_agents: int = 3,
    ):
        # LLM 规划能力（可选注入）：client/model/gateway 由编排器传入；
        # 不注入时 Planner 只走 Fast Path（单任务/规则链/并行）。
        self._client = client
        self._model = model
        self._gateway = gateway
        self._max_tasks = max_tasks
        self._max_agents = max_agents

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def plan(self, req: "Request", domain: IntentDomain, action: IntentAction) -> ExecutionPlan:
        """
        生成 ExecutionPlan：Fast Path 先判；判 single 但"拿不准"时 LLM 规划升级。
        """
        fast = self._fast_plan(req, domain, action)
        if fast.mode != "single" or not self._needs_llm_planning(req):
            return fast
        llm_plan = await self._llm_plan(req)
        if llm_plan is not None:
            return llm_plan
        return fast  # LLM 不可用/输出非法：回落本地（行为不比现状差）

    # ── Fast Path（本地规则）────────────────────────────────────────────────

    def _fast_plan(self, req: "Request", domain: IntentDomain, action: IntentAction) -> ExecutionPlan:
        """本地规则生成：规则链 → 并行 → 单任务。零 LLM 调用。"""
        # 1. 复合规则命中 → 依赖任务链
        rule_tasks = self._apply_rules(req)
        if rule_tasks is not None:
            return ExecutionPlan(rule_tasks, reason="命中复合规则（存在前后依赖）")

        # 2. 多领域 + 显式连接词 → 并行任务
        targets = self._collaboration_targets(req, domain)
        connectors = ("同时", "还要", "并且", "另外", "以及", "顺便", "然后")
        if len(targets) >= 2 and any(word in req.message for word in connectors):
            tasks = [
                Task(
                    task_id=f"t{i}",
                    agent_type=at,
                    goal=self.GOAL_TEMPLATES.get(at, "回答用户的请求"),
                    message=f"{self.GOAL_TEMPLATES.get(at, '回答用户的请求')}。\n用户请求: {req.message}",
                    action=action,
                )
                for i, at in enumerate(targets[: self._max_agents])
            ]
            return ExecutionPlan(tasks, reason="显式复合语义涉及多个校园领域")

        # 3. 单任务（绝大多数请求）
        goal = self.GOAL_TEMPLATES.get(domain, "回答用户的请求")
        return ExecutionPlan(
            [Task(task_id="t0", agent_type=domain, goal=goal, message=req.message, action=action)],
            reason="单领域请求",
        )

    # ── 复合规则 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_errand_content(msg: str) -> str:
        """从用户请求中提取办事内容（"办校园卡" → "校园卡"），避免硬编码业务。"""
        m = re.search(r"(?:办|办理|补办|申请|搞)([\u4e00-\u9fff]{2,8})", msg)
        return m.group(1) if m else "校园事务"

    @classmethod
    def _plan_schedule_errand(cls, req: "Request") -> Optional[List[Task]]:
        """
        规则：个人日程 + 线下办事 + 记待办 的复合请求。

        例："我明天下午有空，想去办校园卡，帮我记个待办"
          t1 查课表（personal, QUERY）→ t2 查办理信息（affairs, QUERY）
          → t3 创建待办（personal, REQUEST，depends_on=[t1,t2]）
        每个任务有自己的 action：t1/t2 是 QUERY（走 QA），t3 是 REQUEST（走 Executor）。
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
                agent_type=IntentDomain.PERSONAL,
                action=IntentAction.QUERY,
                goal="查询课程空闲时间",
                message=f"查询用户课程/空闲时间（如明天下午是否有课）。用户请求: {msg}",
            ),
            Task(
                task_id="t2",
                agent_type=IntentDomain.AFFAIRS,
                action=IntentAction.QUERY,
                goal="查询校园卡办理信息",
                message=f"查询校园卡办理地点和所需材料。用户请求: {msg}",
            ),
            Task(
                task_id="t3",
                agent_type=IntentDomain.PERSONAL,
                action=IntentAction.REQUEST,
                goal="创建校园卡办理待办",
                message=(
                    "根据协作上下文中的课程空闲时间和校园卡办理信息，"
                    "为用户创建一个合适的办理待办/提醒（时间安排在空闲时段）。"
                    f"用户请求: {msg}"
                ),
                depends_on=["t1", "t2"],
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

        任一信号命中即升级：
          1. 消息被切出 ≥3 个从句（信息量大，可能复合）；
          2. 长消息（>24 字）且 ≥2 个从句；
          3. 领域关键词命中 ≥2 个领域但无显式连接词（"隐式复合"）。
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
        return len(hit_domains) >= 2

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
            '"goal": "<任务目标>", "message": "<给该领域助手的自包含请求>", "depends_on": ["<前置任务id>"]}\n'
            "- domain 可选值: academic, campus_life, affairs, it_help, personal\n"
            "- action: query=查询咨询；request=需要系统写数据/产生副作用（创建待办等）\n"
            "- 只有明确需要写操作的任务才是 request，查询类任务一律 query\n"
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
            tasks.append(Task(
                task_id=task_id,
                agent_type=domain,
                action=action,
                goal=goal,
                message=message,
                depends_on=deps,
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
    def _task_goal_fallback(agent_type: IntentDomain) -> str:
        """LLM 未给 goal 时的兜底（与 GOAL_TEMPLATES 同源）。"""
        return TaskPlanner.GOAL_TEMPLATES.get(agent_type, "回答用户的请求")


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
        （由编排器提供，负责按任务的领域角色标签执行任务——执行实体是 QA/EXECUTOR 职责角色，
        领域角色只决定人格/Skills 挂载，执行角色由 task.action 决定）。
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
                    role=Role.QA, content="（该任务因依赖失败已跳过）", success=False,
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

            started = time.monotonic()
            results = await asyncio.gather(
                *[self._run_task(req, t, shared, on_event) for t in wave],
                return_exceptions=True,
            )
            for t, r in zip(wave, results):
                duration_ms = (time.monotonic() - started) * 1000
                if isinstance(r, AgentResponse):
                    status = TASK_SUCCESS if r.success else TASK_FAILED
                    shared.set_result(t.task_id, r, status=status)
                    shared.set_task_meta(t, status, duration_ms)
                else:
                    logger.warning(f"任务 {t.task_id} 执行失败: {r}")
                    shared.set_result(t.task_id, AgentResponse(
                        role=Role.QA, content="（该领域助手处理失败）", success=False,
                    ), status=TASK_FAILED)
                    shared.set_task_meta(t, TASK_FAILED, duration_ms)

        return shared


class Synthesizer:
    """
    协作合成器：一次 LLM 调用把多个任务结果合并为连贯的最终回复。

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
            if r.success and r.content and r.content != "（该领域助手处理失败）"
            and "（该任务因依赖失败已跳过）" not in r.content
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
                        "content": f"用户请求: {req.message}\n\n各领域助手回答:\n{content}",
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
