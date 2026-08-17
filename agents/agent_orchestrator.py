"""
亮点：单 Agent 执行 + 按需多 Agent 协作（领域只做挂载，不做路由）

架构（v3，2026-08 重构）：
  - 职责角色拆分：QAAgent（问答，只读工具面 + 检索/引用规范）与
    ExecutorAgent（执行，全量工具面含写 + 执行确认规范），各配 Fast/Deep
    双 profile；按意图 action 选择角色（REQUEST→Executor，其余→QA）。
    领域分类（IntentDomain）只用来挂载领域人格（DOMAIN_PERSONA）与 Skills，
    不决定工具可见性、不选执行角色 —— "顾问"而非"门卫"；职责 × 领域正交。
  - 工具可见性 = 公共工具层：所有 agent_exposed=True 的工具对任何请求可见，
    门禁两层：注册级 agent_exposed（外部工具默认双重不可见）+
    Action 级读写策略（QUERY/GREETING 拒写，对齐 MCP readOnlyHint/RBAC）。
  - 按需多 Agent 协作（复杂请求，非 Agent 间聊天）：
      Planner 拆分任务 DAG（自包含任务，支持跨任务依赖）
        → Executor 按 depends_on 分波并行执行，结果写入 SharedState
        → 依赖任务执行时注入协作上下文（使用前序 Agent 结果）
        → Synthesizer 合并为最终回复（LLM 失败降级拼接）。
    每个任务是独立上下文的执行实体；任务角色标签沿用领域值（只做
    goal/人格/Skills 挂载键，不构成独立 Agent 身份）。

复杂度判定（规则筛 + LLM 升级确认）：
  意图识别走 LLM 时顺带输出 complexity（single/parallel/dependent + 依赖链，
  同一次调用零额外成本）；否则免费关键词规则先判，规则判 single 但"拿不准"
  （多从句/长句/多领域无连接词）才升级 LLM 确认；LLM 结论必须通过规则校验
  （领域合法/任务无环/数量受限），非法回落关键词规则。
"""
import asyncio
import inspect
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from anthropic import AsyncAnthropic

from core.domains import DOMAIN_KEYWORDS, IntentAction, IntentDomain, keyword_hit
from core.intent_recognizer import IntentCategory, IntentRecognizer
from memory.layered_store import (
    LayeredStore, estimate_tokens, OFFLOAD_CHARS, OFFLOAD_SUMMARY_CHARS,
)
from runtime.policy import ExecutionPolicy
from runtime.runtime import AgentRuntime
from runtime.state import RunState

from agents.verifier import ResponseVerifier

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    """职责角色（执行实体）与协作任务角色标签。

    QA / EXECUTOR：按职责拆分的两个执行角色（不同工具面 + 行为边界），
    人格/Skills 按领域挂载 —— 职责 × 领域正交。
    其余值：协作 DAG 的任务角色标签（领域只用来挂载，不构成执行实体）。
    """
    QA         = "qa"        # 职责角色：问答（只读工具面 + 检索/引用规范）
    EXECUTOR   = "executor"  # 职责角色：执行（全量工具面含写 + 执行确认规范）
    ACADEMIC   = "academic"  # 任务角色：学业支持
    CAMPUS_LIFE = "campus_life"  # 任务角色：校园生活
    AFFAIRS    = "affairs"   # 任务角色：校务咨询
    IT_HELP    = "it_help"   # 任务角色：IT 助手
    PERSONAL   = "personal"  # 任务角色：个人助理


# Action 层工具策略（公共工具层的读写门禁，职责划分：domain = 挂载人格/Skills，action = 怎么处理）。
#   - QUERY：只开放只读/查询类工具，禁止状态修改类工具；
#   - REQUEST：允许按需开放完整工具（含执行类）；
#   - GREETING / FEEDBACK：原则上不开放工具；
#   - COMPLAINT / OTHER：保守策略，只开放只读工具；
#   - None（预置请求/兼容路径）：不额外限制，保持原有 allowlist 行为。
# 新增状态修改类工具时必须登记到 WRITE_TOOLS，否则会在 QUERY 等只读动作下被误开放。
WRITE_TOOLS: FrozenSet[str] = frozenset({"add_todo", "complete_todo"})


def action_allows_tool(action: Optional[IntentAction], tool_name: str) -> bool:
    """Action 层工具过滤（纯函数）：返回该动作下工具是否可暴露/执行。"""
    if action == IntentAction.REQUEST:
        return True  # 完整工具：按需执行
    if action in (IntentAction.GREETING, IntentAction.FEEDBACK):
        return False  # 原则上不开放工具
    if action is None:
        return True  # 动作未知：保持原有 allowlist 行为（兼容既有调用方）
    # QUERY / COMPLAINT / OTHER：保守策略，只开放只读工具
    return tool_name not in WRITE_TOOLS


# Action 行为指引（注入 system prompt）。职责划分：
# domain 决定"挂载什么人格/Skills"（顾问），action 决定"怎么处理"（执行策略）。
ACTION_GUIDANCE: Dict[IntentAction, str] = {
    IntentAction.QUERY: "当前意图为查询：请准确查询并如实回答，不要执行任何修改状态的操作（如新增/删除/完成待办）。",
    IntentAction.REQUEST: "当前意图为请求办理：请积极调用工具解决问题，需要执行操作时按用户指令完成。",
    IntentAction.COMPLAINT: "当前意图为投诉/不满：请先识别具体问题点，再给出明确的解决路径或建议，语气克制。",
    IntentAction.GREETING: "当前意图为问候：请简洁友好回应即可，无需调用工具。",
    IntentAction.FEEDBACK: "当前意图为反馈：请简洁回应并感谢反馈，无需调用工具。",
    IntentAction.OTHER: "当前意图不明确：请保守处理，仅基于已有信息回答，不要执行任何修改状态的操作。",
}


# 领域人格（注入 system prompt 的 [领域人格] 段）。领域分类的唯一产物：
# 挂载行为风格 —— 工具可见性与执行实体都与领域无关。
DOMAIN_PERSONA: Dict[IntentDomain, str] = {
    IntentDomain.ACADEMIC: (
        "当前问题属于学业支持：覆盖选课、课表、考试安排、成绩与绩点、重修、转专业、保研。"
        "回答基于西电教务规则和公开常识，步骤清晰、用语克制。"
        "政策、规定、培养方案或转专业问题必须先调用 knowledge_search；检索结果含 source_url 时，"
        "回答必须给出可点击来源链接、文档标题、更新时间与适用范围，不能把单学院规则泛化为全校规则。"
        "用户提供课程成绩与学分时必须调用 calculate_weighted_score，不要自行心算，并明确它不是官方 GPA。"
        "涉及具体成绩或学籍操作时，提示学生前往教务系统或学院教务老师处确认。"
    ),
    IntentDomain.CAMPUS_LIFE: (
        "当前问题属于校园生活：覆盖宿舍、食堂、校园穿梭车、校园卡、快递、水电、社团、运动场馆。"
        "图书馆/场馆位置或开放时间、校车班次必须先调用 query_campus_info；天气问题必须先调用 get_weather，"
        "包括依赖上一轮实体的「那几点关门」等短追问。回答尽量给出位置（校区/楼栋）和时段。"
        "涉及报修、补办等需现场办理的事项，指引用户到对应服务网点。"
    ),
    IntentDomain.AFFAIRS: (
        "当前问题属于校务办事：覆盖校历、请假流程、奖学金与助学金、各类证明开具、学籍注册、学费缴纳。"
        "回答以办事流程、所需材料、办理地点和系统入口为主，清晰可执行。"
        "校园卡补办、请假、在读证明或缓考问题优先调用 query_affairs_process 获取版本化流程。"
        "涉及实际审批的事项，提示以学院或学生处最新通知为准。"
    ),
    IntentDomain.IT_HELP: (
        "当前问题属于 IT 支持：覆盖教务系统、校园网、VPN、学校邮箱、统一身份认证的故障排查。"
        "遇到上述系统故障时优先调用 diagnose_it_issue，再基于诊断树组织回答，给出清晰的步骤化解决方案。"
        "遇到需要后台操作或账号重置的问题，说明需联系信息化建设处或网络中心处理。"
    ),
    IntentDomain.PERSONAL: (
        "当前问题属于个人事务：覆盖用户自己的课表、待办、考试与 DDL 安排。"
        "查询前先调用工具获取用户个人数据（query_schedule / query_todo / query_ddl 等），"
        "不要凭记忆编造课程或待办。用户未导入课表时，引导其通过「我的课表」上传 .ics 文件或 JSON 课表。"
        "回答按时间组织，带上课时间与地点；涉及考试/DDL 时给出剩余天数。"
    ),
    IntentDomain.OTHER: (
        "当前问题不属于校园领域（如通用知识、编程、GitHub 等外部工具问题）："
        "以通用助手的方式直接回答，可用公共工具（含外部只读工具）辅助，保持简洁准确。"
    ),
}


class ProfileName(Enum):
    FAST = "fast"
    DEEP = "deep"


@dataclass(frozen=True)
class ExecutionProfile:
    """真实执行配置：模型、思考模式、生成预算与检索深度。"""
    name: ProfileName
    model: str
    max_tokens: int
    thinking: bool
    rag_top_k: int
    use_rewrite: bool
    use_rerank: bool


@dataclass
class AgentStats:
    """Agent 运行时统计，供 Monitor 和路由决策使用。"""
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    monitor_penalty: float = 0.0
    in_flight: int = 0

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

    def effective_routing_score(self) -> float:
        """进程内自适应实例池：少量探索并规避正在处理请求的实例。"""
        return self.routing_score() + 0.05 / ((self.total + 1) ** 0.5) - 0.10 * self.in_flight


@dataclass
class AgentResponse:
    agent_type:  AgentType
    content:     str
    success:     bool
    confidence:  float = 1.0
    latency_ms:  float = 0.0
    tools_used:  List[str] = field(default_factory=list)  # 本次调用的工具（供过程可视化）
    tool_evidence: List[Dict[str, Any]] = field(default_factory=list)
    profile: str = "fast"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    offloaded_chars: int = 0   # 上下文卸载：从上下文移出的字符数
    saved_tokens: int = 0      # 上下文卸载：按估算口径省下的 token 数


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
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    confidence: float = 0.0
    classifier_stage: str = "preset"
    profile: Optional[ProfileName] = None
    complexity_mode: str = "single"
    complexity_reason: str = "单领域请求"
    benchmark_strategy: str = "adaptive"
    state: Optional[RunState] = None   # Runtime 运行状态（编排器 run() 创建后回填）
    state_query: Optional[Dict[str, Any]] = None  # 查询理解产出（skills_to_reference/needs_knowledge，观测用）


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    domain:      Optional[IntentDomain] = None
    action:      Optional[IntentAction] = None
    latency_ms:  float = 0.0
    tools_used:  List[str] = field(default_factory=list)  # 本次调用的工具（RAG 等）
    tool_evidence: List[Dict[str, Any]] = field(default_factory=list)
    execution: Dict[str, Any] = field(default_factory=dict)


# ── 轻量多 Agent 协作（Task Planner / Shared State / Synthesizer）─────────────

@dataclass
class Task:
    """多 Agent 协作中的最小执行单元（自包含：不依赖原始对话上下文）。"""
    task_id:     str
    agent_type:  AgentType
    goal:        str                    # 领域化任务目标（给 Agent 的指令）
    message:     str                    # 自包含请求内容
    depends_on:  List[str] = field(default_factory=list)  # 依赖的其他 task_id
    # 后置条件：本任务应落地的写操作（模型忘记调用工具时由 Executor 补执行）。
    # 由 Planner 声明，避免执行器硬编码"t3/校园卡待办"这类任务标识。
    required_tool: Optional[str] = None
    required_tool_args: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ComplexityDecision:
    mode: str
    reason: str
    targets: List[AgentType]
    # LLM 判定的依赖链（mode == "dependent" 时由 _tasks_from_llm 校验产出；
    # 关键词规则产出时保持 None，由 TaskPlanner 现场生成）
    tasks: Optional[List[Task]] = None


class SharedState:
    """协作共享状态：记录每个 Task 的结果，供依赖任务（如汇总）读取。"""

    def __init__(self) -> None:
        self._results: Dict[str, AgentResponse] = {}
        self._task_meta: Dict[str, Dict[str, Any]] = {}

    def set_result(self, task_id: str, resp: AgentResponse) -> None:
        self._results[task_id] = resp

    def get_result(self, task_id: str) -> Optional[AgentResponse]:
        return self._results.get(task_id)

    def done(self, task_id: str) -> bool:
        return task_id in self._results

    def all_results(self) -> Dict[str, AgentResponse]:
        return dict(self._results)

    def set_task_meta(self, task: Task, status: str, duration_ms: float = 0.0) -> None:
        self._task_meta[task.task_id] = {
            "id": task.task_id,
            "agent": task.agent_type.value,
            "depends_on": list(task.depends_on),
            "status": status,
            "duration_ms": round(duration_ms, 1),
        }

    def task_meta(self) -> List[Dict[str, Any]]:
        return list(self._task_meta.values())

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
          t1 查课表（PERSONAL）→ t2 查办理信息（AFFAIRS）
          → t3 创建待办（PERSONAL，depends_on=[t1,t2]，读取 SharedState 结果）
        """
        msg = req.message
        has_schedule = any(keyword_hit(kw, msg) for kw in ("课表", "课程", "空闲", "上课", "没课", "有空"))
        has_errand   = any(keyword_hit(kw, msg) for kw in ("校园卡", "办理", "材料", "缴费", "办证"))
        has_todo     = any(keyword_hit(kw, msg) for kw in ("待办", "提醒", "记一下", "安排上", "安排"))
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
                agent_type=AgentType.AFFAIRS,
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
                required_tool="add_todo",
                required_tool_args={"content": "补办校园卡", "kind": "todo"},
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
        （由编排器提供，负责按任务的领域角色标签执行任务——执行实体是 QA/EXECUTOR 职责角色，
        领域角色只决定人格/Skills 挂载）。
        """
        self._run_task = run_task

    async def execute(
        self,
        req: Request,
        tasks: List[Task],
        on_event: Optional[Any] = None,
        max_tasks: int = 6,  # 任务 DAG 上限（默认 6，可由 ExecutionPolicy 覆盖）
    ) -> SharedState:
        shared = SharedState()
        pending = {t.task_id: t for t in tasks}

        if len(tasks) > max_tasks:
            raise ValueError(f"协作任务数量超过上限 {max_tasks}")

        while pending:
            # 当前波：所有依赖已完成的任务
            wave = [t for t in pending.values()
                    if all(shared.done(dep) for dep in t.depends_on)]
            if not wave:
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
                    shared.set_result(t.task_id, r)
                    shared.set_task_meta(t, "success" if r.success else "failed", duration_ms)
                else:
                    logger.warning(f"任务 {t.task_id} 执行失败: {r}")
                    shared.set_result(t.task_id, AgentResponse(
                        agent_type=t.agent_type, content="（该领域助手处理失败）", success=False,
                    ))
                    shared.set_task_meta(t, "failed", duration_ms)

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
    def _merge_parts(parts: List[Tuple[AgentType, str]]) -> str:
        """规则拼接（Synthesizer LLM 不可用时的兜底）。"""
        return "\n\n".join(f"[{at.value}]\n{text}" for at, text in parts)


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用和统计。"""

    agent_type: AgentType
    system_prompt: str
    write_allowed: bool = True  # 角色级写权限（QA 置 False：只读边界防御纵深）

    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        skill_manager: Optional[Any] = None,
        tool_manager: Optional[Any] = None,
        profile: Optional[ExecutionProfile] = None,
        memory_store: Optional[LayeredStore] = None,
    ):
        self._client = client
        self._model  = model
        self._skill_manager = skill_manager
        self._tool_manager  = tool_manager
        self._memory_store  = memory_store  # 上下文卸载落盘（refs 表），可由编排器注入
        self._runtime = None                # Agent Runtime（编排器注入；None 时钩子全部 no-op）
        self.profile = profile or ExecutionProfile(
            name=ProfileName.FAST,
            model=model,
            max_tokens=1024,
            thinking=False,
            rag_top_k=3,
            use_rewrite=False,
            use_rerank=False,
        )
        self.stats   = AgentStats()

    def _gateway(self) -> Optional[Any]:
        """统一模型调用入口（从注入的 Runtime 取；无 Runtime 时返回 None 走直接调用）。"""
        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            return None
        return getattr(runtime, "model_gateway", None)

    async def _call_model(
        self,
        req: Request,
        system: str,
        messages: List[Dict],
        tools: List[Dict],
        on_event: Optional[Any],
    ) -> Any:
        """经 ModelGateway 的模型调用：真实执行边界（计数/统计/预算/Trace）。

        gateway 可用且 req.state 存在时走统一入口（before_model → provider →
        after_model，step/token 落 RunState）；否则直接调用（测试/无 Runtime
        兼容路径，行为与旧版一致）。
        """
        kwargs = self._model_kwargs()
        kwargs.update(system=system, messages=messages, tools=tools or None)
        gateway = self._gateway()
        if gateway is None or req.state is None:
            if on_event is not None:
                return await self._stream_llm(system, messages, tools, on_event)
            return await self._client.messages.create(**kwargs)
        services = {"skill_manager": self._skill_manager, "history": req.history}
        if on_event is not None:
            result = await gateway.call_stream(
                client=self._client,
                state=req.state,
                services=services,
                on_event=on_event,
                span_name="agent_llm",
                **kwargs,
            )
        else:
            result = await gateway.call(
                client=self._client,
                state=req.state,
                services=services,
                span_name="agent_llm",
                **kwargs,
            )
        return result.response

    async def handle(self, req: Request, on_event: Optional[Any] = None) -> AgentResponse:
        """
        处理一次请求：LLM 工具调用循环（Agentic RAG）。

        on_event: 可选异步回调，接收过程事件（meta/tool/delta），供 SSE 流式输出使用。
        """
        t0 = time.monotonic()
        self.stats.total += 1
        self.stats.in_flight += 1
        try:
            from core.tracing import span

            async with span("agent_handle", agent=self.agent_type.value):
                (content, tools_used, tool_evidence,
                 input_tokens, output_tokens,
                 offloaded_chars, saved_tokens) = await self._call_llm(req, on_event=on_event)
            # 引用是检索链路的执行后置条件，不依赖模型是否自觉把 URL 抄进正文。
            # 仅追加工具返回的公开标题/URL/更新时间，不暴露原始参数或内部上下文。
            cited = []
            seen_urls = set()
            for item in tool_evidence:
                url = str(item.get("source_url") or "").strip()
                if not url or url in seen_urls or url in content:
                    continue
                seen_urls.add(url)
                title = str(item.get("title") or "公开来源").strip()
                updated = str(item.get("updated_at") or "").strip()
                cited.append(f"- [{title}]({url})" + (f"（更新：{updated}）" if updated else ""))
            if cited:
                content = f"{content.rstrip()}\n\n### 可核验来源\n" + "\n".join(cited)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            return AgentResponse(
                agent_type=self.agent_type,
                content=content,
                success=True,
                latency_ms=ms,
                tools_used=tools_used,
                tool_evidence=tool_evidence,
                profile=self.profile.name.value,
                model=self.profile.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                offloaded_chars=offloaded_chars,
                saved_tokens=saved_tokens,
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
                profile=self.profile.name.value,
                model=self.profile.model,
            )
        finally:
            self.stats.in_flight = max(0, self.stats.in_flight - 1)

    # ── Agentic RAG：LLM 工具调用循环 ────────────────────────────────────────

    MAX_TOOL_ROUNDS = 3  # 无 Runtime policy 时的工具循环上限（与 Fast 路径默认一致）
    STAGNANT_ROUND_LIMIT = 2  # 无进展检测：连续重复轮数阈值（可被 ExecutionPolicy 覆盖）

    def _build_tools(self, req: Optional[Request] = None) -> List[Dict[str, Any]]:
        """
        把 MCPToolManager 中注册的工具与 Skill 工具暴露给 LLM（function calling）。

        可见性 = 公共工具层：所有 agent_exposed=True 的工具对任何请求可见，
        不按领域剪裁（领域只挂载人格/Skills）。门禁两层：
          1. 注册级 agent_exposed（外部工具默认双重不可见）；
          2. Action 级读写策略（QUERY/GREETING 等动作下写工具不暴露）。
        Skill 工具（use_skill_*，渐进披露）追加在 MCP 工具之后，同受
        allowlist 与 Action 门禁；完整 SKILL.md 由模型按需加载。
        实例可设 _tool_allowlist 覆盖公共层（测试/定制场景）。
        """
        if self._tool_manager is None and self._skill_manager is None:
            return []
        allowed = getattr(self, "_tool_allowlist", None)
        tools = []
        action = req.action if req is not None else None
        if self._tool_manager is not None:
            for name, tool in self._tool_manager._tools.items():
                if not getattr(tool, "agent_exposed", True):
                    continue
                if not self.write_allowed and name in WRITE_TOOLS:
                    continue  # 角色级只读边界（QA 永远不暴露写工具）
                if allowed is not None and name not in allowed:
                    continue  # 实例级覆盖：显式缩小可见集合
                if action is not None and not action_allows_tool(action, name):
                    continue  # Action 层策略：查询/问候等动作下不暴露执行类工具
                tools.append({
                    "name": name,
                    "description": tool.description,
                    "input_schema": tool.schema,
                })
        if self._skill_manager is not None:
            for tool_def in self._skill_manager.tool_definitions():
                name = tool_def["name"]
                if allowed is not None and name not in allowed:
                    continue
                if action is not None and not action_allows_tool(action, name):
                    continue
                tools.append(tool_def)
        return tools

    def _model_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.profile.model,
            "max_tokens": self.profile.max_tokens,
        }
        kwargs["thinking"] = {"type": "enabled" if self.profile.thinking else "disabled"}
        if self.profile.thinking:
            kwargs["output_config"] = {"effort": "high"}
        return kwargs

    @staticmethod
    def _usage(resp: Any) -> Tuple[int, int]:
        usage = getattr(resp, "usage", None)
        return (
            int(getattr(usage, "input_tokens", 0) or 0),
            int(getattr(usage, "output_tokens", 0) or 0),
        )

    async def _call_llm(self, req: Request, on_event: Optional[Any] = None) -> tuple[str, List[str], List[Dict[str, Any]], int, int, int, int]:
        """
        工具调用循环：
          1. 首次调用带 tools（function calling），让 Agent 自主决定是否检索知识库
          2. 返回 tool_use → 执行工具 → 把 tool_result 回填 → 再次调用
          3. 无工具请求或达到轮次上限 → 返回最终文本

        上下文卸载（对应记忆金字塔之外的短期记忆优化）：
          工具结果超过 OFFLOAD_CHARS 时，完整内容落盘 refs 表（Skill 正文除外，
          必须完整留在上下文），
          上下文只留"前 OFFLOAD_SUMMARY_CHARS 字摘要 + refs/{id} 索引"，
          需要时可按 id 100% 找回 —— 长任务 token 消耗显著下降。

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

        tools = self._build_tools(req)
        system = self._build_system_prompt(req)
        if tools:
            system = (
                f"{system}\n\n[工具使用]\n"
                "你可以调用可用工具来获取信息（如检索校园知识库、查询外部数据源）。"
                "检索到相关资料时，回答末尾用 [1][2] 标注引用来源。"
                "能直接回答就不要调用工具；一次调用无进展或检索不到时如实说明，"
                "不要反复调用同一工具或编造内容。"
            )

        tools_used: List[str] = []
        tool_evidence: List[Dict[str, Any]] = []
        input_tokens = 0
        output_tokens = 0
        offloaded_chars = 0   # 上下文卸载统计（从上下文移出的字符数）
        saved_tokens = 0      # 上下文卸载统计（估算省下的 token）
        hit_tool_use_at_limit = False
        # 工具轮次预算（分级）：Fast 路径用便宜模型可多试几轮，Deep 路径留给复杂
        # 任务；无 Runtime policy 时回落类常量。无进展检测：连续 stagnant_limit
        # 轮工具调用与上一轮完全重复（同名同参，含失败重试）→ 视为死循环强制收尾，
        # 配合分级上限双保险（护栏 = 上限 + 重复检测，而不是只靠硬轮次）。
        max_rounds = self.MAX_TOOL_ROUNDS
        stagnant_limit = self.STAGNANT_ROUND_LIMIT
        if req.state is not None and req.state.policy is not None:
            if self.profile.name == ProfileName.FAST:
                max_rounds = req.state.policy.max_tool_rounds_fast
            else:
                max_rounds = req.state.policy.max_tool_rounds_deep
            stagnant_limit = req.state.policy.stagnant_round_limit
        last_round_sig: Optional[str] = None
        stagnant_rounds = 0
        for _round in range(max_rounds):
            try:
                resp = await self._call_model(req, system, messages, tools, on_event)
            except Exception as ex:
                # 上游不支持 tools → 降级为普通调用（不再带工具重试）
                if tools and _round == 0:
                    logger.warning(f"工具调用模式失败，降级为普通调用: {ex}")
                    tools = []
                    system = self._build_system_prompt(req)
                    resp = await self._call_model(req, system, messages, [], on_event)
                else:
                    raise

            used_in, used_out = self._usage(resp)
            input_tokens += used_in
            output_tokens += used_out
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
                    if name == "knowledge_search" and isinstance(data, list):
                        tool_evidence.extend({
                            "tool": name,
                            "title": item.get("title", ""),
                            "source_url": item.get("source_url", ""),
                            "updated_at": item.get("updated_at", ""),
                            "content": str(item.get("content", ""))[:800],
                        } for item in data if isinstance(item, dict))
                    if on_event is not None:
                        await on_event({
                            "type": "tool", "name": name, "status": "done",
                            "titles": self._tool_result_titles(data),
                        })
                    # 上下文卸载：超长结果落盘 refs 表，上下文只留摘要行 + 索引
                    # （需要时可沿 refs/{id} 100% 找回，代价是上下文里的 token 显著下降）。
                    # Skill 正文（use_skill_*）例外：规范全文必须留在上下文，不做卸载。
                    tool_text = self._clean_text(data) if data is not None else None
                    if (
                        tool_text is not None
                        and len(tool_text) > OFFLOAD_CHARS
                        and self._memory_store is not None
                        and not name.startswith("use_skill_")
                    ):
                        try:
                            ref_id = await self._memory_store.save_ref(
                                req.user_id, req.conv_id, name, tool_text
                            )
                            char_len = len(tool_text)
                            tool_text = (
                                f"{tool_text[:OFFLOAD_SUMMARY_CHARS]}..."
                                f"[完整结果 refs/{ref_id}，共 {char_len} 字符]"
                            )
                            offloaded_chars += char_len - len(tool_text)
                            saved_tokens += estimate_tokens(char_len - len(tool_text))
                        except Exception as ex:
                            # 落盘失败（磁盘/权限等）：保持全量回填，不阻断主链路
                            logger.warning(f"上下文卸载落盘失败，保持全量回填: {ex}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": getattr(block, "id", ""),
                        "content": tool_text if tool_text is not None else f"工具执行失败: {error}",
                    })
                messages.append({"role": "user", "content": tool_results})
                if req.state is not None:
                    req.state.tool_round_count += 1  # 真实工具调用轮次（一轮 = 一次工具批次）
                # 无进展检测：本轮调用签名与上一轮完全一致 → 死循环信号；
                # 连续 stagnant_limit 轮无进展 → 强制收尾（复用轮次上限的收尾路径）
                round_sig = self._tool_round_signature(tool_calls)
                if round_sig is not None and round_sig == last_round_sig:
                    stagnant_rounds += 1
                    if stagnant_rounds >= stagnant_limit:
                        logger.warning(
                            f"{self.agent_type.value} 连续 {stagnant_rounds} 轮无进展（{round_sig}），强制收尾"
                        )
                        hit_tool_use_at_limit = True
                        break
                else:
                    stagnant_rounds = 0
                last_round_sig = round_sig
                hit_tool_use_at_limit = _round == max_rounds - 1
                continue

            # 正常结束：提取文本
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            return text, tools_used, tool_evidence, input_tokens, output_tokens, offloaded_chars, saved_tokens

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
            # 收尾调用同样经统一入口（流式时逐 token 推送，避免割裂体验）
            resp = await self._call_model(req, system, messages, [], on_event)
            used_in, used_out = self._usage(resp)
            input_tokens += used_in
            output_tokens += used_out
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip()
            if text:
                return text, tools_used, tool_evidence, input_tokens, output_tokens, offloaded_chars, saved_tokens

        return "抱歉，处理超时或模型未返回有效内容，请稍后重试。", tools_used, tool_evidence, input_tokens, output_tokens, offloaded_chars, saved_tokens

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
            **self._model_kwargs(),
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
        # Skill 工具拦截（渐进披露）：完整 SKILL.md 正文本地加载，不经过 MCPToolManager；
        # 与普通工具同受 allowlist 与 Action 门禁（防御纵深）。
        if name.startswith("use_skill_"):
            allowed = getattr(self, "_tool_allowlist", None)
            if allowed is not None and name not in allowed:
                logger.warning(f"{self.agent_type.value} 尝试调用权限外工具 {name}，已拒绝")
                return None, f"工具 {name} 不在当前执行权限范围内"
            if req.action is not None and not action_allows_tool(req.action, name):
                logger.warning(
                    f"{self.agent_type.value} 在 {req.action.value} 动作下尝试调用工具 {name}，已拒绝"
                )
                return None, f"工具 {name} 不在当前意图（{req.action.value}）的权限范围内"
            runtime = getattr(self, "_runtime", None)
            if runtime is not None and req.state is not None:
                await runtime.fire_tool_before(req.state, name, params)
            data, error = None, None
            if self._skill_manager is None:
                error = "技能管理器不可用"
            else:
                skill = self._skill_manager.skill_for_tool(name)
                if skill is None:
                    error = f"技能 {name} 不存在或已停用"
                else:
                    data = skill.to_prompt_block(max_chars=12000)
            if runtime is not None and req.state is not None:
                await runtime.fire_tool_after(req.state, name, data, error)
            if error:
                return None, error
            return data, None
        if self._tool_manager is None:
            return None, "工具管理器不可用"
        # 角色级只读边界（防御纵深）：QA 角色即使动作误判也拒绝写工具。
        if not self.write_allowed and name in WRITE_TOOLS:
            logger.warning(f"{self.agent_type.value} 角色尝试调用写工具 {name}，已拒绝")
            return None, f"工具 {name} 不在 {self.agent_type.value} 角色权限范围内"
        # 权限边界（防御纵深）：公共工具层内工具由 Action 层策略把关（见下）；
        # 实例级 _tool_allowlist 覆盖（测试/定制）之外的工具直接拒绝。
        allowed = getattr(self, "_tool_allowlist", None)
        if allowed is not None and name not in allowed:
            logger.warning(f"{self.agent_type.value} 尝试调用权限外工具 {name}，已拒绝")
            return None, f"工具 {name} 不在当前执行权限范围内"
        # Action 层权限（防御纵深，与 _build_tools 暴露层一致）：
        # 查询/问候等动作下，LLM 即使声明了执行类工具也直接拒绝
        if req.action is not None and not action_allows_tool(req.action, name):
            logger.warning(
                f"{self.agent_type.value} 在 {req.action.value} 动作下尝试调用工具 {name}，已拒绝"
            )
            return None, f"工具 {name} 不在当前意图（{req.action.value}）的权限范围内"
        from core.tracing import span

        # 工具边界钩子（Runtime）：计数/预算由中间件处理，注入方无感知
        runtime = getattr(self, "_runtime", None)
        if runtime is not None and req.state is not None:
            await runtime.fire_tool_before(req.state, name, params)
        result = None
        try:
            async with span("tool_call", tool=name, query=str(params.get("query", ""))[:80]):
                result = await self._tool_manager.call(
                    name,
                    params,
                    context={"agent_type": self.agent_type.value, "user_id": req.user_id},
                    rerank_top_k=self.profile.rag_top_k if self.profile.use_rerank else 0,
                    use_rewrite=self.profile.use_rewrite,
                )
        finally:
            if runtime is not None and req.state is not None:
                await runtime.fire_tool_after(
                    req.state,
                    name,
                    result.data if result is not None and result.success else None,
                    result.error if result is not None and not result.success else None,
                )
        if not result.success:
            return None, result.error or "工具执行失败"
        return result.data, None

    @staticmethod
    def _tool_round_signature(tool_calls: List[Any]) -> Optional[str]:
        """本轮工具调用的规范化签名（工具名+参数，参数排序无关）。

        同一工具同一参数连续出现 → 无进展；参数含不可序列化值（罕见）时
        返回 None，调用方跳过本轮检测（宁可不检测也不误判死循环）。
        """
        parts = []
        for block in tool_calls:
            name = getattr(block, "name", "") or ""
            inp = getattr(block, "input", None) or {}
            try:
                sig = json.dumps(inp, sort_keys=True, ensure_ascii=False)
            except (TypeError, ValueError):
                return None
            parts.append(f"{name}:{sig}")
        return "|".join(parts)

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

    # ── Action 行为指引（注入 system prompt）──────────────────────────────────

    def _build_system_prompt(self, req: Request) -> str:
        """
        组装 system prompt：
          1. 职责角色定义（QA/Executor 的 system_prompt）
          2. 领域人格（DOMAIN_PERSONA，按 req.domain 挂载 —— 领域只影响行为风格）
          3. 动态 Skills（业务 SOP：目录 + 命中提示 + 全量内容，模型自主选择）
          4. 意图指引（Action 层行为指令：查询只读 / 请求办理 / 投诉给路径等）
          5. 时间上下文（当前日期/星期/第几节/第几周）—— 所有请求统一注入，
             是"今天有什么课""现在第几节"类问答的前提。
        """
        prompt = self.system_prompt

        if req.domain is not None:
            persona = DOMAIN_PERSONA.get(req.domain)
            if persona:
                prompt = f"{prompt}\n\n[领域人格]\n{persona}"

        if self._skill_manager is not None:
            # 优先读 Runtime SkillMiddleware 的解析缓存（按消息指纹隔离，
            # 结果全链路一致）；无缓存时现场解析（直接调用/无 state 路径兼容）。
            skill_prompt = None
            if req.state is not None:
                prompts = req.state.meta.get("skill_prompt_by_msg")
                key = self._skill_manager.cache_key(req.message, req.history)
                if prompts and key in prompts:
                    skill_prompt = prompts[key]
            if skill_prompt is None:
                skill_prompt = self._skill_manager.prompt_for(req.message, None, req.history)
            if skill_prompt:
                prompt = f"{prompt}\n\n[动态 Skills]\n{skill_prompt}"

        if req.action is not None:
            guidance = ACTION_GUIDANCE.get(req.action)
            if guidance:
                prompt = f"{prompt}\n\n[意图指引]\n{guidance}"

        from personal.time_context import build_time_context
        prompt = f"{prompt}\n\n{build_time_context()}"
        return prompt


class QAAgent(BaseAgent):
    """问答职责角色：只读工具面 + 检索/引用规范。

    处理查询/问候/投诉/其他等非执行类请求；角色级只读边界（write_allowed=False）
    保证即使路由误判，写工具也不会暴露给问答角色（防御纵深）。
    """
    agent_type     = AgentType.QA
    write_allowed  = False
    system_prompt = (
        "你是西电校园智慧助手（EchoGuide）的问答角色，负责查询类请求：知识检索、信息查询与咨询建议。"
        "根据上方 [领域人格] 调整回答风格与侧重点；涉及政策、规定、办事流程或校园设施信息时，"
        "优先调用 knowledge_search 或对应的查询工具，回答末尾用 [1][2] 标注引用来源。"
        "不要编造具体的分数、排名、价格、电话、截止日期或审批结果；"
        "涉及个人数据（课表/待办）只能来自工具结果，涉及学籍、审批等事项时提示以教务处/学院最新通知为准。"
    )


class ExecutorAgent(BaseAgent):
    """执行职责角色：全量工具面（含写）+ 执行确认规范。

    处理请求办理类请求（REQUEST）：操作用户个人数据（待办/日程）与调用外部工具完成指令。
    写操作由 Action 层门禁兜底（非 REQUEST 动作不会路由到此角色）。
    """
    agent_type    = AgentType.EXECUTOR
    system_prompt = (
        "你是西电校园智慧助手（EchoGuide）的执行角色，负责请求办理类任务："
        "操作用户个人数据（待办/日程）与调用工具完成用户指令。"
        "执行前先确认参数齐全；写入类操作完成后在回答中明确回执（如「已添加待办：xxx」），"
        "操作失败时如实说明原因，不要谎报成功。"
        "涉及用户数据的操作只针对当前用户（user_id 由系统注入）；"
        "无法执行的操作给出替代路径（如引导用户自行到对应系统/网点办理）。"
    )


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    西电校园智慧助手的编排器：单 Agent 执行 + 按需多 Agent 协作。

    - 职责角色：QA（问答，只读工具面）/ EXECUTOR（执行，含写工具面），各配
      Fast/Deep 双实例，按 action 选择；领域分类只用来挂载
      领域人格（DOMAIN_PERSONA）与 Skills —— 领域是"顾问"，不是"门卫"；
    - 工具可见性：公共工具层（所有 agent_exposed=True 的工具）+ 双层门禁
      （注册级 agent_exposed + Action 级读写策略），不按领域剪裁；
    - 按需多 Agent 协作：复杂度判定为 parallel/dependent 时，Planner 拆分任务
      DAG（任务角色标签沿用领域值），每个任务独立上下文并行执行、依赖注入
      SharedState，Synthesizer 合并（与 Anthropic MAR 广度并行同构）。
    """

    # 领域 → 协作任务角色标签（仅用于 DAG 拆分与 LLM 任务链校验，
    # 不用于执行实体选择——执行实体是职责角色 QA/EXECUTOR）。
    _DOMAIN_ROLE: Dict[IntentDomain, AgentType] = {
        IntentDomain.ACADEMIC:    AgentType.ACADEMIC,
        IntentDomain.CAMPUS_LIFE: AgentType.CAMPUS_LIFE,
        IntentDomain.AFFAIRS:     AgentType.AFFAIRS,
        IntentDomain.IT_HELP:     AgentType.IT_HELP,
        IntentDomain.PERSONAL:    AgentType.PERSONAL,
    }

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
        # 统一模型调用入口：意图识别 / 工具循环 / 合成 / 出口校验全部经
        # ModelGateway 进出（模型调用计数、token 统计、预算、Trace 口径一致）。
        self._gateway = self._runtime.model_gateway
        self._intent_recognizer = IntentRecognizer(
            api_key=fast_key, base_url=fast_url, model=fast_name,
            gateway=self._gateway,
        )
        # 轻量多 Agent 协作链：Planner（拆分，纯规则无 LLM）→ Executor（分波执行）→ SharedState → Synthesizer（合并）
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

        # 职责角色 × Fast/Deep 双实例：QA（问答）/ EXECUTOR（执行）两个真实
        # 执行角色，工具面与行为规范不同（角色边界），人格/Skills 按领域挂载。
        def agents(cls):
            return [
                cls(fast_client, fast_name, skill_manager, tool_manager, self._profiles[ProfileName.FAST]),
                cls(deep_client, deep_name, skill_manager, tool_manager, self._profiles[ProfileName.DEEP]),
            ]

        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.QA: agents(QAAgent),
            AgentType.EXECUTOR: agents(ExecutorAgent),
        }
        # Runtime 广播到所有 Agent 实例（工具/模型边界钩子），与 set_* 注入同一模式
        for agent_list in self._pool.values():
            for agent in agent_list:
                agent._runtime = self._runtime

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
        """把外部 MCP 工具显式加入公共工具层（任何请求可见，仍受 Action 门禁）。

        外部工具注册时默认 agent_exposed=False（双重不可见：注册级 + 公共层
        只放行 agent_exposed=True）；这里置 True 即进入公共层。v2 起不绑定任何
        领域/Agent——领域不构成工具门禁，新工具接入只需声明读写属性。
        """
        names = set(tool_names)
        if not names or self._tool_manager is None:
            return
        for tool in self._tool_manager._tools.values():
            if tool.name in names:
                tool.agent_exposed = True
        logger.info("外部 MCP 工具已进入公共工具层: %s", sorted(names))

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
                agent_type=AgentType.QA,
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
          意图识别（领域×动作）→ 复杂度判定 → 路由选 Agent → 执行 → 检查升级 → 返回结果

        on_event: 可选异步回调，透传给 Agent（SSE 流式输出 / 工具调用可视化）。
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
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
                # 查询理解产出（v4）：模型建议参考的 Skill / 是否需要知识检索，透出观测
                req.state_query = {
                    "skills_to_reference": list(intent_result.skills_to_reference),
                    "needs_knowledge": intent_result.needs_knowledge,
                }
                if on_event is not None:
                    await on_event({
                        "type": "meta",
                        "domain": req.domain.value if req.domain else "other",
                        "action": req.action.value if req.action else "other",
                        "confidence": intent_result.confidence,
                        "classifier_stage": intent_result.classifier_stage,
                        "skills_to_reference": list(intent_result.skills_to_reference),
                        "needs_knowledge": intent_result.needs_knowledge,
                    })

        # 复杂度判定（意图识别的一部分）：意图识别走了 LLM 时复用其 complexity 输出
        # （零额外调用）；否则走"免费规则 → 拿不准升级 LLM 确认"的级联。
        llm_signal = (
            intent_result.complexity
            if intent_result is not None and req.classifier_stage == "llm"
            else None
        )
        complexity = await self._decide_complexity(req, llm_signal)
        if req.benchmark_strategy == "single_agent" and complexity.mode != "single":
            complexity = ComplexityDecision("single", "Benchmark 单 Agent 基线", [self._role_for(req)])
        req.complexity_mode = complexity.mode
        req.complexity_reason = complexity.reason
        req.profile = (
            ProfileName.DEEP
            if "always_deep" in req.benchmark_strategy
            else self._select_profile(req, complexity)
        )
        if req.state is not None:
            req.state.complexity_mode = complexity.mode
            req.state.profile = req.profile.value if req.profile else ""
        if on_event is not None:
            await on_event({
                "type": "meta",
                "mode": complexity.mode,
                "profile": req.profile.value,
                "complexity_reason": complexity.reason,
            })
        if complexity.mode in ("parallel", "dependent"):
            return await self.run_parallel(req, complexity.targets, on_event, tasks=complexity.tasks)

        # 2. 职责角色：按 action 选择 QA/EXECUTOR（领域只影响人格/Skills 挂载与复杂度判定）
        agent_type = self._role_for(req)
        if on_event is not None:
            await on_event({"type": "meta", "agent": agent_type.value})

        # 3. 执行（含降级）
        response = await self._execute(req, agent_type, on_event)

        # 4. 出口校验（Grounding）：规则全量 + 可选 LLM 判定；只标注不阻断
        verification = await self._verify(req, response)

        execution = self._execution_meta(
            req,
            mode="single",
            agents=[response.agent_type],
            responses=[response],
        )
        execution["verification"] = verification
        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            domain=req.domain,
            action=req.action,
            latency_ms=(time.monotonic() - t0) * 1000,
            tools_used=response.tools_used,
            tool_evidence=response.tool_evidence,
            execution=execution,
        )

    async def run_parallel(
        self,
        req: Request,
        agent_types: List[AgentType],
        on_event: Optional[Any] = None,
        tasks: Optional[List[Task]] = None,
    ) -> OrchestratorResult:
        """
        轻量多 Agent 协作（非 Agent 间聊天），职责链清晰：

          Planner 拆分任务 DAG（自包含任务，支持跨任务依赖）
            → Executor 按 depends_on 分波并行执行，结果写入 SharedState，
              依赖任务执行时注入协作上下文（能看到前序 Agent 结果）
            → Synthesizer 读取 SharedState 合并为最终回复（LLM 失败降级拼接）

        tasks 由调用方给出时（LLM 判定的依赖链，已通过 _tasks_from_llm 校验）
        直接采用；否则 Planner 现场生成。OrchestratorResult 字段保持兼容。
        """
        t0 = time.monotonic()

        # 1. 任务来源：LLM 依赖链（已校验）或 Planner 生成（规则 DAG）
        req.profile = ProfileName.DEEP
        plan = tasks if tasks is not None else TaskPlanner().plan(req, agent_types)
        plan = list(plan)[: self._runtime.policy.max_tasks]

        # 2. Executor：分波执行，产出 SharedState
        shared = await self._executor.execute(
            req, plan, on_event, max_tasks=self._runtime.policy.max_tasks,
        )

        # 3. Synthesizer：合并最终回复
        responses = list(shared.all_results().values())
        final_text = await self._synthesizer.synthesize(req, responses)

        tools_used = [tool for r in responses for tool in r.tools_used]
        tool_evidence = [e for r in responses for e in r.tool_evidence]

        # 4. 出口校验：对合成后的最终回复做 Grounding（证据 = 各任务证据汇总）
        synthesized = AgentResponse(
            agent_type=agent_types[0],
            content=final_text,
            success=True,
            tools_used=tools_used,
            tool_evidence=tool_evidence,
            profile=req.profile.value if req.profile else "deep",
        )
        verification = await self._verify(req, synthesized)

        execution = self._execution_meta(
            req,
            mode=req.complexity_mode,
            agents=agent_types,
            responses=responses,
            tasks=shared.task_meta(),
        )
        execution["verification"] = verification
        return OrchestratorResult(
            request_id=req.request_id,
            response=synthesized.content,
            agent_type=agent_types[0],
            intent=req.intent,
            domain=req.domain,
            action=req.action,
            latency_ms=(time.monotonic() - t0) * 1000,
            tools_used=tools_used,
            tool_evidence=tool_evidence,
            execution=execution,
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
            domain=self._task_domain(task, req),  # 任务角色 → 领域挂载键（人格/Skills）
            action=req.action,
            request_id=req.request_id,
            state=req.state,   # 协作任务继承运行状态（中间件钩子/预算计数不断链）
        )
        task_req.profile = ProfileName.DEEP
        task_req.complexity_mode = req.complexity_mode
        task_req.complexity_reason = req.complexity_reason
        task_req.classifier_stage = req.classifier_stage
        task_req.confidence = req.confidence
        snapshot = shared.snapshot()
        if snapshot:
            # 注入协作上下文：让本任务知道其他 Agent 已给出什么（避免重复检索/重复回答）
            task_req.context = f"{task_req.context}\n\n[协作上下文]\n{snapshot}".strip()
        # 执行职责角色：REQUEST 请求的任务可能包含写操作 → EXECUTOR；否则 QA。
        exec_role = (
            AgentType.EXECUTOR if task_req.action == IntentAction.REQUEST
            else AgentType.QA
        )
        response = await self._execute(task_req, exec_role, on_event)
        # 展示/合成标签回填为任务角色（领域）：执行实体是职责角色，
        # 但协作结果按任务角色区分（Synthesizer 分节标签、execution.agents 可观测）。
        response.agent_type = task.agent_type

        # 依赖 DAG 的终点是一次真实写操作；模型只给出建议而忘记调用工具时，
        # Executor 按任务的 required_tool 后置条件补执行，避免出现
        # "任务 success 但待办未创建"。由 Planner 声明，不硬编码具体任务标识。
        if task.required_tool and task.required_tool not in response.tools_used:
            if req.action is not None and not action_allows_tool(req.action, task.required_tool):
                # Action 层策略（如 QUERY 只读）：补执行写操作被禁止，跳过（策略内不执行即符合预期）
                logger.warning(
                    f"协作任务补执行被 Action 策略拒绝: {task.required_tool} (action={req.action.value})"
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
                context={"agent_type": task.agent_type.value, "user_id": req.user_id},
                use_rewrite=False,
            ) if self._tool_manager is not None else None
            if result is not None and result.success:
                response.tools_used.append(task.required_tool)
                response.content = f"{response.content.rstrip()}\n\n已按协作计划记录待办：补办校园卡。"
                if on_event is not None:
                    await on_event({"type": "tool", "name": task.required_tool, "status": "done", "titles": []})
            else:
                response.success = False
                error = getattr(result, "error", "工具管理器不可用")
                response.content = f"{response.content.rstrip()}\n\n待办写入失败：{error}"
        return response

    # ── 协作任务领域挂载 ──────────────────────────────────────────────────────

    @staticmethod
    def _role_for(req: Request) -> AgentType:
        """职责角色选择：REQUEST（请求办理）→ EXECUTOR；其余动作 → QA（只读）。"""
        if req.action == IntentAction.REQUEST:
            return AgentType.EXECUTOR
        return AgentType.QA

    @staticmethod
    def _task_domain(task: Task, req: Request) -> Optional[IntentDomain]:
        """协作任务的领域挂载键：任务角色 → 领域（角色值与领域值同字面）。

        每个任务是独立上下文的执行实体，其人格/Skills 由任务角色决定，
        而非原请求的领域。
        """
        try:
            return IntentDomain(task.agent_type.value)
        except ValueError:
            return req.domain

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

        # 保持顺序去重。任务角色标签与执行职责解耦：任务由 QA/EXECUTOR
        # 职责角色执行（_run_task 按 action 推导执行角色）。
        return list(dict.fromkeys(targets))

    def _complexity_decision(self, req: Request) -> ComplexityDecision:
        max_agents = self._runtime.policy.max_agents
        dependent = TaskPlanner._plan_schedule_errand(req)
        if dependent:
            targets = list(dict.fromkeys(task.agent_type for task in dependent))[:max_agents]
            return ComplexityDecision("dependent", "日程、办事与待办存在前后依赖", targets)

        targets = self._collaboration_targets(req)
        connectors = ("同时", "还要", "并且", "另外", "以及", "顺便", "然后")
        if len(targets) >= 2 and any(word in req.message for word in connectors):
            return ComplexityDecision("parallel", "显式复合语义涉及多个校园领域", targets[:max_agents])

        return ComplexityDecision("single", "单领域或无显式复合语义", [self._role_for(req)])

    # ── LLM 复杂度判定（规则筛 + LLM 升级确认）──────────────────────────────
    #
    # 三段式：
    #   1. 意图识别已走 LLM（stage=="llm"）→ 复用其 complexity 输出（零额外调用）；
    #   2. 免费关键词规则先判（_complexity_decision，行为与旧版一致）；
    #   3. 规则判 single 但预筛认为"拿不准"（多从句/长句/多领域无连接词）
    #      → judge_complexity 升级确认；LLM 结论通过规则校验则采用（LLM 说了算）。

    # 升级预筛的从句切分器：连接词与 _complexity_decision 的 connectors 同源
    _COMPLEXITY_UPGRADE_CLAUSE_RE = re.compile(
        r"[，。；、,;]+|(?:同时|另外|顺便|然后|再|还要|以及|并且)"
    )

    async def _decide_complexity(
        self,
        req: Request,
        llm_complexity: Optional[Any],
    ) -> ComplexityDecision:
        """三段式复杂度判定（见上方注释）。"""
        if llm_complexity is not None:
            decision = self._complexity_from_llm(llm_complexity, req)
            if decision is not None:
                return decision
            # 复用失败（LLM 输出非法）：规则兜底，不再升级 —— 避免对同一条请求重复调用 LLM
            return self._complexity_decision(req)

        complexity = self._complexity_decision(req)
        if complexity.mode != "single" or not self._needs_llm_complexity(req):
            return complexity
        signal = await self._intent_recognizer.judge_complexity(req.message, req.history, state=req.state)
        llm_decision = self._complexity_from_llm(signal, req)
        return llm_decision if llm_decision is not None else complexity

    def _needs_llm_complexity(self, req: Request) -> bool:
        """
        升级预筛：规则判 single 后，判断这条请求是否值得升级 LLM 确认复杂度。

        任一信号命中即升级：
          1. 消息被切出 ≥3 个从句（信息量大，可能复合）；
          2. 长消息（>24 字）且 ≥2 个从句；
          3. 领域关键词命中 ≥2 个领域但无显式连接词（旧规则会判 single 的"隐式复合"）。
        """
        msg = req.message
        clauses = [c for c in self._COMPLEXITY_UPGRADE_CLAUSE_RE.split(msg) if c.strip()]
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

    def _complexity_from_llm(
        self,
        signal: Optional[Any],
        req: Request,
    ) -> Optional[ComplexityDecision]:
        """
        把 LLM 复杂度信号（IntentResult.complexity / judge_complexity 返回值）校验并
        映射为 ComplexityDecision。LLM 输出不可信：任何非法 → None（调用方回落关键词规则）。

        - targets：领域值 → AgentType，未知领域丢弃、去重、≤3、过滤无实例的；
        - mode=dependent：任务链必须通过 _tasks_from_llm 硬校验，否则整链作废。
        """
        if signal is None:
            return None
        mode = getattr(signal, "mode", None)
        if mode not in ("single", "parallel", "dependent"):
            return None
        reason = str(getattr(signal, "reason", "") or "")[:120]

        targets: List[AgentType] = []
        for value in getattr(signal, "targets", []) or []:
            try:
                domain = IntentDomain(str(value))
            except ValueError:
                continue
            agent_type = self._DOMAIN_ROLE.get(domain)
            if agent_type is not None and agent_type not in targets:
                targets.append(agent_type)

        if mode == "single":
            return ComplexityDecision(
                "single", reason or "LLM 判定单领域", targets or [self._role_for(req)],
            )
        if not targets:
            return None
        if mode == "parallel":
            return ComplexityDecision(
                "parallel", reason or "LLM 判定多领域并行", targets[: self._runtime.policy.max_agents],
            )
        tasks = self._tasks_from_llm(getattr(signal, "tasks", None), req)
        if tasks is None:
            return None
        return ComplexityDecision(
            "dependent", reason or "LLM 判定任务存在依赖", targets[: self._runtime.policy.max_agents], tasks=tasks,
        )

    def _tasks_from_llm(
        self,
        raw_tasks: Optional[Any],
        req: Request,
    ) -> Optional[List[Task]]:
        """
        LLM 任务链硬校验（全部为硬性条件，任一不满足 → None 整链作废）：

          - 1~6 个任务（TaskExecutor 上限）；id 非空且唯一
          - agent 必须是已知领域且池中有实例
          - depends_on 引用的 id 必须存在，且无环（拓扑检查）
          - message 缺失时回落自包含格式（含原始用户请求）
          - required_tool 一律不采用：后置条件是关键词规则链的保险带
            （规则知道精确写参数）；LLM 链中模型会在自己的工具循环里完成写操作，
            无参数的硬执行写工具比不执行更危险。
        """
        if not isinstance(raw_tasks, list) or not (
            1 <= len(raw_tasks) <= self._runtime.policy.max_tasks
        ):
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
                domain = IntentDomain(str(raw.get("agent") or ""))
            except ValueError:
                return None
            agent_type = self._DOMAIN_ROLE.get(domain)
            if agent_type is None:
                return None
            goal = str(raw.get("goal") or "").strip() or self._task_goal_fallback(agent_type)
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
                agent_type=agent_type,
                goal=goal,
                message=message,
                depends_on=deps,
            ))
            seen_ids.add(task_id)
        # 依赖引用与无环校验（第二次遍历：LLM 可能引用后置任务的 id）
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
    def _task_goal_fallback(agent_type: AgentType) -> str:
        """LLM 未给 goal 时的兜底（与 TaskPlanner.GOAL_TEMPLATES 同源）。"""
        return TaskPlanner.GOAL_TEMPLATES.get(agent_type, "回答用户的请求")

    @staticmethod
    def _select_profile(req: Request, complexity: ComplexityDecision) -> ProfileName:
        if complexity.mode != "single":
            return ProfileName.DEEP
        deep_markers = ("转专业", "保研", "培养方案", "政策", "规定", "条件", "比较", "分析", "检索资料", "给出来源")
        if any(marker in req.message for marker in deep_markers):
            return ProfileName.DEEP
        deterministic_markers = (
            "加权", "平均成绩", "校园卡", "请假", "在读证明", "缓考",
            "校园网", "vpn", "统一身份认证", "教务系统", "课表", "待办",
            "校车", "天气", "图书馆", "体育馆",
        )
        if any(marker in req.message.lower() for marker in deterministic_markers):
            return ProfileName.FAST
        # 置信度低于 Embedding 命中线（0.80）视为低置信度 → DEEP，
        # 与 intent_recognizer 的 embedding_threshold 联动（改阈值时同步改这里）
        if req.classifier_stage == "llm" or (req.confidence and req.confidence < 0.80):
            return ProfileName.DEEP
        return ProfileName.FAST

    @staticmethod
    def _execution_meta(
        req: Request,
        *,
        mode: str,
        agents: List[AgentType],
        responses: List[AgentResponse],
        tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "mode": mode,
            "profile": req.profile.value if req.profile else "fast",
            "domain": req.domain.value if req.domain else None,
            "classifier_stage": req.classifier_stage,
            "complexity_reason": req.complexity_reason,
            "agents": list(dict.fromkeys(agent.value for agent in agents)),
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
        # 查询理解产出（v4）：模型建议的 Skill / 检索需求（观测与评测用）
        if req.state_query:
            meta["query_understanding"] = req.state_query
        return meta

    def _best_agent(self, agent_type: Optional[AgentType] = None, profile: Optional[ProfileName] = None) -> Optional[BaseAgent]:
        """
        职责角色实例选择：从该角色池（Fast/Deep 双实例）中按在线表现选实例
        （成功率高、延迟低、避开正在处理的实例）；缺省 QA 角色。

        领域任务角色标签不参与选择——只有职责角色（QA/EXECUTOR）有实体。
        """
        agents = self._pool.get(agent_type or AgentType.QA, [])
        if not agents:
            return None
        if profile is not None:
            preferred = [agent for agent in agents if agent.profile.name == profile]
            if preferred:
                return max(preferred, key=lambda a: a.stats.effective_routing_score())
        return max(agents, key=lambda a: a.stats.effective_routing_score())

    async def _execute(
        self,
        req: Request,
        agent_type: Optional[AgentType] = None,
        on_event: Optional[Any] = None,
    ) -> AgentResponse:
        """按职责角色执行（QA/EXECUTOR）；Fast 失败时同角色重试 Deep。

        agent_type 为职责角色（缺省按 _role_for 从 action 推导）；
        协作任务由 _run_task 传入其执行角色。
        """
        role = agent_type or self._role_for(req)
        agent = self._best_agent(role, req.profile)
        if agent is None:
            return AgentResponse(
                agent_type=role,
                content="助手暂时不可用，请稍后重试，或直接联系辅导员/教务老师。",
                success=False,
            )

        response = await self._handle_with_runtime(req, agent, on_event)

        # Fast 失败 → Deep 重试（同职责角色，受 policy.max_retries 约束）
        if not response.success and req.profile == ProfileName.FAST:
            max_retries = 1
            if req.state is not None and req.state.policy is not None:
                max_retries = req.state.policy.max_retries
            if req.state is not None and req.state.retry_count >= max_retries:
                logger.warning(
                    f"{role.value} Fast 执行失败，已达降级次数上限 {max_retries}，不再重试"
                )
                return response
            logger.warning(f"{role.value} Fast 执行失败，降级重试 Deep")
            if req.state is not None:
                req.state.retry_count += 1
            fallback = self._best_agent(role, ProfileName.DEEP)
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
        result = await self._verifier.verify(
            req=req,
            content=response.content,
            tools_used=response.tools_used,
            tool_evidence=response.tool_evidence,
            profile=response.profile,
            write_tools=WRITE_TOOLS,
        )
        for flag in result.flags:
            self._verification_flags[flag] = self._verification_flags.get(flag, 0) + 1
        if result.disclaimer:
            response.content = f"{response.content.rstrip()}\n\n{result.disclaimer}"
        return result.summary()

    # ── 统计（供 Monitor 读取）────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        result = {}
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}.{agent.profile.name.value}"
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
        接收 Monitor 的在线表现反馈，动态调整路由惩罚项。

        penalties 的 key 使用 get_stats() 中的 agent key，例如 qa.fast / executor.deep。
        兼容旧版 qa_0 / executor_0 键。
        """
        for agent_type, agents in self._pool.items():
            for i, agent in enumerate(agents):
                key = f"{agent_type.value}.{agent.profile.name.value}"
                legacy_key = f"{agent_type.value}_{i}"
                penalty = penalties.get(key, penalties.get(legacy_key, 0.0))
                agent.stats.monitor_penalty = min(max(penalty, 0.0), 0.9)
