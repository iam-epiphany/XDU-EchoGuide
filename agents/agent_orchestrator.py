"""
亮点：多 Agent 路由与编排

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 意图路由 —— 根据 IntentCategory 直接映射到专属 Agent
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 AcademicAgent（校园通用接待）

并行协作：
  - 复杂问题（如"技术问题 + 账单问题"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

升级机制：
  - Agent 置信度低于阈值 → 自动升级到更高级 Agent 或转人工
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class AgentType(Enum):
    ACADEMIC   = "academic"   # 学业支持：选课/课表/考试/成绩/绩点/重修
    CAMPUS_LIFE = "campus_life"  # 校园生活：宿舍/食堂/校车/校园卡/快递
    AFFAIRS    = "affairs"    # 校务咨询：校历/请假/奖学金/办事流程/注册
    IT_HELP    = "it_help"    # IT 助手：教务系统/校园网/VPN/邮箱/统一身份认证
    ESCALATION = "escalation"  # 转人工（辅导员/教务老师）


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


@dataclass
class Request:
    message:     str
    user_id:     str
    conv_id:     str
    context:     str = ""        # 来自 MemoryManager 的格式化上下文
    history:     Optional[List[Dict[str, str]]] = None  # 对话历史，传给意图识别
    intent:      Optional[IntentCategory] = None
    urgency:     Optional[UrgencyLevel]   = None
    request_id:  str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class OrchestratorResult:
    request_id:  str
    response:    str
    agent_type:  AgentType
    intent:      Optional[IntentCategory]
    escalated:   bool  = False
    latency_ms:  float = 0.0


# ── 基础 Agent ────────────────────────────────────────────────────────────────

class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用和统计。"""

    agent_type: AgentType
    system_prompt: str

    def __init__(self, client: AsyncAnthropic, model: str, skill_manager: Optional[Any] = None):
        self._client = client
        self._model  = model
        self._skill_manager = skill_manager
        self.stats   = AgentStats()

    async def handle(self, req: Request) -> AgentResponse:
        t0 = time.monotonic()
        self.stats.total += 1
        try:
            content = await self._call_llm(req)
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

    async def _call_llm(self, req: Request) -> str:
        def _clean(s: str) -> str:
            return s.encode("utf-8", errors="ignore").decode("utf-8")

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        messages.append({"role": "user", "content": _clean(req.message)})

        resp = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=self._build_system_prompt(req),
            messages=messages,
        )
        return resp.content[0].text

    def _build_system_prompt(self, req: Request) -> str:
        """把动态加载的 Skills 拼入 system prompt，让业务规则随请求生效。"""
        if self._skill_manager is None:
            return self.system_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value)
        if not skill_prompt:
            return self.system_prompt
        return f"{self.system_prompt}\n\n[动态 Skills]\n{skill_prompt}"

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


# ── 编排器 ────────────────────────────────────────────────────────────────────

class AgentOrchestrator:
    """
    西电校园智慧助手的多 Agent 编排器。

    路由逻辑（三层）：
      1. 意图 → Agent 类型映射（学业/生活/校务/IT 四大领域）
      2. 同类多实例时按 routing_score() 选最优
      3. 专属 Agent 失败时降级到 AcademicAgent（校园最高频诉求，承担通用接待）
    """

    # 意图 → Agent 类型的静态映射（路由表）
    _INTENT_ROUTING: Dict[IntentCategory, AgentType] = {
        IntentCategory.ACADEMIC:    AgentType.ACADEMIC,
        IntentCategory.CAMPUS_LIFE: AgentType.CAMPUS_LIFE,
        IntentCategory.AFFAIRS:     AgentType.AFFAIRS,
        IntentCategory.IT_HELP:     AgentType.IT_HELP,
        IntentCategory.ESCALATION:  AgentType.ESCALATION,
        # 通用意图（QUERY/REQUEST/GREETING 等）默认 → ACADEMIC（兜底接待）
    }

    def __init__(
        self,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        skill_manager: Optional[Any] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.ACADEMIC:    [AcademicAgent(client, model, skill_manager)],
            AgentType.CAMPUS_LIFE: [CampusLifeAgent(client, model, skill_manager)],
            AgentType.AFFAIRS:     [AffairsAgent(client, model, skill_manager)],
            AgentType.IT_HELP:     [ITHelpAgent(client, model, skill_manager)],
        }

    def set_skill_manager(self, skill_manager: Optional[Any]) -> None:
        """更新 SkillManager 引用，供运行时重载或测试替换使用。"""
        self._skill_manager = skill_manager
        for agents in self._pool.values():
            for agent in agents:
                agent._skill_manager = skill_manager

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def run(self, req: Request) -> OrchestratorResult:
        """
        处理一次请求的完整流程：
          意图识别 → 路由选 Agent → 执行 → 检查升级 → 返回结果
        """
        t0 = time.monotonic()

        # 1. 意图识别（如果调用方已识别则跳过）
        if req.intent is None:
            intent_result = await self._intent_recognizer.recognize(req.message, history=req.history)
            req.intent  = intent_result.intent
            req.urgency = intent_result.urgency

        # 复杂问题自动并行协作，例如同一句同时涉及教务系统故障和选课问题。
        collaboration = self._collaboration_targets(req)
        if len(collaboration) > 1:
            return await self.run_parallel(req, collaboration)

        # 2. 路由：选择 Agent 类型
        agent_type = self._route(req.intent, req.urgency)

        # 3. 执行（含降级）
        response = await self._execute(req, agent_type)

        # 4. 升级检查
        escalated = False
        if response.escalate or req.urgency == UrgencyLevel.CRITICAL or req.intent == IntentCategory.ESCALATION:
            escalated = True
            logger.warning(f"请求 {req.request_id} 触发升级: urgency={req.urgency}")
            # 生产环境：此处创建工单、通知辅导员/教务老师

        return OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    async def run_parallel(self, req: Request, agent_types: List[AgentType]) -> OrchestratorResult:
        """
        并行派发给多个 Agent，合并结果。
        适用于复杂问题（如同时涉及技术和账单）。
        """
        t0 = time.monotonic()
        tasks = [self._execute(req, at) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并：拼接所有成功响应
        parts = []
        for r in responses:
            if isinstance(r, AgentResponse) and r.success:
                parts.append(f"[{r.agent_type.value}]\n{r.content}")

        combined = "\n\n".join(parts) if parts else "抱歉，多个助手模块暂时都没能处理成功，请稍后重试。"
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)

        return OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=agent_types[0],
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    # ── 路由逻辑 ──────────────────────────────────────────────────────────────

    def _route(self, intent: Optional[IntentCategory], urgency: Optional[UrgencyLevel]) -> AgentType:
        """
        三层路由决策：
          1. 意图映射（学业/生活/校务/IT 四大领域）
          2. 紧急度覆盖（CRITICAL 直接升级到转人工）
          3. 默认 ACADEMIC（校园最高频诉求，承担通用接待）
        """
        if urgency == UrgencyLevel.CRITICAL:
            return AgentType.ESCALATION

        if intent and intent in self._INTENT_ROUTING:
            target = self._INTENT_ROUTING[intent]
            # 如果目标类型有可用实例则使用，否则降级
            if target in self._pool and self._pool[target]:
                return target

        return AgentType.ACADEMIC

    def _collaboration_targets(self, req: Request) -> List[AgentType]:
        """
        判断是否需要多个 Agent 并行协作。

        意图识别通常只返回一个主意图；这里用领域关键词补充检测复合问题，
        例如"教务系统打不开 + 选课问题"需要 IT 助手和学业支持 Agent 同时处理。
        """
        msg = req.message.lower()
        targets: List[AgentType] = []

        academic_kws = ["选课", "课表", "考试", "成绩", "绩点", "学分", "重修", "保研", "转专业"]
        campus_kws = ["食堂", "宿舍", "校车", "校园卡", "快递", "水电", "超市", "社团"]
        affairs_kws = ["校历", "请假", "奖学金", "助学金", "证明", "缴费", "学费", "注册"]
        it_kws = ["教务系统", "校园网", "vpn", "邮箱", "登录不上", "报错", "统一身份", "密码重置"]

        if req.intent == IntentCategory.ACADEMIC or any(kw in msg for kw in academic_kws):
            targets.append(AgentType.ACADEMIC)
        if req.intent == IntentCategory.CAMPUS_LIFE or any(kw in msg for kw in campus_kws):
            targets.append(AgentType.CAMPUS_LIFE)
        if req.intent == IntentCategory.AFFAIRS or any(kw in msg for kw in affairs_kws):
            targets.append(AgentType.AFFAIRS)
        if req.intent == IntentCategory.IT_HELP or any(kw in msg for kw in it_kws):
            targets.append(AgentType.IT_HELP)

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

    async def _execute(self, req: Request, agent_type: AgentType) -> AgentResponse:
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

        response = await agent.handle(req)

        # 专属 Agent 失败时降级到 AcademicAgent
        if not response.success and agent_type != AgentType.ACADEMIC:
            logger.warning(f"{agent_type.value} 失败，降级到 AcademicAgent")
            fallback = self._best_agent(AgentType.ACADEMIC)
            if fallback:
                response = await fallback.handle(req)

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
