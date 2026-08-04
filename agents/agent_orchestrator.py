"""
亮点：多 Agent 路由与编排（领域路由）

核心问题：多 Agent 情况下如何做 Routing？

路由策略（三层决策）：
  1. 领域路由 —— 根据 IntentDomain（学业/生活/校务/IT）直接映射到专属 Agent。
     意图体系为「领域 domain × 动作 action」二维，路由只看领域，
     修复旧版"请求句式（帮我/我要）被标成通用 REQUEST 后丢失领域"的缺陷。
  2. 性能路由 —— 同类 Agent 有多个时，选成功率最高、延迟最低的
  3. 降级路由 —— 专属 Agent 不可用时，自动降级到 AcademicAgent（校园通用接待）

并行协作：
  - 复杂问题（如"教务系统故障 + 选课问题"）可同时派发给多个 Agent
  - 结果由 Orchestrator 合并后返回

升级机制：
  - 动作维度为转人工 / 置信度低于阈值 / 紧急度 CRITICAL → 自动升级
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

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

        工具声明使用 Anthropic tool_use 规范：name / description / input_schema。
        """
        if self._tool_manager is None:
            return []
        tools = []
        for name, tool in self._tool_manager._tools.items():
            if not getattr(tool, "agent_exposed", True):
                continue
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
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": getattr(block, "id", ""),
                            "content": self._clean_text(data) if data is not None else f"工具执行失败: {error}",
                        }],
                    })
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
            resp = await self._client.messages.create(
                model=self._model, max_tokens=1024,
                system=self._build_system_prompt(req),
                messages=messages,
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
        """
        stream = self._client.messages.stream(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=tools or None,
        )
        async with stream:
            async for text in stream.text_stream:
                await on_event({"type": "delta", "text": text})
            final = await stream.get_final_message()
        return final

    async def _execute_tool(self, name: str, params: Dict, req: Request) -> tuple[Any, Optional[str]]:
        """执行工具，返回 (结构化数据, 错误信息)。"""
        if self._tool_manager is None:
            return None, "工具管理器不可用"
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
        """把动态加载的 Skills 拼入 system prompt，让业务规则随请求生效。"""
        if self._skill_manager is None:
            return self.system_prompt
        skill_prompt = self._skill_manager.prompt_for(req.message, self.agent_type.value, req.history)
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

    # 领域 → Agent 类型的静态映射（路由表）。路由只看领域，动作只影响行为。
    _INTENT_ROUTING: Dict[IntentDomain, AgentType] = {
        IntentDomain.ACADEMIC:    AgentType.ACADEMIC,
        IntentDomain.CAMPUS_LIFE: AgentType.CAMPUS_LIFE,
        IntentDomain.AFFAIRS:     AgentType.AFFAIRS,
        IntentDomain.IT_HELP:     AgentType.IT_HELP,
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

        self._intent_recognizer = IntentRecognizer(api_key=api_key, base_url=base_url, model=model)
        self._skill_manager = skill_manager
        self._tool_manager  = tool_manager

        # Agent 池：每种类型可有多个实例（水平扩展）
        self._pool: Dict[AgentType, List[BaseAgent]] = {
            AgentType.ACADEMIC:    [AcademicAgent(client, model, skill_manager, tool_manager)],
            AgentType.CAMPUS_LIFE: [CampusLifeAgent(client, model, skill_manager, tool_manager)],
            AgentType.AFFAIRS:     [AffairsAgent(client, model, skill_manager, tool_manager)],
            AgentType.IT_HELP:     [ITHelpAgent(client, model, skill_manager, tool_manager)],
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
        并行派发给多个 Agent，合并结果。
        适用于复杂问题（如同时涉及 IT 故障和选课问题）。
        """
        t0 = time.monotonic()
        tasks = [self._execute(req, at, on_event) for at in agent_types]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # 合并：拼接所有成功响应
        parts = []
        tools_used: List[str] = []
        for r in responses:
            if isinstance(r, AgentResponse) and r.success:
                parts.append(f"[{r.agent_type.value}]\n{r.content}")
                tools_used.extend(r.tools_used)

        combined = "\n\n".join(parts) if parts else "抱歉，多个助手模块暂时都没能处理成功，请稍后重试。"
        escalated = any(isinstance(r, AgentResponse) and r.escalate for r in responses)

        return OrchestratorResult(
            request_id=req.request_id,
            response=combined,
            agent_type=agent_types[0],
            intent=req.intent,
            domain=req.domain,
            action=req.action,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            tools_used=tools_used,
        )

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
