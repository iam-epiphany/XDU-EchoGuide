"""
Task Agent —— 围绕一个 Task 的独立 Agent Run 执行体（Agent Harness 的通用执行器）。

v5 收口：不再有 QAAgent / ExecutorAgent 两个职责 Agent 类，也没有 Role→Agent Pool。
真正的 Agent 单位 = 围绕一个 Task 的一次独立 Agent Run：
  - 每个 Task 有独立 goal / message / domain / action / depends_on / allowed_write_tools；
  - QA / EXECUTOR 降级为 Execution Policy（write_policy_for）：
      非 REQUEST 动作 → READ_ONLY（只读工具面）；
      REQUEST 动作 → WRITE_ALLOWED（可暴露满足策略的写工具）；
    Action 只影响工具可见性、写权限与行为指引，不构成 Agent 身份；
  - 领域（IntentDomain）继续只做人格挂载键（见 persona.py；Skill 平级发现，
    不按领域过滤），
    不参与 Agent 类选择 —— 新增业务领域不需要新增 Agent 类；
  - Fast / Deep 只是 Execution Profile（profiles.py），每 Profile 一个执行实例，
    不是多个同构 Agent 竞争（asyncio 并发已足够，实例间无能力差异）。

工具可见性 = 公共工具层 + 三重门禁（防御纵深，交集决定）：
  Agent-exposed Tools ∩ Action Policy ∩ Task Capability
  1. 注册级 agent_exposed + effect 声明（Tool.is_agent_visible：未声明副作用
     的工具不可见，fail-closed）；
  2. Run 级写策略（write_policy_for：非 REQUEST 一律 READ_ONLY）+ Action 级
     Read/Write 策略（persona.action_allows_tool，写工具集合由
     tool_manager.write_tools() 从 Tool.effect 声明推导，不再手工维护黑名单）；
  3. Task 级写能力（req.allowed_write_tools，Planner 声明的写工具白名单；读工具不受限）：
     暴露与执行前双重校验，REQUEST 任务也只能调用任务允许的写工具。

Agentic RAG 工具循环（LLM 自主决定是否检索/调用工具）、上下文卸载、
Skill 渐进披露都在 BaseAgent/TaskAgent 中实现，是五条主线中 Agentic RAG 的执行体。
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from anthropic import AsyncAnthropic

from core.domains import IntentAction, IntentDomain
from agents.persona import ACTION_GUIDANCE, DOMAIN_PERSONA, action_allows_tool
from agents.profiles import ExecutionProfile, ProfileName
from memory.layered_store import (
    LayeredStore, OFFLOAD_CHARS, OFFLOAD_SUMMARY_CHARS, estimate_tokens,
)

logger = logging.getLogger(__name__)


class WritePolicy(Enum):
    """Task Run 执行策略（由 Task 的 action 决定，不再有角色实体）。"""
    READ_ONLY     = "read_only"      # QUERY/问候/投诉等：只读工具面
    WRITE_ALLOWED = "write_allowed"  # REQUEST：可暴露满足策略的写工具


def write_policy_for(action: Optional[IntentAction]) -> WritePolicy:
    """Action → Run 执行策略（纯函数）：仅 REQUEST 允许写，缺省 READ_ONLY。

    防御纵深：动作未知（None）或误判时一律只读，写工具不会暴露。
    """
    return WritePolicy.WRITE_ALLOWED if action == IntentAction.REQUEST else WritePolicy.READ_ONLY


@dataclass
class AgentStats:
    """Profile 执行统计（供 Monitor 读取；每 Profile 单实例，无实例级路由）。

    保留成功率/延迟/P50/P95/在途作为 Profile 级在线指标；
    不再有 routing_score / monitor_penalty —— 实例间不存在路由竞争。
    """
    total:     int   = 0
    success:   int   = 0
    total_ms:  float = 0.0
    in_flight: int   = 0
    _latencies: "deque[float]" = field(
        default_factory=lambda: deque(maxlen=200), repr=False,
    )

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.total if self.total else 0.0

    def record(self, latency_ms: float) -> None:
        """记录一次执行延迟（P50/P95 分位数依据）。"""
        self._latencies.append(latency_ms)

    @staticmethod
    def _percentile(sorted_latencies: List[float], pct: float) -> float:
        if not sorted_latencies:
            return 0.0
        idx = min(len(sorted_latencies) - 1, int(len(sorted_latencies) * pct))
        return sorted_latencies[idx]

    @property
    def p50_ms(self) -> float:
        return round(self._percentile(sorted(self._latencies), 0.50), 1)

    @property
    def p95_ms(self) -> float:
        return round(self._percentile(sorted(self._latencies), 0.95), 1)


@dataclass
class AgentResponse:
    content:     str
    success:     bool
    agent_type:  str = ""  # 展示标签：任务领域值（task.domain.value），空 = task_agent
    task_id:     str = ""  # 所属 Task（Agent Run 边界标识）
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

    @property
    def label(self) -> str:
        """展示标签：任务领域值优先，否则任务 id，最后回退 task_agent。"""
        return self.agent_type or self.task_id or "task_agent"


class BaseAgent:
    """所有 Agent 的基类，封装 LLM 调用、工具循环与统计。"""

    system_prompt: str = ""

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

    def _write_tools(self) -> FrozenSet[str]:
        """读写门禁的写工具集合：从工具管理器推导（Tool.write 声明）。

        工具管理器未实现 write_tools()（旧测试替身/第三方适配器）时回退空集合。
        """
        if self._tool_manager is None:
            return frozenset()
        derive = getattr(self._tool_manager, "write_tools", None)
        if derive is None:
            return frozenset()
        return derive()

    def _write_policy(self, req: Optional[Any]) -> WritePolicy:
        """本次 Task Run 的执行策略：由 req.action 决定（非 REQUEST 一律只读）。"""
        return write_policy_for(getattr(req, "action", None))

    async def _call_model(
        self,
        req: Any,
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

    async def handle(self, req: Any, on_event: Optional[Any] = None) -> AgentResponse:
        """
        处理一次请求（Task Agent Run）：LLM 工具调用循环（Agentic RAG）。

        on_event: 可选异步回调，接收过程事件（meta/tool/delta），供 SSE 流式输出使用。
        """
        t0 = time.monotonic()
        self.stats.total += 1
        self.stats.in_flight += 1
        tag = self.profile.name.value
        try:
            from core.tracing import span

            async with span("agent_handle", profile=tag):
                (content, tools_used, tool_evidence,
                 input_tokens, output_tokens,
                 offloaded_chars, saved_tokens) = await self._call_llm(req, on_event=on_event)
            # 引用是检索链路的执行后置条件，不依赖模型是否自觉把 URL 抄进正文。
            # v6 升级为 sentence-level citation：剥离模型自觉 [n] → 逐句确定性
            # 匹配证据（Dice + bge 余弦）→ 支持的句子追加 [i]，来源区按 [i]
            # 编号列出 —— 引用正确性由执行层保证，不再依赖模型自觉（core/grounding.py）。
            if tool_evidence:
                from core.grounding import annotate_citations, build_source_section

                annotated = await annotate_citations(content, tool_evidence)
                content = annotated["text"]
                sources = build_source_section(tool_evidence, annotated["citation_indices"])
                if sources:
                    content = f"{content.rstrip()}\n{sources}"
                if annotated["unsupported_sentences"]:
                    logger.info(
                        f"TaskAgent({tag}) {len(annotated['unsupported_sentences'])} 句无证据支持，未加引用"
                    )
            else:
                # 无证据时剥掉模型自标的 [n]：没有来源区的裸引用是噪声
                # （个人数据/工具回答场景模型仍可能照抄引用习惯）。
                from core.grounding import strip_citation_markers

                content = strip_citation_markers(content)
            ms = (time.monotonic() - t0) * 1000
            self.stats.success += 1
            self.stats.total_ms += ms
            self.stats.record(ms)
            return AgentResponse(
                content=content,
                success=True,
                task_id=getattr(req, "task_id", ""),
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
            self.stats.record(ms)
            logger.error(f"TaskAgent({tag}) 处理失败: {ex}")
            return AgentResponse(
                content="抱歉，处理你的请求时出现了问题，请稍后重试，或换个方式描述一下。",
                success=False,
                task_id=getattr(req, "task_id", ""),
                latency_ms=ms,
                profile=self.profile.name.value,
                model=self.profile.model,
            )
        finally:
            self.stats.in_flight = max(0, self.stats.in_flight - 1)

    # ── Agentic RAG：LLM 工具调用循环 ────────────────────────────────────────

    MAX_TOOL_ROUNDS = 3  # 无 Runtime policy 时的工具循环上限（与 Fast 路径默认一致）
    STAGNANT_ROUND_LIMIT = 2  # 无进展检测：连续重复轮数阈值（可被 ExecutionPolicy 覆盖）

    def _build_tools(self, req: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        把 MCPToolManager 中注册的工具与 Skill 工具暴露给 LLM（function calling）。

        可见性 = 公共工具层 ∩ Action 策略 ∩ Task 写能力（三层交集，防御纵深）：
          1. 注册级 is_agent_visible（agent_exposed + 显式 effect 声明：
             未声明副作用 fail-closed 不可见）；
          2. Run 级写策略（write_policy_for：非 REQUEST 一律 READ_ONLY）+
             Action 级读写策略（QUERY/GREETING 等动作下写工具不暴露）；
          3. Task 级写能力（req.allowed_write_tools，Planner 声明的写工具
             白名单）：只限制写工具，读工具不受限（REQUEST 也只拿必要写工具）。
        Skill 通过统一的 load_skill（渐进披露）追加在 MCP 工具之后，同受
        三重门禁；完整 SKILL.md 由模型按需加载。
        实例可设 _tool_allowlist 覆盖公共层（测试/定制场景）。
        """
        if self._tool_manager is None and self._skill_manager is None:
            return []
        allowed = getattr(self, "_tool_allowlist", None)
        capability = getattr(req, "allowed_write_tools", None)
        tools = []
        action = req.action if req is not None else None
        write_tools = self._write_tools()
        write_allowed = self._write_policy(req) == WritePolicy.WRITE_ALLOWED
        if self._tool_manager is not None:
            for name, tool in self._tool_manager._tools.items():
                if not getattr(tool, "is_agent_visible", True):
                    continue  # 注册级：未暴露或未声明副作用（fail-closed）
                if not write_allowed and name in write_tools:
                    continue  # Run 级写策略：非 REQUEST 动作下不暴露写工具
                if allowed is not None and name not in allowed:
                    continue  # 实例级覆盖：显式缩小可见集合
                if action is not None and not action_allows_tool(action, name, write_tools):
                    continue  # Action 层策略：查询/问候等动作下不暴露执行类工具
                if capability is not None and name in write_tools and name not in capability:
                    continue  # Task 级写能力：写工具必须在 Planner 声明的白名单内
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
                if action is not None and not action_allows_tool(action, name, write_tools):
                    continue
                if capability is not None and name in write_tools and name not in capability:
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

    @staticmethod
    def _operational_preflight(req: Any) -> Optional[Tuple[str, Dict[str, str]]]:
        """为具备确定性结构化数据源的领域补齐一次只读查询。

        领域是意图识别的结构化输出；完整用户请求交由各工具的别名/诊断树处理，
        因而新增事项或表达改写不需要在此维护词表。
        """
        domain = getattr(req, "domain", None)
        message = str(getattr(req, "message", "") or "")
        if domain == IntentDomain.AFFAIRS:
            return "query_affairs_process", {"service": message}
        if domain == IntentDomain.IT_HELP:
            return "diagnose_it_issue", {"system": message, "symptom": message}
        if domain == IntentDomain.CAMPUS_LIFE:
            # auto 由工具自身在全部公开结构化数据源上执行只读汇总；不在 Agent
            # 层按校车/图书馆等具体题型维护关键词，避免评测题驱动的过拟合。
            return "query_campus_info", {"category": "auto", "keyword": message}
        return None

    async def _call_llm(self, req: Any, on_event: Optional[Any] = None) -> tuple[str, List[str], List[Dict[str, Any]], int, int, int, int]:
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

        tools_used: List[str] = []
        tool_evidence: List[Dict[str, Any]] = []

        messages = []
        if req.context:
            messages.append({"role": "user", "content": f"[背景信息]\n{_clean(req.context)}"})
            messages.append({"role": "assistant", "content": "好的，我已了解背景信息。"})
        messages.append({"role": "user", "content": _clean(req.message)})
        # 知识类请求：Harness 预检索注入证据（retrieval-first）。
        # 不依赖模型自觉调用 knowledge_search——trace 实测模型大多靠参数记忆
        # 作答并输出越界引用，导致无证据可引用（faithfulness/citation 双低）。
        # 与工具循环同一执行边界（ToolManager.call + 证据采集格式），证据进入
        # tool_evidence 后自动获得引用标注与出口 Grounding 能力。
        if getattr(req, "auto_retrieve", False):
            evidence_text, auto_evidence = await self._auto_retrieve(req)
            if auto_evidence:
                tool_evidence.extend(auto_evidence)
                tools_used.append("knowledge_search")
                messages.append({"role": "user", "content": evidence_text})
                messages.append({"role": "assistant", "content": "好的，我已查看检索资料，将基于资料回答。"})

        # 检索证据覆盖的是政策/说明文本；对于带结构化公共数据的领域，在首轮
        # 同步执行一个由领域决定的只读工具。参数始终是完整用户请求，由工具自身
        # 的版本化别名/诊断规则解析，不在这里添加任何具体题目关键词。
        preflight = self._operational_preflight(req)
        if preflight is not None:
            name, params = preflight
            data, error = await self._execute_tool(name, params, req)
            if error is None:
                tools_used.append(name)
                messages.append({
                    "role": "user",
                    "content": f"[系统已查询结构化工具 {name}]\n{self._clean_text(data)}",
                })
                messages.append({"role": "assistant", "content": "好的，我会结合结构化查询结果回答。"})
            else:
                detail = f"operational preflight {name} failed: {error}"
                logger.warning(detail)
                if getattr(req, "state", None) is not None:
                    req.state.errors.append(detail)

        tools = self._build_tools(req)
        system = self._build_system_prompt(req)
        if tools:
            system = (
                f"{system}\n\n[工具使用]\n"
                "你可以调用可用工具来获取信息（如检索校园知识库、查询外部数据源）。"
                "若上下文中已提供「系统已检索资料」，不要重复调用检索工具；"
                "但这只是一份补充证据，不能替代专门工具：涉及个人状态、数值计算、"
                "结构化流程、故障诊断、地点/班次/开放时段或实时外部信息时，必须先调用"
                "与任务匹配的非检索工具，再结合资料回答。事实性句子后面用 [i] 标注对应资料编号。"
                "未提供资料且需要知识时先调用检索工具；一次调用无进展或检索不到时如实说明，"
                "不要反复调用同一工具或编造内容。"
            )

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
        tag = self.profile.name.value
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
                    # 证据采集泛化：任何返回「含标题/来源 URL 的条目列表」的工具结果
                    # 都视为检索证据（Verifier 不再特殊认识 knowledge_search，
                    # 新增检索类工具自动获得引用校验与出口 Grounding 能力）。
                    if isinstance(data, list) and any(
                        isinstance(item, dict) and (item.get("title") or item.get("source_url"))
                        for item in data
                    ):
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
                    # Skill 正文（load_skill）例外：规范全文必须留在上下文，不做卸载。
                    tool_text = self._clean_text(data) if data is not None else None
                    if (
                        tool_text is not None
                        and len(tool_text) > OFFLOAD_CHARS
                        and self._memory_store is not None
                        and name not in {"load_skill", "load_skill_resource"}
                    ):
                        try:
                            ref_id = await self._memory_store.save_ref(
                                req.user_id, req.conv_id, name, tool_text
                            )
                            char_len = len(tool_text)
                            full_estimate = estimate_tokens(tool_text)
                            tool_text = (
                                f"{tool_text[:OFFLOAD_SUMMARY_CHARS]}..."
                                f"[完整结果 refs/{ref_id}，共 {char_len} 字符]"
                            )
                            offloaded_chars += char_len - len(tool_text)
                            # estimate_tokens 只接受文本：按"全量 token - 摘要行 token"
                            # 统计卸载节省，传入字符数差（int）会抛 TypeError
                            saved_tokens += full_estimate - estimate_tokens(tool_text)
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
                            f"TaskAgent({tag}) 连续 {stagnant_rounds} 轮无进展（{round_sig}），强制收尾"
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
            logger.warning(f"TaskAgent({tag}) 工具调用达到轮次上限，普通调用收尾")
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

    async def _execute_tool(self, name: str, params: Dict, req: Any) -> tuple[Any, Optional[str]]:
        """执行工具，返回 (结构化数据, 错误信息)。"""
        write_tools = self._write_tools()
        write_policy = self._write_policy(req)
        tag = self.profile.name.value
        capability = getattr(req, "allowed_write_tools", None)
        # Task 级写能力（防御纵深）：暴露层之外，真正执行前再次校验 ——
        # 只依赖 Prompt / function schema 的软约束在这里变成硬门禁。
        # 只约束写工具：读工具不受任务白名单影响。
        if capability is not None and name in write_tools and name not in capability:
            logger.warning(f"TaskAgent({tag}) 尝试调用任务写能力外工具 {name}，已拒绝")
            return None, f"工具 {name} 不在当前任务写能力（allowed_write_tools）范围内"
        # Skill 工具拦截（渐进披露）：完整 SKILL.md 正文本地加载，不经过 MCPToolManager；
        # 与普通工具同受 allowlist 与 Action 门禁（防御纵深）。
        if name in {"load_skill", "load_skill_resource"}:
            allowed = getattr(self, "_tool_allowlist", None)
            if allowed is not None and name not in allowed:
                logger.warning(f"TaskAgent({tag}) 尝试调用权限外工具 {name}，已拒绝")
                return None, f"工具 {name} 不在当前执行权限范围内"
            if req.action is not None and not action_allows_tool(req.action, name, write_tools):
                logger.warning(
                    f"TaskAgent({tag}) 在 {req.action.value} 动作下尝试调用工具 {name}，已拒绝"
                )
                return None, f"工具 {name} 不在当前意图（{req.action.value}）的权限范围内"
            runtime = getattr(self, "_runtime", None)
            if runtime is not None and req.state is not None:
                await runtime.fire_tool_before(req.state, name, params)
            data, error = None, None
            if self._skill_manager is None:
                error = "技能管理器不可用"
            else:
                if name == "load_skill":
                    skill_name = str(params.get("skill_name", ""))
                    skill = self._skill_manager.get_skill(skill_name)
                    if skill is None:
                        error = f"技能 {skill_name} 不存在或已停用"
                    else:
                        from core.tracing import span
                        async with span("skill_load", skill=skill_name):
                            data = self._skill_manager.load_skill(skill_name)
                else:
                    skill_name = str(params.get("skill_name", ""))
                    relative_path = str(params.get("path", ""))
                    from core.tracing import span
                    async with span("skill_resource_load", skill=skill_name, path=relative_path):
                        data = self._skill_manager.load_skill_resource(skill_name, relative_path)
            if runtime is not None and req.state is not None:
                await runtime.fire_tool_after(req.state, name, data, error)
            if error:
                return None, error
            return data, None
        if self._tool_manager is None:
            return None, "工具管理器不可用"
        # 副作用声明 fail-closed：未声明 effect 的工具即使被请求也不执行
        # （不依赖手工黑名单，新写工具忘记声明时不会作为只读工具放行）。
        tool_obj = self._tool_manager._tools.get(name)
        if tool_obj is None or not getattr(tool_obj, "is_agent_visible", True):
            logger.warning(f"TaskAgent({tag}) 尝试调用未暴露/未声明副作用的工具 {name}，已拒绝")
            return None, f"工具 {name} 不在当前执行权限范围内"
        # Run 级写策略（防御纵深）：非 REQUEST 动作即使误判也拒绝写工具。
        if write_policy != WritePolicy.WRITE_ALLOWED and name in write_tools:
            logger.warning(
                f"TaskAgent({tag}) 在 {write_policy.value} 策略下尝试调用写工具 {name}，已拒绝"
            )
            return None, f"工具 {name} 不在当前执行策略（{write_policy.value}）权限范围内"
        # 权限边界（防御纵深）：公共工具层内工具由 Action 层策略把关（见下）；
        # 实例级 _tool_allowlist 覆盖（测试/定制）之外的工具直接拒绝。
        allowed = getattr(self, "_tool_allowlist", None)
        if allowed is not None and name not in allowed:
            logger.warning(f"TaskAgent({tag}) 尝试调用权限外工具 {name}，已拒绝")
            return None, f"工具 {name} 不在当前执行权限范围内"
        # Action 层权限（防御纵深，与 _build_tools 暴露层一致）：
        # 查询/问候等动作下，LLM 即使声明了执行类工具也直接拒绝
        if req.action is not None and not action_allows_tool(req.action, name, write_tools):
            logger.warning(
                f"TaskAgent({tag}) 在 {req.action.value} 动作下尝试调用工具 {name}，已拒绝"
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
                    context={
                        "agent_type": req.domain.value if req.domain is not None else "task_agent",
                        "user_id": req.user_id,
                    },
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

    async def _auto_retrieve(self, req: Any) -> Tuple[str, List[Dict[str, Any]]]:
        """Harness 预检索：知识类请求注入一次确定性检索证据（retrieval-first）。

        与模型工具循环同走 ToolManager.call（含改写/重排/缓存/卸载策略），
        返回 (注入文本, 证据列表)。检索失败或为空时返回空，不阻断主链路。
        """
        if self._tool_manager is None:
            return "", []
        name = "knowledge_search"
        tool = self._tool_manager._tools.get(name)
        if tool is None:
            return "", []
        # 追问轮检索补全：省略追问（"那几点关门呢？"）无主题词，直接检索会
        # 召回无关文档；用最近一轮用户消息补全 query（对话式 RAG 的标准做法）。
        query = getattr(req, "message", "") or ""
        if len(query) <= 12:
            history = getattr(req, "history", None) or []
            for m in reversed(history):
                if m.get("role") == "user" and m.get("content"):
                    query = f"{m['content']} {query}"
                    break
        params = {"query": query, "top_k": self.profile.rag_top_k}
        try:
            result = await self._tool_manager.call(
                name,
                params,
                context={
                    "agent_type": req.domain.value if req.domain is not None else "task_agent",
                    "user_id": getattr(req, "user_id", ""),
                },
                rerank_top_k=self.profile.rag_top_k if self.profile.use_rerank else 0,
                use_rewrite=self.profile.use_rewrite,
            )
        except Exception as ex:
            logger.warning(f"预检索失败: {ex}")
            return "", []
        if not result.success or not isinstance(result.data, list):
            return "", []
        items = result.data[: self.profile.rag_top_k]
        evidence = [
            {
                "tool": name,
                "title": str(item.get("title", "")),
                "source_url": str(item.get("source_url", "")),
                "updated_at": str(item.get("updated_at", "")),
                "content": str(item.get("content", ""))[:800],
            }
            for item in items
        ]
        if not evidence:
            return "", []
        lines = [
            f"[{i + 1}] {str(item.get('title', ''))}: {str(item.get('content', ''))}"
            for i, item in enumerate(items)
        ]
        text = "[系统已检索资料，请严格基于以下资料回答：]\n" + "\n".join(lines)
        return text, evidence

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

    # ── Task 语境组装（system prompt）──────────────────────────────────────

    def _build_system_prompt(self, req: Any) -> str:
        """
        组装 system prompt（Task-scoped context，每次 Run 独立组装）：
          1. TaskAgent 基座定义
          2. [任务目标]：本 Task 的 goal（独立于其他 Task 的指令）
          3. 领域人格（DOMAIN_PERSONA，按 req.domain 挂载 —— 领域只影响行为风格）
          4. 动态 Skills（业务 SOP：目录 + 命中提示 + 全量内容，模型自主选择）
          5. 意图指引（Action 层行为指令：查询只读 / 请求办理 / 投诉给路径等）
          6. 时间上下文（当前日期/星期/第几节/第几周）—— 所有请求统一注入，
             是"今天有什么课""现在第几节"类问答的前提。
        """
        prompt = self.system_prompt

        if getattr(req, "goal", ""):
            prompt = f"{prompt}\n\n[任务目标]\n{req.goal}"

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


class TaskAgent(BaseAgent):
    """围绕一个 Task 的独立 Agent Run 执行体（唯一 Agent 类）。

    不是业务领域 Agent，也不是 QA/Executor 职责 Agent：本类实例由编排器
    按 Execution Profile（Fast/Deep）实例化，每次 Run 通过 Request 注入
    独立的 task_id / goal / domain / action / 上下文，执行策略（读写门禁）
    由 action 决定（write_policy_for）。领域新增时不需要新增 Agent 类。
    """

    system_prompt = (
        "你是西电校园智慧助手（EchoGuide）的 Task Agent，负责执行分配给你的子任务。"
        "根据上方 [任务目标] 完成该任务，并结合 [领域人格] 调整回答风格与侧重点。\n"
        "查询类任务：优先调用只读工具（如 knowledge_search）获取准确信息；"
        "涉及政策、规定、办事流程或校园设施信息时，回答末尾用 [1][2] 标注引用来源；"
        "不要编造具体的分数、排名、价格、电话、截止日期或审批结果。\n"
        "执行类任务（REQUEST）：操作用户个人数据（待办/日程）或调用工具完成指令，"
        "写入完成后在回答中明确回执（如「已添加待办：xxx」），操作失败时如实说明原因，不要谎报成功；"
        "涉及个人数据（课表/待办）只能来自工具结果，涉及学籍、审批等事项时提示以教务处/学院最新通知为准；"
        "无法执行的操作给出替代路径（如引导用户自行到对应系统/网点办理）。"
    )
