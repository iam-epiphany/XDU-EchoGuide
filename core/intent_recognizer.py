"""
亮点：端到端意图识别（层次化意图 Hierarchical Intent）

从「单维扁平意图」升级为「领域 domain × 动作 action」二维体系：

  - 领域 IntentDomain（academic/campus_life/affairs/it_help/other）
      —— 路由的唯一依据。修复了旧版 P0 缺陷：请求句式（"帮我…/我要…"）被
         few-shot 标成通用 REQUEST 后丢失领域信息，校务问题被学业 Agent 回答。
  - 动作 IntentAction（query/request/greeting/complaint/feedback）
      —— 行为决策依据。

级联识别策略（宁多付成本、不静默误判）：
  1. 追问形态 → 直接 LLM（带最近对话，由 LLM 结合上下文裁决）。
     Embedding 是无上下文概念的匹配器，省略追问只能靠残留疑问词猜领域——
     猜对是运气，猜错是静默误路由；强信号（那/再/还/别的…）无条件判追问，
     弱信号（极短疑问句）仅当 pattern 无信号时判追问，完整问句放行免费路径。
  2. Pattern 高置信 + Embedding 双确认 → 免费直返
     （关键词子串可能误配，如"电子图书馆怎么登录？"被"图书馆"命中；
     双确认要求 Embedding 方向一致且达到命中阈值 0.80 —— 低于阈值即使方向
     一致也只是弱证据；方向分歧或未达阈值则升级 LLM 仲裁，成本仅 ms 级 bge 推理）
  3. Embedding 达到阈值且与第二候选有足够间隔时返回
     （与 pattern 弱信号方向矛盾时升级 LLM 仲裁，不静默路由歧义句）
  4. 未命中或低置信度请求调用 LLM（携带最近对话）

复杂度判定（意图识别的一部分）：
  - LLM 参与意图识别时顺带输出 complexity（single/parallel/dependent，同一次调用零额外成本）
  - mode=dependent 时 LLM 直接输出任务依赖链（tasks），合法性由编排器校验，非法回落关键词规则
  - 编排器另有"规则拿不准 → judge_complexity 升级确认"路径（见 agents/agent_orchestrator.py）

追问处理（对话感知）：
  - 追问形态是级联的最高优先级：识别为追问（指代承接/极短省略句）就直接进
    LLM，不做本地继承（"谢谢"也会继承领域——误路由风险大于省下的 LLM 调用
    成本），也不让 Embedding 猜（它无上下文概念）。LLM prompt 携带最近对话，
    由 LLM 结合上下文判断领域。
  - 结果缓存 key 加入对话历史指纹，同一句追问在不同上下文不会命中陈旧意图。

领域关键词的唯一来源在 core/domains.py，本模块与 Orchestrator、API 层共用，
消除三处重复维护的漂移问题。
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.domains import (
    IntentAction,
    IntentDomain,
    action_hit_score,
    domain_hit_score,
)

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    """兼容枚举：保留旧版语义（领域或动作），供 API / 评测 / 前端兼容使用。"""
    QUERY      = "query"        # 信息查询
    REQUEST    = "request"      # 请求操作
    GREETING   = "greeting"     # 问候
    COMPLAINT  = "complaint"    # 投诉不满
    FEEDBACK   = "feedback"     # 正面反馈
    # 西电校园场景的领域意图（兼容旧版路由）
    ACADEMIC   = "academic"     # 学业支持
    CAMPUS_LIFE = "campus_life" # 校园生活
    AFFAIRS    = "affairs"      # 校务咨询
    IT_HELP    = "it_help"      # IT 助手
    PERSONAL   = "personal"     # 个人助理（课表/待办/日程）
    OTHER      = "other"


# 领域 → 兼容意图 的映射（旧版消费方只需要领域值）
_DOMAIN_TO_CATEGORY = {
    IntentDomain.ACADEMIC:    IntentCategory.ACADEMIC,
    IntentDomain.CAMPUS_LIFE: IntentCategory.CAMPUS_LIFE,
    IntentDomain.AFFAIRS:     IntentCategory.AFFAIRS,
    IntentDomain.IT_HELP:     IntentCategory.IT_HELP,
    IntentDomain.PERSONAL:    IntentCategory.PERSONAL,
}

_ACTION_TO_CATEGORY = {
    IntentAction.QUERY:     IntentCategory.QUERY,
    IntentAction.REQUEST:   IntentCategory.REQUEST,
    IntentAction.GREETING:  IntentCategory.GREETING,
    IntentAction.COMPLAINT: IntentCategory.COMPLAINT,
    IntentAction.FEEDBACK:  IntentCategory.FEEDBACK,
}


@dataclass
class IntentResult:
    domain:     IntentDomain     # 领域（免费路径/历史回溯回填，仅用于人格挂载与观测）
    action:     IntentAction     # 动作（行为依据：角色选择 + 写门禁）
    intent:     IntentCategory   # 兼容字段（domain 优先，其次 action）
    confidence: float
    entities:   Dict[str, List[str]]
    reasoning:  str
    latency_ms: float
    classifier_stage: str = "llm"
    complexity: Optional["ComplexitySignal"] = None  # LLM 参与识别时顺带判定的复杂度
    skills_to_reference: List[str] = field(default_factory=list)  # LLM 建议参考的 Skill（观测/评测）
    needs_knowledge: bool = True                     # LLM 判定是否需要知识检索（观测/评测）


@dataclass
class ComplexitySignal:
    """
    LLM 输出的复杂度判定（意图识别的一部分）。

    只有 LLM 参与了意图识别（classifier_stage == "llm"）或走升级路径时才存在；
    模式判定与领域/动作同一次 LLM 调用产出，不额外付费。
    tasks 是 LLM 原始任务链（dict 列表），合法性由编排器校验（本模块只做形状解析）。
    """
    mode: str                                        # single / parallel / dependent
    targets: List[str] = field(default_factory=list) # 涉及的领域值（如 "campus_life"）
    reason: str = ""
    tasks: Optional[List[Dict[str, Any]]] = None     # dependent 时的任务链（原始 dict）


# 复杂度模式枚举值（LLM 输出校验用）
COMPLEXITY_MODES = ("single", "parallel", "dependent")


# ── Few-shot 模板 ─────────────────────────────────────────────────────────────
# 领域模板：用于 LLM 示例与 Embedding 匹配；动作模板：用于 LLM 示例。
_DOMAIN_TEMPLATES: Dict[IntentDomain, List[str]] = {
    IntentDomain.ACADEMIC:    ["这学期选课什么时候开始？", "绩点怎么算的？", "重修怎么报名？", "保研有什么条件？", "培养方案学分要求是什么？"],
    IntentDomain.CAMPUS_LIFE: ["南校区食堂几点关门？", "校车最后一班几点？", "宿舍怎么报修？", "校园卡在哪充值？"],
    IntentDomain.AFFAIRS:     ["奖学金什么时候评？", "请假流程怎么走？", "在读证明在哪开？", "学费缴费方式有哪些？", "我要请假怎么走流程", "校园卡丢了怎么补办？"],
    IntentDomain.IT_HELP:     ["教务系统登录不上", "校园网连不上", "VPN怎么配置？", "学校邮箱收不到邮件"],
    IntentDomain.PERSONAL:    ["今天有什么课？", "帮我查一下我的课表", "明天第几节在哪上课？", "这周周几没课？", "帮我记个待办，周三前交实验报告", "我最近的考试安排？", "还有什么没做完？"],
}

_ACTION_TEMPLATES: Dict[IntentAction, List[str]] = {
    IntentAction.QUERY:       ["西电校历这学期什么时候放假？", "图书馆几点开门？", "南校区快递站在哪？"],
    IntentAction.REQUEST:     ["帮我查一下选课时间", "帮我查一下校园卡余额"],
    IntentAction.GREETING:    ["你好", "嗨", "在吗", "早上好"],
    IntentAction.COMPLAINT:   ["宿舍热水一直不来！", "校车等了半小时还没来", "食堂排队太久了"],
    IntentAction.FEEDBACK:    ["这个助手很实用！", "回答得很清楚，谢谢", "帮我大忙了"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。

    维度不一致时返回 0.0（不参与匹配）：真实 Embedding 384 维、n-gram 兜底
    256 维，若嵌入器中途降级导致混维，继续计算会产生无意义分数。
    """
    if len(a) != len(b):
        logger.warning(f"向量维度不一致（{len(a)} vs {len(b)}），跳过相似度计算")
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器（领域 × 动作）。

    初始化时不加载任何本地模型，所有 AI 能力通过 Anthropic API 调用。
    模板 Embedding 在首次请求时懒加载并缓存，后续复用。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        pattern_threshold: float = 0.90,
        embedding_threshold: float = 0.80,
        embedding_margin: float = 0.10,
        gateway: Optional[Any] = None,  # 统一模型调用入口（编排器注入；None 时直接调用）
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self._gateway  = gateway
        # 阈值可用环境变量覆盖（ECHOGUIDE_INTENT_*）：有 LLM 兜底时阈值宁紧勿松——
        # 高阈值只是让"拿不准"的请求多付一次 LLM 调用，低阈值则会把低分误判静默
        # 落入错误领域（LLM 托底只保护漏判，不保护误判）。
        # 默认值按真实 bge 标定（probe_intent_thresholds.py）：同构嵌入下模板原文
        # 1.000、命中区最低 0.820、miss 区最高 0.655，0.80 在分离空档内且不误判；
        # 0.85 只会把"学费怎么交？"这类高频问句白送到 LLM。
        self.pattern_threshold = float(
            os.getenv("ECHOGUIDE_INTENT_PATTERN_THRESHOLD", str(pattern_threshold)))
        self.embedding_threshold = float(
            os.getenv("ECHOGUIDE_INTENT_EMBEDDING_THRESHOLD", str(embedding_threshold)))
        self.embedding_margin = float(
            os.getenv("ECHOGUIDE_INTENT_EMBEDDING_MARGIN", str(embedding_margin)))
        # 真实 Embedding：本地 bge 中文模型（mcp.embeddings，与知识库 RAG 同源，
        # 模板走 embed_documents、用户消息走 embed_query 指令前缀）。
        # 模型不可用（如离线环境）时自动回退字符 n-gram 哈希向量，保证链路可用。
        self._embedding_enabled = True
        self._embedder = None   # 惰性初始化（get_embedder 单例）

        self._tpl_embeddings: Dict[IntentDomain, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        force_llm: bool = False,
        state: Optional[Any] = None,  # RunState（有则经 ModelGateway 统计模型调用）
    ) -> IntentResult:
        """
        识别用户意图（领域 + 动作）。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        缓存 key 包含最近对话指纹 —— 同一句追问在不同上下文不会命中陈旧结果。
        """
        key = self._cache_key(message, history) + (":llm" if force_llm else "")
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        from core.tracing import span

        async with span("intent_recognize"):
            pat = self._pattern_recognize(message)
            action = pat.get("action", IntentAction.OTHER)
            domain = pat.get("domain", IntentDomain.OTHER)
            confidence = 0.0
            reasoning = ""
            stage = "pattern"
            complexity = None   # 只有 LLM 参与识别时才可能携带复杂度信号
            skills_to_reference: List[str] = []
            needs_knowledge = True

            if force_llm:
                llm = await self._llm_recognize(message, history, state=state)
                action = llm.get("action", action)
                confidence = float(llm.get("confidence", 0.0))
                reasoning = llm.get("reasoning", "")
                stage = "llm"
                complexity = self._normalize_complexity(llm.get("complexity"))
                skills_to_reference = llm.get("skills_to_reference") or []
                needs_knowledge = bool(llm.get("needs_knowledge", True))
                domain = self._domain_fallback(message, history)  # LLM 不输出领域：免费回填
            elif self._is_followup_shaped(message, pat.get("domain") != IntentDomain.OTHER):
                # 追问形态 → 直接 LLM（带最近对话，由 LLM 结合上下文裁决 action 与
                # 查询理解；领域不再由 LLM 输出，改由历史关键词回溯免费回填）：
                #   Embedding 是无上下文概念的匹配器，对省略追问只能靠残留疑问词
                #   去猜——猜对是运气，猜错是静默误判；强信号（那/再/还/别的…）
                #   即使有 pattern 弱信号也不让 Embedding 猜（如"那选课呢？"），
                #   弱信号（极短疑问句）仅当 pattern 完全无信号时判追问，
                #   完整问句（"绩点怎么算的？"有主题词）放行 Embedding。
                llm = await self._llm_recognize(message, history, state=state)
                action = llm.get("action", action)
                confidence = float(llm.get("confidence", 0.0))
                reasoning = llm.get("reasoning", "")
                stage = "llm"
                complexity = self._normalize_complexity(llm.get("complexity"))
                skills_to_reference = llm.get("skills_to_reference") or []
                needs_knowledge = bool(llm.get("needs_knowledge", True))
                domain = self._domain_fallback(message, history)  # 追问继承：历史关键词回溯
            elif (
                pat.get("domain") != IntentDomain.OTHER
                and pat.get("confidence", 0.0) >= self.pattern_threshold
            ):
                # Pattern 高置信 + 双确认：关键词子串可能误配（"电子图书馆怎么
                # 登录？"被"图书馆"命中 campus_life 直返），Embedding 方向一致
                # 且达到命中阈值（≥embedding_threshold）才免费直返；方向分歧或
                # 分数低于阈值（含未命中/失败）→ 升级 LLM 仲裁。
                # 0.80 是 bge 标定的命中区/未命中区分隔线（probe_intent_thresholds.py：
                # 命中区最低 0.820、miss 区最高 0.655）：低于它即使方向一致也只是
                # 噪声级巧合，不能算"双确认"（宁多花钱不误判）。
                # margin 在此不要求：pattern 已用 ≥0.90 的关键词证据消歧，
                # margin 只约束独立 Embedding 路径的 top-2 模糊。
                emb = await self._embedding_recognize(message) if self._embedding_enabled else {
                    "domain": IntentDomain.OTHER,
                    "action": IntentAction.OTHER,
                    "confidence": 0.0,
                    "margin": 0.0,
                }
                if (
                    emb.get("domain") == pat["domain"]
                    and emb.get("confidence", 0.0) >= self.embedding_threshold
                ):
                    domain = pat["domain"]
                    confidence = float(pat["confidence"])
                    reasoning = "关键词高置信命中（Embedding 双确认）"
                else:
                    llm = await self._llm_recognize(message, history, state=state)
                    action = llm.get("action", action)
                    confidence = float(llm.get("confidence", 0.0))
                    if emb.get("domain") == pat["domain"]:
                        reason_detail = (
                            f"Embedding 同向但分数 {emb.get('confidence', 0.0):.2f} "
                            f"低于阈值 {self.embedding_threshold}"
                        )
                    else:
                        reason_detail = (
                            f"关键词与 Embedding 分歧（{emb.get('domain') or '未命中'}）"
                        )
                    reasoning = f"{reason_detail}，LLM 仲裁"
                    stage = "llm"
                    complexity = self._normalize_complexity(llm.get("complexity"))
                    skills_to_reference = llm.get("skills_to_reference") or []
                    needs_knowledge = bool(llm.get("needs_knowledge", True))
                    domain = self._domain_fallback(message, history)  # LLM 不输出领域：免费回填
            else:
                    emb = await self._embedding_recognize(message) if self._embedding_enabled else {
                        "domain": IntentDomain.OTHER,
                        "action": IntentAction.OTHER,
                        "confidence": 0.0,
                        "margin": 0.0,
                    }
                    if (
                        emb.get("domain") != IntentDomain.OTHER
                        and emb.get("confidence", 0.0) >= self.embedding_threshold
                        and emb.get("margin", 0.0) >= self.embedding_margin
                    ):
                        # 分歧仲裁：Embedding 高置信命中但与 pattern 弱信号方向
                        # 矛盾（如 pattern 命中 academic 而 Embedding 判 personal
                        # 的歧义句）→ 不静默判定，升级 LLM 结合上下文裁决
                        if (
                            pat.get("domain") != IntentDomain.OTHER
                            and emb["domain"] != pat["domain"]
                        ):
                            llm = await self._llm_recognize(message, history, state=state)
                            action = llm.get("action", action)
                            confidence = float(llm.get("confidence", 0.0))
                            reasoning = f"Pattern 弱命中 {pat['domain'].value} 与 Embedding {emb['domain'].value} 分歧，LLM 仲裁"
                            stage = "llm"
                            complexity = self._normalize_complexity(llm.get("complexity"))
                            skills_to_reference = llm.get("skills_to_reference") or []
                            needs_knowledge = bool(llm.get("needs_knowledge", True))
                            domain = self._domain_fallback(message, history)  # LLM 不输出领域：免费回填
                        else:
                            domain = emb["domain"]
                            confidence = float(emb["confidence"])
                            reasoning = "Embedding 高置信度命中"
                            stage = "embedding"
                    else:
                        llm = await self._llm_recognize(message, history, state=state)
                        action = llm.get("action", action)
                        confidence = float(llm.get("confidence", 0.0))
                        reasoning = llm.get("reasoning", "")
                        stage = "llm"
                        complexity = self._normalize_complexity(llm.get("complexity"))
                        skills_to_reference = llm.get("skills_to_reference") or []
                        needs_knowledge = bool(llm.get("needs_knowledge", True))
                        domain = self._domain_fallback(message, history)  # LLM 不输出领域：免费回填

            entities = self._extract_entities_local(message)

        result = IntentResult(
            domain=domain,
            action=action,
            intent=self._legacy_intent(domain, action),
            confidence=round(confidence, 4),
            entities=entities,
            reasoning=reasoning,
            latency_ms=(time.monotonic() - t0) * 1000,
            classifier_stage=stage,
            complexity=complexity,
            skills_to_reference=list(dict.fromkeys(skills_to_reference))[:8],
            needs_knowledge=needs_knowledge,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: IntentDomain) -> None:
        """在线学习：将纠正样本加入领域模板，清除对应 Embedding 缓存。"""
        tpls = _DOMAIN_TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    async def judge_complexity(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        state: Optional[Any] = None,
    ) -> Optional[ComplexitySignal]:
        """
        复杂度专用判断（升级路径）：意图识别走完免费规则仍"拿不准"时，
        编排器调用本方法做一次轻量 LLM 确认（只问复杂度，不重复问领域/动作）。

        不写缓存 —— 升级路径低频，且结论依赖上下文，避免污染意图缓存。
        返回 None 表示 LLM 不可用或输出非法，调用方应回落关键词规则。
        """
        llm = await self._llm_recognize(message, history, complexity_only=True, state=state)
        return self._normalize_complexity(llm.get("complexity"))

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    # 复杂度判定指引：意图识别与复杂度专用调用共用同一段说明，保证口径一致。
    # 刻意要求"普通问题不要过度拆分"——复杂度误判的成本不对称（误判复杂比漏判贵）。
    _COMPLEXITY_GUIDE = (
        "同时判断请求复杂度 complexity（普通问题不要过度拆分，绝大多数请求都是 single）：\n"
        '- mode: "single"（单个领域、无依赖的普通问题）\n'
        '- mode: "parallel"（涉及多个校园领域、可并行处理，如"食堂几点关门，顺便帮我查下明天的课表"）\n'
        '- mode: "dependent"（多个诉求有先后依赖，需要先查再办，如"我明天下午有空，想去办校园卡，帮我记个待办"）\n'
        "- targets: 涉及的领域值列表，可选值: academic, campus_life, affairs, it_help, personal\n"
        '- 只有 mode == "dependent" 时才输出 tasks 任务链，每个任务格式:\n'
        '  {"id": "t1", "agent": "<领域值>", "goal": "<任务目标>", '
        '"message": "<给该领域助手的自包含请求>", "depends_on": ["<前置任务id>"]}\n'
        "  depends_on 引用前面已定义任务的 id；无前置依赖的任务省略或为空数组。"
    )

    # ── 领域回填（LLM 不再输出领域）──────────────────────────────────────────
    # v4：领域只用于人格挂载与观测，全部由免费路径产出——
    #   当前消息关键词 → 最近 4 轮用户消息关键词回溯（追问继承）→ OTHER。
    @staticmethod
    def _domain_fallback(
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentDomain:
        domain, _ = domain_hit_score(message)
        if domain is not None:
            return domain
        if history:
            for m in reversed(history[-4:]):
                if m.get("role") != "user":
                    continue
                domain, _ = domain_hit_score(str(m.get("content", "")))
                if domain is not None:
                    return domain
        return IntentDomain.OTHER

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
        complexity_only: bool = False,
        state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        策略 1：LLM 查询理解（Few-shot + 上下文）。

        - 正常模式：输出 action + entities + skills_to_reference + needs_knowledge
          + complexity（v4：不再输出领域——领域由免费路径回填，Skill 选择交给模型）；
        - complexity_only=True：只输出 complexity（编排器"规则拿不准"时的升级路径，
          轻量 prompt，不重复问动作/查询理解）。
        """
        message = self._clean_text(message)

        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        if complexity_only:
            prompt = f"""你是西电校园智慧助手（EchoGuide）的复杂度分析模块。判断用户请求是否需要多个校园领域助手协作、是否存在先后依赖。
{ctx}
用户消息: "{message}"

{self._COMPLEXITY_GUIDE}
返回格式（仅 JSON，不要其他文字）:
{{"complexity": {{"mode": "<single/parallel/dependent>", "targets": [...], "reason": "<一句话>", "tasks": [...]}}}}
"""
        else:
            action_examples = []
            for action, tpls in _ACTION_TEMPLATES.items():
                for t in tpls[:2]:
                    action_examples.append(f'  消息: "{t}" → action: {action.value}')
            examples_text = "\n".join(action_examples[:12])

            prompt = f"""你是西电校园智慧助手（EchoGuide）的查询理解模块。分析用户消息并输出结构化 JSON（v4：不需要输出领域）。

动作 action 可选值: {", ".join(a.value for a in IntentAction)}
（query=查询信息；request=请求系统执行操作（写待办/日程等）；greeting=问候；complaint=投诉不满；feedback=反馈）

动作示例:
{examples_text}

{ctx}
用户消息: "{message}"

{self._COMPLEXITY_GUIDE}
返回格式（仅 JSON，不要其他文字）:
{{"action": "<动作值>", "confidence": <0-1>, "reasoning": "<一句话说明>", "entities": {{"term": ["时间词"], "content": ["待办内容/主体"]}}, "skills_to_reference": ["<建议参考的技能名，无则空数组>"], "needs_knowledge": true, "complexity": {{"mode": "<single/parallel/dependent>", "targets": [...], "reason": "<一句话>", "tasks": [...]}}}}

要求：
- action 表示用户希望系统做什么（查询/操作/问候/投诉/反馈等）。
- entities 抽取用户消息里的关键实体（时间、待办内容、地点等），没有的字段给空数组。
- skills_to_reference 从系统提示中的技能目录选择建议参考的技能名（如"学业咨询规范"），不确定就空数组。
- needs_knowledge 表示该问题是否需要检索校园知识库（政策/流程/规则类 true；闲聊/个人数据操作 false）。
- 追问（如"那几点开门？"）应结合最近对话推断 action，并给出 entities。"""
        prompt = self._clean_text(prompt)

        try:
            if self._gateway is not None:
                # 经统一模型调用入口：模型调用计数/统计/预算/Trace 与 Agent 链路口径一致
                result = await self._gateway.call(
                    client=self.client,
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    state=state,
                    span_name="intent_llm",
                    max_tokens=256,
                    temperature=0.1,
                    thinking={"type": "disabled"},
                )
                resp = result.response
            else:
                resp = await self.client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    temperature=0.1,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}],
                )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            if complexity_only:
                return {"complexity": self._parse_complexity(data.get("complexity"))}
            try:
                data["action"] = IntentAction(data.get("action", "other"))
            except ValueError:
                data["action"] = IntentAction.OTHER
            data["complexity"] = self._parse_complexity(data.get("complexity"))
            entities = data.get("entities")
            if not isinstance(entities, dict):
                entities = {}
            data["entities"] = {
                str(k): [str(v) for v in vs if str(v).strip()]
                for k, vs in entities.items() if isinstance(vs, list)
            }
            skills = data.get("skills_to_reference")
            data["skills_to_reference"] = [
                str(x) for x in skills if isinstance(x, str) and x.strip()
            ] if isinstance(skills, list) else []
            data["needs_knowledge"] = bool(data.get("needs_knowledge", True))
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            if complexity_only:
                return {"complexity": None}
            return {
                "action": IntentAction.OTHER,
                "confidence": 0.0,
                "reasoning": "LLM 失败",
                "failed": True,
            }

    @staticmethod
    def _normalize_complexity(value: Any) -> Optional[ComplexitySignal]:
        """
        统一复杂度信号形态：已是 ComplexitySignal 直接用；dict（如测试 fake 或
        外部调用方）走 _parse_complexity；其余（None/非法）返回 None。
        """
        if isinstance(value, ComplexitySignal):
            return value
        return IntentRecognizer._parse_complexity(value)

    @staticmethod
    def _parse_complexity(raw: Any) -> Optional[ComplexitySignal]:
        """
        解析 LLM 输出的复杂度字段（宽容模式）：

        - mode 必须合法（single/parallel/dependent）；
        - targets 只收非空字符串，最多 3 个；
        - tasks 必须是 dict 列表且仅 dependent 模式保留；
        任何畸形返回 None —— 编排器随之回落关键词规则，不让 LLM 坏输出打穿链路。
        """
        if not isinstance(raw, dict):
            return None
        mode = raw.get("mode")
        if mode not in COMPLEXITY_MODES:
            return None
        targets = raw.get("targets")
        if not isinstance(targets, list):
            targets = []
        targets = [str(t) for t in targets if isinstance(t, str) and t.strip()][:3]
        reason = str(raw.get("reason") or "")[:120]
        tasks = raw.get("tasks")
        if tasks is not None:
            if mode != "dependent" or not isinstance(tasks, list) or not all(isinstance(t, dict) for t in tasks):
                tasks = None
        return ComplexitySignal(mode=mode, targets=targets, reason=reason, tasks=tasks)

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配（按领域模板）。

        用户消息与模板**同构嵌入**（都不带 bge-zh 指令前缀）：领域模板是"用户
        问法原型"，与用户消息同为 query 形态——指令前缀只该用于 RAG 检索
        （query vs passage 异构），用在模板匹配会把同义文本的相似度从 ~1.0
        压到 ~0.79（实测），导致阈值再紧都无法命中、Embedding 级联空转。
        """
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message, is_query=False)

            scored: List[tuple[float, IntentDomain]] = []
            for domain, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                scored.append((score, domain))
            scored.sort(key=lambda item: item[0], reverse=True)
            best_score, best_domain = scored[0] if scored else (0.0, IntentDomain.OTHER)
            second_score = scored[1][0] if len(scored) > 1 else 0.0
            return {
                "domain": best_domain,
                "action": IntentAction.OTHER,
                "confidence": best_score,
                "margin": max(0.0, best_score - second_score),
            }
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"domain": IntentDomain.OTHER, "action": IntentAction.OTHER, "confidence": 0.0, "margin": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """
        策略 3：关键词模式匹配（同步，零延迟兜底）。

        领域与动作独立匹配：领域关键词（选课/食堂/请假/教务系统等）比通用疑问词
        （怎么/几点/什么时候）更具判别力，因此领域维度先行；动作维度用通用模式兜底，
        两者不再互相抢占（修复旧版"请求句式吞掉领域"的问题）。
        """
        domain, domain_score = domain_hit_score(message)
        action, action_score = action_hit_score(message)

        return {
            "domain": domain or IntentDomain.OTHER,
            "action": action or IntentAction.OTHER,
            "confidence": max(domain_score, action_score) or 0.0,
        }

    # ── 追问形态检测（防 Embedding 误判）───────────────────────────────────
    # 省略追问（"那几点开门呢？"/"几点？"/"下午呢？"）没有主题词，Embedding
    # 无上下文概念、只能靠残留疑问词猜领域——猜对是运气，猜错是静默误路由。
    # 级联中最优先：判为追问 → 直接 LLM（带最近对话，由 LLM 结合上下文裁决）。
    # 两级信号：
    #   强信号（指代承接词）→ 无条件判追问——即使有 pattern 弱信号也不让
    #     Embedding 猜（"那选课呢？"主题词弱命中 academic，但语义依赖上文）；
    #   弱信号（极短疑问句/呢结尾）→ 仅当 pattern 完全无信号时判追问——
    #     完整问句（"绩点怎么算的？"有主题词，Embedding 1.000 免费命中）
    #     必须放行；"几点？""什么时候？"无主题词才进 LLM。
    #   注意："这"不放强信号——"这学期/这周"等时间名词极常见，
    #     "这学期选课什么时候开始？"（13 字）是完整问句，Embedding 1.000 命中。

    _FOLLOWUP_STRONG = ("那", "再", "还", "然后", "别的", "其他", "也", "又", "接着", "另外", "另一个", "它")
    _FOLLOWUP_QUESTION_WORDS = ("几点", "多少", "什么", "哪", "怎么", "几号", "几")

    @classmethod
    def _is_followup_shaped(cls, message: str, has_pattern_signal: bool = False) -> bool:
        """
        追问形态启发式（两级信号，级联最优先路由到 LLM）。

        - 强信号：含指代承接词（那/再/还/别的/其他/也/又/接着/另外/它…）
          且 ≤14 字 → 承接上文话题，无条件判追问（"那几点开门呢？"/"那选课呢？"）；
        - 弱信号（需 pattern 无信号）：
          · 去标点后以"呢"结尾且 ≤8 字 → 省略追问（"下午呢？"）；
          · 极短且含疑问词（≤8 字）→ 省略疑问（"几点？"/"什么时候？"）。
        完整问句（"绩点怎么算的？"有主题词信号）、社交语（"谢谢/好的"）
        返回 False，放行 Pattern/Embedding 免费路径。
        """
        msg = (message or "").strip()
        if not msg:
            return False
        compact = re.sub(r"[\s，。！？、,.!?]", "", msg)
        n = len(compact)
        if 0 < n <= 14 and any(tok in compact for tok in cls._FOLLOWUP_STRONG):
            return True
        if has_pattern_signal:
            return False  # 弱信号要求 pattern 无主题词信号
        if 0 < n <= 8 and compact.endswith("呢"):
            return True
        if 0 < n <= 8 and any(tok in compact for tok in cls._FOLLOWUP_QUESTION_WORDS):
            return True
        return False

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_entities_local(message: str) -> Dict[str, List[str]]:
        """本地提取高频校园实体，避免级联命中后再次产生 LLM 调用。"""
        text = message or ""
        campuses = [name for name in ("南校区", "北校区", "太白校区", "长安校区") if name in text]
        systems = [name for name in ("教务系统", "校园网", "VPN", "邮箱", "统一身份认证") if name.lower() in text.lower()]
        terms = re.findall(r"(?:今天|明天|后天|本周|这周|下周|周[一二三四五六日天]|\d{1,2}月\d{1,2}日)", text)
        locations = re.findall(r"(?:[A-Ga-g]楼|[A-Ga-g]栋|信远楼|图书馆|体育馆|行政楼)", text)
        course_matches = re.findall(r"([\u4e00-\u9fffA-Za-z]{2,16})(?:课|课程)", text)
        return {
            "course": list(dict.fromkeys(course_matches)),
            "term": list(dict.fromkeys(terms)),
            "location": list(dict.fromkeys(locations)),
            "campus": list(dict.fromkeys(campuses)),
            "system": list(dict.fromkeys(systems)),
        }

    async def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """兼容旧调用方；实体提取现为本地确定性实现。"""
        return self._extract_entities_local(message)

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有领域模板的 Embedding（只在首次调用时执行）。"""
        missing = [d for d in _DOMAIN_TEMPLATES if d not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for d in missing for t in _DOMAIN_TEMPLATES[d]]
        # 模板按文档侧嵌入（不带 bge-zh 指令前缀）
        vecs = [await self._embed_text(text, is_query=False) for text in all_texts]
        idx = 0
        for domain in missing:
            n = len(_DOMAIN_TEMPLATES[domain])
            self._tpl_embeddings[domain] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str, *, is_query: bool = False) -> List[float]:
        """
        生成文本向量：本地 bge Embedding 优先，n-gram 哈希兜底。

        - 优先使用 mcp.embeddings 的本地 bge 中文模型（与知识库 RAG 同源，
          512 维）。bge-zh 指令前缀只用于 RAG 检索的 query 侧（知识库文档
          passage 侧不加）；意图识别的模板匹配必须同构嵌入（两侧都不加，
          见 _embedding_recognize 说明），否则同义文本相似度被压到 ~0.79；
        - 模型不可用（如离线环境）时永久回退本地 n-gram 哈希向量，
          保证三路融合链路在任何环境都不中断。
        """
        if self._embedding_enabled:
            if self._embedder is None:
                try:
                    from mcp.embeddings import get_embedder

                    self._embedder = get_embedder()
                    if self._embedder is None:
                        raise RuntimeError("本地 Embedding 模型不可用")
                except Exception as ex:
                    logger.warning(f"本地 Embedding 模型不可用，回退 n-gram 向量: {ex}")
                    self._embedding_enabled = False
            if self._embedder is not None:
                try:
                    embed = (self._embedder.embed_query if is_query
                             else self._embedder.embed_documents)
                    vec = await asyncio.to_thread(embed, [text])
                    return [float(x) for x in vec[0]]
                except Exception as ex:
                    logger.warning(f"Embedding 计算失败，回退 n-gram 向量: {ex}")
                    self._embedding_enabled = False

        return self._local_embedding(text)

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    @staticmethod
    def _legacy_intent(domain: IntentDomain, action: IntentAction) -> IntentCategory:
        """兼容字段：领域优先，其次动作，最后 OTHER。"""
        if domain in _DOMAIN_TO_CATEGORY:
            return _DOMAIN_TO_CATEGORY[domain]
        if action in _ACTION_TO_CATEGORY:
            return _ACTION_TO_CATEGORY[action]
        return IntentCategory.OTHER

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]]) -> str:
        """
        缓存 key = 消息 + 最近 3 轮对话指纹。
        追问依赖上下文，纯消息 key 会返回陈旧意图 —— 这是旧版的一个真实缺陷。
        """
        fp = ""
        if history:
            tail = "|".join(
                f"{m.get('role', '')}:{self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )
            fp = hashlib.md5(tail.encode("utf-8")).hexdigest()[:8]
        return f"{self._clean_text(message)[:200]}#{fp}"

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
