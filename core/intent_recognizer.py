"""
亮点：端到端意图识别（领域 domain × 动作 action）

职责收口（v5 Task-scoped 架构）：Intent 只负责"理解用户想做什么"——
  - 领域 IntentDomain（academic/campus_life/affairs/it_help/other）
      —— 人格挂载键（顾问），不做 Agent 路由（真正的 Agent 单位
         是 Task，执行体是唯一 TaskAgent，见 agents/roles.py）。
  - 动作 IntentAction（query/request/greeting/complaint/feedback）
      —— 行为决策依据（Run 执行策略 write_policy_for + 工具读写门禁）。
  - needs_knowledge —— 是否需要知识检索（由 Verifier 消费：判定需要
    但最终执行链无检索证据时标记异常）。

复杂度判定（single/parallel/dependent、任务链）已从本模块移出，
统一交给 Planner（agents/workflow.py）——意图不再负责"怎么拆任务"。

级联识别策略（宁多付成本、不静默误判）：
  1. 追问形态 → 直接 LLM（带最近对话，由 LLM 结合上下文裁决）。
     Embedding 是无上下文概念的匹配器，省略追问只能靠残留疑问词猜领域——
     猜对是运气，猜错是静默误路由；强信号（那/再/还/别的…）无条件判追问，
     弱信号（极短疑问句）仅当 pattern 无信号时判追问，完整问句放行免费路径。
   2. Pattern 高置信 + Embedding 双确认 → 免费直返
     （关键词子串可能误配，如"电子图书馆怎么登录？"被"图书馆"命中；
      双确认要求 Embedding 方向一致、达到命中阈值 0.80，且与第二候选的
      间隔不小于 0.10 —— 任一条件不满足即升级 LLM 仲裁。）
   3. Pattern 未达高置信、Embedding 未通过双确认或二者方向分歧 → 直接 LLM
      （不保留 Embedding 单独直返路径，避免本地单信号静默路由）

追问处理（对话感知）：
  - 追问形态是级联的最高优先级：识别为追问（指代承接/极短省略句）就直接进
    LLM，不做本地继承（"谢谢"也会继承领域——误路由风险大于省下的 LLM 调用
    成本），也不让 Embedding 猜（它无上下文概念）。LLM prompt 携带最近对话，
    由 LLM 结合上下文判断动作。
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
    reasoning:  str
    latency_ms: float
    classifier_stage: str = "llm"
    needs_knowledge: bool = False                # 是否需要知识检索（Verifier 消费）


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
    IntentAction.QUERY:       ["西电校历这学期什么时候放假？", "图书馆几点开门？", "南校区快递站在哪？", "帮我查一下选课时间"],
    IntentAction.REQUEST:     ["帮我添加一个补办校园卡的待办", "把这个待办标记完成", "记一下明天交实验报告"],
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
        ablation_mode: str = "full",    # 消融档位：full=完整级联 / pattern_only=仅关键词 / no_llm=免费路径（无 LLM 仲裁）
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

        # 消融模式（评测专用，生产恒为 full）：pattern_only 与 no_llm 用于
        # 量化"关键词 → +Embedding → +LLM 仲裁"每一档的贡献（evaluation/run_intent_eval.py）。
        self._ablation_mode = ablation_mode

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        force_llm: bool = False,
        state: Optional[Any] = None,  # RunState（有则经 ModelGateway 统计模型调用）
        _trace: Optional[Dict[str, Any]] = None,  # 路由决策 trace（评测错误分析用）
    ) -> IntentResult:
        """
        识别用户意图（领域 + 动作 + 是否需要知识检索）。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        缓存 key 包含最近对话指纹 —— 同一句追问在不同上下文不会命中陈旧结果。

        _trace: 传入 dict 时记录各级信号（pattern / embedding 候选 / 最终
        决策），供 evaluation/error_analysis 与可观测性消费；不传则零开销。
        """
        key = self._cache_key(message, history) + (":llm" if force_llm else "")
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        # 消融模式（评测专用）：不缓存、不计延迟，直接返回对应档位的结果
        if self._ablation_mode == "pattern_only":
            return self._ablate_pattern_only(message)
        if self._ablation_mode == "no_llm":
            return await self._ablate_no_llm(message)

        t0 = time.monotonic()

        from core.tracing import span

        async with span("intent_recognize"):
            pat = self._pattern_recognize(message)
            action = pat.get("action", IntentAction.OTHER)
            domain = pat.get("domain", IntentDomain.OTHER)
            confidence = 0.0
            reasoning = ""
            stage = "pattern"
            needs_knowledge = False

            if _trace is not None:
                # 路由决策 trace：关键词信号 / 追问形态 / Embedding 候选，
                # 供评测错误分析与可观测性复盘（与生产路径同一实现）。
                _trace["pattern"] = {
                    "domain": pat.get("domain").value if pat.get("domain") else None,
                    "action": pat.get("action").value if pat.get("action") else None,
                    "confidence": round(float(pat.get("confidence", 0.0)), 4),
                }
                _trace["is_followup"] = self._is_followup_shaped(
                    message, pat.get("domain") != IntentDomain.OTHER
                )
                # Embedding 只用于与高置信 Pattern 的双确认。先写空值，避免
                # 诊断 trace 改变正常分流（例如本应直接走 LLM 的追问/弱 Pattern
                # 请求不应为了观测而额外调用 Embedding）。
                _trace["embedding_candidates"] = []

            if force_llm:
                llm = await self._llm_recognize(message, history, state=state)
                if _trace is not None:
                    _trace["llm_domain"] = (
                        llm.get("domain").value if isinstance(llm.get("domain"), IntentDomain) else None
                    )
                    _trace["llm_domain_confidence"] = round(float(llm.get("domain_confidence", 0.0)), 4)
                action = llm.get("action", action)
                confidence = float(llm.get("confidence", 0.0))
                reasoning = llm.get("reasoning", "")
                stage = "llm"
                needs_knowledge = bool(llm.get("needs_knowledge", False))
                domain = self._resolve_domain(llm, message, history)
            elif self._is_followup_shaped(message, pat.get("domain") != IntentDomain.OTHER):
                # 追问形态 → 直接 LLM（带最近对话，由 LLM 结合上下文裁决 action 与
                # 查询理解；领域由历史关键词回溯免费回填）：
                #   Embedding 是无上下文概念的匹配器，对省略追问只能靠残留疑问词
                #   去猜——猜对是运气，猜错是静默误判；强信号（那/再/还/别的…）
                #   即使有 pattern 弱信号也不让 Embedding 猜（如"那选课呢？"），
                #   弱信号（极短疑问句）仅当 pattern 完全无信号时判追问，
                #   完整问句若 Pattern 足够强，才放行 Embedding 做双确认。
                llm = await self._llm_recognize(message, history, state=state)
                action = llm.get("action", action)
                confidence = float(llm.get("confidence", 0.0))
                reasoning = llm.get("reasoning", "")
                stage = "llm"
                needs_knowledge = bool(llm.get("needs_knowledge", False))
                domain = self._resolve_domain(llm, message, history)
            elif (
                pat.get("domain") != IntentDomain.OTHER
                and pat.get("confidence", 0.0) >= self.pattern_threshold
            ):
                # Pattern 高置信 + 双确认：关键词子串可能误配（"电子图书馆怎么
                # 登录？"被"图书馆"命中 campus_life 直返），Embedding 方向一致
                # 且达到命中阈值（≥embedding_threshold）并拉开候选间隔
                # （margin ≥ embedding_margin）才免费直返；其余情况直接升级 LLM。
                # 0.80 是 bge 标定的命中区/未命中区分隔线（probe_intent_thresholds.py：
                # 命中区最低 0.820、miss 区最高 0.655）：低于它即使方向一致也只是
                # 噪声级巧合，不能算"双确认"（宁多花钱不误判）。
                emb = await self._embedding_recognize(message) if self._embedding_enabled else {
                    "domain": IntentDomain.OTHER,
                    "action": IntentAction.OTHER,
                    "confidence": 0.0,
                    "margin": 0.0,
                }
                if _trace is not None and emb.get("domain") != IntentDomain.OTHER:
                    candidates = emb.get("candidates") or [{
                        "domain": emb["domain"],
                        "score": emb.get("confidence", 0.0),
                    }]
                    _trace["embedding_candidates"] = [
                        {"domain": candidate["domain"].value, "score": round(float(candidate["score"]), 4)}
                        for candidate in candidates
                    ]
                if (
                    emb.get("domain") == pat["domain"]
                    and emb.get("confidence", 0.0) >= self.embedding_threshold
                    and emb.get("margin", 0.0) >= self.embedding_margin
                ):
                    domain = pat["domain"]
                    confidence = float(pat["confidence"])
                    reasoning = "关键词高置信命中（Embedding 双确认）"
                else:
                    llm = await self._llm_recognize(message, history, state=state)
                    action = llm.get("action", action)
                    confidence = float(llm.get("confidence", 0.0))
                    if emb.get("domain") != pat["domain"]:
                        reason_detail = (
                            f"关键词与 Embedding 分歧（{emb.get('domain') or '未命中'}）"
                        )
                    elif emb.get("confidence", 0.0) < self.embedding_threshold:
                        reason_detail = (
                            f"Embedding 同向但分数 {emb.get('confidence', 0.0):.2f} "
                            f"低于阈值 {self.embedding_threshold}"
                        )
                    else:
                        reason_detail = (
                            f"Embedding 同向且分数达标，但 margin {emb.get('margin', 0.0):.2f} "
                            f"低于阈值 {self.embedding_margin}"
                        )
                    reasoning = f"{reason_detail}，LLM 仲裁"
                    stage = "llm"
                    needs_knowledge = bool(llm.get("needs_knowledge", False))
                    domain = self._resolve_domain(llm, message, history)
            else:
                # 不存在足够强的 Pattern 时不让 Embedding 单独决定领域：
                # 它只能作为双确认的第二票，而不是一条独立的免费路径。
                llm = await self._llm_recognize(message, history, state=state)
                action = llm.get("action", action)
                confidence = float(llm.get("confidence", 0.0))
                reasoning = llm.get("reasoning", "Pattern 未达高置信，LLM 仲裁")
                stage = "llm"
                needs_knowledge = bool(llm.get("needs_knowledge", False))
                domain = self._resolve_domain(llm, message, history)

        if _trace is not None:
            _trace["stage"] = stage
            _trace["confidence"] = round(confidence, 4)
            _trace["reasoning"] = reasoning
            _trace["domain"] = domain.value if domain else None
            _trace["action"] = action.value if action else None

        result = IntentResult(
            domain=domain,
            action=action,
            intent=self._legacy_intent(domain, action),
            confidence=round(confidence, 4),
            reasoning=reasoning,
            latency_ms=(time.monotonic() - t0) * 1000,
            classifier_stage=stage,
            needs_knowledge=needs_knowledge,
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
        self._cache[key] = result
        return result

    # ── 消融档位（评测专用，evaluation/run_intent_eval.py）────────────────────
    # pattern_only：只跑关键词匹配（最朴素基线，零成本、可离线复现）。
    # no_llm：仅保留 Pattern + Embedding 双确认的免费路径，跳过全部 LLM 仲裁——
    #   追问继承、低置信度兜底这两类"花钱买正确"的行为在无 LLM 档位下
    #   会表现为准确率回落，从而量化 LLM 仲裁档的贡献。

    def _ablate_pattern_only(self, message: str) -> IntentResult:
        pat = self._pattern_recognize(message)
        domain, action = pat["domain"], pat["action"]
        return IntentResult(
            domain=domain,
            action=action,
            intent=self._legacy_intent(domain, action),
            confidence=round(float(pat["confidence"]), 4),
            reasoning="ablation:pattern_only",
            latency_ms=0.0,
            classifier_stage="pattern",
        )

    async def _ablate_no_llm(self, message: str) -> IntentResult:
        pat = self._pattern_recognize(message)
        domain, action = IntentDomain.OTHER, pat["action"]
        reasoning = "ablation:no_llm（未通过双确认，无 LLM 兜底）"
        stage = "no_match"
        emb = await self._embedding_recognize(message) if self._embedding_enabled else {
            "domain": IntentDomain.OTHER, "action": IntentAction.OTHER,
            "confidence": 0.0, "margin": 0.0,
        }
        if (
            pat["domain"] != IntentDomain.OTHER
            and pat["confidence"] >= self.pattern_threshold
            and emb.get("domain") == pat["domain"]
            and emb.get("confidence", 0.0) >= self.embedding_threshold
            and emb.get("margin", 0.0) >= self.embedding_margin
        ):
            domain = pat["domain"]
            stage = "pattern"
            reasoning = "ablation:no_llm（Pattern + Embedding 双确认）"

        return IntentResult(
            domain=domain,
            action=action,
            intent=self._legacy_intent(domain, action),
            confidence=round(float(pat["confidence"]), 4),
            reasoning=reasoning,
            latency_ms=0.0,
            classifier_stage=stage,
        )

    def learn(self, message: str, correct: IntentDomain) -> None:
        """在线学习：将纠正样本加入领域模板，清除对应 Embedding 缓存。"""
        tpls = _DOMAIN_TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._tpl_embeddings.pop(correct, None)  # 下次重新计算
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 领域回填（LLM 仲裁路径）────────────────────────────────────────────
    # v5：LLM 结构化输出 domain（带领域定义，错误分析补强），失败/低置信时
    # 回退关键词回填——免费路径信号（当前消息 → 最近 4 轮用户消息 → OTHER）
    # 作为确定性兜底，不静默落 OTHER。
    @staticmethod
    def _resolve_domain(
        llm: Dict[str, Any],
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentDomain:
        llm_domain = llm.get("domain")
        if (
            isinstance(llm_domain, IntentDomain)
            and llm_domain != IntentDomain.OTHER
            and float(llm.get("domain_confidence", 0.0)) >= 0.60
        ):
            return llm_domain
        return IntentRecognizer._domain_fallback(message, history)

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
        state: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        LLM 查询理解（Few-shot + 上下文）。

        输出 action + domain + confidence + reasoning + needs_knowledge。

        domain 输出是错误分析后的补强（v5）：LLM 仲裁路径下领域回填只靠
        关键词（_domain_fallback），词表覆盖不到的问法（"预选和正选有什么
        区别？"）会静默落回 OTHER —— 而 LLM 的 reasoning 明显理解领域。
        结构化输出让 LLM 仲裁路径的领域决策由 LLM 给出（带领域定义），
        免费路径（pattern+embedding 双确认）优先级不变，LLM 失败时回退
        关键词回填。
        """
        message = self._clean_text(message)

        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        action_examples = []
        for action, tpls in _ACTION_TEMPLATES.items():
            for t in tpls[:2]:
                action_examples.append(f'  消息: "{t}" → action: {action.value}')
        examples_text = "\n".join(action_examples[:12])

        domain_defs = "\n".join(
            f"- {d.value}: {_DOMAIN_TEMPLATES[d][0]}"
            for d in IntentDomain
            if d != IntentDomain.OTHER
        )

        # 领域边界说明（错误分析补强：affairs/academic 的"教务"常识混淆、
        # 地点类按对象归属而非查询类型判断）。只描述 taxonomy 边界，
        # 不针对具体评测问题。
        boundary_notes = """领域判断边界：
- 归属以"对象/事项"为准，不以查询形式为准：问"学生处在哪""图书馆几点关门"都是地点/时间查询，但对象学生处=affairs、图书馆=campus_life。
- academic 只含学业规则（选课/绩点/重修/保研/培养方案/考试规则）；涉及行政流程的事项即使与学业相关（缓考、注册、休学、奖学金、助学金、困难认定、校历、缴费、在读证明）一律归 affairs。
- 校园卡相关：余额/充值/消费归 campus_life；补办/挂失/补卡归 affairs。
- personal 只含个人数据与日程（我的课表、待办、作业、提醒、空闲时间、截止日期、考试倒计时）；通用教务规则（期末考时间）归 academic。
- 电子图书馆/网上图书馆的登录、无法访问问题归 it_help；图书馆的开放时间、资源访问说明归 campus_life。"""

        prompt = f"""你是西电校园智慧助手（EchoGuide）的查询理解模块。分析用户消息并输出结构化 JSON。

动作 action 可选值: {", ".join(a.value for a in IntentAction)}
定义：
- query = 查询、咨询、分析，不产生系统状态修改（"帮我查一下课表"是 query，即使有"帮我"）
- request = 需要系统真正执行写操作或产生副作用（"帮我添加一个补办校园卡的待办"、"把这个待办标记完成"是 request）
- greeting/complaint/feedback = 问候/投诉不满/正面反馈

动作示例:
{examples_text}

领域 domain 可选值: {", ".join(d.value for d in IntentDomain)}
领域定义与典型问法：
{domain_defs}
{boundary_notes}

{ctx}
用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"action": "<动作值>", "domain": "<领域值>", "domain_confidence": <0-1>, "confidence": <0-1>, "reasoning": "<一句话说明>", "needs_knowledge": <true/false>}}

要求：
- action 表示用户希望系统做什么（查询/操作/问候/投诉/反馈等）；只有明确需要系统写数据或产生副作用才是 request。
- domain 按领域定义与边界说明判断消息归属；拿不准时给最合理的领域（不要用 other 逃避）。
- needs_knowledge 表示该问题是否需要检索校园知识库（政策/流程/规则类 true；闲聊/个人数据操作 false）。
- 追问（如"那几点开门？"）应结合最近对话推断 action 与 domain。"""
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
            try:
                data["action"] = IntentAction(data.get("action", "other"))
            except ValueError:
                data["action"] = IntentAction.OTHER
            try:
                data["domain"] = IntentDomain(data.get("domain", "other"))
            except ValueError:
                data["domain"] = IntentDomain.OTHER
            data["needs_knowledge"] = bool(data.get("needs_knowledge", False))
            data["domain_confidence"] = float(data.get("domain_confidence", 1.0))
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {
                "action": IntentAction.OTHER,
                "domain": IntentDomain.OTHER,
                "confidence": 0.0,
                "reasoning": "LLM 失败",
                "failed": True,
            }

    async def _embedding_candidates(
        self, message: str, top_n: int = 3
    ) -> List[tuple[float, IntentDomain]]:
        """按模板相似度降序返回 Top-N 领域候选（(score, domain)）。

        供 _embedding_recognize 与路由 trace（评测错误分析）共用：
        错误分析需要"正确意图是否在候选内"来区分 Embedding 召回失败
        与阈值/仲裁问题，单一最佳值无法回答。
        """
        await self._load_template_embeddings()
        msg_vec = await self._embed_text(message, is_query=False)
        scored: List[tuple[float, IntentDomain]] = []
        for domain, vecs in self._tpl_embeddings.items():
            score = max(_cosine(msg_vec, v) for v in vecs)
            scored.append((score, domain))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_n]

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配（按领域模板）。

        用户消息与模板**同构嵌入**（都不带 bge-zh 指令前缀）：领域模板是"用户
        问法原型"，与用户消息同为 query 形态——指令前缀只该用于 RAG 检索
        （query vs passage 异构），用在模板匹配会把同义文本的相似度从 ~1.0
        压到 ~0.79（实测），导致阈值再紧都无法命中、Embedding 级联空转。
        """
        try:
            cands = await self._embedding_candidates(message, top_n=3)
            best_score, best_domain = cands[0] if cands else (0.0, IntentDomain.OTHER)
            second_score = cands[1][0] if len(cands) > 1 else 0.0
            return {
                "domain": best_domain,
                "action": IntentAction.OTHER,
                "confidence": best_score,
                "margin": max(0.0, best_score - second_score),
                "candidates": [
                    {"domain": domain, "score": score}
                    for score, domain in cands
                ],
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
    #     完整问句（"绩点怎么算的？"有主题词）则交由后续 Pattern 高置信
    #     + Embedding 双确认；"几点？""什么时候？"无主题词才进 LLM。
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
        返回 False，放行 Pattern；若 Pattern 高置信，再由 Embedding 双确认。
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
