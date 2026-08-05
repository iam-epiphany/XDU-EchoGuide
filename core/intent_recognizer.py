"""
亮点：端到端意图识别（层次化意图 Hierarchical Intent）

从「单维扁平意图」升级为「领域 domain × 动作 action」二维体系：

  - 领域 IntentDomain（academic/campus_life/affairs/it_help/other）
      —— 路由的唯一依据。修复了旧版 P0 缺陷：请求句式（"帮我…/我要…"）被
         few-shot 标成通用 REQUEST 后丢失领域信息，校务问题被学业 Agent 回答。
  - 动作 IntentAction（query/request/greeting/complaint/feedback/escalation）
      —— 行为决策依据（是否升级、是否转人工、紧急度）。

三路融合策略（与旧版一致的权重哲学）：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底（评分改为边际衰减，修正旧版
     hits/len(kws) 导致的长关键词表永远低分问题）

追问处理（对话感知）：
  - 无领域命中的短句（"那几点开门？"）自动从最近对话继承领域，保证追问路由正确。
  - 结果缓存 key 加入对话历史指纹，同一句追问在不同上下文不再返回陈旧意图。

领域关键词的唯一来源在 core/domains.py，本模块与 Orchestrator、API 层共用，
消除三处重复维护的漂移问题。
"""
import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.domains import (
    URGENCY_KEYWORDS,
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
    ESCALATION = "escalation"   # 转人工/升级
    # 西电校园场景的领域意图（兼容旧版路由）
    ACADEMIC   = "academic"     # 学业支持
    CAMPUS_LIFE = "campus_life" # 校园生活
    AFFAIRS    = "affairs"      # 校务咨询
    IT_HELP    = "it_help"      # IT 助手
    PERSONAL   = "personal"     # 个人助理（课表/待办/日程）
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


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
    IntentAction.ESCALATION: IntentCategory.ESCALATION,
}


@dataclass
class IntentResult:
    domain:     IntentDomain     # 领域（路由依据）
    action:     IntentAction     # 动作（行为依据）
    intent:     IntentCategory   # 兼容字段（domain 优先，其次 action）
    confidence: float
    urgency:    UrgencyLevel
    entities:   Dict[str, List[str]]
    reasoning:  str
    latency_ms: float


# ── Few-shot 模板 ─────────────────────────────────────────────────────────────
# 领域模板：用于 LLM 示例与 Embedding 匹配；动作模板：用于 LLM 示例。
_DOMAIN_TEMPLATES: Dict[IntentDomain, List[str]] = {
    IntentDomain.ACADEMIC:    ["这学期选课什么时候开始？", "绩点怎么算的？", "重修怎么报名？", "保研有什么条件？", "培养方案学分要求是什么？"],
    IntentDomain.CAMPUS_LIFE: ["南校区食堂几点关门？", "校车最后一班几点？", "宿舍怎么报修？", "校园卡在哪充值？", "校园卡丢了怎么补办？"],
    IntentDomain.AFFAIRS:     ["奖学金什么时候评？", "请假流程怎么走？", "在读证明在哪开？", "学费缴费方式有哪些？", "我要请假怎么走流程"],
    IntentDomain.IT_HELP:     ["教务系统登录不上", "校园网连不上", "VPN怎么配置？", "学校邮箱收不到邮件"],
    IntentDomain.PERSONAL:    ["今天有什么课？", "帮我查一下我的课表", "明天第几节在哪上课？", "这周周几没课？", "帮我记个待办，周三前交实验报告", "我最近的考试安排？", "还有什么没做完？"],
}

_ACTION_TEMPLATES: Dict[IntentAction, List[str]] = {
    IntentAction.QUERY:       ["西电校历这学期什么时候放假？", "图书馆几点开门？", "南校区快递站在哪？"],
    IntentAction.REQUEST:     ["帮我查一下选课时间", "帮我查一下校园卡余额"],
    IntentAction.GREETING:    ["你好", "嗨", "在吗", "早上好"],
    IntentAction.COMPLAINT:   ["宿舍热水一直不来！", "校车等了半小时还没来", "食堂排队太久了"],
    IntentAction.FEEDBACK:    ["这个助手很实用！", "回答得很清楚，谢谢", "帮我大忙了"],
    IntentAction.ESCALATION:  ["我要找辅导员", "转人工老师", "这个问题得找教务处"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
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
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        # 真实 Embedding：本地 all-MiniLM-L6-v2（与知识库同源，384 维）。
        # 与 base_url 无关 —— DeepSeek 等兼容端点同样使用真向量。
        # 模型不可用（如离线环境）时自动回退字符 n-gram 哈希向量，保证链路可用。
        self._embedding_enabled = True
        self._embedder = None   # 惰性初始化（DefaultEmbeddingFunction）

        self._tpl_embeddings: Dict[IntentDomain, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图（领域 + 动作）。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        缓存 key 包含最近对话指纹 —— 同一句追问在不同上下文不会命中陈旧结果。
        """
        key = self._cache_key(message, history)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        from core.tracing import span

        async with span("intent_recognize"):
            # LLM 和 Embedding 并行（Embedding 不可用时跳过）
            llm_task = asyncio.create_task(self._llm_recognize(message, history))
            emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
            pat      = self._pattern_recognize(message)

            if emb_task:
                llm, emb = await asyncio.gather(llm_task, emb_task)
            else:
                llm = await llm_task
                emb = {"domain": IntentDomain.OTHER, "action": IntentAction.OTHER, "confidence": 0.0}

            domain = self._vote_domain(llm, emb, pat)
            action = self._vote_action(llm, pat)

            # 追问继承：无领域命中的短句从最近对话继承领域
            domain = self._inherit_domain(message, history, domain, action)

            entities = await self._extract_entities(message)
            urgency  = self._urgency(message, action, domain)

        result = IntentResult(
            domain=domain,
            action=action,
            intent=self._legacy_intent(domain, action),
            confidence=self._voted_confidence(llm, emb, pat, domain),
            urgency=urgency,
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
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

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文），输出 domain + action。"""
        message = self._clean_text(message)
        examples = []
        for domain, tpls in _DOMAIN_TEMPLATES.items():
            for t in tpls[:1]:
                examples.append(f'  消息: "{t}" → domain: {domain.value}, action: query')
        for action, tpls in _ACTION_TEMPLATES.items():
            for t in tpls[:1]:
                examples.append(f'  消息: "{t}" → domain: other, action: {action.value}')
        examples_text = "\n".join(examples[:16])  # 控制 prompt 长度

        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是西电校园智慧助手（EchoGuide）的意图分析模块。请同时判断用户消息的【领域 domain】和【动作 action】。

领域 domain 可选值: {", ".join(d.value for d in IntentDomain)}
动作 action 可选值: {", ".join(a.value for a in IntentAction)}

示例:
{examples_text}

{ctx}
用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"domain": "<领域值>", "action": "<动作值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

要求：
- domain 表示用户问题属于哪个校园领域（学业/生活/校务/IT/个人助理）；无法判断时用 other。
- personal（个人助理）指与"我的"日程相关：我的课表、待办、考试安排、DDL 倒计时；
  而 academic 指教务规则类问题（选课流程、绩点算法、培养方案等）。
- action 表示用户希望系统做什么（查询/操作/问候/投诉/反馈/转人工等）。
- 追问（如"那几点开门？"）应结合最近对话推断 domain。"""
        prompt = self._clean_text(prompt)

        try:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["domain"] = IntentDomain(data.get("domain", "other"))
            except ValueError:
                data["domain"] = IntentDomain.OTHER
            try:
                data["action"] = IntentAction(data.get("action", "other"))
            except ValueError:
                data["action"] = IntentAction.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {
                "domain": IntentDomain.OTHER,
                "action": IntentAction.OTHER,
                "confidence": 0.0,
                "reasoning": "LLM 失败",
                "failed": True,
            }

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配（按领域模板）。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_domain, best_score = IntentDomain.OTHER, 0.0
            for domain, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_domain = score, domain

            return {"domain": best_domain, "action": IntentAction.OTHER, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"domain": IntentDomain.OTHER, "action": IntentAction.OTHER, "confidence": 0.0}

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

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote_domain(self, llm: Dict, emb: Dict, pat: Dict) -> IntentDomain:
        """领域维度加权投票。embedding 不可用时权重自动转移到 LLM 和 Pattern。"""
        if llm.get("failed"):
            if emb.get("domain") != IntentDomain.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["domain"]
            if pat.get("domain") != IntentDomain.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["domain"]
            return IntentDomain.OTHER

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentDomain, float] = {}
        for result, w in weights:
            domain = result.get("domain", IntentDomain.OTHER)
            conf   = result.get("confidence", 0.0)
            scores[domain] = scores.get(domain, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        return best if scores[best] >= self.threshold else IntentDomain.OTHER

    def _vote_action(self, llm: Dict, pat: Dict) -> IntentAction:
        """动作维度投票：LLM 主导，Pattern 兜底。"""
        if llm.get("failed"):
            if pat.get("action") != IntentAction.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["action"]
            return IntentAction.OTHER

        scores: Dict[IntentAction, float] = {}
        for result, w in [(llm, 0.85), (pat, 0.15)]:
            action = result.get("action", IntentAction.OTHER)
            conf   = result.get("confidence", 0.0)
            scores[action] = scores.get(action, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        return best if scores[best] >= self.threshold else IntentAction.OTHER

    def _voted_confidence(self, llm: Dict, emb: Dict, pat: Dict, domain: IntentDomain) -> float:
        """返回最终投票结果的置信度（而非 LLM 单路置信度，修正旧版失真）。"""
        if llm.get("failed"):
            return emb.get("confidence", 0.0) or pat.get("confidence", 0.0) or 0.0
        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        return round(sum(
            w * r.get("confidence", 0.0)
            for r, w in weights
            if r.get("domain") == domain
        ), 4)

    # ── 追问继承（对话感知）──────────────────────────────────────────────────

    @staticmethod
    def _inherit_domain(
        message: str,
        history: Optional[List[Dict[str, str]]],
        domain: IntentDomain,
        action: IntentAction,
    ) -> IntentDomain:
        """
        追问继承：当前消息无领域命中、且属于短句追问时，
        从最近几轮用户消息的领域关键词中继承领域。

        例如：上一轮"南校区食堂几点关门？" → 追问"那几点开门呢？" → campus_life。
        """
        if domain != IntentDomain.OTHER:
            return domain
        if action not in (IntentAction.QUERY, IntentAction.REQUEST, IntentAction.OTHER):
            return domain
        if not history:
            return domain

        # 只从 user 消息里找领域线索
        for m in reversed(history[-6:]):
            if m.get("role") != "user":
                continue
            text = str(m.get("content", ""))
            hit_domain, score = domain_hit_score(text)
            if hit_domain is not None and hit_domain != IntentDomain.OTHER and score >= 0.55:
                logger.info(f"追问领域继承: {message[:20]!r} → {hit_domain.value}（来自历史: {text[:20]!r}）")
                return hit_domain
        return domain

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    async def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用 LLM 从消息中提取结构化实体（西电校园场景）。"""
        message = self._clean_text(message)
        prompt = f"""从西电校园用户消息中提取实体，返回 JSON（字段值为列表，没有则为空列表）:
消息: "{message}"
格式: {{"course":[],"term":[],"location":[],"campus":[],"system":[]}}
（course=课程名, term=学期/时间, location=地点, campus=南校区/北校区, system=教务系统/校园网/VPN/邮箱等）"""
        prompt = self._clean_text(prompt)
        try:
            resp = await self.client.messages.create(
                model=self.model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[s:e])
        except Exception:
            return {"course": [], "term": [], "location": [], "campus": [], "system": []}

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    async def _load_template_embeddings(self) -> None:
        """懒加载所有领域模板的 Embedding（只在首次调用时执行）。"""
        missing = [d for d in _DOMAIN_TEMPLATES if d not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for d in missing for t in _DOMAIN_TEMPLATES[d]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for domain in missing:
            n = len(_DOMAIN_TEMPLATES[domain])
            self._tpl_embeddings[domain] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量：真实 Embedding 优先，n-gram 哈希兜底。

        优先使用 chromadb 的 DefaultEmbeddingFunction（all-MiniLM-L6-v2，
        384 维）——与知识库 RAG 同源模型，容器镜像已预下载、本机已缓存，
        零额外下载。模型加载失败时永久回退本地 n-gram 哈希向量，
        保证三路融合链路在任何环境都不中断。
        """
        if self._embedding_enabled:
            if self._embedder is None:
                try:
                    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

                    self._embedder = DefaultEmbeddingFunction()
                except Exception as ex:
                    logger.warning(f"本地 Embedding 模型不可用，回退 n-gram 向量: {ex}")
                    self._embedding_enabled = False
            if self._embedder is not None:
                try:
                    vec = await asyncio.to_thread(self._embedder, [text])
                    # DefaultEmbeddingFunction 返回 np.float32 向量：
                    # 转纯 float，避免 float32 混入置信度导致 JSON 序列化失败
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

    def _urgency(self, message: str, action: IntentAction, domain: IntentDomain) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return UrgencyLevel[level.upper()]
        if action == IntentAction.ESCALATION:
            return UrgencyLevel.HIGH
        if action == IntentAction.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

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
