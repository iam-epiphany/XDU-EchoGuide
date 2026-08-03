"""
亮点：端到端意图识别

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。
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

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    # 通用维度（与具体业务领域无关）
    QUERY      = "query"        # 信息查询
    REQUEST    = "request"      # 请求操作
    GREETING   = "greeting"     # 问候
    COMPLAINT  = "complaint"    # 投诉不满
    FEEDBACK   = "feedback"     # 正面反馈
    ESCALATION = "escalation"   # 转人工/升级
    # 西电校园场景的领域意图
    ACADEMIC   = "academic"     # 学业支持：选课/课表/考试/成绩/绩点/重修
    CAMPUS_LIFE = "campus_life" # 校园生活：宿舍/食堂/校车/校园卡/快递
    AFFAIRS    = "affairs"      # 校务咨询：校历/请假/奖学金/办事流程/注册
    IT_HELP    = "it_help"      # IT 助手：教务系统/校园网/VPN/邮箱/统一身份认证
    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
# 围绕西电校园场景的典型表达，覆盖四个领域 Agent + 通用维度。
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:       ["西电校历这学期什么时候放假？", "图书馆几点开门？", "南校区快递站在哪？"],
    IntentCategory.REQUEST:     ["帮我查一下选课时间", "我要请假怎么走流程", "校园卡丢了怎么补办"],
    IntentCategory.GREETING:    ["你好", "嗨", "在吗", "早上好"],
    IntentCategory.COMPLAINT:   ["宿舍热水一直不来！", "校车等了半小时还没来", "食堂排队太久了"],
    IntentCategory.FEEDBACK:    ["这个助手很实用！", "回答得很清楚，谢谢", "帮我大忙了"],
    IntentCategory.ESCALATION:  ["我要找辅导员", "转人工老师", "这个问题得找教务处"],
    IntentCategory.ACADEMIC:    ["这学期选课什么时候开始？", "绩点怎么算的？", "重修怎么报名？", "保研有什么条件？"],
    IntentCategory.CAMPUS_LIFE: ["南校区食堂几点关门？", "校车最后一班几点？", "宿舍怎么报修？", "校园卡在哪充值？"],
    IntentCategory.AFFAIRS:     ["奖学金什么时候评？", "请假流程怎么走？", "在读证明在哪开？", "学费缴费方式有哪些？"],
    IntentCategory.IT_HELP:     ["教务系统登录不上", "校园网连不上", "VPN怎么配置？", "学校邮箱收不到邮件"],
}

# 紧急关键词
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻", "马上要交"],
    UrgencyLevel.HIGH:     ["今天", "马上", "尽快", "hurry", "now", "截止前"],
    UrgencyLevel.MEDIUM:   ["这周", "soon", "快点", "这学期"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

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
        # 第三方兼容 API（如 DeepSeek）通常不支持 Embedding，禁用该策略。
        # 官方 Anthropic SDK 当前没有 embeddings 资源，因此下面会使用稳定的
        # 本地字符 n-gram 向量作为轻量兜底，保证三路融合链路真实可跑。
        self._embedding_enabled = not bool(base_url)

        self._tpl_embeddings: Dict[IntentCategory, List[List[float]]] = {}
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
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和 Embedding 并行（Embedding 不可用时跳过）
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}

        intent = self._vote(llm, emb, pat)
        entities = await self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=llm["confidence"],
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

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
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
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是西电校园智慧助手（EchoGuide）的意图分析模块。根据示例判断用户意图，返回 JSON。

示例:
{examples}

{ctx}
用户消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
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
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            await self._load_template_embeddings()
            msg_vec = await self._embed_text(message)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in self._tpl_embeddings.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": best_score}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """
        策略 3：关键词模式匹配（同步，零延迟兜底）。

        领域关键词（选课/食堂/请假/教务系统等）比通用疑问词（怎么/几点/什么时候）
        更具判别力，因此先匹配领域意图；无领域命中时才用通用意图兜底。
        """
        msg = message.lower()
        domain_patterns = {
            IntentCategory.ACADEMIC:   ["选课", "课表", "考试", "成绩", "绩点", "学分", "重修", "保研", "转专业", "挂科"],
            IntentCategory.CAMPUS_LIFE:["食堂", "宿舍", "校车", "校园卡", "快递", "水电", "超市", "运动场", "社团"],
            IntentCategory.AFFAIRS:    ["校历", "请假", "奖学金", "助学金", "证明", "缴费", "学费", "注册", "办事"],
            IntentCategory.IT_HELP:    ["教务系统", "校园网", "vpn", "邮箱", "登录不上", "报错", "密码重置", "统一身份"],
            IntentCategory.ESCALATION: ["转人工", "找辅导员", "教务处", "找老师", "escalate"],
        }
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in domain_patterns.items():
            hits = sum(1 for kw in kws if kw in msg)
            if hits:
                score = hits / len(kws)
                if score > best_score:
                    best_score, best_cat = score, cat
        if best_cat is not IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score}

        generic_patterns = {
            IntentCategory.COMPLAINT:  ["太差", "糟糕", "等了很久", "一直没人"],
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "几点", "在哪", "什么时候"],
            IntentCategory.REQUEST:    ["帮我", "需要", "我要", "怎么申请", "怎么办理"],
            IntentCategory.GREETING:   ["你好", "嗨", "hello", "hi", "在吗"],
        }
        for cat, kws in generic_patterns.items():
            hits = sum(1 for kw in kws if kw in msg)
            if hits:
                score = hits / len(kws)
                if score > best_score:
                    best_score, best_cat = score, cat
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> IntentCategory:
        """加权投票。embedding 不可用时权重自动转移到 LLM 和 Pattern。"""
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"]
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"]
            return IntentCategory.OTHER

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            scores[cat] = scores.get(cat, 0.0) + w * conf

        best = max(scores, key=scores.get)  # type: ignore
        return best if scores[best] >= self.threshold else IntentCategory.OTHER

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
        """懒加载所有模板的 Embedding（只在首次调用时执行）。"""
        missing = [cat for cat in _TEMPLATES if cat not in self._tpl_embeddings]
        if not missing:
            return

        all_texts = [t for cat in missing for t in _TEMPLATES[cat]]
        vecs = [await self._embed_text(text) for text in all_texts]
        idx = 0
        for cat in missing:
            n = len(_TEMPLATES[cat])
            self._tpl_embeddings[cat] = vecs[idx: idx + n]
            idx += n

    async def _embed_text(self, text: str) -> List[float]:
        """
        生成文本向量。

        如果未来接入的官方/兼容客户端提供 embeddings.create，会优先使用远端向量；
        当前 Anthropic SDK 没有该资源时，退化为字符 n-gram 哈希向量。这样不会因为
        Embedding 服务缺失导致三路融合中断。
        """
        embeddings = getattr(self.client, "embeddings", None)
        if embeddings is not None:
            try:
                resp = await embeddings.create(model="voyage-3-lite", input=[text])
                return list(resp.data[0].embedding)
            except Exception as ex:
                logger.warning(f"远端 Embedding 失败，使用本地向量兜底: {ex}")

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

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(kw in msg for kw in kws):
                return level
        if intent == IntentCategory.ESCALATION:
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str) -> str:
        return self._clean_text(message)[:200]

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
