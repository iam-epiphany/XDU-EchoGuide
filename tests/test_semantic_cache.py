"""语义缓存测试：上下文依赖性判定 + 双层读写路由（Global/User/强依赖 bypass）。

不依赖真实 ChromaDB：
  - 纯函数（classify_context_dependence / cache_tier / cache_read_tier / _entry_id）直接单测；
  - SemanticCache 用最小 FakeCollection（只做 where 过滤）验证跨用户隔离、
    强上下文依赖 bypass（计 bypass 不算 miss）、user 层不回退 Global；
  - API 层用记录型 FakeCache 验证读取发生在记忆上下文之后、dependence 传递与写入层路由。

覆盖 P0：
  1. 事实查询（即使有历史上下文）→ Global 层语义匹配（不再被指纹硬隔离）；
  2. 追问/省略句/指代 → 直接 bypass（计 bypass，不算 miss，不打 warning）；
  3. 同 query + 不同 user_id 在 User Cache 中独立并存（doc_id 含 user_id）；
  4. user 层请求不回退 Global（不绕过个性化 Agent 推理）；
  5. /chat 与 /chat/stream 行为一致；
  6. 写入侧叠加编排信号（personal/history/request → skip 不落库）。
"""
from __future__ import annotations

import asyncio
import hashlib

from mcp.semantic_cache import (
    SemanticCache,
    _entry_id,
    cache_read_tier,
    cache_tier,
    classify_context_dependence,
)


# ── 上下文依赖性判定 ────────────────────────────────────────────────────────

def test_classify_short_followup_with_deictic_skips():
    """含指代词的短追问 → skip（答案依赖上文话题，且优先于第一人称/疑问句式）。"""
    ctx = "[最近对话]\nuser: 南校区食堂几点关门？\nassistant: 22:00"
    assert classify_context_dependence("那几点开门？", ctx_text=ctx) == "skip"
    assert classify_context_dependence("这个怎么办理？", ctx_text=ctx) == "skip"
    assert classify_context_dependence("它几点开？", ctx_text=ctx) == "skip"
    # 指代优先于第一人称：即使含"我的"也判 skip
    assert classify_context_dependence("我的这个怎么弄？", ctx_text=ctx) == "skip"


def test_classify_ellipsis_skips():
    """省略句/极短问句 → skip（依赖上文话题，语义匹配不可靠）。"""
    ctx = "[最近对话]\nuser: 南校区食堂几点关门？\nassistant: 22:00"
    assert classify_context_dependence("图书馆呢？", ctx_text=ctx) == "skip"
    assert classify_context_dependence("然后呢？", ctx_text=ctx) == "skip"
    assert classify_context_dependence("怎么样？", ctx_text=ctx) == "skip"
    assert classify_context_dependence("几点开？", ctx_text=ctx) == "skip"
    assert classify_context_dependence("选课吗？", ctx_text=ctx) == "skip"


def test_classify_ma_not_a_followup_signal():
    """"吗"不是追问信号：完整疑问句按事实查询走 Global。"""
    assert classify_context_dependence("图书馆几点关门吗？", ctx_text="[用户画像]") == "global"


def test_classify_first_person_goes_user():
    """第一人称指代 → user（答案依赖用户画像），且"我们学校…"不误伤。"""
    ctx = "[用户画像]\n{\"preferences\": [\"清淡\"]}"
    assert classify_context_dependence("我适合什么选修课？", ctx_text=ctx) == "user"
    assert classify_context_dependence("我的课表几点更新？", ctx_text=ctx) == "user"
    assert classify_context_dependence("推荐一下食堂", ctx_text=ctx) == "user"
    # "我们学校图书馆几点关门？"是公共问题 → global
    assert classify_context_dependence("我们学校图书馆几点关门？", ctx_text=ctx) == "global"


def test_classify_fact_query_goes_global_even_with_context():
    """事实查询（含疑问词）即使有历史上下文 → global（核心行为变化）。"""
    ctx = "[最近对话]\nuser: 你好\nassistant: 你好！有什么可以帮你？"
    assert classify_context_dependence("选课什么时候开始？", ctx_text=ctx) == "global"
    assert classify_context_dependence("图书馆几点关门？", ctx_text=ctx) == "global"
    assert classify_context_dependence("南校区食堂几点关门？", ctx_text=ctx) == "global"


def test_classify_no_context_goes_global():
    assert classify_context_dependence("南校区食堂几点关门？", ctx_text="") == "global"
    assert classify_context_dependence("你好", ctx_text="") == "global"


def test_classify_fallback_user_with_context():
    """有上下文但无任何信号 → 保守 user（防公共答案绕过个性化推理）。"""
    assert classify_context_dependence("推荐一下", ctx_text="[用户画像]") == "user"
    assert classify_context_dependence("南校区食堂", ctx_text="[最近对话]") == "user"


def test_classify_orchestrator_signals_skip():
    """编排信号（写入侧）→ skip：personal 领域/请求动作。"""
    ctx = "[用户画像]"
    assert classify_context_dependence("我的课表几点更新？", ctx_text=ctx, domain="personal") == "skip"
    assert classify_context_dependence("帮我查一下食堂", ctx_text=ctx, action="request") == "skip"
    assert classify_context_dependence("今天有什么安排？", ctx_text=ctx, domain="personal", action="query") == "skip"


def test_classify_empty_message_skips():
    assert classify_context_dependence("", ctx_text="...") == "skip"
    assert classify_context_dependence("   ", ctx_text="...") == "skip"


# ── tier 纯映射 ─────────────────────────────────────────────────────────────

def test_cache_tier_pure_mapping():
    """cache_tier 只做三态 → 写入层映射，不重复判断 domain/action。"""
    assert cache_tier("global", None) == "global"
    assert cache_tier("global", "u1") == "global"
    assert cache_tier("global", "anonymous") == "global"
    assert cache_tier("user", "u1") == "user"
    assert cache_tier("user", "xdu_2024") == "user"
    assert cache_tier("user", "anonymous") is None
    assert cache_tier("user", "") is None
    assert cache_tier("user", None) is None
    assert cache_tier("skip", "u1") is None
    assert cache_tier("skip", None) is None


def test_cache_read_tier_pure_mapping():
    """读取层映射：user + 有效身份 → 只读 User（不回退 Global）；skip/匿名 → 跳过。"""
    assert cache_read_tier("global", "u1") == "global"
    assert cache_read_tier("global", "anonymous") == "global"
    assert cache_read_tier("user", "u1") == "user"
    assert cache_read_tier("user", "anonymous") is None
    assert cache_read_tier("user", "") is None
    assert cache_read_tier("user", None) is None
    assert cache_read_tier("skip", "u1") is None
    assert cache_read_tier("skip", None) is None


def test_entry_id_isolation_between_users():
    query = "推荐一下食堂"
    # 同 query + 不同 user_id → 不同 ID（User 缓存互不覆盖）
    assert _entry_id(query, user_id="A") != _entry_id(query, user_id="B")
    # 同 user_id + 同 query → 稳定 ID（upsert 覆盖只留最新）
    assert _entry_id(query, user_id="A") == _entry_id(query, user_id="A")
    # Global 保持 md5(query)（不破坏现有行为，且与 User ID 区分）
    assert _entry_id(query) == hashlib.md5(query.encode("utf-8")).hexdigest()
    assert _entry_id(query) != _entry_id(query, user_id="A")


# ── SemanticCache 行为（FakeCollection 只做 where 过滤）──────────────────────

class _FakeCollection:
    """最小 ChromaDB collection 替身：只按 where 过滤，距离固定 0.0（相似度 1.0）。"""

    def __init__(self):
        self.entries: dict = {}  # doc_id -> (doc, meta)

    def upsert(self, ids, documents, metadatas):
        for i, doc, meta in zip(ids, documents, metadatas):
            self.entries[i] = (doc, meta)

    def query(self, query_texts, n_results=1, where=None):
        docs, metas, dists = [], [], []
        for doc, meta in self.entries.values():
            if self._match(meta, where):
                docs.append(doc)
                metas.append(meta)
                dists.append(0.0)
        n = min(n_results, len(docs))
        return {
            "documents": [docs[:n]],
            "metadatas": [metas[:n]],
            "distances": [dists[:n]],
        }

    @staticmethod
    def _match(meta, where):
        if not where:
            return True
        if "$and" in where:
            return all(meta.get(k) == v for cond in where["$and"] for k, v in cond.items())
        return all(meta.get(k) == v for k, v in where.items())


def _cache():
    """构造不连接真实 ChromaDB 的 SemanticCache（替换为 FakeCollection）。"""
    cache = SemanticCache.__new__(SemanticCache)
    cache.enabled = True
    cache.threshold = 0.85
    cache.ttl_s = 86400
    cache._hits = 0
    cache._misses = 0
    cache._bypass = 0
    cache._global = _FakeCollection()
    cache._user = _FakeCollection()
    return cache, cache._global, cache._user


def test_user_cache_cross_user_isolation():
    """P0-3：同 query + 不同 user_id 独立并存，互不覆盖、互不命中。"""
    cache, g, user = _cache()
    cache.put("推荐一下食堂", "用户A的个性化回答（偏好清淡，推荐一食堂）", domain="campus_life", user_id="A", dependence="user")
    cache.put("推荐一下食堂", "用户B的个性化回答（偏好无辣，推荐二食堂）", domain="campus_life", user_id="B", dependence="user")

    # 两条独立记录（upsert 未覆盖）
    assert len(user.entries) == 2
    # A 只能命中 A 的回答，B 只能命中 B 的回答
    hit_a = cache.get("推荐一下食堂", user_id="A", dependence="user")
    assert hit_a and hit_a["response"] == "用户A的个性化回答（偏好清淡，推荐一食堂）"
    hit_b = cache.get("推荐一下食堂", user_id="B", dependence="user")
    assert hit_b and hit_b["response"] == "用户B的个性化回答（偏好无辣，推荐二食堂）"


def test_user_cache_hit_ignores_context_change():
    """User 层不再按上下文指纹硬隔离：同一用户画像/历史变化后，相同问题仍可语义命中。"""
    cache, g, user = _cache()
    cache.put("推荐一下食堂", "推荐一食堂（偏好清淡），午餐人较少，建议错峰就餐。", domain="campus_life", user_id="A", dependence="user")
    # 无 where 之外的指纹过滤，同用户重复查询直接命中
    hit = cache.get("推荐一下食堂", user_id="A", dependence="user")
    assert hit and hit["response"] == "推荐一食堂（偏好清淡），午餐人较少，建议错峰就餐。"
    # 同一问题再次语义改写（FakeCollection 距离固定 1.0）仍命中
    hit2 = cache.get("推荐个食堂", user_id="A", dependence="user")
    assert hit2 and hit2["response"] == "推荐一食堂（偏好清淡），午餐人较少，建议错峰就餐。"


def test_skip_read_counts_bypass_not_miss():
    """P0-2：强上下文依赖（skip）读取 → 计 bypass，不计入 miss（不影响 hit_rate）。"""
    cache, g, user = _cache()
    for _ in range(3):
        assert cache.get("那几点开门？", user_id="u1", dependence="skip") is None
    assert cache._bypass == 3
    assert cache._misses == 0
    assert cache._hits == 0


def test_skip_write_silently_skipped():
    """skip 写入 → 静默跳过（不入 Global/User，也不打 warning）。"""
    cache, g, user = _cache()
    cache.put("那几点开门？", "食堂 7:00 开门，早餐时段人较少，建议错峰就餐。", domain="campus_life", user_id="123", dependence="skip")
    assert len(g.entries) == 0 and len(user.entries) == 0


def test_user_tier_never_falls_back_to_global():
    """P0-4：user 层请求（依赖用户画像）Global 命中也不复用（不绕过个性化推理）。"""
    cache, g, user = _cache()
    cache.put("南校区食堂几点关门？", "公共答案：南校区食堂一般 22:00 关门。", domain="campus_life", dependence="global")

    # global → 读 Global 命中
    assert cache.get("南校区食堂几点关门？", dependence="global") is not None
    # user（如个性化上下文）→ 只查 User 层，Global 条目不可见 → miss
    assert cache.get("南校区食堂几点关门？", user_id="u1", dependence="user") is None


def test_anonymous_user_skips_read_and_write():
    """匿名 + user：读跳过（计 bypass）、写跳过（不污染 Global/User）。"""
    cache, g, user = _cache()
    # 写：user 但匿名 → 不入任何缓存
    cache.put("推荐一下食堂", "上下文相关回答", domain="campus_life", user_id="anonymous", dependence="user")
    assert len(g.entries) == 0 and len(user.entries) == 0
    # 读：跳过（计 bypass，不算 miss）
    assert cache.get("推荐一下食堂", user_id="anonymous", dependence="user") is None
    assert cache._bypass == 1
    assert cache._misses == 0


def test_global_cache_normal_scenario_unchanged():
    """上下文无关问题：写 Global、读 Global，任意用户可复用。"""
    cache, g, user = _cache()
    cache.put("选课分几个阶段？", "选课一般分为预选、正选、退改选几个阶段。", domain="academic", dependence="global")
    assert len(g.entries) == 1
    hit = cache.get("选课分几个阶段？", dependence="global")
    assert hit and hit["response"] == "选课一般分为预选、正选、退改选几个阶段。"
    assert hit["tier"] == "global"
    # 任意用户都可读 Global
    assert cache.get("选课分几个阶段？", user_id="u1", dependence="global") is not None
    assert cache.get("选课分几个阶段？", user_id="anonymous", dependence="global") is not None


def test_stats_reports_bypass():
    cache, g, user = _cache()
    cache.get("那几点开门？", user_id="u1", dependence="skip")
    cache.get("选课什么时候开始？", dependence="global")
    s = cache.stats
    assert s["bypass"] == 1
    assert s["misses"] == 1  # global miss 计入 miss
    assert s["hits"] == 0


# ── API 层读写路由（/chat 与 /chat/stream 共用同一逻辑）──────────────────────

class _RecordingCache:
    """记录 get/put 调用参数，模拟语义缓存。"""

    def __init__(self):
        self.gets = []
        self.puts = []

    def get(self, query, user_id=None, dependence="global"):
        self.gets.append((query, user_id, dependence))
        return None

    def put(self, query, response, domain="other", agent_type="", user_id=None, dependence="global", knowledge_used=False):
        self.puts.append({"query": query, "domain": domain, "user_id": user_id, "dependence": dependence})


class _FakeMemory:
    def __init__(self, context_text=""):
        self._context_text = context_text

    async def get_context(self, user_id, conv_id, query=""):
        class Ctx:
            recent_messages = []

            def __init__(self, text):
                self._text = text

            def to_prompt_text(self):
                return self._text

        return Ctx(self._context_text)

    async def add_message(self, *args, **kwargs):
        return None

    async def update_profile(self, *args, **kwargs):
        return None


class _FakeOrchestrator:
    async def run(self, req, on_event=None):
        from agents.agent_orchestrator import AgentType, OrchestratorResult
        from core.domains import IntentAction, IntentDomain

        return OrchestratorResult(
            request_id="r1",
            response="南校区食堂一般晚上七点关门。",
            agent_type=AgentType.CAMPUS_LIFE,
            intent=None,
            domain=IntentDomain.CAMPUS_LIFE,
            action=IntentAction.QUERY,
            latency_ms=12.3,
            tools_used=[],
        )


def _run_chat(user_id, context_text="", message="南校区食堂几点关门？"):
    """打桩跑一遍 /chat 非流式主链路，返回缓存调用记录。"""
    import api.main as m
    from fastapi import Response

    m._orchestrator = _FakeOrchestrator()
    m._memory = _FakeMemory(context_text)
    cache = _RecordingCache()
    m._semantic_cache = cache

    req = m.ChatRequest(message=message, user_id=user_id)
    resp = asyncio.run(m.chat(req, Response()))
    assert resp.response  # 正常返回
    return cache


def test_chat_fact_query_reads_global():
    """无记忆上下文的事实查询 → dependence="global"（只读 Global）。"""
    cache = _run_chat("u1")
    assert cache.gets and cache.gets[0] == ("南校区食堂几点关门？", "u1", "global")


def test_chat_fact_query_with_history_still_reads_global():
    """P0-1（API 层）：事实查询即使有历史上下文也判 global（不再被指纹强制进 User 层）。"""
    cache = _run_chat("u1", context_text="[最近对话]\nuser: 你好\nassistant: 你好！")
    assert cache.gets and cache.gets[0][0] == "南校区食堂几点关门？"
    assert cache.gets[0][2] == "global"


def test_chat_followup_skips_cache():
    """追问（强上下文依赖）→ dependence="skip"，get 内部直接 bypass。"""
    cache = _run_chat("u1", context_text="[最近对话]\nuser: 南校区食堂几点关门？\nassistant: 22:00", message="那几点开门？")
    assert cache.gets and cache.gets[0][2] == "skip"


def test_chat_write_fact_query_goes_global_even_with_context():
    """事实查询 → 即使有历史上下文也写入 Global 层（核心行为变化）。"""
    cache = _run_chat("u1", context_text="[最近对话]\nuser: 你好")
    assert len(cache.puts) == 1
    assert cache.puts[0]["domain"] == "campus_life"
    assert cache.puts[0]["user_id"] is None
    assert cache.puts[0]["dependence"] == "global"


def test_chat_write_personalized_goes_user_tier():
    """依赖用户画像的问题 → 写入 User 层（user_id 分区）。"""
    cache = _run_chat("u1", context_text="[用户画像]\n{\"preferences\": [\"清淡\"]}", message="推荐一下食堂")
    assert len(cache.puts) == 1
    assert cache.puts[0]["user_id"] == "u1"
    assert cache.puts[0]["dependence"] == "user"


def test_chat_anonymous_followup_not_written():
    """匿名 + 追问 → 读跳过、不写缓存。"""
    cache = _run_chat("anonymous", context_text="[最近对话]\nuser: 你好", message="那几点开门？")
    assert cache.gets and cache.gets[0][2] == "skip"
    assert cache.puts == []
