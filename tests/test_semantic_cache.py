"""双层语义缓存测试：隔离语义（user_id + 上下文指纹）+ API 层读写路由。

不依赖真实 ChromaDB：
  - 纯函数（cache_tier / context_fingerprint / cache_read_tier / _entry_id）直接单测；
  - SemanticCache 用最小 FakeCollection（只做 where 过滤）验证跨用户/跨上下文隔离、
    上下文相关不回退 Global、匿名跳过缓存；
  - API 层用记录型 FakeCache 验证读取发生在记忆上下文之后、指纹传递与写入层路由。

覆盖 P0：
  1. 同 query + 不同 user_id 在 User Cache 中独立并存（doc_id 含 user_id + context_fp）；
  2. 同一用户不同对话上下文（"那几点开门？"等追问）互不污染；
  3. 有明显用户上下文的请求不回退 Global（不绕过个性化 Agent 推理）；
  4. /chat 与 /chat/stream 行为一致；
  5. Global / User 普通场景不受影响。
"""
from __future__ import annotations

import asyncio
import hashlib

from mcp.semantic_cache import (
    SemanticCache,
    _entry_id,
    cache_read_tier,
    cache_tier,
    context_fingerprint,
)


# ── 纯函数 ──────────────────────────────────────────────────────────────────

def test_cache_tier_personal_and_other_never_cached():
    assert cache_tier("personal", has_user_context=False, user_id="u1") is None
    assert cache_tier("personal", has_user_context=True, user_id="u1") is None
    assert cache_tier("other", has_user_context=False, user_id=None) is None


def test_cache_tier_no_context_goes_global():
    assert cache_tier("academic", has_user_context=False, user_id=None) == "global"
    assert cache_tier("academic", has_user_context=False, user_id="u1") == "global"
    assert cache_tier("campus_life", has_user_context=False, user_id="anonymous") == "global"


def test_cache_tier_user_context_with_identity_goes_user():
    assert cache_tier("academic", has_user_context=True, user_id="u1") == "user"
    assert cache_tier("affairs", has_user_context=True, user_id="xdu_2024") == "user"


def test_cache_tier_anonymous_with_context_skipped():
    assert cache_tier("academic", has_user_context=True, user_id="anonymous") is None
    assert cache_tier("academic", has_user_context=True, user_id="") is None
    assert cache_tier("academic", has_user_context=True, user_id=None) is None


def test_context_fingerprint_empty_for_no_context():
    assert context_fingerprint("") == ""
    assert context_fingerprint("   ") == ""


def test_context_fingerprint_stable_and_sensitive():
    fp1 = context_fingerprint("[用户画像]\n{\"preferences\": [\"清淡\"]}")
    fp2 = context_fingerprint("[用户画像]\n{\"preferences\": [\"清淡\"]}")
    assert fp1 == fp2 and fp1
    # 不同对话上下文（历史不同）→ 指纹不同
    ctx_canteen = "[最近对话]\nuser: 南校区食堂几点关门？\nassistant: 22:00"
    ctx_library = "[最近对话]\nuser: 图书馆几点关门？\nassistant: 22:30"
    assert context_fingerprint(ctx_canteen) != context_fingerprint(ctx_library)


def test_cache_read_tier_routing():
    # 有上下文 + 有效身份 → 只读 User（不回退 Global）
    assert cache_read_tier("fp", "u1") == "user"
    # 有上下文 + 匿名 → 跳过缓存
    assert cache_read_tier("fp", "anonymous") is None
    assert cache_read_tier("fp", "") is None
    assert cache_read_tier("fp", None) is None
    # 无上下文 → 只读 Global
    assert cache_read_tier("", "u1") == "global"
    assert cache_read_tier("", "anonymous") == "global"


def test_entry_id_isolation_between_users_and_contexts():
    query = "推荐一下食堂"
    # 同 query + 不同 user_id → 不同 ID（User 缓存互不覆盖）
    assert _entry_id(query, user_id="A", context_fp="fp") != _entry_id(query, user_id="B", context_fp="fp")
    # 同 user_id + 不同上下文 → 不同 ID（跨对话不覆盖）
    assert _entry_id(query, user_id="A", context_fp="fp1") != _entry_id(query, user_id="A", context_fp="fp2")
    # 同 user_id + 同上下文 + 同 query → 稳定 ID（可命中）
    assert _entry_id(query, user_id="A", context_fp="fp1") == _entry_id(query, user_id="A", context_fp="fp1")
    # Global 保持 md5(query)（不破坏现有行为，且与 User ID 区分）
    assert _entry_id(query) == hashlib.md5(query.encode("utf-8")).hexdigest()
    assert _entry_id(query) != _entry_id(query, user_id="A", context_fp="fp")


# ── SemanticCache 隔离语义（FakeCollection 只做 where 过滤）──────────────────

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
    cache._global = _FakeCollection()
    cache._user = _FakeCollection()
    return cache, cache._global, cache._user


def test_user_cache_cross_user_isolation():
    """P0-1：同 query + 不同 user_id 独立并存，互不覆盖、互不命中。"""
    cache, g, user = _cache()
    cache.put("推荐一下食堂", "用户A的个性化回答（偏好清淡，推荐一食堂）", domain="campus_life", user_id="A", context_fp="fp")
    cache.put("推荐一下食堂", "用户B的个性化回答（偏好无辣，推荐二食堂）", domain="campus_life", user_id="B", context_fp="fp")

    # 两条独立记录（upsert 未覆盖）
    assert len(user.entries) == 2
    # A 只能命中 A 的回答，B 只能命中 B 的回答
    hit_a = cache.get("推荐一下食堂", user_id="A", context_fp="fp")
    assert hit_a and hit_a["response"] == "用户A的个性化回答（偏好清淡，推荐一食堂）"
    hit_b = cache.get("推荐一下食堂", user_id="B", context_fp="fp")
    assert hit_b and hit_b["response"] == "用户B的个性化回答（偏好无辣，推荐二食堂）"


def test_user_cache_cross_context_isolation():
    """P0-2：同一用户不同对话上下文（追问）互不污染。"""
    cache, g, user = _cache()
    ctx_canteen = "[最近对话]\nuser: 南校区食堂几点关门？\nassistant: 22:00"
    ctx_library = "[最近对话]\nuser: 图书馆几点关门？\nassistant: 22:30"
    fp_canteen = context_fingerprint(ctx_canteen)
    fp_library = context_fingerprint(ctx_library)

    cache.put("那几点开门？", "食堂 7:00 开门，早餐时段人较少，建议错峰就餐。", domain="campus_life", user_id="123", context_fp=fp_canteen)

    # 图书馆话题下同样追问 → 不同指纹 → miss（走正常 Agent 推理）
    assert cache.get("那几点开门？", user_id="123", context_fp=fp_library) is None
    # 食堂话题下再次追问 → 命中
    hit = cache.get("那几点开门？", user_id="123", context_fp=fp_canteen)
    assert hit and hit["response"] == "食堂 7:00 开门，早餐时段人较少，建议错峰就餐。"
    # 两个上下文各自写入后并存
    cache.put("那几点开门？", "图书馆 8:00 开门，周三闭馆休息，周末延长开放。", domain="campus_life", user_id="123", context_fp=fp_library)
    assert len(user.entries) == 2


def test_context_dependent_never_falls_back_to_global():
    """P0：有明显用户上下文的请求，Global 命中也不复用（不绕过个性化推理）。"""
    cache, g, user = _cache()
    cache.put("南校区食堂几点关门？", "公共答案：南校区食堂一般 22:00 关门。", domain="campus_life")

    # 无上下文 → 读 Global 命中
    assert cache.get("南校区食堂几点关门？", context_fp="") is not None
    # 有上下文（如画像/历史）→ 只查 User，Global 条目不可见 → miss
    assert cache.get("南校区食堂几点关门？", user_id="u1", context_fp="fp") is None
    assert cache.get("南校区食堂几点关门？", user_id="anonymous", context_fp="fp") is None


def test_anonymous_context_dependent_skips_read_and_write():
    """anonymous + 上下文：读跳过、写跳过（不污染 Global/User）。"""
    cache, g, user = _cache()
    # 写：上下文相关但匿名 → 不入任何缓存
    cache.put("今天有什么安排？", "上下文相关回答", domain="personal", user_id="anonymous", context_fp="fp")
    assert len(g.entries) == 0 and len(user.entries) == 0
    # 读：跳过
    assert cache.get("今天有什么安排？", user_id="anonymous", context_fp="fp") is None


def test_global_cache_normal_scenario_unchanged():
    """上下文无关问题：写 Global、读 Global，行为与之前一致。"""
    cache, g, user = _cache()
    cache.put("选课分几个阶段？", "选课一般分为预选、正选、退改选几个阶段。", domain="academic")
    assert len(g.entries) == 1
    hit = cache.get("选课分几个阶段？", context_fp="")
    assert hit and hit["response"] == "选课一般分为预选、正选、退改选几个阶段。"
    assert hit["tier"] == "global"
    # 任意用户都可读 Global（无 user_id 或带 user_id 但无上下文）
    assert cache.get("选课分几个阶段？", user_id="u1", context_fp="") is not None


# ── API 层读写路由（/chat 与 /chat/stream 共用同一逻辑）──────────────────────

class _RecordingCache:
    """记录 get/put 调用参数，模拟语义缓存。"""

    def __init__(self):
        self.gets = []
        self.puts = []

    def get(self, query, user_id=None, context_fp=""):
        self.gets.append((query, user_id, context_fp))
        return None

    def put(self, query, response, domain="other", agent_type="", user_id=None, context_fp=""):
        self.puts.append({"query": query, "domain": domain, "user_id": user_id, "context_fp": context_fp})


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


def _run_chat(user_id, context_text=""):
    """打桩跑一遍 /chat 非流式主链路，返回缓存调用记录。"""
    import api.main as m
    from fastapi import Response

    m._orchestrator = _FakeOrchestrator()
    m._memory = _FakeMemory(context_text)
    cache = _RecordingCache()
    m._semantic_cache = cache

    req = m.ChatRequest(message="南校区食堂几点关门？", user_id=user_id)
    resp = asyncio.run(m.chat(req, Response()))
    assert resp.response  # 正常返回
    return cache


def test_chat_context_free_reads_global():
    """无用户上下文 → 缓存读取发生在记忆上下文之后，context_fp 为空（只读 Global）。"""
    cache = _run_chat("u1")
    assert cache.gets and cache.gets[0] == ("南校区食堂几点关门？", "u1", "")


def test_chat_context_dependent_reads_user_with_fp():
    """有用户上下文 → context_fp 非空（只读 User 层，按指纹隔离）。"""
    cache = _run_chat("u1", context_text="[用户画像]\n{\"preferences\": [\"清淡\"]}")
    assert cache.gets and cache.gets[0][0] == "南校区食堂几点关门？"
    assert cache.gets[0][1] == "u1"
    assert cache.gets[0][2]  # context_fp 非空


def test_chat_write_no_context_goes_global():
    """无画像/历史 → 写入 Global 层（不带 user_id / context_fp）。"""
    cache = _run_chat("u1")
    assert len(cache.puts) == 1
    assert cache.puts[0]["domain"] == "campus_life"
    assert cache.puts[0]["user_id"] is None
    assert cache.puts[0]["context_fp"] == ""


def test_chat_write_with_context_goes_user_tier_with_fp():
    """有画像/历史 + 有效身份 → 写入 User 层（user_id + 上下文指纹）。"""
    cache = _run_chat("u1", context_text="[用户画像]\n{\"preferences\": [\"清淡\"]}")
    assert len(cache.puts) == 1
    assert cache.puts[0]["user_id"] == "u1"
    assert cache.puts[0]["context_fp"]  # 指纹随写入传递


def test_chat_write_anonymous_with_context_skipped():
    """匿名 + 有上下文 → 读跳过、不写缓存（防匿名会话间串扰）。"""
    cache = _run_chat("anonymous", context_text="[最近对话]\nuser: 你好")
    # 读取仍会发起（返回 None 后走正常 Agent 推理），但绝不写缓存
    assert cache.gets and cache.gets[0][2]  # context_fp 非空
    assert cache.puts == []
