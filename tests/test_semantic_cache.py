"""双层语义缓存测试：cache_tier 决策 + API 层读写路由（不依赖真实 ChromaDB）。

覆盖：
  1. cache_tier 纯函数：personal/other 不缓存、无上下文 → Global、
     有上下文且身份有效 → User、匿名带上下文 → 不缓存
  2. API 层：读缓存带 user_id；写缓存按上下文/身份路由到 user/global 层
"""
from __future__ import annotations

import asyncio

from mcp.semantic_cache import cache_tier


# ── cache_tier 决策 ─────────────────────────────────────────────────────────

def test_cache_tier_personal_and_other_never_cached():
    assert cache_tier("personal", has_user_context=False, user_id="u1") is None
    assert cache_tier("personal", has_user_context=True, user_id="u1") is None
    assert cache_tier("other", has_user_context=False, user_id=None) is None


def test_cache_tier_no_context_goes_global():
    # 无画像/历史 → 答案不依赖用户上下文 → Global（任何身份均可写入）
    assert cache_tier("academic", has_user_context=False, user_id=None) == "global"
    assert cache_tier("academic", has_user_context=False, user_id="u1") == "global"
    assert cache_tier("campus_life", has_user_context=False, user_id="anonymous") == "global"


def test_cache_tier_user_context_with_identity_goes_user():
    # 回答受画像/历史影响 → User 缓存，按 user_id 隔离
    assert cache_tier("academic", has_user_context=True, user_id="u1") == "user"
    assert cache_tier("affairs", has_user_context=True, user_id="xdu_2024") == "user"


def test_cache_tier_anonymous_with_context_skipped():
    # 匿名 + 有上下文：写 user 库会与其他匿名会话串扰 → 不缓存
    assert cache_tier("academic", has_user_context=True, user_id="anonymous") is None
    assert cache_tier("academic", has_user_context=True, user_id="") is None
    assert cache_tier("academic", has_user_context=True, user_id=None) is None


# ── API 层读写路由 ──────────────────────────────────────────────────────────

class _RecordingCache:
    """记录 get/put 调用参数，模拟语义缓存。"""

    def __init__(self):
        self.gets = []
        self.puts = []

    def get(self, query, user_id=None):
        self.gets.append((query, user_id))
        return None

    def put(self, query, response, domain="other", agent_type="", user_id=None):
        self.puts.append({"query": query, "domain": domain, "user_id": user_id})


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


def test_chat_read_always_passes_user_id():
    """读缓存带 user_id：先查 User 层（隔离），再回退 Global 层。"""
    cache = _run_chat("u1")
    assert cache.gets and cache.gets[0] == ("南校区食堂几点关门？", "u1")


def test_chat_write_no_context_goes_global():
    """无画像/历史 → 写入 Global 层（不带 user_id）。"""
    cache = _run_chat("u1")
    assert len(cache.puts) == 1
    assert cache.puts[0]["domain"] == "campus_life"
    assert cache.puts[0]["user_id"] is None  # global 层


def test_chat_write_with_context_goes_user_tier():
    """有画像/历史上下文 + 有效身份 → 写入 User 层（按 user_id 隔离）。"""
    cache = _run_chat("u1", context_text="[用户画像]\n{\"preferences\": [\"清淡\"]}")
    assert len(cache.puts) == 1
    assert cache.puts[0]["user_id"] == "u1"  # user 层


def test_chat_write_anonymous_with_context_skipped():
    """匿名 + 有上下文 → 不写缓存（防匿名会话间串扰）。"""
    cache = _run_chat("anonymous", context_text="[最近对话]\nuser: 你好")
    assert cache.puts == []
