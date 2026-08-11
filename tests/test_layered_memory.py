"""分层记忆（L0-L3 金字塔 + 上下文卸载）单元测试。

照项目现有模式：
  - 纯逻辑（token 估算、MemoryContext 组装）直接断言，零外部依赖
  - LayeredStore 用 pytest tmp_path 的 SQLite 直测
  - MemoryManager 行为用 _FakeClient（顺序响应）+ _FakeCollection（upsert 记录）
    + monkeypatch 替换内部方法，不依赖真实 Redis / ChromaDB / LLM
"""
from __future__ import annotations

import asyncio
import json

import pytest

from memory.conversation_memory import (
    MemoryContext, MemoryManager, Message, MsgRole,
)
from memory.layered_store import LayeredStore, estimate_tokens


# ── 纯逻辑：token 估算 ───────────────────────────────────────────────────────

def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1          # ASCII 4 字符 ≈ 1 token
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("中文") == 2         # 中文 1 字符 ≈ 1 token
    assert estimate_tokens("你好 world") == 2 + 2  # 2 中文 + 5 ASCII


def test_memory_context_to_prompt_includes_facts():
    """L1 原子事实注入 prompt，且位于画像之前。"""
    ctx = MemoryContext(
        recent_messages=[Message(role=MsgRole.USER, content="hi")],
        relevant_history=[],
        user_profile={"preferences": ["p"]},
        summary="",
        facts=["用户在准备考研", "用户在南校区"],
        memory_trace={},
    )
    text = ctx.to_prompt_text()
    assert "[用户事实]\n- 用户在准备考研" in text
    assert "- 用户在南校区" in text
    assert text.index("[用户事实]") < text.index("[用户画像]")  # 事实先于画像


# ── LayeredStore：L0 原文 ────────────────────────────────────────────────────

def test_layered_raw_turns_and_trace(tmp_path):
    store = LayeredStore(str(tmp_path / "m.db"))

    async def scenario():
        assert await store.append_raw("u1", "c1", "user", "第一句") > 0
        assert await store.append_raw("u1", "c1", "assistant", "回答一") > 0
        assert await store.append_raw("u1", "c1", "user", "第二句") > 0
        # turn_id 会话内自增
        assert await store.get_last_turn("u1", "c1") == 3
        assert await store.get_last_turn("u1", "c2") == 0  # 新会话从 0 开始
        # 用户隔离：其他用户看不到
        assert await store.count_raw("u2") == 0
        # 溯源：按 turn 取原文
        by_turn = await store.get_raw_by_turns("u1", "c1", [1, 3])
        assert by_turn == {1: "第一句", 3: "第二句"}
        rows = await store.get_raw_range("u1", "c1", start_turn=2)
        assert [r["content"] for r in rows] == ["回答一", "第二句"]

    asyncio.run(scenario())


# ── LayeredStore：L1 原子事实（去重 + 失效治理）──────────────────────────────

def test_layered_facts_dedup_and_deactivate(tmp_path):
    store = LayeredStore(str(tmp_path / "m.db"))

    async def scenario():
        facts = [
            {"fact": "用户在准备考研", "category": "status", "source_conv": "c1", "source_turn": 3},
            {"fact": "用户在南校区", "category": "entity", "source_conv": "c1", "source_turn": 5},
        ]
        assert await store.add_facts("u1", facts) == 2
        # 重复提炼按文本去重（零新增）
        assert await store.add_facts("u1", [dict(facts[0])]) == 0
        # 空事实/坏输入不落库
        assert await store.add_facts("u1", [{"fact": "  "}, {"category": "x"}]) == 0
        listed = await store.list_facts("u1")
        assert len(listed) == 2
        # 失效标记：不物理删除，但不再参与读取
        fid = listed[0]["id"]
        assert await store.deactivate_fact("u1", fid) is True
        assert await store.deactivate_fact("u1", 99999) is False
        assert len(await store.list_facts("u1")) == 1
        assert await store.count_facts("u1") == 1

    asyncio.run(scenario())


# ── LayeredStore：L3 画像版本历史（可回滚）──────────────────────────────────

def test_layered_profile_versions(tmp_path):
    store = LayeredStore(str(tmp_path / "m.db"))

    async def scenario():
        for i in range(3):
            await store.save_profile_version(
                "u1", json.dumps({"preferences": [f"偏好{i}"]}, ensure_ascii=False), reason="signal"
            )
        versions = await store.list_profile_versions("u1")
        assert len(versions) == 3
        # 倒序：最新在前；回滚读取到最老版本
        assert "偏好2" in versions[0]["profile_json"]
        oldest = await store.get_profile_version("u1", versions[-1]["id"])
        assert oldest is not None and "偏好0" in oldest["profile_json"]
        # 用户隔离
        assert await store.count_profile_versions("u2") == 0

    asyncio.run(scenario())


# ── LayeredStore：refs 卸载落盘（100% 找回）─────────────────────────────────

def test_layered_refs_roundtrip(tmp_path):
    store = LayeredStore(str(tmp_path / "m.db"))

    async def scenario():
        rid = await store.save_ref("u1", "c1", "knowledge_search", "长文档" * 500)
        ref = await store.get_ref("u1", rid)
        assert ref is not None and ref["content"] == "长文档" * 500
        assert ref["char_len"] == len("长文档" * 500)
        # 用户隔离：其他用户拿不到
        assert await store.get_ref("u2", rid) is None

    asyncio.run(scenario())


# ── LayeredStore：治理清理（prune）──────────────────────────────────────────

def test_layered_prune(tmp_path):
    store = LayeredStore(str(tmp_path / "m.db"))

    async def scenario():
        await store.append_raw("u1", "c1", "user", "旧对话")
        await store.save_ref("u1", "c1", "tool", "旧结果")
        await store.add_facts("u1", [{"fact": "旧事实", "category": "status"}])
        fid = (await store.list_facts("u1"))[0]["id"]
        await store.deactivate_fact("u1", fid)  # 失效后才能被清理
        for i in range(3):
            await store.save_profile_version("u1", "{}", "r")

        stats = await store.prune(
            "u1", raw_ttl_days=0, ref_ttl_days=0, fact_ttl_days=0, max_profile_versions=1
        )
        assert stats["raw"] == 1
        assert stats["refs"] == 1
        assert stats["facts"] == 1
        assert stats["profiles"] == 2  # 3 版 → 保留 1 版，清 2 版
        assert await store.count_raw("u1") == 0
        assert await store.count_profile_versions("u1") == 1

    asyncio.run(scenario())


# ── 伪客户端（照 test_orchestrator._FakeClient 模式）────────────────────────

class _FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResp:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, fake):
        self._fake = fake

    async def create(self, **kwargs):
        return self._fake._create(kwargs)


class _FakeClient:
    """顺序返回预设响应的伪客户端（记录每次调用参数）。"""

    def __init__(self, *texts):
        self.messages = _FakeMessages(self)
        self._texts = list(texts)
        self.seen = []

    def _create(self, kwargs):
        self.seen.append(kwargs)
        return _FakeResp(self._texts.pop(0))


class _FakeCollection:
    """最小 Chroma 替身：只记录 upsert（画像写入）。"""

    def __init__(self):
        self.upserts = []

    def upsert(self, ids, documents, metadatas):
        self.upserts.append((ids, documents, metadatas))


class _FakeEmbedding:
    """伪 embedding function（避免加载真实 ONNX 模型）。"""

    def __call__(self, input):
        return [[0.1] * 8 for _ in input]


def _make_manager(tmp_path, monkeypatch):
    """构造 MemoryManager：SQLite 用 tmp，Chroma 用本地 PersistentClient + 伪 embedding。"""
    import memory.conversation_memory as cm

    monkeypatch.setattr(cm, "get_embedder", lambda: _FakeEmbedding())
    mgr = MemoryManager(
        redis_url="redis://localhost:6399/0",   # 不会真正连接
        chroma_host="127.0.0.1",                # 连不上 → 本地嵌入式
        chroma_port=1,
        chroma_path=str(tmp_path / "chroma"),
        api_key="sk-test-not-used",
        model="test-model",
        layered_store=LayeredStore(str(tmp_path / "memory.db")),
    )
    return mgr


# ── update_profile：一次 LLM 调用双产出（画像 + 原子事实）───────────────────

def test_update_profile_dual_output(tmp_path, monkeypatch):
    mgr = _make_manager(tmp_path, monkeypatch)
    fake_profile = _FakeCollection()
    mgr._profile = fake_profile
    mgr._client = _FakeClient(json.dumps({
        "preferences": ["喜欢晚上学习"],
        "entities": {"院系专业": ["通信工程"], "年级": [], "校区": [], "诉求类型": []},
        "facts": [
            {"fact": "用户在准备考研", "category": "status"},
            {"fact": "用户是通信工程学院大二学生", "category": "entity"},
        ],
    }, ensure_ascii=False))

    async def fake_wm(user_id, conv_id):
        return [
            Message(role=MsgRole.USER, content="我最近在准备考研"),          # 画像信号
            Message(role=MsgRole.ASSISTANT, content="已记录"),              # 助手消息
            Message(role=MsgRole.USER, content="我是通信工程学院大二的"),     # 画像信号
        ]

    async def fake_get_profile(user_id):
        return {}

    monkeypatch.setattr(mgr, "_get_working_memory", fake_wm)
    monkeypatch.setattr(mgr, "_get_profile", fake_get_profile)

    async def scenario():
        await mgr.update_profile("u1", "c1")
        # 仅 1 次 LLM 调用（画像 + 事实双产出，零额外成本）
        assert mgr.llm_call_count == 1
        # L1 事实落库，带证据链（source_turn = 当前 L0 最大轮次）
        facts = await mgr._layered.list_facts("u1")
        assert {f["fact"] for f in facts} == {"用户在准备考研", "用户是通信工程学院大二学生"}
        assert all(f["source_conv"] == "c1" for f in facts)
        assert all(f["source_turn"] >= 0 for f in facts)
        # L3 画像 upsert + 版本历史
        assert len(fake_profile.upserts) == 1
        assert await mgr._layered.count_profile_versions("u1") == 1
        # 重复提炼去重：facts 不重复
        await mgr.update_profile("u1", "c1")
        assert await mgr._layered.count_facts("u1") == 2

    asyncio.run(scenario())


def test_update_profile_skips_without_signal(tmp_path, monkeypatch):
    """无画像信号时不调用 LLM（成本控制核心断言）。"""
    mgr = _make_manager(tmp_path, monkeypatch)
    mgr._profile = _FakeCollection()
    mgr._client = _FakeClient("{}")  # 若被调用会 pop 空列表报错，恰好作为哨兵

    async def fake_wm(user_id, conv_id):
        return [Message(role=MsgRole.USER, content="图书馆几点关门？")]

    async def fake_get_profile(user_id):
        return {}

    monkeypatch.setattr(mgr, "_get_working_memory", fake_wm)
    monkeypatch.setattr(mgr, "_get_profile", fake_get_profile)

    async def scenario():
        await mgr.update_profile("u1", "c1")
        assert mgr.llm_call_count == 0
        assert await mgr._layered.count_facts("u1") == 0
        assert await mgr._layered.count_profile_versions("u1") == 0

    asyncio.run(scenario())


# ── get_context：四层融合 + 场景优先 + memory_trace ─────────────────────────

def test_get_context_layers_and_trace(tmp_path, monkeypatch):
    mgr = _make_manager(tmp_path, monkeypatch)

    async def fake_wm(user_id, conv_id):
        return [Message(role=MsgRole.USER, content="今天有空吗")]

    async def fake_search(user_id, query):
        return (["场景：用户咨询选课与校园卡办理", "普通历史片段"],
                {"scenario": 1, "segment": 1})

    async def fake_facts(user_id):
        return [{"fact": "用户在准备考研"}, {"fact": "用户在南校区"}]

    async def fake_profile(user_id):
        return {"preferences": ["喜欢晚上学习"]}

    async def fake_redis_get(name):
        return "会话摘要：讨论选课"

    monkeypatch.setattr(mgr, "_get_working_memory", fake_wm)
    monkeypatch.setattr(mgr, "_search_episodic", fake_search)
    monkeypatch.setattr(mgr, "_list_facts", fake_facts)
    monkeypatch.setattr(mgr, "_get_profile", fake_profile)
    monkeypatch.setattr(mgr._redis, "get", fake_redis_get)

    async def scenario():
        ctx = await mgr.get_context("u1", "c1", query="选课")
        # L2 场景块排在普通片段之前（场景优先注入）
        assert ctx.relevant_history[0].startswith("场景")
        assert "普通历史片段" in ctx.relevant_history
        # L1 事实注入（上限 8 条）
        assert ctx.facts == ["用户在准备考研", "用户在南校区"]
        # L0/L3 计数来自分层存储
        trace = ctx.memory_trace["layers"]
        assert trace["scenario"] == 1 and trace["segments"] == 1
        assert trace["facts"] == 2
        assert trace["raw"] == 0 and trace["profile_versions"] == 0
        # 摘要来自工作记忆
        assert ctx.summary == "会话摘要：讨论选课"

    asyncio.run(scenario())
