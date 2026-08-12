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
    _fact_relevant_to_query, _fact_subsumed_by_profile,
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


# ── 纯逻辑：L1/L3 分工去重 + L1 按需召回 ────────────────────────────────────

def test_fact_subsumed_by_profile():
    """画像已覆盖的事实（条目是事实子串 / 事实全文在画像中）不落 L1。"""
    profile = {"preferences": ["准备考研", "喜欢晚上学习"],
               "entities": {"院系专业": ["通信工程"], "年级": ["大二"]}}
    # 偏好条目"准备考研"是事实的规范化子串 → 已覆盖（同一信息不双写）
    assert _fact_subsumed_by_profile("用户在准备考研", profile)
    # 实体条目"通信工程"是事实子串 → 已覆盖（身份归 L3 聚合画像）
    assert _fact_subsumed_by_profile("用户是通信工程学院大二学生", profile)
    # 事实全文已在画像文本中（逐字重复）
    assert _fact_subsumed_by_profile("准备考研", profile)
    # 画像未覆盖的细粒度事实（决定/状态/细节）→ 保留
    assert not _fact_subsumed_by_profile("用户决定周三下午去校医院补办校园卡", profile)
    # 空画像 → 不覆盖；空事实 → 视为覆盖（不落库）
    assert not _fact_subsumed_by_profile("用户在准备考研", {})
    assert _fact_subsumed_by_profile("  ", profile)
    # 标点差异不影响判定
    assert _fact_subsumed_by_profile("用户，在准备考研！", profile)


def test_fact_relevant_to_query():
    """L1 按需召回：与查询共享 ≥1 个非停用 bigram 才注入。"""
    fact = "用户决定周三下午去校医院补办校园卡"
    # 共享"补办/校园/园卡"等 bigram → 相关
    assert _fact_relevant_to_query("补办校园卡需要什么材料", fact)
    # 无关查询 → 不注入
    assert not _fact_relevant_to_query("食堂几点关门", fact)
    # 只有停用 bigram（"用户"）不算相关；空查询无 bigram → 不相关
    assert not _fact_relevant_to_query("用户", "用户在南校区")
    assert not _fact_relevant_to_query("", fact)
    # 时间词"今天"是停用 bigram，但共享的非停用 bigram 仍判定相关
    assert _fact_relevant_to_query("今天去补办校园卡", fact)


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
    llm_output = json.dumps({
        "preferences": ["喜欢晚上学习"],
        "entities": {"院系专业": ["通信工程"], "年级": [], "校区": [], "诉求类型": []},
        "facts": [
            {"fact": "用户在准备考研", "category": "status"},
            {"fact": "用户是通信工程学院大二学生", "category": "entity"},
        ],
    }, ensure_ascii=False)
    # 两条相同响应：第二次调用用于验证"重复提炼去重"（原测试只给一条响应，
    # 第二次调用因 LLM 响应耗尽而静默失败，去重实际未被真正断言）
    mgr._client = _FakeClient(llm_output, llm_output)

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
        # L1/L3 分工去重：画像实体"通信工程"覆盖的身份事实不落 L1，
        # 未被画像覆盖的"用户在准备考研"保留（带证据链：source_turn 锚点）
        facts = await mgr._layered.list_facts("u1")
        assert {f["fact"] for f in facts} == {"用户在准备考研"}
        assert all(f["source_conv"] == "c1" for f in facts)
        assert all(f["source_turn"] >= 0 for f in facts)
        # L3 画像 upsert + 版本历史
        assert len(fake_profile.upserts) == 1
        assert await mgr._layered.count_profile_versions("u1") == 1
        # 重复提炼去重：同一事实再次提炼 → 零新增（按文本去重）
        await mgr.update_profile("u1", "c1")
        assert mgr.llm_call_count == 2
        assert await mgr._layered.count_facts("u1") == 1

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
        # L1 按需召回：与"考研"共享 bigram 的事实注入，无关的"用户在南校区"不注入
        ctx = await mgr.get_context("u1", "c1", query="考研")
        assert ctx.facts == ["用户在准备考研"]
        # L2 场景块排在普通片段之前（场景优先注入）
        assert ctx.relevant_history[0].startswith("场景")
        assert "普通历史片段" in ctx.relevant_history
        # L0/L1/L3 计数来自分层存储：facts = 注入条数，facts_total = 可用总数
        trace = ctx.memory_trace["layers"]
        assert trace["scenario"] == 1 and trace["segments"] == 1
        assert trace["facts"] == 1 and trace["facts_total"] == 2
        assert trace["raw"] == 0 and trace["profile_versions"] == 0
        # 摘要来自工作记忆
        assert ctx.summary == "会话摘要：讨论选课"

        # query 与 L1 事实无关联 → 不注入（L2 历史 + L3 画像仍携带上下文）
        ctx2 = await mgr.get_context("u1", "c1", query="食堂几点关门")
        assert ctx2.facts == []
        assert ctx2.relevant_history[0].startswith("场景")

    asyncio.run(scenario())


def test_get_context_facts_not_duplicate_profile(tmp_path, monkeypatch):
    """L1 与 L3 不重复注入：画像已覆盖的事实即使与 query 相关也不注入（兼容存量数据）。"""
    mgr = _make_manager(tmp_path, monkeypatch)

    async def fake_wm(user_id, conv_id):
        return [Message(role=MsgRole.USER, content="我喜欢晚上学习")]

    async def fake_search(user_id, query):
        return (["场景：咨询学习安排"], {"scenario": 1, "segment": 0})

    async def fake_facts(user_id):
        return [
            {"fact": "用户喜欢晚上学习"},                    # 与画像偏好逐字重复（存量数据）
            {"fact": "用户决定周三下午去校医院补办校园卡"},    # 画像未覆盖
        ]

    async def fake_profile(user_id):
        return {"preferences": ["喜欢晚上学习"]}

    async def fake_redis_get(name):
        return ""

    monkeypatch.setattr(mgr, "_get_working_memory", fake_wm)
    monkeypatch.setattr(mgr, "_search_episodic", fake_search)
    monkeypatch.setattr(mgr, "_list_facts", fake_facts)
    monkeypatch.setattr(mgr, "_get_profile", fake_profile)
    monkeypatch.setattr(mgr._redis, "get", fake_redis_get)

    async def scenario():
        # "用户喜欢晚上学习"与画像偏好"喜欢晚上学习"重复 → 即使 query 相关也不注入
        ctx = await mgr.get_context("u1", "c1", query="晚上学习")
        assert ctx.facts == []
        # 画像未覆盖且与 query 相关的事实正常注入
        ctx2 = await mgr.get_context("u1", "c1", query="补办校园卡")
        assert ctx2.facts == ["用户决定周三下午去校医院补办校园卡"]

    asyncio.run(scenario())
