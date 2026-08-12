"""知识库切分与导入测试：递归分隔符切分、页码映射、元数据透传。

不依赖真实 ChromaDB：
  - 切分纯函数直接单测（KnowledgeBase.__new__ 构造，不触发 __init__ 连接）；
  - add_documents 用最小 FakeCollection（记录 upsert 的 metadatas）验证
    format / page_start / page_end 元数据与 doc_id 幂等性。
"""
from __future__ import annotations

from mcp.knowledge_base import KnowledgeBase


class _FakeCollection:
    """最小 ChromaDB collection 替身：只记录 upsert 内容。"""

    def __init__(self) -> None:
        self.data: dict[str, tuple[str, dict]] = {}

    def upsert(self, ids, documents, metadatas) -> None:
        for i, d, m in zip(ids, documents, metadatas):
            self.data[i] = (d, m)

    def count(self) -> int:
        return len(self.data)


def _kb() -> KnowledgeBase:
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = _FakeCollection()
    return kb


# ── 递归分隔符切分 ─────────────────────────────────────────────────────────

def test_short_text_single_chunk():
    assert _kb()._chunk_text("你好，西电。") == ["你好，西电。"]


def test_empty_text_no_chunks():
    assert _kb()._chunk_text("   \n  ") == []


def test_long_sentence_hard_split_by_char():
    """回归：无任何可拆分隔符的超长句子 → 字符级硬切，不撑爆单块。"""
    chunks = _kb()._chunk_text("长" * 1000, chunk_size=100, overlap=20)
    assert chunks
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == "长" * 1000  # 内容无损


def test_long_sentence_split_by_comma():
    """含逗号的长句 → 逗号级切分，块不超过上限。"""
    text = "，".join(["甲" * 40] * 30)
    chunks = _kb()._chunk_text(text, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    assert all(c in text for c in chunks)  # 不丢字、不引入新字


def test_chunks_within_size_and_content_preserved():
    text = ("第一句是开头。" + "这是一个比较长的句子，中间有逗号，还有更多内容。" * 30 + "结尾句。")
    chunks = _kb()._chunk_text(text, chunk_size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    # overlap 会使相邻块尾部重复，但整体覆盖原文本且不丢字
    assert all(c in text for c in chunks)
    assert chunks[0] == text[:len(chunks[0])]     # 从开头开始
    # 覆盖到结尾（末尾分隔符会被丢弃，与 LangChain splitter 行为一致，正文完整）
    assert text[:-1].endswith(chunks[-1])
    for i in range(len(chunks) - 1):
        assert chunks[i + 1][:20] == chunks[i][-20:]  # 相邻块首尾相接，覆盖连续


def test_overlap_carried_exactly():
    """每块开头精确携带上一块尾部 overlap 字，跨块语义连续。"""
    text = "句一。" + "内容，非常丰富。" * 40 + "句尾。"
    chunks = _kb()._chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    for i in range(len(chunks) - 1):
        assert chunks[i + 1][:20] == chunks[i][-20:]


def test_paragraph_boundary_preferred():
    """段落优先成块：两段各 400 字、chunk_size 500 → 第二块从第一块尾部 overlap 起。"""
    para = "甲" * 400 + "\n\n" + "乙" * 400
    chunks = _kb()._chunk_text(para, chunk_size=500, overlap=60)
    assert len(chunks) == 2
    assert chunks[0] == "甲" * 400                       # 第一段独立成块
    assert chunks[1] == "甲" * 60 + "\n\n" + "乙" * 400  # overlap 跨段携带


# ── PDF 页码映射 ───────────────────────────────────────────────────────────

def test_pages_small_merged_chunk_spans_pages():
    """页内容小于 chunk_size → 两页合成一块，page_start=1 / page_end=2。"""
    full = "一" * 300 + "\n\n" + "二" * 200
    offsets = [(0, 300, 1), (302, 502, 2)]
    chunks = _kb()._chunk_with_pages(full, offsets, chunk_size=400, overlap=60)
    assert [(c[1], c[2]) for c in chunks] == [(1, 1), (1, 2)]
    assert chunks[1][0] == "一" * 60 + "\n\n" + "二" * 200


def test_pages_large_chunks_keep_page_and_overlap():
    """页内容大于 chunk_size → 各自切块；overlap 跨页携带时页码区间覆盖两页。"""
    full = "甲" * 250 + "\n\n" + "乙" * 250
    offsets = [(0, 250, 1), (252, 502, 2)]
    chunks = _kb()._chunk_with_pages(full, offsets, chunk_size=300, overlap=60)
    assert [(len(c[0]), c[1], c[2]) for c in chunks] == [(250, 1, 1), (312, 1, 2)]


# ── add_documents 元数据与幂等 ─────────────────────────────────────────────

def test_add_documents_pdf_metadata_passthrough():
    kb = _kb()
    full = "一" * 300 + "\n\n" + "二" * 200
    n = kb.add_documents([{
        "title": "测试手册", "content": full, "format": "pdf",
        "page_offsets": [(0, 300, 1), (302, 502, 2)],
    }])
    assert n == 2
    metas = [m for _, m in kb._collection.data.values()]
    assert [m["format"] for m in metas] == ["pdf", "pdf"]
    assert (metas[0]["page_start"], metas[0]["page_end"]) == ("1", "1")
    assert (metas[1]["page_start"], metas[1]["page_end"]) == ("1", "2")
    assert metas[0]["total_chunks"] == 2


def test_add_documents_plain_text_default_format():
    kb = _kb()
    kb.add_documents([{"title": "普通文档", "content": "短内容。"}])
    meta = next(m for _, m in kb._collection.data.values())
    assert meta["format"] == "text"
    assert "page_start" not in meta


def test_add_documents_idempotent_same_doc():
    """doc_id 由 title+chunk 内容 md5 生成 → 重复导入 upsert 覆盖，不产生重复切片。"""
    kb = _kb()
    doc = {"title": "校历", "content": "第一句。" + "内容。" * 200}
    first = kb.add_documents([doc])
    second = kb.add_documents([doc])
    assert first == second
    assert kb._collection.count() == first  # 第二次导入未增长


def test_add_documents_empty_content_skipped():
    kb = _kb()
    n = kb.add_documents([{"title": "空", "content": "   "}])
    assert n == 0


# ── 跨模型迁移（MiniLM v2 / L2 legacy → bge v3）────────────────────────────

class _LegacyCollection:
    """旧 collection 替身：get() 原样返回原始文本（不触发 embedding）。"""

    def __init__(self, name, ids, docs, metas):
        self.name = name
        self.ids, self.docs, self.metas = ids, docs, metas

    def get(self, include=None):
        return {"ids": self.ids, "documents": self.docs, "metadatas": self.metas}


class _FakeChroma:
    def __init__(self, collections):
        self._by_name = {c.name: c for c in collections}

    def get_collection(self, name):
        if name not in self._by_name:
            raise Exception(f"collection 不存在: {name}")
        return self._by_name[name]


def _kb_for_migration(collection_name, client, embedding_function=object()):
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = _FakeCollection()
    kb._collection.name = collection_name
    kb._client = client
    kb._embedding_function = embedding_function
    return kb


def test_migrate_v2_to_v3():
    """bge 空间（v3）为空时：从 MiniLM 空间（v2）读原始文本重嵌入。"""
    legacy = _LegacyCollection(
        "knowledge_base_v2",
        ["id1", "id2"],
        ["选课流程第一段", "食堂开放时间"],
        [{"title": "选课", "domain": "academic"}, {"title": "食堂", "domain": "campus_life"}],
    )
    kb = _kb_for_migration("knowledge_base_v3", _FakeChroma([legacy]))
    kb._migrate_previous_collections()
    assert kb._collection.count() == 2
    assert ("选课流程第一段", {"title": "选课", "domain": "academic"}) in kb._collection.data.values()


def test_migrate_skips_when_current_not_empty():
    """当前 collection 已有数据时不做迁移（幂等）。"""
    legacy = _LegacyCollection(
        "knowledge_base_v2", ["id1"], ["旧文本"], [{"title": "t"}],
    )
    kb = _kb_for_migration("knowledge_base_v3", _FakeChroma([legacy]))
    kb._collection.upsert(["new"], ["新文本"], [{"title": "新"}])
    kb._migrate_previous_collections()
    # 只有新数据，没有旧数据（未迁移）
    assert list(kb._collection.data.keys()) == ["new"]


def test_migrate_legacy_l2_directly_to_v3_when_v2_empty():
    """v2 为空时回看最早期 L2 空间（knowledge_base），直接迁入 v3。"""
    legacy = _LegacyCollection(
        "knowledge_base", ["old"], ["最早期的文本"], [{"title": "old"}],
    )
    kb = _kb_for_migration("knowledge_base_v3", _FakeChroma([legacy]))
    kb._migrate_previous_collections()
    assert kb._collection.count() == 1
    assert next(iter(kb._collection.data.values())) == ("最早期的文本", {"title": "old"})


def test_fallback_mode_migrates_only_legacy_to_v2():
    """模型不可用（回退 v2 空间）时：只迁移最早期 L2 空间，不跨到 bge 语义。"""
    v2 = _LegacyCollection(
        "knowledge_base_v2", ["v2id"], ["v2 文本"], [{"title": "v2"}],
    )
    legacy = _LegacyCollection(
        "knowledge_base", ["old"], ["旧文本"], [{"title": "old"}],
    )
    kb = _kb_for_migration("knowledge_base_v2", _FakeChroma([v2, legacy]),
                           embedding_function=None)
    kb._migrate_previous_collections()
    assert kb._collection.count() == 1
    assert next(iter(kb._collection.data.values())) == ("旧文本", {"title": "old"})


def test_migrate_no_source_collections_is_silent():
    """任何旧 collection 都不存在时静默跳过（全新部署）。"""
    kb = _kb_for_migration("knowledge_base_v3", _FakeChroma([]))
    kb._migrate_previous_collections()
    assert kb._collection.count() == 0


# ── Markdown 结构感知分块 ─────────────────────────────────────────────────

def test_plain_text_falls_back_to_recursive():
    """无标题/表格/代码块的纯文本：行为与递归分隔符切分一致（回归）。"""
    kb = _kb()
    text = "甲" * 400 + "\n\n" + "乙" * 400
    assert kb._chunk_text(text, chunk_size=500, overlap=60) == kb._chunk_text(text, chunk_size=500, overlap=60)
    # 与 _split_recursive 结果一致（纯文本路径）
    expected = [c for c, _, _ in kb._split_recursive(text, kb._CHUNK_SEPARATORS, 500, 60)]
    assert kb._chunk_text(text, chunk_size=500, overlap=60) == expected


def test_heading_chain_injected_and_headings_split_blocks():
    """标题链注入块首；每个标题是硬边界，开新块。"""
    text = (
        "# 校园指南\n\n"
        "## 校车时刻\n\n"
        "工作日校车每 15 分钟一班。\n\n"
        "## 食堂介绍\n\n"
        "南校区食堂共三层。"
    )
    chunks = _kb()._chunk_text(text, chunk_size=500, overlap=60)
    assert len(chunks) == 2
    # 每块块首携带「文档标题 > 小节标题」链
    assert chunks[0].startswith("校园指南 > 校车时刻\n")
    assert chunks[1].startswith("校园指南 > 食堂介绍\n")
    # 标题链不进内容区，内容完整保留
    assert "工作日校车每 15 分钟一班。" in chunks[0]
    assert "南校区食堂共三层。" in chunks[1]


def test_heading_level_updates_chain():
    """三级标题出现后，链为 文档 > 部分 > 小节；同级/上级标题替换链。"""
    text = (
        "# 面试面经\n\n"
        "## 第一部分\n\n"
        "### Q1. 项目目标\n\n"
        "内容一。\n\n"
        "## 第二部分\n\n"
        "内容二。"
    )
    chunks = _kb()._chunk_text(text, chunk_size=500, overlap=60)
    assert chunks[0].startswith("面试面经 > 第一部分 > Q1. 项目目标\n")
    assert chunks[1].startswith("面试面经 > 第二部分\n")


def test_table_kept_atomic():
    """表格整体成块，行不被打散；表格与其标题合并到同一块。"""
    text = (
        "# 校历\n\n"
        "| 周次 | 事项 |\n"
        "| --- | --- |\n"
        "| 第1周 | 报到 |\n"
        "| 第2周 | 选课 |\n"
        "\n"
        "补充说明：以上时间以教务系统为准。"
    )
    chunks = _kb()._chunk_text(text, chunk_size=500, overlap=60)
    # 表格是原子单元独立成块；后续段落单独成块（均带链头）
    assert len(chunks) == 2
    table_block = chunks[0]
    assert table_block.startswith("校历\n")
    assert "| 周次 | 事项 |" in table_block
    assert "| 第1周 | 报到 |" in table_block
    assert "| 第2周 | 选课 |" in table_block
    # 表格行之间不被其他内容插入
    assert table_block == "校历\n| 周次 | 事项 |\n| --- | --- |\n| 第1周 | 报到 |\n| 第2周 | 选课 |"
    assert chunks[1] == "校历\n补充说明：以上时间以教务系统为准。"


def test_oversized_table_own_block_even_over_budget():
    """超长表格是原子单元：宁可独占一块超预算，也不拆散结构。"""
    rows = "\n".join(f"| 第{i}周 | 事项{i} |" for i in range(1, 30))
    text = f"# 校历\n\n| 周次 | 事项 |\n| --- | --- |\n{rows}"
    chunks = _kb()._chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) == 1
    # header + 分隔行 + 29 数据行全部完整保留（表格是原子单元）
    table_lines = [l for l in chunks[0].split("\n") if l.startswith("|")]
    assert len(table_lines) == 31
    assert "| 第29周 | 事项29 |" in chunks[0]


def test_code_block_kept_atomic():
    """代码块整体成块，围栏内容不被打散。"""
    text = (
        "# API 示例\n\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "调用方式见上文。"
    )
    chunks = _kb()._chunk_text(text, chunk_size=500, overlap=60)
    assert len(chunks) == 2
    code_block = chunks[0]
    assert code_block.startswith("API 示例\n")
    assert "```python" in code_block
    assert "def add(a, b):" in code_block
    assert code_block.endswith("```")
    assert chunks[1] == "API 示例\n调用方式见上文。"


def test_chunk_size_budget_includes_heading_chain():
    """标题链计入 chunk_size 预算：块总长（链头+正文）不超过 chunk_size。"""
    text = (
        "# 校园指南\n\n"
        "## 校车时刻\n\n"
        + "工作日班次说明。" * 100  # 超长段落 → 递归拆分
    )
    chunks = _kb()._chunk_text(text, chunk_size=150, overlap=20)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c) <= 150, f"块超预算: {len(c)} > 150"


def test_table_without_heading_still_protected():
    """无标题文档中的表格同样保护（链头为空时退化但表格仍原子）。"""
    text = (
        "| 课程 | 时间 |\n"
        "| --- | --- |\n"
        "| 高数 | 周一 |\n"
        "\n"
        "课表以教务系统为准。"
    )
    chunks = _kb()._chunk_text(text, chunk_size=500, overlap=60)
    assert len(chunks) == 2
    assert chunks[0] == "| 课程 | 时间 |\n| --- | --- |\n| 高数 | 周一 |"
    assert chunks[1] == "课表以教务系统为准。"
