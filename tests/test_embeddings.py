"""本地向量模型模块测试（mcp/embeddings.py）：ONNX 推理 + 池化数学 + 降级链。

不依赖真实模型下载/onnxruntime 加载：
  - 用最小 FakeTokenizer/FakeSession 替身验证 mean pooling + L2 归一化数学、
    指令前缀行为、reranker sigmoid 排序；
  - 下载/加载失败路径直接 mock，验证冷却降级与预下载报错。

覆盖 P0：
  1. mean pooling（attention_mask 加权）+ L2 归一化正确性；
  2. bge-zh 指令前缀只加在 query 侧（embed_query），embed_documents 不加；
  3. chromadb 协议 __call__ 的前缀模式（none 默认 / both 可切换）；
  4. reranker sigmoid 分数 ∈ [0,1] 且重排保持 top_k 语义；
  5. 模型文件不可用 → 抛错（_ensure_model），单例冷却期内不重试；
  6. 空输入安全返回。
"""
from __future__ import annotations

import threading
from unittest.mock import patch

import numpy as np

from mcp import embeddings
from mcp.embeddings import (
    LocalEmbedder,
    LocalReranker,
    QUERY_INSTRUCTION,
    _ensure_model,
    _get_singleton,
    get_embedder,
    get_reranker,
    reset_singletons,
)


# ── 最小替身（结构对齐 tokenizers / onnxruntime）─────────────────────────────

class _FakeEnc:
    def __init__(self, ids, mask, type_ids):
        self.ids = ids
        self.attention_mask = mask
        self.type_ids = type_ids


class _FakeTokenizer:
    """把文本编码为伪 ids（[CLS]+字符码+[SEP]），支持句子对与 batch padding。"""

    def encode_batch(self, texts):
        rows = []
        for t in texts:
            seq = t if isinstance(t, str) else t[0] + "[SEP]" + t[1]
            rows.append([101] + [ord(c) % 1000 for c in seq] + [102])
        maxlen = max(len(r) for r in rows)
        return [
            _FakeEnc(
                r + [0] * (maxlen - len(r)),
                [1] * len(r) + [0] * (maxlen - len(r)),
                [0] * maxlen,
            )
            for r in rows
        ]

    def enable_truncation(self, **kwargs):
        pass

    def enable_padding(self, **kwargs):
        pass


class _FakeSession:
    """输出确定性 last_hidden_state / logits 的假 onnxruntime session。

    隐藏状态 = input_ids 的确定性函数（相同 token 序列 → 相同向量），
    使「同文本在不同 batch」的池化结果可比较（验证 padding 不稀释）。
    """

    def __init__(self, hidden_dim=4, logits=False):
        self.hidden_dim = hidden_dim
        self.logits = logits  # True=输出 [batch,1] logits（reranker 形状）
        self.inputs = [type("_In", (), {"name": n})() for n in
                       ("input_ids", "attention_mask", "token_type_ids")]

    def get_inputs(self):
        return self.inputs

    def run(self, _, feeds):
        ids = feeds["input_ids"]  # [batch, seq]
        batch, seq = ids.shape
        if self.logits:
            # 确定性 logits：随输入 id 求和变化，sigmoid 后分数可排序
            logits = (ids.sum(axis=1, keepdims=True) % 7 - 3).astype(np.float32)
            return [logits]
        hidden = np.zeros((batch, seq, self.hidden_dim), dtype=np.float32)
        for k in range(self.hidden_dim):
            hidden[:, :, k] = (ids % (k + 2)).astype(np.float32)
        return [hidden]


def _attach_fakes(model, hidden_dim=4, logits=False):
    """注入假 session/tokenizer，跳过真实下载与加载。"""
    model._model._session = _FakeSession(hidden_dim, logits=logits)
    model._model._tokenizer = _FakeTokenizer()
    return model


# ── 1. pooling 数学 ──────────────────────────────────────────────────────────

def test_mean_pooling_and_l2_normalization():
    """mean pooling（attention_mask 加权）后向量 L2 范数为 1（cosine 空间）。"""
    emb = _attach_fakes(LocalEmbedder())
    vecs = emb.embed_texts(["选课什么时候开始", "食堂几点关门"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 4  # hidden_dim
    for v in vecs:
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_pooling_ignores_padding_tokens():
    """attention_mask 为 0 的位置不参与 mean pooling（长文本 pad 不稀释向量）。"""
    emb = _attach_fakes(LocalEmbedder())
    # 同文本不同长度（短文本会被 pad 到 batch 最大长度）：语义不变
    v1 = emb.embed_texts(["a" * 3])
    v2 = emb.embed_texts(["a" * 3, "b" * 30])[0]
    # 相同内容向量应完全一致（池化不受 padding 影响）
    assert np.allclose(v1[0], v2, atol=1e-6)


# ── 2. bge-zh 指令前缀 ──────────────────────────────────────────────────────

def test_instruction_prefix_only_on_query_side():
    """embed_query 加指令前缀；embed_documents 不加（BAAI 要求 query-only）。"""
    emb = _attach_fakes(LocalEmbedder())
    seen: list[list[str]] = []
    orig = emb._model.tokenize

    def spy(texts):
        seen.append(list(texts))
        return orig(texts)

    emb._model.tokenize = spy
    emb.embed_query(["选课"])
    assert seen[-1][0] == QUERY_INSTRUCTION + "选课"
    emb.embed_documents(["选课"])
    assert seen[-1][0] == "选课"


def test_call_prefix_mode_none_default():
    """chromadb 协议入口：默认不加前缀（0.5.x 无法区分 query/document）。"""
    emb = _attach_fakes(LocalEmbedder())
    seen: list[list[str]] = []
    orig = emb._model.tokenize

    def spy(texts):
        seen.append(list(texts))
        return orig(texts)

    emb._model.tokenize = spy
    emb(["你好"])
    assert seen[-1][0] == "你好"


def test_call_prefix_mode_both():
    """ECHOGUIDE_EMBED_PREFIX_MODE=both 时两侧都加指令前缀。"""
    emb = _attach_fakes(LocalEmbedder())
    emb._prefix_mode = "both"
    seen: list[list[str]] = []
    orig = emb._model.tokenize

    def spy(texts):
        seen.append(list(texts))
        return orig(texts)

    emb._model.tokenize = spy
    emb(["你好"])
    assert seen[-1][0] == QUERY_INSTRUCTION + "你好"


# ── 3. reranker ──────────────────────────────────────────────────────────────

def test_reranker_sigmoid_scores_in_unit_range():
    """logit → sigmoid 分数 ∈ [0,1]（可解释为相关概率）。"""
    rr = _attach_fakes(LocalReranker(), logits=True)
    scores = rr.score("选课流程", ["选课指南", "食堂指南"])
    assert len(scores) == 2
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_reranker_rerank_topk_preserves_items():
    """rerank 只排序不修改 item，返回 top_k 且长度不超过输入。"""
    rr = _attach_fakes(LocalReranker(), logits=True)
    items = [{"t": "食堂"}, {"t": "选课"}, {"t": "图书馆"}]
    top = rr.rerank("选课", items, 2)
    assert len(top) == 2
    assert all(item in items for item in top)
    assert rr.rerank("x", [], 3) == []
    assert rr.rerank("x", items, 0) == []


# ── 4. 降级路径 ──────────────────────────────────────────────────────────────

def test_ensure_model_raises_when_all_files_unavailable():
    """模型文件全部下载失败 → 抛错（调用方降级，不静默返回坏模型）。"""
    with patch("mcp.embeddings._download_file", return_value=False):
        try:
            _ensure_model("fake/repo", ["onnx/model.onnx"], embeddings.model_cache_dir())
        except RuntimeError as ex:
            assert "fake/repo" in str(ex)
        else:
            raise AssertionError("应抛 RuntimeError")


def test_singleton_cooldown_skips_retry():
    """加载失败后冷却期内 get_embedder 返回 None，不重复尝试下载。"""
    reset_singletons()
    with patch("mcp.embeddings.LocalEmbedder") as mock_cls:
        mock_cls.return_value.available = False
        mock_cls.return_value._model = type("_M", (), {"error": "x"})()
        assert get_embedder() is None
        # 冷却期内：不再构造新实例
        assert get_embedder() is None
        assert mock_cls.call_count == 1


def test_singleton_success_cached():
    """加载成功后单例复用同一实例。"""
    reset_singletons()
    emb = _attach_fakes(LocalEmbedder())
    with patch("mcp.embeddings.LocalEmbedder", return_value=emb):
        assert get_embedder() is emb
        assert get_embedder() is emb


def test_empty_inputs_safe():
    """空输入不触发模型加载。"""
    emb = _attach_fakes(LocalEmbedder())
    assert emb.embed_texts([]) == []
    rr = _attach_fakes(LocalReranker())
    assert rr.score("q", []) == []
    assert rr.rerank("q", [], 3) == []


def test_get_singleton_thread_safe():
    """并发获取单例只初始化一次（锁保护）。"""
    reset_singletons()
    holder = {"instance": None, "failed_at": 0.0}
    created = []

    def factory():
        created.append(1)
        return _attach_fakes(LocalEmbedder())

    results = []
    threads = [threading.Thread(
        target=lambda: results.append(_get_singleton(holder, factory)))
        for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(created) == 1
    assert all(r is results[0] for r in results)
    reset_singletons()
