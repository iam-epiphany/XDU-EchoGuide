"""RAG 检索硬指标测试：HitRate@K / Recall@K / MRR 纯函数 + 引用正确性（无需 LLM）。"""
from __future__ import annotations

from evaluation.evaluator import citation_correctness, compute_retrieval_metrics


def test_retrieval_metrics_hit_recall_mrr():
    results = [
        [{"title": "选课指南"}, {"title": "校历"}],   # 相关在 rank1 → MRR 1.0
        [{"title": "食堂"}, {"title": "选课指南"}],    # 相关在 rank2 → MRR 0.5
        [{"title": "图书馆"}, {"title": "宿舍"}],      # 无相关 → MRR 0
    ]
    relevant = [["选课指南"], ["选课指南"], ["选课指南"]]
    m = compute_retrieval_metrics(results, relevant, top_k=2)
    assert m["hit_rate@K"] == round(2 / 3, 4)
    assert m["recall@K"] == round(2 / 3, 4)   # 前两个用例的相关被召回，第三个未召回
    assert m["mrr"] == round((1.0 + 0.5 + 0.0) / 3, 4)
    assert m["total"] == 3 and m["top_k"] == 2


def test_retrieval_recall_partial():
    """相关文档只召回一部分 → Recall@K < 1，但 HitRate 仍可能为 1。"""
    results = [[{"title": "选课指南"}]]
    relevant = [["选课指南", "校历"]]
    m = compute_retrieval_metrics(results, relevant, top_k=5)
    assert m["recall@K"] == 0.5
    assert m["hit_rate@K"] == 1.0
    assert m["mrr"] == 1.0


def test_citation_correctness_all_valid():
    answer = "选课分预选、正选两个阶段 [1]，退改选时间见通知 [2]。"
    c = citation_correctness(answer, ["文档A", "文档B", "文档C"])
    assert c["total"] == 2 and c["valid"] == 2 and c["invalid"] == []
    assert c["score"] == 1.0 and c["has_citation"] is True


def test_citation_correctness_invalid_index():
    answer = "选课说明 [1]，细节见 [5]。"
    c = citation_correctness(answer, ["文档A", "文档B"])
    assert c["total"] == 2 and c["valid"] == 1
    assert c["invalid"] == [5]
    assert c["score"] == 0.5


def test_citation_correctness_no_citation():
    c = citation_correctness("没有任何引用的回答", ["文档A"])
    assert c["total"] == 0 and c["score"] is None and c["has_citation"] is False
    assert c["has_citation"] is False
