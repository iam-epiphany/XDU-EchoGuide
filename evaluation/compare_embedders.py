"""Embedding 模型检索质量 A/B 对比：MiniLM-L6-v2（旧） vs bge-small-zh-v1.5（新）。

为什么做这个对比：
  旧链路 Embedding 用 ChromaDB 内置 all-MiniLM-L6-v2（英文模型，中文语义弱），
  新链路改用本地 bge-small-zh-v1.5（中文优化，ONNX）。本脚本在同一份文档集、
  同一批检索用例上分别跑两种模型，量化检索质量差异（HitRate@K / Recall@K / MRR）。

用法：
  python evaluation/compare_embedders.py [--top-k 5] [--cases evaluation/retrieval_cases.json]

说明：
  - 文档集与运行时导入一致：默认知识库（mcp.knowledge_base.DEFAULT_DOCS）
    + data/public/academic_policies.json + data/knowledge_docs/ 投放目录；
  - 纯 numpy 余弦检索（不依赖 ChromaDB 服务），两个模型各自独立向量空间；
  - 指标复用 evaluation/evaluator.py 的 compute_retrieval_metrics（与生产口径一致）；
  - 任一模型不可用（无本地缓存/无网络）时跳过该模型并提示，其余照常对比；
  - 结果同时输出到 evaluation/embedding_compare_result.json 与 stdout。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 支持直接 python evaluation/compare_embedders.py 运行

from evaluation.evaluator import compute_retrieval_metrics  # noqa: E402
from mcp.knowledge_base import DEFAULT_DOCS, KnowledgeBase  # noqa: E402

DEFAULT_CASES = ROOT / "evaluation" / "retrieval_cases.json"
OUTPUT_PATH = ROOT / "evaluation" / "embedding_compare_result.json"


# ── 文档集（与运行时导入同源）──────────────────────────────────────────────

def _load_document_set() -> List[Dict[str, Any]]:
    docs = [dict(d) for d in DEFAULT_DOCS]
    public = ROOT / "data" / "public" / "academic_policies.json"
    if public.exists():
        docs.extend(json.loads(public.read_text(encoding="utf-8")))
    docs_dir = ROOT / "data" / "knowledge_docs"
    if docs_dir.is_dir():
        from mcp.document_parser import SUPPORTED_EXTENSIONS, parse_document
        for path in sorted(docs_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                docs.extend(parse_document(path.name, path.read_bytes()))
    return docs


def _chunk_documents(docs: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """与知识库一致的分块（500 字 / 60 字 overlap），返回 [(title, chunk), ...]。"""
    kb = KnowledgeBase.__new__(KnowledgeBase)  # 只取纯函数，不触发 chroma 连接
    chunks: List[Tuple[str, str]] = []
    for doc in docs:
        for chunk in kb._chunk_text(doc.get("content", "")):
            if chunk.strip():
                chunks.append((doc.get("title", ""), chunk))
    return chunks


# ── 向量化检索（纯 numpy，独立于 ChromaDB）─────────────────────────────────

def _embed_all(embedder: Callable[[List[str]], Any], texts: List[str]) -> np.ndarray:
    """批量嵌入并 L2 归一化（cosine 空间：归一化后点积即相似度）。"""
    vecs: List[np.ndarray] = []
    for i in range(0, len(texts), 64):
        for v in embedder(texts[i:i + 64]):
            arr = np.asarray(v, dtype=np.float32)
            norm = np.linalg.norm(arr)
            vecs.append(arr / norm if norm > 0 else arr)
    return np.stack(vecs)


def _retrieve(
    chunk_vecs: np.ndarray,
    chunks: List[Tuple[str, str]],
    query_vec: np.ndarray,
    top_k: int,
) -> List[Dict[str, Any]]:
    sims = chunk_vecs @ query_vec
    order = np.argsort(-sims)[:top_k]
    return [
        {"title": chunks[i][0], "content": chunks[i][1], "score": float(sims[i])}
        for i in order
    ]


# ── 各 embedder 工厂（与生产链路同款实现）──────────────────────────────────

def _make_minilm() -> Callable[[List[str]], Any]:
    """旧链路：chromadb 内置 all-MiniLM-L6-v2（DefaultEmbeddingFunction）。"""
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


def _make_bge() -> Callable[[List[str]], Any]:
    """新链路：本地 bge-small-zh-v1.5（mcp.embeddings，chromadb 协议入口无前缀）。"""
    from mcp.embeddings import LocalEmbedder
    embedder = LocalEmbedder()
    if not embedder.available:
        raise RuntimeError(f"bge 模型不可用: {embedder._model.error}")
    return embedder


# ── 主流程 ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding 模型检索质量 A/B 对比")
    parser.add_argument("--top-k", type=int, default=5, help="检索返回条数（默认 5）")
    parser.add_argument("--cases", type=str, default=str(DEFAULT_CASES), help="检索用例 JSON")
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    queries = [c["query"] for c in cases]
    relevant = [c["relevant_titles"] for c in cases]

    docs = _chunk_documents(_load_document_set())
    print(f"文档集: {len(docs)} 个片段，{len(queries)} 个检索用例，Top-{args.top_k}\n")

    factories = {
        "MiniLM-L6-v2 (旧)": _make_minilm,
        "bge-small-zh-v1.5 (新)": _make_bge,
    }
    report: Dict[str, Any] = {"top_k": args.top_k, "cases": len(queries), "models": {}}
    for name, factory in factories.items():
        try:
            embedder = factory()
        except Exception as ex:
            print(f"[跳过] {name}: {ex}")
            continue
        chunk_vecs = _embed_all(embedder, [c for _, c in docs])
        query_vecs = _embed_all(embedder, queries)
        results = [_retrieve(chunk_vecs, docs, qv, args.top_k) for qv in query_vecs]
        metrics = compute_retrieval_metrics(results, relevant, top_k=args.top_k)
        report["models"][name] = {
            "hit_rate@K": metrics["hit_rate@K"],
            "recall@K": metrics["recall@K"],
            "mrr": metrics["mrr"],
        }
        print(f"  {name}")
        print(f"    HitRate@{args.top_k} = {metrics['hit_rate@K']:.4f}"
              f"  Recall@{args.top_k} = {metrics['recall@K']:.4f}"
              f"  MRR = {metrics['mrr']:.4f}")

    # 汇总对比行
    models = report["models"]
    if len(models) >= 2:
        old, new = models["MiniLM-L6-v2 (旧)"], models["bge-small-zh-v1.5 (新)"]
        print("\n  对比（新 - 旧）:")
        for k in ("hit_rate@K", "recall@K", "mrr"):
            delta = new[k] - old[k]
            print(f"    {k}: {delta:+.4f}")
        report["delta"] = {k: round(new[k] - old[k], 4) for k in ("hit_rate@K", "recall@K", "mrr")}
    else:
        print("\n  仅 1 个模型可用，无法对比；请在模型缓存/网络可用的环境运行。")

    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {OUTPUT_PATH}")
    return 0 if "bge-small-zh-v1.5 (新)" in models else 1


if __name__ == "__main__":
    sys.exit(main())
