"""RAG 检索评测 runner：三档对比 HitRate@K / Recall@K / MRR。

用法：
  python evaluation/run_retrieval_eval.py --mode baseline  # 单查询直检（离线）
  python evaluation/run_retrieval_eval.py --mode rerank    # +本地重排（离线）
  python evaluation/run_retrieval_eval.py --mode full      # 改写+并行召回+去重+重排（需 API key）

消融口径：
  - baseline：KnowledgeBase.search 单查询向量检索（最朴素基线，可离线复现）
  - rerank：多召回后本地 bge-reranker 重排取 Top-K（离线，零 token 成本）
  - full：与 Agentic RAG 主链路同一套 search_with_rewrite（LLM 查询改写 →
    并行召回 → 去重 → 重排），量化完整优化链路的增益

指标（evaluation/evaluator.py compute_retrieval_metrics，纯 Python 确定性）：
  HitRate@K / Recall@K / MRR，K=5。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
load_dotenv()  # 加载 .env（ANTHROPIC_API_KEY 等）

from evaluation.cases import load_retrieval_cases
from evaluation.evaluator import compute_retrieval_metrics
from mcp.knowledge_base import KnowledgeBase
from mcp.tool_manager import MCPToolManager, Tool, ToolEffect


def _llm_config() -> dict:
    key = os.getenv("ECHOGUIDE_FAST_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
    base_url = os.getenv("ECHOGUIDE_FAST_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL", "")
    model = os.getenv("ECHOGUIDE_FAST_MODEL") or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    return {"api_key": key, "base_url": base_url, "model": model}


async def _run_baseline(kb: KnowledgeBase, cases, top_k: int = 5) -> list:
    """单查询直检：无改写、无重排。"""
    return [await asyncio.to_thread(kb.search, c.query, top_k=top_k) for c in cases]


async def _run_rerank(kb: KnowledgeBase, manager: MCPToolManager, cases, top_k: int = 5) -> list:
    """多召回（Top-10）→ 本地重排 → Top-K。重排后端 local，模型不可用自动降级 LLM。"""
    results = []
    for c in cases:
        items = await asyncio.to_thread(kb.search, c.query, top_k=max(top_k * 2, 10))
        reranked = await manager._rerank(c.query, items, top_k)
        results.append(reranked)
    return results


async def _run_full(kb: KnowledgeBase, manager: MCPToolManager, cases, top_k: int = 5) -> list:
    """完整优化链路：查询改写 → 并行召回 → 去重 → 重排（与主链路同实现）。"""
    manager.register(Tool(
        name="knowledge_search",
        description="检索校园知识库",
        handler=kb.search_handler,
        schema={"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}},
        cache_ttl=300.0,
        use_rewrite=False,  # search_with_rewrite 内部自管理改写，避免递归
        effect=ToolEffect.READ,
    ))
    results = []
    for c in cases:
        result = await manager.search_with_rewrite("knowledge_search", c.query, top_k=top_k)
        results.append(result.data if result.success else [])
    return results


async def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索评测（baseline / rerank / full）")
    parser.add_argument("--mode", choices=["baseline", "rerank", "full"], default="baseline")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default=None, help="输出 JSON 路径（默认 data/eval/retrieval_<mode>.json）")
    parser.add_argument("--cases", default=None, help="标注集路径（默认 evaluation/cases/retrieval_cases.json）")
    args = parser.parse_args()

    cases = load_retrieval_cases(args.cases)
    kb = KnowledgeBase()

    if args.mode == "full" and not _llm_config()["api_key"]:
        print("[错误] full 模式需要 LLM 配置（查询改写）：设置 ANTHROPIC_API_KEY（或 ECHOGUIDE_FAST_API_KEY）。"
              "离线档位请用 --mode baseline / rerank。")
        return 2

    if args.mode == "baseline":
        results = await _run_baseline(kb, cases, args.top_k)
        mode_desc = "单查询向量检索（无改写无重排）"
    else:
        cfg = _llm_config()
        manager = MCPToolManager(
            api_key=cfg["api_key"] or "sk-ablation-offline",
            base_url=cfg["base_url"] or None,
            model=cfg["model"],
            rerank_backend="local",
        )
        if args.mode == "rerank":
            results = await _run_rerank(kb, manager, cases, args.top_k)
            mode_desc = "多召回 + 本地重排（无改写）"
        else:
            results = await _run_full(kb, manager, cases, args.top_k)
            mode_desc = "查询改写 + 并行召回 + 去重 + 重排（完整链路）"

    relevant = [c.relevant_titles for c in cases]
    metrics = compute_retrieval_metrics(results, relevant, top_k=args.top_k)
    report = {
        "mode": args.mode,
        "description": mode_desc,
        "total": len(cases),
        "metrics": {k: v for k, v in metrics.items() if k != "cases"},
    }

    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).resolve().parents[1] / "data" / "eval" / f"retrieval_{args.mode}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
