"""意图识别错误分析：跑完整级联 + 路由 trace，把每个错误分类到根因桶。

用法：
  python evaluation/error_analysis/intent_error_analysis.py
  python evaluation/error_analysis/intent_error_analysis.py --out data/eval/error_analysis/intent_errors.json
  python evaluation/error_analysis/intent_error_analysis.py --ablation no_llm   # 离线分析免费路径

输出（每条用例）：
  {
    "query", "gold_domain", "predict_domain", "gold_action", "predict_action",
    "router_stage", "confidence", "reason",
    "pattern": {...}, "is_followup": bool, "embedding_candidates": [...],
    "category": "A"|"B"|"C", "category_reason": "..."
  }

分类口径（仅统计领域维度错误；动作维度错误单独列出）：
  - C（LLM 仲裁错误）：最终决策经 LLM（stage=llm）且判错；
      若关键词/Embedding 已给出正确信号 → 子类 C1「未采纳强信号」，
      否则 → 子类 C2「无信号硬猜」。
  - B（Embedding 召回错误）：免费路径（pattern/embedding）判错，
      且正确领域不在 Embedding Top-3 候选内（模板/阈值体系问题）。
  - A（领域边界 / 追问继承）：免费路径判错，但正确领域在 Embedding
      候选内（信号存在、决策或阈值组合问题），或追问形态下领域回填失败。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv()

from core.intent_recognizer import IntentDomain, IntentRecognizer
from evaluation.cases import load_intent_cases


def _llm_config() -> dict:
    key = os.getenv("ECHOGUIDE_FAST_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
    base_url = os.getenv("ECHOGUIDE_FAST_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL", "")
    model = os.getenv("ECHOGUIDE_FAST_MODEL") or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    return {"api_key": key, "base_url": base_url, "model": model}


def _classify(record: dict) -> dict:
    """按 trace 信号把错误分类到 A/B/C 桶，返回 (category, reason)。"""
    gold = record["gold_domain"]
    pred = record["predict_domain"]
    stage = record["router_stage"]
    pat = record.get("pattern") or {}
    cands = record.get("embedding_candidates") or []
    pat_hit = pat.get("domain") == gold
    emb_top = cands[0]["domain"] if cands else None
    emb_hit = any(c["domain"] == gold for c in cands)
    emb_top_score = cands[0]["score"] if cands else 0.0

    def emb_desc() -> str:
        return "、".join(f"{c['domain']}({c['score']:.2f})" for c in cands[:3]) or "无"

    if stage == "llm":
        if pat_hit or emb_hit:
            return "C1", (
                f"LLM 仲裁判 {pred}，但信号本可给出 {gold}（关键词命中="
                f"{pat_hit}，Embedding 候选[{emb_desc()}]）；LLM 未采纳强信号"
            )
        return "C2", (
            f"LLM 仲裁判 {pred}（置信 {record['confidence']:.2f}），免费路径无信号"
            f"（pattern={pat.get('domain')}，Embedding 候选[{emb_desc()}]），硬猜失败"
        )
    if stage == "embedding":
        if emb_top == gold:
            return "A", f"Embedding 判 {pred} 但 Top-1 是 {gold}（决策与信号矛盾）"
        return "B", (
            f"Embedding 直返 {pred}（Top-1 相似度 {emb_top_score:.2f}），"
            f"正确领域 {gold} 不在候选[{emb_desc()}]：Embedding 召回失败"
        )
    if stage == "pattern":
        if record.get("is_followup"):
            return "A", f"追问形态（{record['reason']}），领域回填为 {pred}，未继承 {gold}"
        if emb_hit:
            return "A", (
                f"关键词直返 {pred}，但正确领域 {gold} 在 Embedding 候选"
                f"[{emb_desc()}]：领域边界/双确认阈值问题"
            )
        return "B", (
            f"关键词直返 {pred}（pattern={pat.get('domain')} {pat.get('confidence', 0):.2f}），"
            f"正确领域 {gold} 不在候选[{emb_desc()}]：关键词与 Embedding 都召回失败"
        )
    return "C2", f"未知阶段 {stage}，LLM 兜底判 {pred}"


async def main() -> int:
    parser = argparse.ArgumentParser(description="意图识别错误分析")
    parser.add_argument("--ablation", choices=["full", "pattern_only", "no_llm"], default="full")
    parser.add_argument("--out", default=None, help="输出 JSON（默认 data/eval/error_analysis/intent_errors.json）")
    parser.add_argument("--cases", default=None, help="标注集路径（默认 evaluation/cases/intent_cases.json）")
    args = parser.parse_args()

    cases = load_intent_cases(args.cases)
    cfg = _llm_config()
    if args.ablation == "full" and not cfg["api_key"]:
        print("[错误] full 分析需要 LLM 配置；离线分析请用 --ablation pattern_only / no_llm")
        return 2

    recognizer = IntentRecognizer(
        api_key=cfg["api_key"] or "sk-analysis-offline",
        base_url=cfg["base_url"] or None,
        model=cfg["model"],
        ablation_mode=args.ablation,
    )

    domain_values = {d.value for d in IntentDomain}
    records = []
    for case in cases:
        expected = case.expected_intent
        trace: dict = {}
        result = await recognizer.recognize(
            case.message,
            history=(case.context or {}).get("history") if case.context else None,
            _trace=trace,
        )
        is_domain_case = expected in domain_values
        gold = expected if is_domain_case else None
        pred_domain = result.domain.value
        pred_action = result.action.value
        record = {
            "query": case.message,
            "gold_domain": gold,
            "predict_domain": pred_domain,
            "gold_action": None if is_domain_case else expected,
            "predict_action": pred_action,
            "router_stage": result.classifier_stage,
            "confidence": result.confidence,
            "reason": result.reasoning,
            "pattern": trace.get("pattern"),
            "is_followup": trace.get("is_followup", False),
            "embedding_candidates": trace.get("embedding_candidates", []),
            "correct": (pred_domain == gold) if is_domain_case else (pred_action == expected),
        }
        if not record["correct"]:
            category, category_reason = _classify(record)
            record["category"] = category
            record["category_reason"] = category_reason
        records.append(record)

    # ── 统计 ────────────────────────────────────────────────────────────────
    domain_cases = [r for r in records if r["gold_domain"]]
    action_cases = [r for r in records if r["gold_action"]]
    errors = [r for r in domain_cases if not r["correct"]]
    cat_counts: dict = {}
    for r in errors:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    per_domain: dict = {}
    for r in domain_cases:
        g = r["gold_domain"]
        per_domain.setdefault(g, {"total": 0, "correct": 0})
        per_domain[g]["total"] += 1
        per_domain[g]["correct"] += int(r["correct"])

    confusion: dict = {}
    for r in errors:
        key = f"{r['gold_domain']}→{r['predict_domain']}"
        confusion[key] = confusion.get(key, 0) + 1

    stage_counts: dict = {}
    for r in errors:
        stage_counts[r["router_stage"]] = stage_counts.get(r["router_stage"], 0) + 1

    report = {
        "ablation": args.ablation,
        "total_domain_cases": len(domain_cases),
        "domain_accuracy": round(sum(1 for r in domain_cases if r["correct"]) / len(domain_cases), 4),
        "domain_correct": sum(1 for r in domain_cases if r["correct"]),
        "action_accuracy": round(sum(1 for r in action_cases if r["correct"]) / len(action_cases), 4) if action_cases else None,
        "error_categories": cat_counts,
        "error_stages": stage_counts,
        "per_domain": {k: {**v, "accuracy": round(v["correct"] / v["total"], 4)} for k, v in sorted(per_domain.items())},
        "confusion": dict(sorted(confusion.items(), key=lambda kv: -kv[1])),
        "errors": [r for r in errors],
        "all": records,
    }

    out = args.out or ROOT / "data" / "eval" / "error_analysis" / "intent_errors.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"领域准确率: {report['domain_accuracy']:.4f} "
          f"({report['domain_correct']}/{report['total_domain_cases']})")
    print(f"动作准确率: {report['action_accuracy']}")
    print("错误分类:", json.dumps(cat_counts, ensure_ascii=False))
    print("错误路由阶段:", json.dumps(stage_counts, ensure_ascii=False))
    print("每领域准确率:")
    for d, v in report["per_domain"].items():
        print(f"  {d:12s} {v['accuracy']:.4f} ({v['correct']}/{v['total']})")
    print("混淆对:")
    for k, v in report["confusion"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
