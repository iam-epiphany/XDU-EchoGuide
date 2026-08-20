"""意图识别评测 runner：加载标注集，跑指定消融档位，输出指标 JSON。

用法：
  python evaluation/run_intent_eval.py                         # full 级联（需 API key）
  python evaluation/run_intent_eval.py --ablation pattern_only # 仅关键词（离线）
  python evaluation/run_intent_eval.py --ablation no_llm       # 关键词+Embedding（离线）
  python evaluation/run_intent_eval.py --held-out 0.2          # 留出 20% 未见用例
  python evaluation/run_intent_eval.py --out data/eval/intent_xxx.json

消融档位（core/intent_recognizer.py ablation_mode）：
  - pattern_only：关键词匹配（最朴素基线）
  - no_llm：级联免费路径（关键词 + Embedding），无 LLM 仲裁
  - full：完整级联（追问→LLM / 双确认直返 / Embedding 直返 / LLM 仲裁）

统计口径：
  - expected 为 IntentDomain 值（academic/campus_life/affairs/it_help/personal/other）
    时比较预测 domain；为 IntentAction 值（query/request/greeting/complaint/feedback）
    时比较预测 action —— 与 IntentEvaluator 约定一致，两组分开统计。
  - Accuracy / Macro-F1 / per-class P/R/F1 由 evaluation/evaluator.py 纯 Python 计算。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import random
import sys

from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
load_dotenv()  # 加载 .env（ANTHROPIC_API_KEY 等）

from core.intent_recognizer import IntentDomain, IntentRecognizer
from evaluation.cases import load_intent_cases
from evaluation.evaluator import IntentEvaluator, IntentTestCase


def _llm_config() -> dict:
    """读取 LLM 配置（与 api/state.py 环境变量约定一致）。"""
    key = os.getenv("ECHOGUIDE_FAST_API_KEY") or os.getenv("ANTHROPIC_API_KEY", "")
    base_url = os.getenv("ECHOGUIDE_FAST_BASE_URL") or os.getenv("ANTHROPIC_BASE_URL", "")
    model = os.getenv("ECHOGUIDE_FAST_MODEL") or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    return {"api_key": key, "base_url": base_url, "model": model}


def _split_held_out(cases: list, ratio: float, seed: int = 42):
    """留出 held-out 未见用例（固定 seed 可复现），返回 (训练子集, 留出子集)。"""
    if ratio <= 0:
        return cases, []
    rng = random.Random(seed)
    pool = list(cases)
    rng.shuffle(pool)
    n = max(1, round(len(pool) * ratio))
    return pool[n:], pool[:n]


def _group_by_dim(cases: list) -> dict:
    """按 expected 取值把用例分成 domain 组与 action 组（口径：两组分开统计）。"""
    domain_values = {d.value for d in IntentDomain}
    groups = {"domain": [], "action": []}
    for case in cases:
        groups["domain" if case.expected_intent in domain_values else "action"].append(case)
    return groups


def _summarize_metrics(metrics: dict) -> dict:
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "total": metrics["total"],
        "correct": metrics["correct"],
    }


async def _run(cases: list, recognizer: IntentRecognizer) -> dict:
    evaluator = IntentEvaluator(recognizer)
    groups = _group_by_dim(cases)
    result = {}
    for dim, group in groups.items():
        if group:
            result[dim] = _summarize_metrics(await evaluator.evaluate(group))
    result["all"] = _summarize_metrics(await evaluator.evaluate(cases))
    return result


async def main() -> int:
    parser = argparse.ArgumentParser(description="意图识别评测（含消融与 held-out）")
    parser.add_argument("--ablation", choices=["full", "pattern_only", "no_llm"], default="full")
    parser.add_argument("--held-out", type=float, default=0.0, help="留出比例 0-1（固定 seed 可复现）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="输出 JSON 路径（默认 data/eval/intent_<mode>.json）")
    parser.add_argument("--cases", default=None, help="标注集路径（默认 evaluation/cases/intent_cases.json）")
    args = parser.parse_args()

    cases = load_intent_cases(args.cases)
    cfg = _llm_config()
    if args.ablation == "full" and not cfg["api_key"]:
        print("[错误] full 级联需要 LLM 配置：设置 ANTHROPIC_API_KEY（或 ECHOGUIDE_FAST_API_KEY）。"
              "离线档位请用 --ablation pattern_only / no_llm。")
        return 2

    recognizer = IntentRecognizer(
        api_key=cfg["api_key"] or "sk-ablation-offline",  # 离线档位不会发起请求
        base_url=cfg["base_url"] or None,
        model=cfg["model"],
        ablation_mode=args.ablation,
    )

    train_cases, held_out = _split_held_out(cases, args.held_out, seed=args.seed)
    report = {
        "mode": args.ablation,
        "total_cases": len(cases),
        "held_out_ratio": args.held_out,
        "train": await _run(train_cases, recognizer),
    }
    if held_out:
        report["held_out"] = await _run(held_out, recognizer)

    out = pathlib.Path(args.out) if args.out else pathlib.Path(__file__).resolve().parents[1] / "data" / "eval" / f"intent_{args.ablation}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
