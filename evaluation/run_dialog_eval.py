"""端到端对话评测 runner：真实调用 Orchestrator 生成回复，LLM-as-Judge 四维评分。

评测内容（与 /eval/run 同一实现 evaluation/evaluator.py EndToEndEvaluator）：
  - 四维质量评分：relevance / accuracy / completeness / helpfulness（Judge 独立可配）
  - Answer Correctness：与 golden_answer 要点一致性（标注集人工审核）
  - Faithfulness：回答是否被检索证据支持（反幻觉）
  - Citation Correctness：回答中的 [n] 引用是否都在来源范围内（确定性）

用法：
  python evaluation/run_dialog_eval.py                  # 全量评测并保存基线
  python evaluation/run_dialog_eval.py --smoke          # 只跑前 3 组（联调）
  python evaluation/run_dialog_eval.py --no-baseline    # 不写回基线

说明：需要 .env 中配置 LLM API Key（生成与 Judge 可同模型，也可用 EVAL_JUDGE_* 分离）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv()  # 加载 .env

from evaluation.cases import load_dialog_cases


async def main() -> int:
    parser = argparse.ArgumentParser(description="端到端对话评测（LLM-as-Judge）")
    parser.add_argument("--smoke", action="store_true", help="只跑前 3 组（联调用）")
    parser.add_argument("--no-baseline", action="store_true", help="不写回评测基线")
    parser.add_argument("--out", default=None, help="输出 JSON 路径（默认 data/eval/e2e_dialog.json）")
    args = parser.parse_args()

    from api import state

    state._build_runtime()
    evaluator = state._evaluator
    if evaluator is None:
        print("[错误] 评测器未初始化")
        return 2

    cases = load_dialog_cases()
    if args.smoke:
        cases = cases[:3]

    report = await evaluator.run(
        dialog_cases=cases,
        dataset="dialog_cases_v1",
        promote_baseline=not args.no_baseline,
    )

    # 汇总关键指标
    summary = {
        "timestamp": report.timestamp,
        "dataset": report.provenance.get("dataset"),
        "dataset_counts": report.provenance.get("dataset_counts"),
        "code_commit": report.provenance.get("code_commit"),
        "generator_model": report.provenance.get("generator_model"),
        "judge_model": report.provenance.get("judge_model"),
        "judge_independent": report.judge.get("judge_independent"),
        "pass_rate": report.pass_rate,
        "total_cases": report.total,
        "passed": report.passed,
        "avg_scores": report.avg_scores,
        "regressions": report.regressions,
    }
    out = args.out or ROOT / "data" / "eval" / "e2e_dialog.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
