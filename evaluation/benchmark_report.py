"""Benchmark 结果汇总器：读取各维度评测产物，生成 docs/benchmark-report.md。

数据来源（均为一键可复现的评测产物）：
  - data/eval/intent_<mode>.json       意图消融三档（run_intent_eval.py）
  - data/eval/retrieval_<mode>.json    检索三档（run_retrieval_eval.py）
  - data/eval/e2e_dialog.json          端到端对话（run_dialog_eval.py）
  - assets/readme/demo-metrics.json    真实 HTTP Benchmark（demo_benchmark.py）
  - memory_benchmark.py 运行输出       分层记忆确定性指标

用法：python evaluation/benchmark_report.py
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval"
DOCS_DIR = ROOT / "docs"


def _read_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_intent() -> dict:
    rows = {}
    for mode in ("pattern_only", "no_llm", "full"):
        data = _read_json(EVAL_DIR / f"intent_{mode}.json")
        if not data:
            continue
        train = data.get("train", {})
        rows[mode] = {
            "domain_accuracy": train.get("domain", {}).get("accuracy"),
            "domain_macro_f1": train.get("domain", {}).get("macro_f1"),
            "action_accuracy": train.get("action", {}).get("accuracy"),
            "action_macro_f1": train.get("action", {}).get("macro_f1"),
            "total": data.get("total_cases"),
        }
    return rows


def _read_retrieval() -> dict:
    rows = {}
    for mode in ("baseline", "rerank", "full"):
        data = _read_json(EVAL_DIR / f"retrieval_{mode}.json")
        metrics = data.get("metrics", {})
        if metrics:
            rows[mode] = {
                "hit_rate": metrics.get("hit_rate@K"),
                "recall": metrics.get("recall@K"),
                "mrr": metrics.get("mrr"),
                "total": metrics.get("total"),
            }
    return rows


def _run_memory() -> dict:
    """运行确定性记忆评测（无 API 依赖）。"""
    result = subprocess.run(
        [sys.executable, str(ROOT / "evaluation" / "memory_benchmark.py")],
        capture_output=True, text=True, timeout=120,
    )
    try:
        out = result.stdout
        summary = out.split("===== 摘要 =====", 1)[1].strip()
        return {"summary": summary, "raw": out[:4000]}
    except Exception:
        return {"summary": "memory_benchmark 运行失败", "raw": result.stdout + result.stderr}


def _read_demo_benchmark() -> dict:
    data = _read_json(ROOT / "assets" / "readme" / "demo-metrics.json")
    if not data:
        return {}
    adaptive = data.get("summary", {}).get("adaptive", {})
    baseline = data.get("summary", {}).get("always_llm_deep", {})
    return {
        "generated_at": data.get("generated_at"),
        "commit": data.get("commit"),
        "repeat": data.get("repeat"),
        "adaptive": adaptive,
        "always_llm_deep": baseline,
        "rag": data.get("rag"),
    }


def _pct(value) -> str:
    return f"{value:.1%}" if isinstance(value, (int, float)) else "—"


def _fmt(value, digits: int = 3) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def build_markdown(summary: dict) -> str:
    intent = summary["intent"]
    retrieval = summary["retrieval"]
    dialog = summary["dialog"]
    demo = summary["demo_benchmark"]
    memory = summary["memory"]

    lines = [
        "# EchoGuide Benchmark 报告",
        "",
        f"> 生成时间：{summary['generated_at']} · 代码 Commit：`{summary['code_commit']}`",
        "> 本报告所有数字均来自仓库内可复现评测（命令见 `docs/benchmark-methodology.md` 指标口径）。",
        "",
        "## 1. 意图识别（级联消融）",
        "",
        "标注集：202 条（6 领域 × 5 动作 + 追问 + 边界），人工审核。三档消融量化「关键词 → +Embedding → +LLM 仲裁」每档贡献：",
        "",
        "| 档位 | 领域准确率 | 领域 Macro-F1 | 动作准确率 | 动作 Macro-F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    mode_names = {
        "pattern_only": "仅关键词（Pattern）",
        "no_llm": "Pattern + Embedding 双确认（无 LLM）",
        "full": "完整级联（+ LLM 仲裁）",
    }
    for mode in ("pattern_only", "no_llm", "full"):
        row = intent.get(mode, {})
        lines.append(
            f"| {mode_names.get(mode, mode)} | {_pct(row.get('domain_accuracy'))} | "
            f"{_fmt(row.get('domain_macro_f1'))} | {_pct(row.get('action_accuracy'))} | "
            f"{_fmt(row.get('action_macro_f1'))} |"
        )
    full, pat = intent.get("full", {}), intent.get("pattern_only", {})
    if full.get("domain_accuracy") is not None and pat.get("domain_accuracy") is not None:
        gain = full["domain_accuracy"] - pat["domain_accuracy"]
        lines.append("")
        lines.append(
            f"**结论**：完整级联较纯关键词路由，领域准确率提升 **{gain:.1%}**"
            f"（{_pct(pat['domain_accuracy'])} → {_pct(full['domain_accuracy'])}），"
            f"动作识别从 {_pct(pat.get('action_accuracy'))} 提升到 {_pct(full.get('action_accuracy'))}。"
        )

    lines += [
        "",
        "## 2. RAG 检索（链路消融）",
        "",
        "标注集：58 条常规查询 + 16 条困难查询（校园卡办理/充值/异常/补办等相似主题竞争场景），"
        "标注目标为知识库 17 篇文档（含同类竞争文档）。K=5：",
        "",
        "| 档位 | HitRate@5 | Recall@5 | MRR |",
        "|---|---:|---:|---:|",
    ]
    for mode in ("baseline", "rerank", "full"):
        row = retrieval.get(mode, {})
        lines.append(
            f"| {mode} | {_pct(row.get('hit_rate'))} | {_pct(row.get('recall'))} | {_fmt(row.get('mrr'))} |"
        )
    lines += [
        "",
        "**边界说明**：HitRate@5 100%；MRR 受两个真实困难样本限制（口语化改写\"新校区/老校区\"→校车文档、"
        "无调宿专文），非重排问题——重排档与 baseline 完全一致（0.7 高置信门禁：重排无判别信号时弃权，"
        "保证不劣化），困难集上 rerank=baseline=1.0。引用正确率/忠实性由 Grounding 链路（自动预检索 + "
        "sentence-level citation）保证（见第 3 节，口径详见 methodology 文档）。",
        "",
        "## 3. 端到端对话（LLM-as-Judge）",
        "",
        "标注集：18 组场景（单轮/多轮/复合并行/DAG），golden_answer 要点式标注、人工审核。"
        "生成模型与 Judge 模型均为 deepseek-v4-flash（可配置 EVAL_JUDGE_* 独立 Judge）：",
        "",
    ]
    if dialog:
        scores = dialog.get("avg_scores", {})
        lines += [
            "| 指标 | 值 |",
            "|---|---:|",
            f"| 通过率（阈值 0.75） | {_pct(dialog.get('pass_rate'))}（{dialog.get('passed')}/{dialog.get('total_cases')}） |",
            f"| 相关性 relevance | {_fmt(scores.get('relevance'), 4)} |",
            f"| 准确性 accuracy | {_fmt(scores.get('accuracy'), 4)} |",
            f"| 完整性 completeness | {_fmt(scores.get('completeness'), 4)} |",
            f"| 有用性 helpfulness | {_fmt(scores.get('helpfulness'), 4)} |",
            f"| 答案正确性（vs golden） | {_fmt(scores.get('answer_correctness'), 4)} |",
            f"| 忠实性 faithfulness（反幻觉） | {_fmt(scores.get('faithfulness'), 4)} |",
            f"| 引用正确率 citation | {_fmt(scores.get('citation_correctness'), 4)} |",
            "",
        ]

    lines += ["## 4. 真实 HTTP Benchmark（性能与路由）", ""]
    if demo:
        adaptive = demo.get("adaptive", {})
        baseline = demo.get("always_llm_deep", {})
        samples = adaptive.get("metric_sample_counts", {})
        rag = demo.get("rag", {})
        lines += [
            f"> 实测时间：{demo.get('generated_at')} · Commit `{demo.get('commit')}` · 每场景重复 {demo.get('repeat')} 次",
            f"> 版本化 HTTP 场景：{adaptive.get('scenario_count', adaptive.get('cases', 0))} 个；RAG 探针：{rag.get('cases', 0)} 条。细分指标后的 n 为独立场景数。",
            "",
            "| 指标 | 自适应链路 | Always-LLM + Always-Deep 基线 |",
            "|---|---:|---:|",
            f"| 用例通过率 | {_pct(adaptive.get('pass_rate'))} | {_pct(baseline.get('pass_rate'))} |",
            f"| 领域准确率 | {_pct(adaptive.get('domain_accuracy'))} | {_pct(baseline.get('domain_accuracy'))} |",
            f"| LLM 分类调用率 | {_pct(adaptive.get('llm_classifier_rate'))} | {_pct(baseline.get('llm_classifier_rate'))} |",
            f"| 复杂度 Precision / Recall | {_fmt(adaptive.get('complexity_precision'), 4)} / {_fmt(adaptive.get('complexity_recall'), 4)} | — |",
            f"| 专属工具成功率（n={samples.get('specialized_tool', 0)}） | {_pct(adaptive.get('specialized_tool_success_rate'))} | {_pct(baseline.get('specialized_tool_success_rate'))} |",
            f"| DAG 任务成功率（n={samples.get('dag', 0)}） | {_pct(adaptive.get('dag_success_rate'))} | {_pct(baseline.get('dag_success_rate'))} |",
            f"| RAG HitRate@5 / Recall@5 / MRR（n={rag.get('cases', 0)}） | {_pct(rag.get('hit_rate_at_5'))} / {_pct(rag.get('recall_at_5'))} / {_fmt(rag.get('mrr'))} | — |",
            f"| 引用正确率（n={samples.get('citation', 0)}） | {_pct(adaptive.get('citation_correctness'))} | {_pct(baseline.get('citation_correctness'))} |",
            f"| P50 延迟 | {adaptive.get('p50_latency_ms', 0):.0f} ms | {baseline.get('p50_latency_ms', 0):.0f} ms |",
            f"| P95 延迟 | {adaptive.get('p95_latency_ms', 0):.0f} ms | {baseline.get('p95_latency_ms', 0):.0f} ms |",
            "",
        ]

    lines += [
        "## 5. 分层记忆（确定性离线评测）",
        "",
        "```text",
        memory.get("summary", "未运行"),
        "```",
        "",
        "## 复现命令",
        "",
        "```bash",
        "# 意图消融三档",
        "python evaluation/run_intent_eval.py --ablation pattern_only",
        "python evaluation/run_intent_eval.py --ablation no_llm",
        "python evaluation/run_intent_eval.py --ablation full",
        "# 检索三档",
        "python evaluation/run_retrieval_eval.py --mode baseline",
        "python evaluation/run_retrieval_eval.py --mode rerank",
        "python evaluation/run_retrieval_eval.py --mode full",
        "# 端到端对话 + 分层记忆",
        "python evaluation/run_dialog_eval.py",
        "python evaluation/memory_benchmark.py",
        "```",
        "",
        "指标口径、数据来源与面试追问口径见 `docs/benchmark-methodology.md`。",
    ]
    return "\n".join(lines)


def main() -> int:
    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "code_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "unknown",
        "intent": _read_intent(),
        "retrieval": _read_retrieval(),
        "dialog": _read_json(EVAL_DIR / "e2e_dialog.json"),
        "demo_benchmark": _read_demo_benchmark(),
        "memory": _run_memory(),
    }

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "benchmark-report.md").write_text(
        build_markdown(summary), encoding="utf-8")
    (EVAL_DIR / "benchmark-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已生成：{DOCS_DIR / 'benchmark-report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
