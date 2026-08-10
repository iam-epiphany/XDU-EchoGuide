"""真实 HTTP Demo Benchmark：准备专用用户、运行用例并更新 README 指标块。"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import statistics
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evaluation" / "demo_cases.json"
RESULT_PATH = ROOT / "assets" / "readme" / "demo-metrics.json"
README_PATH = ROOT / "README.md"
# 专用 demo 账号可覆盖；自定义用户名时脚本会跳过个人数据清空（防误删真实数据）
USERNAME = os.getenv("ECHOGUIDE_DEMO_USER", "echoguide_demo")
PASSWORD = os.getenv("ECHOGUIDE_DEMO_PASSWORD", "EchoGuideDemo2026!")
CASE_BY_ID = {case["id"]: case for case in json.loads(CASES_PATH.read_text(encoding="utf-8"))}


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


@functools.lru_cache(maxsize=1)
def load_cases() -> List[Dict[str, Any]]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def git_revision() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    ).stdout.strip() or "working-tree"
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    ).stdout.strip())
    return f"{head}-dirty" if dirty else head


def login_or_register(client: httpx.Client) -> None:
    response = client.post("/auth/register", json={"username": USERNAME, "password": PASSWORD})
    if response.status_code not in (201, 409, 400):
        response.raise_for_status()
    response = client.post("/auth/login", json={"username": USERNAME, "password": PASSWORD})
    response.raise_for_status()


def prepare_demo_data(client: httpx.Client) -> None:
    # 只清空专用 demo 账号的数据；自定义账号（可能是真实用户）时跳过删除操作，
    # 仅导入测试课表，避免误清真实个人数据。
    if USERNAME != "echoguide_demo":
        print(f"[benchmark] 使用自定义账号 {USERNAME}，跳过个人数据清空（仅导入测试课表）")
    else:
        client.delete("/personal/schedule").raise_for_status()
        todos = client.get("/personal/todo", params={"status": "all"})
        if todos.status_code == 200:
            payload = todos.json()
            items = payload.get("todos", []) if isinstance(payload, dict) else payload
            for item in items:
                client.delete(f"/personal/todo/{item['id']}")

    today = date.today()
    courses = [
        {"course": "智能系统导论", "day_of_week": today.weekday(), "start_time": "10:10", "end_time": "11:55", "location": "南校区B楼-203", "weeks": []},
        {"course": "计算机网络", "day_of_week": (today + timedelta(days=1)).weekday(), "start_time": "08:30", "end_time": "10:05", "location": "南校区A楼-101", "weeks": []},
    ]
    client.post("/personal/schedule/import", json={"courses": courses}).raise_for_status()
    client.post("/personal/todo", json={"content": "提交 Agent 实验报告", "kind": "ddl", "due_at": (today + timedelta(days=5)).isoformat()}).raise_for_status()


def run_case(client: httpx.Client, case: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    conv_id = f"demo-{case['id']}-{strategy}-{time.time_ns()}"
    headers = {"X-EchoGuide-Benchmark-Strategy": strategy}
    if case.get("prelude"):
        first = client.post("/chat", headers=headers, json={"message": case["prelude"], "conv_id": conv_id})
        first.raise_for_status()
    started = time.perf_counter()
    response = client.post("/chat", headers=headers, json={"message": case["question"], "conv_id": conv_id})
    elapsed_ms = (time.perf_counter() - started) * 1000
    expected_status = int(case.get("expected_status", 200))
    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    if response.status_code != expected_status:
        return {"case_id": case["id"], "strategy": strategy, "ok": False, "status": response.status_code, "error": body, "latency_ms": elapsed_ms}
    if expected_status != 200:
        return {"case_id": case["id"], "strategy": strategy, "ok": True, "status": response.status_code, "latency_ms": elapsed_ms}

    execution = body.get("execution") or {}
    tools = execution.get("tools") or []
    checks = {
        "domain": body.get("domain") == case.get("expected_domain"),
        "profile": execution.get("profile") == case.get("expected_profile"),
        "mode": execution.get("mode") == case.get("expected_mode"),
        "tools": set(case.get("required_tools", [])).issubset(tools),
    }
    if case.get("expected_classifier_stage"):
        checks["classifier_stage"] = execution.get("classifier_stage") == case["expected_classifier_stage"]
    if case.get("expected_mode") == "dependent":
        tasks = execution.get("tasks") or []
        # 通用 DAG 断言：至少一个任务带依赖且全部任务成功。
        # 不硬编码任务 id 命名/数量，避免与 Planner 实现细节强耦合。
        checks["dag"] = (
            len(tasks) >= 2
            and all(task.get("status") == "success" for task in tasks)
            and any(task.get("depends_on") for task in tasks)
        )
    if case.get("expected_citation_domain"):
        # 引用检查的域名来自用例配置（与数据源 source_url 对应），
        # 不硬编码在代码里，数据/模型措辞变化时只需改 demo_cases.json。
        answer = str(body.get("response", ""))
        checks["citation"] = bool(body.get("knowledge_used")) and case["expected_citation_domain"] in answer
    return {
        "case_id": case["id"], "strategy": strategy, "ok": all(checks.values()),
        "status": response.status_code, "checks": checks, "domain": body.get("domain"),
        "execution": execution, "latency_ms": float(body.get("latency_ms") or elapsed_ms),
        "knowledge_used": bool(body.get("knowledge_used")),
        "answer_preview": str(body.get("response", ""))[:240],
    }


def aggregate(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    # 调用方传入生成器（逐 strategy 过滤），必须先物化再多次遍历
    records = list(records)
    rows = [row for row in records if row.get("status") == 200]
    latencies = [float(row.get("latency_ms", 0)) for row in rows]
    executions = [row.get("execution") or {} for row in rows]
    labels = sorted({CASE_BY_ID[row["case_id"]].get("expected_domain") for row in rows if CASE_BY_ID[row["case_id"]].get("expected_domain")})
    f1_values = []
    for label in labels:
        tp = sum(row.get("domain") == label and CASE_BY_ID[row["case_id"]].get("expected_domain") == label for row in rows)
        fp = sum(row.get("domain") == label and CASE_BY_ID[row["case_id"]].get("expected_domain") != label for row in rows)
        fn = sum(row.get("domain") != label and CASE_BY_ID[row["case_id"]].get("expected_domain") == label for row in rows)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1_values.append(2 * precision * recall / max(1e-12, precision + recall))
    expected_complex = [CASE_BY_ID[row["case_id"]].get("expected_mode") != "single" for row in rows]
    predicted_complex = [(row.get("execution") or {}).get("mode") != "single" for row in rows]
    gate_tp = sum(expected and predicted for expected, predicted in zip(expected_complex, predicted_complex))
    specialized = [row for row in rows if row["case_id"] in {"weighted_score", "affairs_card", "it_network"}]
    dag_rows = [row for row in rows if row["case_id"] == "dependent_dag"]
    citation_rows = [row for row in rows if row["case_id"] == "academic_policy"]
    return {
        "cases": len(rows),
        # 通过率分母为全部记录（含期望非 200 的安全用例，如 Guard 403）：
        # 拦截成功同样算作通过，避免"只罚不奖"的不对称口径。
        "pass_rate": round(sum(bool(row.get("ok")) for row in records) / max(1, len(records)), 4),
        "domain_accuracy": round(sum((row.get("checks") or {}).get("domain", False) for row in rows) / max(1, len(rows)), 4),
        "domain_macro_f1": round(sum(f1_values) / max(1, len(f1_values)), 4),
        "profile_accuracy": round(sum((row.get("checks") or {}).get("profile", False) for row in rows) / max(1, len(rows)), 4),
        "complexity_accuracy": round(sum((row.get("checks") or {}).get("mode", False) for row in rows) / max(1, len(rows)), 4),
        "complexity_precision": round(gate_tp / max(1, sum(predicted_complex)), 4),
        "complexity_recall": round(gate_tp / max(1, sum(expected_complex)), 4),
        "tool_success_rate": round(sum((row.get("checks") or {}).get("tools", False) for row in rows) / max(1, len(rows)), 4),
        "specialized_tool_success_rate": round(sum((row.get("checks") or {}).get("tools", False) for row in specialized) / max(1, len(specialized)), 4),
        "dag_success_rate": round(sum((row.get("checks") or {}).get("dag", False) for row in dag_rows) / max(1, len(dag_rows)), 4),
        "citation_correctness": round(sum((row.get("checks") or {}).get("citation", False) for row in citation_rows) / max(1, len(citation_rows)), 4),
        "llm_classifier_rate": round(sum(exe.get("classifier_stage") == "llm" for exe in executions) / max(1, len(executions)), 4),
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": round(percentile(latencies, 0.95), 1),
        "input_tokens": sum(int(exe.get("input_tokens", 0)) for exe in executions),
        "output_tokens": sum(int(exe.get("output_tokens", 0)) for exe in executions),
    }


def measure_rag(client: httpx.Client) -> Dict[str, Any]:
    """用公开政策标题作为唯一相关文档，实测 Top-5 HitRate/Recall/MRR。"""
    query = "西电转专业报名审核考核流程"
    response = client.post("/search", params={"query": query, "top_k": 5})
    response.raise_for_status()
    results = response.json().get("results") or []
    ranks = [i for i, item in enumerate(results, 1) if "转专业公开实施方案" in str(item.get("title", ""))]
    rank = ranks[0] if ranks else 0
    return {
        "query": query,
        "hit_rate_at_5": 1.0 if rank else 0.0,
        "recall_at_5": 1.0 if rank else 0.0,
        "mrr": round(1.0 / rank, 4) if rank else 0.0,
        "rank": rank,
        "returned": len(results),
    }


def markdown_block(report: Dict[str, Any]) -> str:
    adaptive = report["summary"]["adaptive"]
    baseline = report["summary"].get("always_llm_deep", {})
    return "\n".join([
        "<!-- BENCHMARK:START -->",
        f"> 实测时间：{report['generated_at']} · Commit `{report['commit']}` · 每场景重复 {report['repeat']} 次",
        "",
        "| 指标 | 自适应链路 | Always-LLM + Always-Deep 基线 |",
        "|---|---:|---:|",
        f"| 用例通过率 | {adaptive['pass_rate']:.1%} | {baseline.get('pass_rate', 0):.1%} |",
        f"| 领域准确率 | {adaptive['domain_accuracy']:.1%} | {baseline.get('domain_accuracy', 0):.1%} |",
        f"| 领域 Macro-F1 | {adaptive['domain_macro_f1']:.1%} | {baseline.get('domain_macro_f1', 0):.1%} |",
        f"| LLM 分类调用率 | {adaptive['llm_classifier_rate']:.1%} | {baseline.get('llm_classifier_rate', 0):.1%} |",
        f"| Profile 路由准确率 | {adaptive['profile_accuracy']:.1%} | {baseline.get('profile_accuracy', 0):.1%} |",
        f"| 复杂度 Precision / Recall | {adaptive['complexity_precision']:.1%} / {adaptive['complexity_recall']:.1%} | {baseline.get('complexity_precision', 0):.1%} / {baseline.get('complexity_recall', 0):.1%} |",
        f"| 专属工具成功率 | {adaptive['specialized_tool_success_rate']:.1%} | {baseline.get('specialized_tool_success_rate', 0):.1%} |",
        f"| DAG 任务成功率 | {adaptive['dag_success_rate']:.1%} | {baseline.get('dag_success_rate', 0):.1%} |",
        f"| RAG HitRate@5 / Recall@5 / MRR | {report['rag']['hit_rate_at_5']:.1%} / {report['rag']['recall_at_5']:.1%} / {report['rag']['mrr']:.2f} | — |",
        f"| 引用正确率 | {adaptive['citation_correctness']:.1%} | {baseline.get('citation_correctness', 0):.1%} |",
        f"| P50 延迟 | {adaptive['p50_latency_ms']:.0f} ms | {baseline.get('p50_latency_ms', 0):.0f} ms |",
        f"| P95 延迟 | {adaptive['p95_latency_ms']:.0f} ms | {baseline.get('p95_latency_ms', 0):.0f} ms |",
        f"| 输入 / 输出 Token | {adaptive['input_tokens']} / {adaptive['output_tokens']} | {baseline.get('input_tokens', 0)} / {baseline.get('output_tokens', 0)} |",
        "",
        "> 消融：专属工具成功率 {adaptive_specialized}，改用通用 RAG 后为 {generic_specialized}；依赖 DAG 成功率 {adaptive_dag}，强制单 Agent 后为 {single_dag}。".format(
            adaptive_specialized=f"{adaptive['specialized_tool_success_rate']:.1%}",
            generic_specialized=f"{report['summary'].get('generic_rag', {}).get('specialized_tool_success_rate', 0):.1%}",
            adaptive_dag=f"{adaptive['dag_success_rate']:.1%}",
            single_dag=f"{report['summary'].get('single_agent', {}).get('dag_success_rate', 0):.1%}",
        ),
        "<!-- BENCHMARK:END -->",
    ])


def update_readme(report: Dict[str, Any]) -> None:
    text = README_PATH.read_text(encoding="utf-8")
    start, end = "<!-- BENCHMARK:START -->", "<!-- BENCHMARK:END -->"
    if start not in text or end not in text:
        raise RuntimeError("README 缺少 Benchmark 标记")
    before = text.split(start, 1)[0]
    after = text.split(end, 1)[1]
    README_PATH.write_text(before + markdown_block(report) + after, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    repeat = 1 if args.smoke else max(1, min(args.repeat, 5))
    cases = load_cases()
    records: List[Dict[str, Any]] = []
    rag_metrics: Dict[str, Any] = {}
    with httpx.Client(base_url=args.base_url, timeout=120.0, follow_redirects=True) as client:
        login_or_register(client)
        prepare_demo_data(client)
        strategies = ("adaptive",) if args.smoke else ("adaptive", "always_llm_deep")
        for strategy in strategies:
            for _ in range(repeat):
                for case in cases:
                    records.append(run_case(client, case, strategy))
        if not args.smoke:
            for strategy, ids in (("generic_rag", {"weighted_score", "affairs_card", "it_network"}), ("single_agent", {"dependent_dag"})):
                for case in cases:
                    if case["id"] in ids:
                        records.append(run_case(client, case, strategy))
        rag_metrics = measure_rag(client)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "commit": git_revision(),
        "repeat": repeat,
        "models": {"fast": "deepseek-v4-flash", "deep": "deepseek-v4-pro"},
        "rag": rag_metrics,
        "summary": {
            strategy: aggregate(row for row in records if row.get("strategy") == strategy)
            for strategy in ("adaptive", "always_llm_deep", "generic_rag", "single_agent")
        },
        "records": records,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.update_readme:
        update_readme(report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["adaptive"]["pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
