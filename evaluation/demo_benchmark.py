"""真实 HTTP Demo Benchmark：准备专用用户、运行用例并更新 README 指标块。"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import re
import statistics
import subprocess
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

import httpx


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "evaluation" / "demo_cases.json"
RAG_CASES_PATH = ROOT / "evaluation" / "cases" / "http_rag_cases.json"
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
    try:
        if case.get("prelude"):
            first = client.post("/chat", headers=headers, json={"message": case["prelude"], "conv_id": conv_id})
            first.raise_for_status()
        started = time.perf_counter()
        response = client.post("/chat", headers=headers, json={"message": case["question"], "conv_id": conv_id})
        elapsed_ms = (time.perf_counter() - started) * 1000
    except httpx.HTTPError as ex:
        # 超时/连接错误是实测结果的一部分；记录失败并继续下一条，不能让单个
        # 外部依赖卡住整轮 Benchmark，也不能把该请求从通过率分母中悄悄移除。
        return {
            "case_id": case["id"], "strategy": strategy, "ok": False,
            "status": 0, "error": f"{type(ex).__name__}: {ex}",
            "latency_ms": (time.perf_counter() - started) * 1000 if "started" in locals() else 0.0,
        }
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
        # profile_policy 是复杂度策略的原始选择；profile 是实际模型，可被
        # Monitor 在 Fast 健康异常时临时升级为 Deep。二者分别统计，避免把
        # 故障转移误判成分类错误。
        "profile": execution.get("profile_policy", execution.get("profile")) == case.get("expected_profile"),
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
    elif case.get("expect_citation"):
        # 对未绑定单一官网域名的知识问答，检查执行链确实使用了知识库且最终答案
        # 留下编号引用；这样既覆盖多来源文档，也不把某个 URL 写进评测代码。
        answer = str(body.get("response", ""))
        checks["citation"] = bool(body.get("knowledge_used")) and bool(re.search(r"\[\d+\]", answer))
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
    def tagged(tag: str) -> List[Dict[str, Any]]:
        return [row for row in rows if tag in CASE_BY_ID[row["case_id"]].get("tags", [])]

    def rate(group: List[Dict[str, Any]], check: str) -> float | None:
        if not group:
            return None
        return round(sum((row.get("checks") or {}).get(check, False) for row in group) / len(group), 4)

    specialized = tagged("specialized_tool")
    dag_rows = tagged("dag")
    citation_rows = tagged("citation")
    scenario_count = len({row["case_id"] for row in records})
    return {
        "cases": len(rows),
        "scenario_count": scenario_count,
        # 通过率分母为全部记录（含期望非 200 的安全用例，如 Guard 403）：
        # 拦截成功同样算作通过，避免"只罚不奖"的不对称口径。
        "pass_rate": round(sum(bool(row.get("ok")) for row in records) / max(1, len(records)), 4),
        "domain_accuracy": round(sum((row.get("checks") or {}).get("domain", False) for row in rows) / max(1, len(rows)), 4),
        "domain_macro_f1": round(sum(f1_values) / max(1, len(f1_values)), 4),
        "profile_accuracy": round(sum((row.get("checks") or {}).get("profile", False) for row in rows) / max(1, len(rows)), 4),
        "runtime_deep_fallback_rate": round(sum(
            (row.get("execution") or {}).get("profile_policy") == "fast"
            and (row.get("execution") or {}).get("profile") == "deep"
            for row in rows
        ) / max(1, len(rows)), 4),
        "complexity_accuracy": round(sum((row.get("checks") or {}).get("mode", False) for row in rows) / max(1, len(rows)), 4),
        "complexity_precision": round(gate_tp / max(1, sum(predicted_complex)), 4),
        "complexity_recall": round(gate_tp / max(1, sum(expected_complex)), 4),
        "tool_success_rate": round(sum((row.get("checks") or {}).get("tools", False) for row in rows) / max(1, len(rows)), 4),
        "specialized_tool_success_rate": rate(specialized, "tools"),
        "dag_success_rate": rate(dag_rows, "dag"),
        "citation_correctness": rate(citation_rows, "citation"),
        "metric_sample_counts": {
            "specialized_tool": len({row["case_id"] for row in specialized}),
            "dag": len({row["case_id"] for row in dag_rows}),
            "citation": len({row["case_id"] for row in citation_rows}),
        },
        "llm_classifier_rate": round(sum(exe.get("classifier_stage") == "llm" for exe in executions) / max(1, len(executions)), 4),
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0.0,
        "p95_latency_ms": round(percentile(latencies, 0.95), 1),
        "input_tokens": sum(int(exe.get("input_tokens", 0)) for exe in executions),
        "output_tokens": sum(int(exe.get("output_tokens", 0)) for exe in executions),
    }


def measure_rag(client: httpx.Client) -> Dict[str, Any]:
    """对版本化 HTTP RAG 探针逐条请求，避免单一查询的 100% 误导性。"""
    payload = json.loads(RAG_CASES_PATH.read_text(encoding="utf-8"))
    cases = payload["cases"]
    records = []
    for case in cases:
        try:
            response = client.post("/search", params={"query": case["query"], "top_k": 5})
            response.raise_for_status()
            results = response.json().get("results") or []
        except httpx.HTTPError:
            results = []
        titles = [str(item.get("title", "")) for item in results]
        expected = case["relevant_titles"]
        matched_targets = {
            target for target in expected
            if any(target in title or title in target for title in titles)
        }
        rank = next(
            (index for index, title in enumerate(titles, 1)
             if any(target in title or title in target for target in expected)),
            0,
        )
        records.append({
            "case_id": case["id"],
            "query": case["query"],
            "expected_titles": expected,
            "returned_titles": titles,
            "rank": rank,
            "hit": bool(rank),
            "recall": round(len(matched_targets) / len(expected), 4),
        })
    return {
        "cases": len(records),
        "hit_rate_at_5": round(sum(row["hit"] for row in records) / max(1, len(records)), 4),
        "recall_at_5": round(sum(row["recall"] for row in records) / max(1, len(records)), 4),
        "mrr": round(sum(1.0 / row["rank"] if row["rank"] else 0.0 for row in records) / max(1, len(records)), 4),
        "records": records,
    }


def markdown_block(report: Dict[str, Any]) -> str:
    adaptive = report["summary"]["adaptive"]
    baseline = report["summary"].get("always_llm_deep", {})
    def percentage(value: Any) -> str:
        return f"{float(value or 0):.1%}"
    samples = adaptive.get("metric_sample_counts", {})
    return "\n".join([
        "<!-- BENCHMARK:START -->",
        f"> 实测时间：{report['generated_at']} · Commit `{report['commit']}` · 每场景重复 {report['repeat']} 次",
        f"> 版本化 HTTP 场景：{adaptive.get('scenario_count', 0)} 个；RAG 探针：{report['rag'].get('cases', 0)} 条。指标后的 n 为独立场景数。",
        "",
        "| 指标 | 自适应链路 | Always-LLM + Always-Deep 基线 |",
        "|---|---:|---:|",
        f"| 用例通过率 | {percentage(adaptive['pass_rate'])} | {percentage(baseline.get('pass_rate'))} |",
        f"| 领域准确率 | {percentage(adaptive['domain_accuracy'])} | {percentage(baseline.get('domain_accuracy'))} |",
        f"| 领域 Macro-F1 | {percentage(adaptive['domain_macro_f1'])} | {percentage(baseline.get('domain_macro_f1'))} |",
        f"| LLM 分类调用率 | {percentage(adaptive['llm_classifier_rate'])} | {percentage(baseline.get('llm_classifier_rate'))} |",
        f"| Profile 路由准确率 | {percentage(adaptive['profile_accuracy'])} | {percentage(baseline.get('profile_accuracy'))} |",
        f"| Fast→Deep 运行时降级率 | {percentage(adaptive['runtime_deep_fallback_rate'])} | {percentage(baseline.get('runtime_deep_fallback_rate'))} |",
        f"| 复杂度 Precision / Recall | {adaptive['complexity_precision']:.1%} / {adaptive['complexity_recall']:.1%} | {baseline.get('complexity_precision', 0):.1%} / {baseline.get('complexity_recall', 0):.1%} |",
        f"| 专属工具成功率（n={samples.get('specialized_tool', 0)}） | {percentage(adaptive['specialized_tool_success_rate'])} | {percentage(baseline.get('specialized_tool_success_rate'))} |",
        f"| DAG 任务成功率（n={samples.get('dag', 0)}） | {percentage(adaptive['dag_success_rate'])} | {percentage(baseline.get('dag_success_rate'))} |",
        f"| RAG HitRate@5 / Recall@5 / MRR（n={report['rag'].get('cases', 0)}） | {report['rag']['hit_rate_at_5']:.1%} / {report['rag']['recall_at_5']:.1%} / {report['rag']['mrr']:.2f} | — |",
        f"| 引用正确率（n={samples.get('citation', 0)}） | {percentage(adaptive['citation_correctness'])} | {percentage(baseline.get('citation_correctness'))} |",
        f"| P50 延迟 | {adaptive['p50_latency_ms']:.0f} ms | {baseline.get('p50_latency_ms', 0):.0f} ms |",
        f"| P95 延迟 | {adaptive['p95_latency_ms']:.0f} ms | {baseline.get('p95_latency_ms', 0):.0f} ms |",
        f"| 输入 / 输出 Token | {adaptive['input_tokens']} / {adaptive['output_tokens']} | {baseline.get('input_tokens', 0)} / {baseline.get('output_tokens', 0)} |",
        "",
        "> 消融：专属工具成功率 {adaptive_specialized}，改用通用 RAG 后为 {generic_specialized}；依赖 DAG 成功率 {adaptive_dag}，强制单 Agent 后为 {single_dag}。".format(
            adaptive_specialized=percentage(adaptive['specialized_tool_success_rate']),
            generic_specialized=percentage(report['summary'].get('generic_rag', {}).get('specialized_tool_success_rate')),
            adaptive_dag=percentage(adaptive['dag_success_rate']),
            single_dag=percentage(report['summary'].get('single_agent', {}).get('dag_success_rate')),
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
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="单个 HTTP 请求超时秒数；超时会记为失败并继续后续用例。")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    repeat = 1 if args.smoke else max(1, min(args.repeat, 5))
    cases = load_cases()
    records: List[Dict[str, Any]] = []
    rag_metrics: Dict[str, Any] = {}
    request_timeout = max(10.0, min(args.timeout, 180.0))
    with httpx.Client(base_url=args.base_url, timeout=request_timeout, follow_redirects=True) as client:
        login_or_register(client)
        prepare_demo_data(client)
        strategies = ("adaptive",) if args.smoke else ("adaptive", "always_llm_deep")
        for strategy in strategies:
            for run_index in range(repeat):
                for case in cases:
                    record = run_case(client, case, strategy)
                    records.append(record)
                    print(
                        f"[{strategy} {run_index + 1}/{repeat}] {case['id']}: "
                        f"{'PASS' if record.get('ok') else 'FAIL'} "
                        f"(HTTP {record.get('status')})",
                        flush=True,
                    )
        if not args.smoke:
            # 消融集按 tags 选择，新增场景后不需要回头修改硬编码 ID 集合。
            for strategy, tag in (("generic_rag", "specialized_tool"), ("single_agent", "dag")):
                for case in cases:
                    if tag in case.get("tags", []):
                        record = run_case(client, case, strategy)
                        records.append(record)
                        print(
                            f"[{strategy}] {case['id']}: "
                            f"{'PASS' if record.get('ok') else 'FAIL'} "
                            f"(HTTP {record.get('status')})",
                            flush=True,
                        )
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
