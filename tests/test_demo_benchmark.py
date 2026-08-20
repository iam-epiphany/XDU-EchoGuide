"""真实 HTTP Benchmark 的数据集驱动汇总回归测试。"""
from __future__ import annotations

from evaluation.demo_benchmark import aggregate


def _record(case_id: str, checks: dict[str, bool], mode: str = "single") -> dict:
    return {
        "case_id": case_id,
        "status": 200,
        "ok": all(checks.values()),
        "checks": checks,
        "domain": "academic",
        "execution": {"mode": mode, "classifier_stage": "pattern"},
        "latency_ms": 12.0,
    }


def test_aggregate_uses_case_tags_for_small_metric_denominators():
    """新增 tagged case 后，专属工具/DAG/引用分母无需改评测脚本。"""
    result = aggregate([
        _record("weighted_score_variant", {"domain": True, "profile": True, "mode": True, "tools": True}),
        _record("affairs_leave", {"domain": True, "profile": True, "mode": True, "tools": False}),
        _record("dependent_dag_paraphrase", {"domain": True, "profile": True, "mode": True, "tools": True, "dag": True}, "dependent"),
        _record("academic_course_policy", {"domain": True, "profile": True, "mode": True, "tools": True, "citation": True}),
    ])

    assert result["metric_sample_counts"] == {
        "specialized_tool": 2,
        "dag": 1,
        "citation": 1,
    }
    assert result["specialized_tool_success_rate"] == 0.5
    assert result["dag_success_rate"] == 1.0
    assert result["citation_correctness"] == 1.0
    assert result["scenario_count"] == 4
