from __future__ import annotations

import asyncio

import pytest

from tools.academic_tool import calculate_weighted_score_handler
from tools.affairs_tool import query_affairs_process_handler
from tools.it_tool import diagnose_it_issue_handler


def test_weighted_score_is_deterministic_and_not_gpa():
    result = asyncio.run(calculate_weighted_score_handler({"courses": [
        {"name": "高数", "credits": 4, "score": 90},
        {"name": "物理", "credits": 3, "score": 84},
        {"name": "英语", "credits": 2, "score": 92},
    ]}, {}))
    assert result["total_credits"] == 9
    assert result["weighted_score"] == pytest.approx(88.44, abs=0.01)
    assert "不是学校官方 GPA" in result["disclaimer"]


@pytest.mark.parametrize("credits,score", [(0, 80), (2, 101), (2, -1)])
def test_weighted_score_rejects_invalid_values(credits, score):
    with pytest.raises(ValueError):
        asyncio.run(calculate_weighted_score_handler({"courses": [{"credits": credits, "score": score}]}, {}))


def test_affairs_process_returns_versioned_provenance():
    result = asyncio.run(query_affairs_process_handler({"service": "校园卡补办"}, {}))
    assert result["found"] is True
    process = result["processes"][0]
    assert process["materials"] and process["steps"]
    assert process["source_url"].startswith("https://")
    assert process["updated_at"] and process["version"]


def test_affairs_process_unknown_is_explicit():
    result = asyncio.run(query_affairs_process_handler({"service": "未知业务xyz"}, {}))
    assert result["found"] is False
    assert result["available_services"]


def test_it_diagnostic_matches_network_branch():
    result = asyncio.run(diagnose_it_issue_handler({
        "system": "校园网", "symptom": "认证成功但网页打不开",
    }, {}))
    assert result["matched"] is True
    diagnosis = result["diagnosis"]
    assert diagnosis["steps"] and diagnosis["next_branch"]
    assert diagnosis["source_url"].startswith("https://")


def test_it_diagnostic_unknown_falls_back_cleanly():
    result = asyncio.run(diagnose_it_issue_handler({"system": "量子终端", "symptom": "闪烁"}, {}))
    assert result["matched"] is False
