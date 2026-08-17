"""EndToEndEvaluator 质量不达标 / Judge 失败日志测试。

覆盖两类日志契约：
  - [Eval] 质量不达标：主评分低于及格线（或扩展指标低分）时，问题/回答/分数落日志
  - [Eval] Judge 失败：Judge 自身故障单独记录，不得误报为质量退化
"""
import asyncio
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

_ROOT = str(Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from evaluation.evaluator import EndToEndEvaluator, LLMJudge, QualityScores

_LOG_LOGGER = "evaluation.evaluator"


class _FakeIntent:
    value = "query"


class _FakeOrchResult:
    """orchestrator.run 的返回替身，携带评测需要的字段。"""

    def __init__(self, response: str, tool_evidence=None):
        self.response = response
        self.agent_type = "campus_life"  # 展示标签（str）
        self.intent = _FakeIntent()
        self.tool_evidence = tool_evidence or []


def _make_evaluator(judge_result=None, tool_evidence=None):
    """构造不真正调用 LLM 的评测器；judge_result 给定时替换 judge.judge。"""
    orchestrator = MagicMock()
    orchestrator.run = AsyncMock(return_value=_FakeOrchResult(
        "食堂晚上七点关门，次日六点开门。", tool_evidence=tool_evidence,
    ))
    evaluator = EndToEndEvaluator(
        orchestrator=orchestrator,
        recognizer=None,
        api_key="test-key",
        judge_model="judge-test",
    )
    if judge_result is not None:
        evaluator._judge.judge = AsyncMock(return_value=judge_result)
    return evaluator


def test_low_quality_logs_question_scores_and_response(caplog):
    """主评分不达标：问题、综合分、四维低分指标、Agent 类型与回答都要落日志。"""
    low = QualityScores(relevance=0.3, accuracy=0.2, completeness=0.4, helpfulness=0.5)
    evaluator = _make_evaluator(low)
    with caplog.at_level(logging.WARNING, logger=_LOG_LOGGER):
        results = asyncio.run(evaluator._evaluate_dialog_case({"question": "南校区食堂几点关门？"}, 0))
    assert results[0].passed is False
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[Eval] 质量不达标" in text
    assert "question='南校区食堂几点关门？'" in text
    assert "overall=0.350" in text
    assert "低分指标=" in text and "0.2" in text          # accuracy=0.2 在低分集合中
    assert "agent_type=campus_life" in text
    assert "judge_model=judge-test" in text
    assert "response=" in text                            # Agent 回答已落日志


def test_judge_failure_logged_separately(caplog):
    """Judge 自身失败：单独记录，不得误报为质量不达标。"""
    failed = QualityScores(0.5, 0.5, 0.5, 0.5, judge_failed=True, error="Judge 连续 2 次输出无法解析")
    evaluator = _make_evaluator(failed)
    with caplog.at_level(logging.WARNING, logger=_LOG_LOGGER):
        results = asyncio.run(evaluator._evaluate_dialog_case({"question": "教务系统登录不上怎么办？"}, 0))
    assert results[0].passed is False
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[Eval] Judge 失败" in text
    assert "error=Judge 连续 2 次输出无法解析" in text
    assert "question='教务系统登录不上怎么办？'" in text
    assert "[Eval] 质量不达标" not in text


def test_extension_metric_low_score_logged_even_if_main_passes(caplog):
    """主评分达标但扩展指标（Faithfulness）低分：同样记录，捕获幻觉等主评分发现不了的问题。"""
    good = QualityScores(relevance=0.9, accuracy=0.9, completeness=0.9, helpfulness=0.9)
    evaluator = _make_evaluator(good, tool_evidence=[
        {"title": "食堂与餐饮", "content": "南校区食堂晚上七点关门。"},
    ])
    evaluator._retrieval_evaluator = object()  # 仅需非 None 以启用扩展指标评测
    evaluator._judge.judge_faithfulness = AsyncMock(return_value=(0.4, False))
    with caplog.at_level(logging.WARNING, logger=_LOG_LOGGER):
        results = asyncio.run(evaluator._evaluate_dialog_case({"question": "南校区食堂几点关门？"}, 0))
    assert results[0].passed is True
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[Eval] 质量不达标" in text
    assert "faithfulness" in text and "0.4" in text


def test_judge_retry_log_includes_question(caplog):
    """Judge 调用失败重试：日志必须带问题片段，便于定位是哪条用例。"""
    client = MagicMock()
    client.messages.create = AsyncMock(side_effect=RuntimeError("API 不可用"))
    judge = LLMJudge(client, "judge-test")
    with caplog.at_level(logging.WARNING, logger=_LOG_LOGGER):
        result = asyncio.run(judge.judge("食堂几点关门？", "晚上七点"))
    assert result.judge_failed is True
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "LLM Judge 第 1 次失败" in text
    assert "question='食堂几点关门？'" in text
