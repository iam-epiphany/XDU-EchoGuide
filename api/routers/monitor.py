"""可观测性路由：/monitor、/metrics、/traces 与评测入口 /eval/run。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field, model_validator
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from api import state
from api.deps import require_admin, require_observability

router = APIRouter(tags=["观测"])


@router.get("/monitor")
async def monitor_summary(_admin=Depends(require_observability)):
    """实时监控摘要：Agent 成功率、工具统计、告警。"""
    if state._monitor is None:
        raise HTTPException(503, "服务未就绪")
    return state._monitor.summary()


@router.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus 指标入口（只读，无认证）。

    安全权衡：指标不含敏感数据，供 Prometheus 无认证抓取（config/prometheus.yml
    已按此配置）；生产环境应通过网络层（防火墙/内网）限制 /metrics 的暴露面。
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/traces")
async def traces_list(limit: int = 20, _admin=Depends(require_observability)):
    """最近的全链路 trace（排障/演示用）。"""
    from core.tracing import list_traces

    return {"traces": list_traces(limit=max(1, min(limit, 200)))}


@router.get("/traces/{trace_id}")
async def trace_detail(trace_id: str, _admin=Depends(require_observability)):
    """单条 trace 详情：request → intent → agent → tool → LLM 逐跳耗时。"""
    from core.tracing import get_trace

    record = get_trace(trace_id)
    if record is None:
        raise HTTPException(404, f"trace 不存在: {trace_id}")
    return record


# ── 评测 ──────────────────────────────────────────────────────────────────────

class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str = Field(min_length=1, max_length=2000)
    expected_intent: str = Field(min_length=1, max_length=80)
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮；可选 golden_answer（Answer Correctness）。"""
    question: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    turns: Optional[List[str]] = Field(default=None, min_length=1, max_length=12)
    user_id: Optional[str] = Field(default=None, max_length=64)
    conv_id: Optional[str] = Field(default=None, max_length=80)
    golden_answer: Optional[str] = Field(default=None, max_length=10000)

    @model_validator(mode="after")
    def require_question_or_turns(self):
        if not self.question and not self.turns:
            raise ValueError("question 或 turns 至少提供一个")
        return self


class RetrievalCaseInput(BaseModel):
    """RAG 检索硬指标评测用例。"""
    query: str = Field(min_length=1, max_length=500)
    relevant_titles: List[str] = Field(min_length=1, max_length=20)


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None
    routing_cases: Optional[List[Dict[str, Any]]] = None
    retrieval_cases: Optional[List[RetrievalCaseInput]] = None
    promote_baseline: bool = False


@router.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None, _admin=Depends(require_admin)):
    """运行内置评测用例，返回评测报告。"""
    if state._evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import (
        DEFAULT_DIALOG_CASES,
        DEFAULT_INTENT_CASES,
        DEFAULT_RETRIEVAL_CASES,
        DEFAULT_ROUTING_CASES,
        IntentTestCase,
        RetrievalTestCase,
    )

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    routing_cases = body.routing_cases if body and body.routing_cases is not None else DEFAULT_ROUTING_CASES

    if body and body.retrieval_cases is not None:
        retrieval_cases = [
            RetrievalTestCase(query=c.query, relevant_titles=c.relevant_titles)
            for c in body.retrieval_cases
        ]
    else:
        retrieval_cases = DEFAULT_RETRIEVAL_CASES

    custom_cases = bool(
        body
        and (
            body.intent_cases is not None
            or body.dialog_cases is not None
            or body.routing_cases is not None
            or body.retrieval_cases is not None
        )
    )
    report = await state._evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
        routing_cases=routing_cases,
        retrieval_cases=retrieval_cases,
        promote_baseline=bool(body and body.promote_baseline),
        dataset="custom_cases" if custom_cases else "built_in_cases_v1",
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "retrieval":       report.retrieval,
        "provenance":      report.provenance,
        "judge":           report.judge,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }
