"""轻量 Trace 测试：span 记录、contextvars 隔离、环形缓冲。"""
from __future__ import annotations

import asyncio

from core.tracing import begin_trace, end_trace, list_traces, span, sync_span


def test_trace_records_spans_with_duration():
    trace = begin_trace("test")
    try:
        with sync_span("sync_op", key="v"):
            pass

        async def async_op():
            async with span("async_op", agent="academic"):
                await asyncio.sleep(0.01)

        asyncio.run(async_op())
    finally:
        record = end_trace()

    assert record is not None
    record_dict = record.to_dict()
    names = [s["name"] for s in record_dict["spans"]]
    assert names == ["sync_op", "async_op"]
    assert record_dict["spans"][1]["duration_ms"] >= 5
    assert record_dict["spans"][0]["meta"]["key"] == "v"
    assert "trace_id" in record_dict


def test_span_without_trace_is_noop():
    # 未 begin_trace 时，span 不应报错
    async def run():
        async with span("orphan"):
            pass

    asyncio.run(run())
    with sync_span("orphan_sync"):
        pass
    assert True


def test_traces_buffer_returns_recent():
    begin_trace("a")
    end_trace()
    traces = list_traces(limit=10)
    assert len(traces) >= 1
    assert traces[-1]["name"] == "a"


def test_begin_trace_resets_context():
    begin_trace("first")
    trace2 = begin_trace("second")
    record2 = end_trace()
    assert record2.trace_id == trace2.trace_id
    # 结束后再结束应为 None（上下文已清空）
    assert end_trace() is None
