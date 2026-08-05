"""
轻量全链路 Trace —— 不依赖 OpenTelemetry 依赖库的零成本方案。

设计：
  - contextvars 传递当前 Trace，无需把 trace 对象层层传参
  - span(name, **meta) 自动记录耗时（async 版 + sync 版）
  - 环形缓冲保留最近 N 条 trace，供 /traces 接口排障
  - 每个 span 输出一条 JSON 日志，可被日志采集系统聚合

接入点：
  - api /chat、/chat/stream：request 级 trace（X-Trace-Id 响应头）
  - BaseAgent.handle：agent_handle span
  - BaseAgent._execute_tool：tool_call span
  - IntentRecognizer.recognize：intent_recognize span
  - KnowledgeBase.search：kb_search span
"""
import asyncio
import contextvars
import json
import logging
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MAX_TRACES = 200          # 环形缓冲容量
TRACE_TTL_S = 3600        # trace 保留时长

_current: contextvars.ContextVar = contextvars.ContextVar("echoguide_trace", default=None)

_traces: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()


class Span:
    __slots__ = ("name", "started", "duration_ms", "meta")

    def __init__(self, name: str, **meta: Any):
        self.name = name
        self.started = time.monotonic()
        self.duration_ms = 0.0
        self.meta = meta

    def finish(self) -> None:
        self.duration_ms = (time.monotonic() - self.started) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "meta": {k: str(v)[:120] for k, v in self.meta.items()},
        }


class Trace:
    def __init__(self, name: str = "request"):
        self.trace_id = uuid.uuid4().hex[:12]
        self.name = name
        self.started = time.monotonic()
        self.created = time.time()
        self.spans: List[Span] = []
        self.tags: Dict[str, Any] = {}

    @property
    def duration_ms(self) -> float:
        return (time.monotonic() - self.started) * 1000

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "ts": self.created,
            "duration_ms": round(self.duration_ms, 2),
            "tags": {k: str(v)[:200] for k, v in self.tags.items()},
            "spans": [span.to_dict() for span in self.spans],
        }


def begin_trace(name: str = "request") -> Trace:
    """开启新 trace 并绑定到当前上下文。"""
    trace = Trace(name)
    _current.set(trace)
    return trace


def end_trace() -> Optional[Trace]:
    """结束当前 trace：输出 JSON 日志并存入环形缓冲。"""
    trace = _current.get()
    if trace is None:
        return None
    _current.set(None)
    record = trace.to_dict()
    logger.info(f"[trace] {json.dumps(record, ensure_ascii=False)}")
    _traces[trace.trace_id] = record
    # 环形缓冲：超容量清最旧；超 TTL 的旧 trace 一并清理
    while len(_traces) > MAX_TRACES:
        _traces.pop(next(iter(_traces)))
    now = time.time()
    for tid, rec in list(_traces.items()):
        if now - float(rec.get("ts", now)) > TRACE_TTL_S:
            _traces.pop(tid, None)
    return trace


def current_trace() -> Optional[Trace]:
    return _current.get()


@asynccontextmanager
async def span(name: str, **meta: Any):
    """异步 span：记录耗时与元数据。"""
    trace = _current.get()
    if trace is None:
        yield None
        return
    s = Span(name, **meta)
    trace.spans.append(s)
    try:
        yield s
    finally:
        s.finish()


@contextmanager
def sync_span(name: str, **meta: Any):
    """同步 span（用于知识库检索等同步操作）。"""
    trace = _current.get()
    if trace is None:
        yield None
        return
    s = Span(name, **meta)
    trace.spans.append(s)
    try:
        yield s
    finally:
        s.finish()


def get_trace(trace_id: str) -> Optional[Dict[str, Any]]:
    return _traces.get(trace_id)


def list_traces(limit: int = 20) -> List[Dict[str, Any]]:
    return list(_traces.values())[-limit:]
