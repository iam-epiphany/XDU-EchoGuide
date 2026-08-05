"""EchoGuide Guard ASGI sidecar.

The service proxies four observable surfaces from AgentRange:

* OpsPilot ingress, to bind a verified actor and original task to a trace.
* OpenAI-compatible chat completions, to observe plans and untrusted context.
* MCP JSON-RPC calls, to enforce policy before a tool handler runs.
* Langflow ingress, to stop the two framework exploit classes before dispatch.

The detector intentionally treats trace/instance IDs as opaque correlation keys.
It never parses scenario prefixes and never accesses the LLM stub's admin API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import httpx
import jwt
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from echoguide_guard.config import Settings
from echoguide_guard.models import Actor, PolicyDecision
from echoguide_guard.policy import PolicyEngine
from echoguide_guard.redaction import redact_data
from echoguide_guard.runtime import TraceRegistry
from echoguide_guard.static_scan import StaticScanner
from echoguide_guard.store import AuditStore

logger = logging.getLogger(__name__)

MAX_PROXY_BODY = 2 * 1024 * 1024
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "content-encoding",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}

DECISIONS = Counter(
    "echoguide_guard_policy_decisions_total",
    "Runtime policy decisions",
    ("surface", "decision"),
)
PROXY_REQUESTS = Counter(
    "echoguide_guard_proxy_requests_total",
    "Requests observed by EchoGuide Guard",
    ("surface", "status"),
)
DECISION_LATENCY = Histogram(
    "echoguide_guard_policy_decision_seconds",
    "Synchronous policy decision latency",
    ("surface",),
)


class GuardState:
    def __init__(self, settings: Settings):
        settings.ensure_runtime_directory()
        self.settings = settings
        self.registry = TraceRegistry(ttl_s=settings.trace_ttl_s)
        self.policy = PolicyEngine()
        self.store = AuditStore(settings.database_path)
        self.client = httpx.AsyncClient(timeout=settings.request_timeout_s, follow_redirects=False)

    async def close(self) -> None:
        await self.client.aclose()
        self.store.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    app.state.guard = GuardState(settings)
    target = Path(settings.target_source)
    if target.exists():
        try:
            report = StaticScanner(str(target)).scan()
            app.state.guard.store.save_scan(report)
            logger.info("EchoGuide Guard initial asset scan completed: %s", report.get("summary", {}))
        except Exception as exc:  # startup must not break runtime protection
            logger.warning("initial asset scan failed: %s", exc)
    yield
    await app.state.guard.close()


app = FastAPI(
    title="EchoGuide Guard Agent Runtime Security",
    version="0.1.0",
    lifespan=lifespan,
)


def _state(request: Request) -> GuardState:
    return request.app.state.guard


def _trace_id(request: Request) -> str:
    # Opaque correlation only. Do not split, classify, or score this value.
    return request.headers.get("x-trace-id") or str(uuid.uuid4())


def _request_headers(request: Request, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }
    if extra:
        headers.update(extra)
    return headers


def _response_headers(response: httpx.Response) -> Dict[str, str]:
    return {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


async def _body(request: Request) -> bytes:
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_PROXY_BODY:
        raise HTTPException(status_code=413, detail="proxy request body is too large")
    data = await request.body()
    if len(data) > MAX_PROXY_BODY:
        raise HTTPException(status_code=413, detail="proxy request body is too large")
    return data


def _json(data: bytes) -> Dict[str, Any]:
    if not data:
        return {}
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _actor_from_request(request: Request, settings: Settings) -> Optional[Actor]:
    authorization = request.headers.get("authorization", "")
    if not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except Exception:
        return None
    return Actor(
        sub=str(payload.get("sub") or "unknown"),
        role=str(payload.get("role") or "unknown"),
        team=str(payload.get("team") or "unknown"),
        scope=dict(payload.get("scope") or {}),
    )


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def _skill_allowed_tools(source: str, skill_name: Optional[str]) -> list[str]:
    if not skill_name or not re.fullmatch(r"[A-Za-z0-9_.-]+", skill_name):
        return []
    relative = f"skills/{skill_name}/SKILL.md"
    path = Path(source)
    try:
        if path.is_file() and path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                text = archive.read(relative).decode("utf-8")
        else:
            text = (path / relative).read_text(encoding="utf-8")
    except (OSError, KeyError, zipfile.BadZipFile):
        return []
    match = re.search(r"(?mi)^allowed[-_]tools\s*:\s*\[([^]]*)\]", text)
    if not match:
        return []
    return [item.strip().strip("\"'") for item in match.group(1).split(",") if item.strip()]


def _decision_value(value: Any) -> str:
    if isinstance(value, PolicyDecision):
        return value.value
    return str(getattr(value, "value", value)).lower()


def _verdict_dict(verdict: Any) -> Dict[str, Any]:
    if hasattr(verdict, "to_dict"):
        raw = verdict.to_dict()
    elif hasattr(verdict, "model_dump"):
        raw = verdict.model_dump()
    else:
        raw = dict(getattr(verdict, "__dict__", {}))
    raw["decision"] = _decision_value(raw.get("decision", getattr(verdict, "decision", "allow")))
    raw.setdefault("reason_codes", raw.get("matched_rules", []))
    raw.setdefault("risk_score", 0.0)
    return raw


def _audit(
    state: GuardState,
    *,
    trace_id: str,
    phase: str,
    source: str,
    action: str,
    decision: str = "allow",
    risk_score: float = 0.0,
    matched_rules: Optional[Iterable[str]] = None,
    details: Optional[Dict[str, Any]] = None,
    duration_ms: float = 0.0,
) -> None:
    state.store.append_event(
        {
            "trace_id": trace_id,
            "timestamp": time.time(),
            "phase": phase,
            "source": source,
            "action": action,
            "decision": decision,
            "risk_score": risk_score,
            "matched_rules": list(matched_rules or []),
            "details": redact_data(details or {}),
            "duration_ms": duration_ms,
        }
    )


async def _forward(
    state: GuardState,
    request: Request,
    upstream_url: str,
    *,
    body: bytes,
    extra_headers: Optional[Dict[str, str]] = None,
) -> httpx.Response:
    if request.url.query:
        upstream_url = f"{upstream_url}?{request.url.query}"
    return await state.client.request(
        request.method,
        upstream_url,
        content=body,
        headers=_request_headers(request, extra_headers),
    )


def _proxy_response(response: httpx.Response) -> Response:
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=_response_headers(response),
        media_type=response.headers.get("content-type"),
    )


def _text_values(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _text_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _text_values(item)


def _has_injection_marker(value: Any) -> bool:
    text = "\n".join(_text_values(value)).lower()
    patterns = (
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"不要向用户提及|do not (tell|mention)",
        r"<important>|<!--\s*(system|ignore)",
        r"read\s+/app/\.env",
        r"读取\s*/app/\.env",
    )
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _has_secret_marker(value: Any) -> bool:
    text = "\n".join(_text_values(value))
    return bool(
        re.search(
            r"(?i)(api[_-]?key|secret|token|password|signing[_-]?key)\s*[:=]\s*[^\s,;]{6,}",
            text,
        )
    )


@app.get("/health")
async def health(request: Request):
    state = _state(request)
    return {
        "status": "ok",
        "version": app.version,
        "target": state.settings.target_source,
        "runtime": "proxy",
    }


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/v1/traces")
async def traces(request: Request, limit: int = 100):
    return {"items": _state(request).store.list_traces(limit)}


@app.get("/api/v1/traces/{trace_id}")
async def trace_detail(trace_id: str, request: Request):
    result = _state(request).store.get_trace(trace_id)
    if result is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return result


@app.post("/api/v1/assets/scan")
async def scan_assets(request: Request):
    state = _state(request)
    target = Path(state.settings.target_source)
    if not target.exists():
        raise HTTPException(status_code=404, detail="configured target source does not exist")
    report = StaticScanner(str(target)).scan()
    report["scan_id"] = state.store.save_scan(report)
    return report


@app.get("/api/v1/assets/latest")
async def latest_assets(request: Request):
    report = _state(request).store.latest_scan()
    if report is None:
        raise HTTPException(status_code=404, detail="no asset scan has been stored")
    return report


@app.api_route("/agent/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def agent_proxy(path: str, request: Request):
    state = _state(request)
    body = await _body(request)
    trace_id = _trace_id(request)
    payload = _json(body)

    if path.strip("/") == "run" and request.method == "POST":
        actor = _actor_from_request(request, state.settings)
        prompt = str(payload.get("prompt") or payload.get("input") or "")
        skill = str(payload.get("skill")) if payload.get("skill") else None
        allowed_tools = _skill_allowed_tools(state.settings.target_source, skill)
        state.registry.upsert(
            trace_id,
            actor=actor,
            prompt=prompt,
            selected_skill=skill,
            skill_allowed_tools=allowed_tools,
        )
        _audit(
            state,
            trace_id=trace_id,
            phase="request",
            source="agent-ingress",
            action="run",
            details={
                "actor": actor.to_dict() if actor and hasattr(actor, "to_dict") else getattr(actor, "__dict__", None),
                "prompt_sha256": _hash_text(prompt),
                "prompt_length": len(prompt),
                "selected_skill": skill,
                "skill_allowed_tools": allowed_tools,
            },
        )

    started = time.perf_counter()
    response = await _forward(
        state,
        request,
        f"{state.settings.agent_upstream}/{path.lstrip('/')}",
        body=body,
        extra_headers={"X-Trace-Id": trace_id},
    )
    duration = (time.perf_counter() - started) * 1000
    PROXY_REQUESTS.labels("agent", str(response.status_code)).inc()
    _audit(
        state,
        trace_id=trace_id,
        phase="response",
        source="agent-ingress",
        action=path or "/",
        details={"status_code": response.status_code},
        duration_ms=duration,
    )
    return _proxy_response(response)


@app.post("/v1/chat/completions")
async def llm_proxy(request: Request):
    state = _state(request)
    body = await _body(request)
    payload = _json(body)
    trace_id = _trace_id(request)
    context = state.registry.get(trace_id)
    if context is None:
        context = state.registry.upsert(trace_id)

    messages = payload.get("messages") if isinstance(payload.get("messages"), list) else []
    suspicious_sources = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if _has_injection_marker(content):
            source = f"llm-message:{message.get('role', 'unknown')}:{index}"
            suspicious_sources.append(source)
            state.registry.add_taint(trace_id, "untrusted_instruction")

    _audit(
        state,
        trace_id=trace_id,
        phase="model-input",
        source="openai-proxy",
        action="chat.completions",
        decision="allow",
        risk_score=35.0 if suspicious_sources else 0.0,
        matched_rules=["CONTENT_UNTRUSTED_INSTRUCTION"] if suspicious_sources else [],
        details={
            "message_count": len(messages),
            "model": payload.get("model"),
            "suspicious_sources": suspicious_sources,
        },
    )

    started = time.perf_counter()
    response = await _forward(
        state,
        request,
        f"{state.settings.llm_upstream}/v1/chat/completions",
        body=body,
    )
    duration = (time.perf_counter() - started) * 1000
    PROXY_REQUESTS.labels("llm", str(response.status_code)).inc()

    response_payload = _json(response.content)
    choices = response_payload.get("choices") if isinstance(response_payload.get("choices"), list) else []
    tool_calls = []
    for choice in choices:
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        for call in message.get("tool_calls", []) if isinstance(message, dict) else []:
            if not isinstance(call, dict):
                continue
            function = call.get("function", {})
            tool_calls.append({
                "id": call.get("id"),
                "name": function.get("name") if isinstance(function, dict) else None,
                "arguments": function.get("arguments") if isinstance(function, dict) else None,
            })

    _audit(
        state,
        trace_id=trace_id,
        phase="model-output",
        source="openai-proxy",
        action="tool-plan" if tool_calls else "final-answer",
        details={"status_code": response.status_code, "tool_calls": tool_calls},
        duration_ms=duration,
    )
    return _proxy_response(response)


@app.post("/mcp/{server}")
async def mcp_proxy(server: str, request: Request):
    state = _state(request)
    body = await _body(request)
    payload = _json(body)
    trace_id = _trace_id(request)
    context = state.registry.get(trace_id) or state.registry.upsert(trace_id)
    method = payload.get("method")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}

    if method == "tools/call":
        tool = str(params.get("name") or "")
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        started = time.perf_counter()
        verdict = state.policy.evaluate(context, server, tool, arguments)
        duration = (time.perf_counter() - started) * 1000
        data = _verdict_dict(verdict)
        decision = data["decision"]
        DECISIONS.labels("mcp", decision).inc()
        DECISION_LATENCY.labels("mcp").observe(duration / 1000)
        _audit(
            state,
            trace_id=trace_id,
            phase="tool-preflight",
            source="mcp-proxy",
            action=f"{server}/{tool}",
            decision=decision,
            risk_score=float(data.get("risk_score") or 0.0),
            matched_rules=data.get("reason_codes") or [],
            details={"arguments": arguments, "message": data.get("message")},
            duration_ms=duration,
        )
        if decision in {PolicyDecision.BLOCK.value, PolicyDecision.ASK.value}:
            state.registry.mark_blocked(trace_id)
            code = -32001 if decision == PolicyDecision.BLOCK.value else -32002
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": payload.get("id"),
                    "error": {
                        "code": code,
                        "message": f"EchoGuide Guard decision: {decision}",
                        "data": {
                            "reason_codes": data.get("reason_codes") or [],
                            "risk_score": data.get("risk_score") or 0.0,
                        },
                    },
                }
            )

    try:
        upstream = state.settings.mcp_upstream(server)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    started = time.perf_counter()
    response = await _forward(state, request, upstream, body=body)
    duration = (time.perf_counter() - started) * 1000
    PROXY_REQUESTS.labels("mcp", str(response.status_code)).inc()
    response_payload = _json(response.content)

    result = response_payload.get("result")
    quarantined = False
    matched_rules = []
    if method == "tools/call" and server in ("notice", "calendar"):
        state.registry.add_taint(trace_id, "external_untrusted")
    if _has_injection_marker(result):
        state.registry.add_taint(trace_id, "untrusted_instruction")
        matched_rules.append("MCP_RESULT_PROMPT_INJECTION")
        quarantined = True
    if _has_secret_marker(result):
        state.registry.add_taint(trace_id, "secret")
        matched_rules.append("MCP_RESULT_SECRET")
    if server == "student-db" and isinstance(result, dict) and result.get("rows"):
        state.registry.add_taint(trace_id, "pii")

    _audit(
        state,
        trace_id=trace_id,
        phase="tool-result",
        source="mcp-proxy",
        action=f"{server}/{params.get('name', method)}",
        decision="block" if quarantined else "allow",
        risk_score=70.0 if quarantined else (45.0 if matched_rules else 0.0),
        matched_rules=matched_rules,
        details={
            "status_code": response.status_code,
            "result_sha256": _hash_text(json.dumps(result, ensure_ascii=False, default=str)),
            "quarantined": quarantined,
        },
        duration_ms=duration,
    )

    if quarantined:
        safe_payload = {
            "jsonrpc": response_payload.get("jsonrpc", "2.0"),
            "id": response_payload.get("id", payload.get("id")),
            "result": {
                "quarantined": True,
                "source": server,
                "reason": "untrusted instructions were removed from tool output",
            },
        }
        return JSONResponse(safe_payload, status_code=response.status_code)
    return _proxy_response(response)


@app.api_route("/langflow/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
async def langflow_proxy(path: str, request: Request):
    state = _state(request)
    body = await _body(request)
    trace_id = _trace_id(request)
    lowered_path = "/" + path.lower().lstrip("/")
    decoded = body.decode("utf-8", errors="ignore")
    rules = []

    if lowered_path.endswith("/api/v1/validate/code") and re.search(
        r"(?is)(__import__\s*\(|subprocess\.|shell\s*=\s*true|os\.system\s*\()",
        decoded,
    ):
        rules.append("LANGFLOW_CODE_EXECUTION_PAYLOAD")
    if "/api/v2/files" in lowered_path and (
        "../" in decoded or "..\\" in decoded or re.search(r"(?i)/etc/(cron|systemd)", decoded)
    ):
        rules.append("LANGFLOW_PATH_TRAVERSAL_UPLOAD")

    if rules:
        DECISIONS.labels("langflow", "block").inc()
        _audit(
            state,
            trace_id=trace_id,
            phase="framework-preflight",
            source="langflow-proxy",
            action=lowered_path,
            decision="block",
            risk_score=100.0,
            matched_rules=rules,
            details={"body_sha256": hashlib.sha256(body).hexdigest()},
        )
        return JSONResponse(
            {"detail": "blocked by EchoGuide Guard", "reason_codes": rules},
            status_code=403,
        )

    response = await _forward(
        state,
        request,
        f"{state.settings.langflow_upstream}/{path.lstrip('/')}",
        body=body,
        extra_headers={"X-Trace-Id": trace_id},
    )
    PROXY_REQUESTS.labels("langflow", str(response.status_code)).inc()
    return _proxy_response(response)


if __name__ == "__main__":
    import uvicorn

    cfg = Settings.from_env()
    uvicorn.run("echoguide_guard.app:app", host=cfg.host, port=cfg.port, reload=False)
