from __future__ import annotations

import json

import httpx
import jwt
from fastapi.testclient import TestClient

from echoguard.app import app


def test_agent_identity_reaches_mcp_policy_and_audit(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOGUARD_TARGET", str(tmp_path))
    monkeypatch.setenv("ECHOGUARD_AUDIT_DB", str(tmp_path / "audit.sqlite3"))
    monkeypatch.setenv("ECHOGUARD_JWT_SECRET", "test-secret")

    upstream_calls: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(str(request.url))
        if request.url.host == "opspilot-app":
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 2, "result": {"rows": []}},
        )

    token = jwt.encode(
        {
            "sub": "counselor-1",
            "role": "counselor",
            "team": "communication",
            "scope": {"tenant": "communication"},
        },
        "test-secret",
        algorithm="HS256",
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Trace-Id": "opaque-trace-1",
    }

    with TestClient(app) as client:
        guard = client.app.state.guard
        previous_client = guard.client
        guard.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        client.portal.call(previous_client.aclose)

        response = client.post(
            "/agent/run",
            headers=headers,
            json={"prompt": "query student records", "skill": None},
        )
        assert response.status_code == 200

        blocked = client.post(
            "/mcp/student-db",
            headers={"X-Trace-Id": "opaque-trace-1"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "query", "arguments": {"tenant": "other-college"}},
            },
        )
        assert blocked.status_code == 200
        assert blocked.json()["error"]["data"]["reason_codes"] == ["tenant.cross_boundary"]

        trace = client.get("/api/v1/traces/opaque-trace-1")
        assert trace.status_code == 200
        events = trace.json()["events"]
        assert any(event["decision"] == "block" for event in events)
        serialized = json.dumps(events)
        assert "query student records" not in serialized
        assert any("opspilot-app" in url for url in upstream_calls)


def test_direct_mcp_call_without_identity_is_fail_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHOGUARD_TARGET", str(tmp_path))
    monkeypatch.setenv("ECHOGUARD_AUDIT_DB", str(tmp_path / "audit.sqlite3"))

    with TestClient(app) as client:
        response = client.post(
            "/mcp/shell-runner",
            headers={"X-Trace-Id": "opaque-trace-2"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "run", "arguments": {"command": "echo harmless"}},
            },
        )

    assert response.status_code == 200
    assert response.json()["error"]["data"]["reason_codes"] == ["role.unmanaged"]
