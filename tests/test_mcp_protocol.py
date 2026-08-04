"""MCP JSON-RPC 协议层测试：initialize / tools/list / tools/call / 错误处理。"""
from __future__ import annotations

import asyncio
import json

from mcp.protocol import MCPServer
from mcp.tool_manager import MCPToolManager, Tool

FAKE_KEY = "sk-test-not-used"


def _make_server():
    tm = MCPToolManager(api_key=FAKE_KEY)

    async def echo_handler(params, context):
        return {"echo": params.get("text", ""), "context": bool(context)}

    tm.register(Tool(
        name="echo",
        description="回显工具",
        handler=echo_handler,
        schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    ))
    return MCPServer(tm)


def _run(coro):
    return asyncio.run(coro)


def test_initialize_returns_capabilities():
    server = _make_server()
    resp = _run(server.handle(json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"clientInfo": {"name": "test", "version": "0.0.1"}},
    })))
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["capabilities"]["tools"] == {"listChanged": False}
    assert resp["result"]["serverInfo"]["name"] == "echoguide-mcp"


def test_tools_list_returns_registered_tools():
    server = _make_server()
    resp = _run(server.handle(json.dumps({
        "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
    })))
    names = [t["name"] for t in resp["result"]["tools"]]
    assert names == ["echo"]
    assert resp["result"]["tools"][0]["inputSchema"]["required"] == ["text"]


def test_tools_call_executes_handler():
    server = _make_server()
    resp = _run(server.handle(json.dumps({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "echo", "arguments": {"text": "你好"}},
    })))
    assert resp["result"]["isError"] is False
    content = resp["result"]["content"][0]
    assert content["type"] == "text"
    assert json.loads(content["text"])["echo"] == "你好"


def test_unknown_method_returns_jsonrpc_error():
    server = _make_server()
    resp = _run(server.handle(json.dumps({
        "jsonrpc": "2.0", "id": 4, "method": "unknown/method", "params": {},
    })))
    assert resp["error"]["code"] == -32601


def test_unknown_tool_returns_internal_error():
    server = _make_server()
    resp = _run(server.handle(json.dumps({
        "jsonrpc": "2.0", "id": 5, "method": "tools/call",
        "params": {"name": "not_exist", "arguments": {}},
    })))
    assert resp["error"]["code"] == -32603


def test_malformed_json_returns_parse_error():
    server = _make_server()
    resp = _run(server.handle("{not json"))
    assert resp["error"]["code"] == -32700


def test_notification_returns_none():
    server = _make_server()
    resp = _run(server.handle(json.dumps({
        "jsonrpc": "2.0", "method": "notifications/initialized", "params": {},
    })))
    assert resp is None


def test_batch_requests_supported():
    server = _make_server()
    batch = json.dumps([
        {"jsonrpc": "2.0", "id": 10, "method": "ping", "params": {}},
        {"jsonrpc": "2.0", "id": 11, "method": "tools/list", "params": {}},
    ])
    responses = _run(server.handle(batch))
    assert len(responses) == 2
    assert responses[0]["id"] == 10
    assert responses[1]["id"] == 11
