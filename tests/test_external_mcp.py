"""外部 MCP 工具源测试：客户端握手 / 工具映射 / 只读过滤 / 降级 / 暴露策略。

离线：用项目自带 MCPServer（mcp/protocol.py）做协议对端，经 httpx
ASGITransport 直连（零网络、CI 可跑），与 test_mcp_protocol.py 同风格
（FAKE_KEY + asyncio.run，无 pytest-asyncio）。
"""
from __future__ import annotations

import asyncio
import json

import httpx

from agents.agent_orchestrator import AgentOrchestrator
from agents.profiles import ProfileName
from mcp.external_client import ExternalMCPSource, StreamableHTTPClient
from mcp.protocol import MCPServer
from mcp.tool_manager import MCPToolManager, Tool, ToolEffect

FAKE_KEY = "sk-test-not-used"


# ── 协议对端：MCPServer 的 ASGI 包装（模拟远程 MCP server）───────────────────

def _make_server() -> MCPServer:
    tm = MCPToolManager(api_key=FAKE_KEY)

    async def search_handler(params, context):
        return [{"title": f"repo-{params.get('query', '')}", "url": "https://github.com/example"}]

    async def create_issue_handler(params, context):
        return {"created": True, "number": 1}

    tm.register(Tool(
        name="search_repositories",
        description="按关键词搜索 GitHub 仓库",
        handler=search_handler,
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        },
    ))
    tm.register(Tool(
        name="create_issue",
        description="创建 issue（写操作）",
        handler=create_issue_handler,
        schema={"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]},
    ))
    tm.register(Tool(
        name="analyze_repository",
        description="分析仓库（未知命名，非白名单非黑名单）",
        handler=create_issue_handler,
        schema={"type": "object", "properties": {"repo": {"type": "string"}}},
    ))
    return MCPServer(tm)


def _asgi_app(server: MCPServer, status: int = 200):
    async def app(scope, receive, send):
        assert scope["type"] == "http"
        body = b""
        while True:
            msg = await receive()
            if msg["type"] == "http.request":
                body += msg.get("body", b"")
                if not msg.get("more_body", False):
                    break
        result = await server.handle(body.decode("utf-8"))
        resp_body = json.dumps(result).encode() if result is not None else b""
        await send({
            "type": "http.response.start",
            "status": 202 if result is None else status,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": resp_body})
    return app


def _transport(server: MCPServer):
    return httpx.ASGITransport(app=_asgi_app(server))


def _run(coro):
    return asyncio.run(coro)


# ── StreamableHTTPClient：握手 / 枚举 / 转发 ──────────────────────────────────

def test_client_initialize_and_list_tools():
    client = StreamableHTTPClient("http://test", token="pat", transport=_transport(_make_server()))
    try:
        result = _run(client.initialize())
        assert result["capabilities"]["tools"]["listChanged"] is False
        tools = _run(client.list_tools())
        assert {t["name"] for t in tools} == {"search_repositories", "create_issue", "analyze_repository"}
        assert tools[0]["inputSchema"]["required"] == ["query"]
    finally:
        _run(client.aclose())


def test_client_call_tool_roundtrip():
    client = StreamableHTTPClient("http://test", token="pat", transport=_transport(_make_server()))
    try:
        text, is_error = _run(client.call_tool("search_repositories", {"query": "deepseek"}))
        assert is_error is False
        data = json.loads(text)
        assert data[0]["title"] == "repo-deepseek"
    finally:
        _run(client.aclose())


def test_parse_sse_frame():
    from mcp.external_client import _parse_sse

    frame = 'event: message\ndata: {"jsonrpc": "2.0", "id": 1, "result": {}}\n\n'
    assert json.loads(_parse_sse(frame))["id"] == 1


# ── ExternalMCPSource：只读过滤 / 白名单 / 降级 ───────────────────────────────

def test_source_registers_prefixed_readonly_tools():
    tm = MCPToolManager(api_key=FAKE_KEY)
    source = ExternalMCPSource("http://test", prefix="github", transport=_transport(_make_server()))
    registered = _run(source.setup(tm))

    # create_issue 命中写关键词、analyze_repository 非白名单命名 → 都被跳过
    assert registered == ["github_search_repositories"]
    tool = tm._tools["github_search_repositories"]
    assert tool.agent_exposed is False  # 默认不暴露
    assert tool.schema["required"] == ["query"]
    assert "搜索" in tool.description  # 描述保留


def test_read_only_filter_boundaries():
    """只读过滤边界：白名单放行 / 写关键词拒绝 / 未知命名保守跳过。"""
    from mcp.external_client import is_read_only_tool

    # 真实 GitHub server 命名：get_*/list_*/search_*/*_read 只读
    assert is_read_only_tool("get_me")
    assert is_read_only_tool("list_issues")
    assert is_read_only_tool("search_repositories")
    assert is_read_only_tool("issue_read")
    # 真实写工具：write/push/fork/run 命名
    assert not is_read_only_tool("push_files")
    assert not is_read_only_tool("issue_write")
    assert not is_read_only_tool("fork_repository")
    assert not is_read_only_tool("run_secret_scanning")
    # 未知命名：宁紧勿松，保守跳过
    assert not is_read_only_tool("analyze_repository")


def test_source_whitelist_overrides_readonly_filter():
    tm = MCPToolManager(api_key=FAKE_KEY)
    source = ExternalMCPSource("http://test", prefix="github", transport=_transport(_make_server()))
    # 白名单显式放行写工具（支持带前缀全名或原始名）
    registered = _run(source.setup(tm, tool_whitelist={"github_create_issue"}))
    assert registered == ["github_create_issue"]
    assert "github_search_repositories" not in tm._tools


def test_source_connection_failure_degrades():
    """连接失败（HTTP 500）→ 返回空列表，不抛异常，服务照常启动。"""
    tm = MCPToolManager(api_key=FAKE_KEY)
    server = _make_server()
    app = _asgi_app(server, status=500)
    source = ExternalMCPSource("http://test", prefix="github", transport=httpx.ASGITransport(app=app))
    registered = _run(source.setup(tm))
    assert registered == []
    assert tm._tools == {}


# ── expose_external_tools：加入公共工具层（任何请求可见）─────────────────────

def test_expose_external_tools_visibility():
    tm = MCPToolManager(api_key=FAKE_KEY)
    # 本地工具（模拟 knowledge_search）与外部工具并存
    async def kb_handler(params, context):
        return [{"title": "本地知识"}]
    tm.register(Tool(
        name="knowledge_search",
        description="搜索校园知识库",
        handler=kb_handler,
        schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        effect=ToolEffect.READ,  # 显式副作用声明（fail-closed）
    ))
    source = ExternalMCPSource("http://test", prefix="github", transport=_transport(_make_server()))
    registered = _run(source.setup(tm))

    orch = AgentOrchestrator(api_key=FAKE_KEY)
    orch.set_tool_manager(tm)
    orch.expose_external_tools(registered)

    agent = orch._agents[ProfileName.FAST]
    exposed = [t["name"] for t in agent._build_tools(req=None)]
    assert "github_search_repositories" in exposed
    assert "knowledge_search" in exposed  # 本地工具不受影响（同一公共层）
    assert tm._tools["github_search_repositories"].agent_exposed is True


def test_unexposed_external_tools_invisible():
    """未调用 expose 时外部工具不可见（agent_exposed=False 双重不可见）。"""
    tm = MCPToolManager(api_key=FAKE_KEY)
    source = ExternalMCPSource("http://test", prefix="github", transport=_transport(_make_server()))
    _run(source.setup(tm))

    orch = AgentOrchestrator(api_key=FAKE_KEY)
    orch.set_tool_manager(tm)
    agent = orch._agents[ProfileName.FAST]
    assert "github_search_repositories" not in [t["name"] for t in agent._build_tools(req=None)]
