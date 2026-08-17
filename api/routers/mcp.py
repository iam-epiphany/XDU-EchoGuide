"""MCP 协议路由：POST /mcp（Streamable HTTP tools 子集）与 /mcp/info。"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from api import state
from api.deps import optional_user

router = APIRouter(prefix="/mcp", tags=["MCP"])


@router.post("")
async def mcp_endpoint(request: Request):
    """
    标准 MCP 协议端点（JSON-RPC 2.0 / Streamable HTTP transport）。

    支持 MCP 2025-11-25 Streamable HTTP 的 tools 子集。请求必须声明 JSON 与
    SSE Accept；通知按规范返回 202。它不是完整的账号或 RBAC 服务。
    用户身份来自签名登录 Cookie；未登录的 MCP 调用仍可使用公开工具，但个人工具拒绝访问。
    """
    if state._tool_manager is None:
        raise HTTPException(503, "工具管理器未初始化")
    from mcp.protocol import MCPServer, SUPPORTED_PROTOCOL_VERSIONS

    state._validate_mcp_origin(request)
    state._validate_mcp_accept(request)
    version = request.headers.get("MCP-Protocol-Version")
    if version and version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise HTTPException(400, "不支持的 MCP-Protocol-Version")
    user = optional_user(request)
    user_id = user.id if user else "anonymous"
    server = MCPServer(state._tool_manager, user_id=user_id)
    try:
        raw = (await request.body()).decode("utf-8")
    except UnicodeDecodeError:
        # 非法 UTF-8：返回标准 JSON-RPC PARSE_ERROR（-32700），
        # 而不是静默丢弃字节后让协议层给出通用解析错误。
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error: 请求体不是合法 UTF-8"},
        }, status_code=400)
    result = await server.handle(raw)
    if result is None:
        return Response(status_code=202)
    return JSONResponse(result)


@router.get("")
async def mcp_method_not_allowed():
    """Streamable HTTP 在本服务不提供 SSE GET 流；信息改由 /mcp/info 提供。"""
    return Response(status_code=405, headers={"Allow": "POST"})


@router.get("/info")
async def mcp_info():
    """EchoGuide 支持的 MCP transport 与工具说明（非协议握手端点）。"""
    if state._tool_manager is None:
        raise HTTPException(503, "工具管理器未初始化")
    from mcp.protocol import PROTOCOL_VERSION

    tools = [
        {"name": name, "description": t.description, "inputSchema": t.schema}
        for name, t in state._tool_manager._tools.items()
    ]
    return {
        "server": "echoguide-mcp",
        "protocolVersion": PROTOCOL_VERSION,
        "tools": tools,
        "note": "POST /mcp 为 MCP Streamable HTTP tools 子集；GET /mcp 明确返回 405。",
    }
