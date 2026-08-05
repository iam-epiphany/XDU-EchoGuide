"""
MCP（Model Context Protocol）JSON-RPC 协议层。

把内部 MCPToolManager 以标准 MCP 协议暴露出去：
  - 支持 initialize / tools/list / tools/call 等核心方法
  - HTTP Streamable transport（POST /mcp 单端点）
  - 协议格式遵循 MCP 规范：jsonrpc 2.0，方法名 tools/list、tools/call

这样 EchoGuide 的工具不仅能被 Agent 内部调用，也能被任何 MCP 客户端
（Claude Desktop、Cursor、自研客户端等）即插即用 —— 工具层真正"协议化"。

面试点：从"自研工具注册表"升级为"标准 MCP Server"，工具即插即用。
"""
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# MCP 协议常量
JSONRPC_VERSION = "2.0"
SERVER_NAME = "echoguide-mcp"
SERVER_VERSION = "0.1.0"

# JSON-RPC 错误码
PARSE_ERROR        = -32700
INVALID_REQUEST    = -32600
METHOD_NOT_FOUND   = -32601
INVALID_PARAMS     = -32602
INTERNAL_ERROR     = -32603


class MCPServer:
    """基于内部 MCPToolManager 的标准 MCP Server 实现。"""

    def __init__(self, tool_manager, user_id: str = "anonymous"):
        self._tool_manager = tool_manager
        self._user_id = user_id   # 调用方（HTTP 层）注入的用户身份，供个人工具使用
        self._initialized = False
        self._client_info: Dict[str, Any] = {}

    # ── 协议入口 ──────────────────────────────────────────────────────────────

    async def handle(self, raw_body: str) -> Dict[str, Any]:
        """
        处理单个 JSON-RPC 请求（含批量请求）。
        返回标准 JSON-RPC 响应。
        """
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return self._error(None, PARSE_ERROR, "Parse error: 非法 JSON")

        if isinstance(payload, list):
            results = [await self._dispatch(item) for item in payload]
            return [r for r in results if r is not None]
        return await self._dispatch(payload)

    async def _dispatch(self, req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(req, dict):
            return self._error(None, INVALID_REQUEST, "Invalid Request: 请求必须是 JSON 对象")

        request_id = req.get("id")
        if req.get("jsonrpc") != JSONRPC_VERSION:
            return self._error(request_id, INVALID_REQUEST, "Invalid Request: 缺少 jsonrpc=2.0")

        method = req.get("method", "")

        # 通知类消息（无 id 是合法的）不返回响应
        if method.startswith("notifications/"):
            if method == "notifications/initialized":
                self._initialized = True
            return None

        # 非通知类请求必须带 id；缺 id 视为 Invalid Request
        if request_id is None and method:
            return self._error(None, INVALID_REQUEST, "Invalid Request: 请求缺少 id")

        params = req.get("params", {}) or {}
        if params is None:
            params = {}
        # params 非对象（数组/字符串）→ Invalid Params（-32602）
        if not isinstance(params, dict):
            return self._error(request_id, INVALID_PARAMS, "Invalid Params: params 必须是对象")

        handlers = {
            "initialize": self._initialize,
            "tools/list": self._tools_list,
            "tools/call": self._tools_call,
            "ping":       self._ping,
        }
        handler = handlers.get(method)
        if handler is None:
            return self._error(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")

        try:
            result = await handler(params, request_id)
            return self._result(request_id, result)
        except MCPError as ex:
            return self._error(request_id, ex.code, ex.message)
        except Exception as ex:
            logger.exception(f"MCP 方法 {method} 执行异常")
            return self._error(request_id, INTERNAL_ERROR, f"Internal error: {ex}")

    # ── MCP 方法实现 ──────────────────────────────────────────────────────────

    async def _initialize(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        self._client_info = params.get("clientInfo", {})
        self._initialized = True
        logger.info(f"MCP 客户端初始化: {self._client_info}")
        return {
            "protocolVersion": "2025-03-26",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "EchoGuide MCP Server：校园知识库检索等工具。"
                "tools/call 参数与工具声明中的 inputSchema 一致。"
            ),
        }

    async def _tools_list(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        tools = []
        for name, tool in self._tool_manager._tools.items():
            tools.append({
                "name": name,
                "description": tool.description,
                "inputSchema": tool.schema,
            })
        return {"tools": tools}

    async def _tools_call(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        name = params.get("name")
        arguments = params.get("arguments", {}) or {}
        if not name:
            raise MCPError(INVALID_PARAMS, "tools/call 缺少 name")

        # 透传调用方注入的 user_id：个人工具（课表/待办/DDL）在 MCP 路径下按身份生效。
        # 信任模型与前端一致（HTTP 层通过 X-User-Id 请求头传入，同 user_id 软身份）。
        result = await self._tool_manager.call(
            name, arguments,
            context={"mcp": True, "user_id": self._user_id},
        )
        if not result.success:
            raise MCPError(INTERNAL_ERROR, f"工具执行失败: {result.error}")

        # 内容序列化为 MCP 规范的 text content
        text = json.dumps(result.data, ensure_ascii=False)
        return {
            "content": [{"type": "text", "text": text}],
            "isError": False,
            "structuredContent": result.data if _jsonable(result.data) else None,
        }

    async def _ping(self, params: Dict[str, Any], request_id: Any) -> Dict[str, Any]:
        return {}

    # ── JSON-RPC 编解码 ───────────────────────────────────────────────────────

    @staticmethod
    def _result(request_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
        return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "result": result}

    @staticmethod
    def _error(request_id: Any, code: int, message: str) -> Dict[str, Any]:
        return {
            "jsonrpc": JSONRPC_VERSION,
            "id": request_id,
            "error": {"code": code, "message": message},
        }


class MCPError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _jsonable(value: Any) -> bool:
    """判断结果是否可 JSON 序列化（避免 structuredContent 携带不可序列化对象）。"""
    try:
        json.dumps(value, ensure_ascii=False)
        return True
    except (TypeError, ValueError):
        return False
