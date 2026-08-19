"""
外部 MCP 工具源 —— 把远程 MCP server（如 GitHub 官方 MCP server）的工具
作为工具源接入 MCPToolManager。

与 mcp/protocol.py 的服务端对称：那边把本地工具暴露给外部 MCP 客户端；
这里用 httpx 手写极简 Streamable HTTP 客户端（零新依赖），
initialize 握手 → tools/list 枚举 → tools/call 转发，把外部工具包装成 Tool
注册进工具管理器，自动获得熔断/超时/降级/缓存/上下文卸载等既有工程能力。

设计原则（对齐项目全链路降级哲学）：
  - 连接失败/超时/鉴权失败只记日志并返回空列表，服务照常启动；
  - 默认只读过滤：写操作工具不注册（不依赖 WRITE_TOOLS——那是给本地工具登记的）；
  - 工具名加前缀（默认 github_）避免与本地工具冲突；
  - 注册的工具默认 agent_exposed=False，由编排器显式暴露给指定 Agent。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import httpx

from mcp.tool_manager import MCPToolManager, Tool, ToolEffect

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-11-25"

# 只读命名白名单：工具名以这些前缀开头或以 _read 结尾 → 视为只读直接放行。
# 基于真实 GitHub MCP server 的命名规律（get_*/list_*/search_* 全部只读，
# *_read 只读），同时兼容 query_/fetch_ 等常见只读命名。
_READ_PREFIXES = ("get_", "list_", "search_", "query_", "fetch_", "read_")
_READ_SUFFIX = "_read"

# 写操作关键词黑名单：命中任一即视为写工具，默认跳过注册。
# 真实 server（GitHub）用 write/push/fork 等命名而非 create_*，黑名单必须覆盖。
_WRITE_KEYWORDS = (
    "create", "update", "delete", "remove", "merge", "close", "comment",
    "review", "add", "assign", "transfer", "rename", "move", "mark", "set",
    "write", "push", "fork", "run", "edit", "change", "open", "reply",
)


def is_read_only_tool(name: str) -> bool:
    """默认只读策略（宁紧勿松）：白名单命名放行，写关键词拒绝，其余保守跳过。"""
    lower = name.lower()
    if any(kw in lower for kw in _WRITE_KEYWORDS):
        return False
    return lower.startswith(_READ_PREFIXES) or lower.endswith(_READ_SUFFIX)


class MCPProtocolError(Exception):
    """外部 MCP 协议错误（JSON-RPC error / 响应格式异常）。"""


def _parse_sse(text: str) -> str:
    """解析 text/event-stream 响应，返回最后一个 data 字段的 JSON 字符串。

    Streamable HTTP 的 SSE 帧形如 `event: message\\ndata: {...}\\n\\n`，
    非流式单响应场景通常只有一帧。
    """
    data_lines: List[str] = []
    for block in text.split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
    if not data_lines:
        raise MCPProtocolError("SSE 响应无 data 字段")
    return "\n".join(data_lines)


class StreamableHTTPClient:
    """极简 MCP Streamable HTTP 客户端（tools 子集）。

    覆盖 initialize / tools/list / tools/call 三个方法，JSON-RPC 响应
    （application/json）与 SSE 帧（text/event-stream）两种形态都支持；
    处理 Mcp-Session-Id 会话头。transport 可注入（测试用 ASGITransport 直连）。
    """

    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        proxy: Optional[str] = None,
        timeout_s: float = 15.0,
        protocol_version: str = PROTOCOL_VERSION,
        transport: Optional[Any] = None,
    ):
        base = url.rstrip("/")
        self._url = base if base.endswith("/mcp") else f"{base}/mcp"
        self._protocol_version = protocol_version
        self._session_id: Optional[str] = None
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": protocol_version,
        }
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            proxy=proxy or None,
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """发送 JSON-RPC 请求，解析响应（JSON 或 SSE），校验 JSON-RPC error。"""
        headers = dict(self._headers)
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        resp = await self._client.post(self._url, json=payload, headers=headers)
        resp.raise_for_status()
        session = resp.headers.get("Mcp-Session-Id")
        if session:
            self._session_id = session
        content_type = resp.headers.get("Content-Type", "")
        if "text/event-stream" in content_type:
            body = _parse_sse(resp.text)
        else:
            body = resp.text
        try:
            parsed = json.loads(body)
        except (TypeError, ValueError) as ex:
            raise MCPProtocolError(f"响应不是合法 JSON: {ex}")
        if isinstance(parsed, dict) and "error" in parsed:
            err = parsed["error"]
            raise MCPProtocolError(f"JSON-RPC 错误 {err.get('code')}: {err.get('message')}")
        return parsed

    async def initialize(self) -> Dict[str, Any]:
        """initialize 握手；返回 server 能力声明（result）。"""
        resp = await self._post({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": self._protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "echoguide", "version": "0.1.0"},
            },
        })
        return resp.get("result", {})

    async def list_tools(self) -> List[Dict[str, Any]]:
        """tools/list → 工具声明列表 [{name, description, inputSchema, ...}]。"""
        resp = await self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        return resp.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> tuple[str, bool]:
        """tools/call → (文本内容, is_error)。"""
        resp = await self._post({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        })
        result = resp.get("result", {})
        parts = [
            c.get("text", "")
            for c in result.get("content", [])
            if isinstance(c, dict) and c.get("type") == "text"
        ]
        return "\n".join(parts), bool(result.get("isError", False))


class ExternalMCPSource:
    """外部 MCP 工具源：连接远程 server，把只读工具包装注册进 MCPToolManager。

    setup() 幂等注册（重名跳过）；任何异常只记日志返回空列表，不阻断服务启动。
    """

    def __init__(
        self,
        url: str,
        token: Optional[str] = None,
        proxy: Optional[str] = None,
        timeout_s: float = 15.0,
        prefix: str = "github",
        transport: Optional[Any] = None,
    ):
        self._url = url
        self._token = token
        self._proxy = proxy
        self._timeout_s = timeout_s
        self._prefix = prefix
        self._transport = transport

    def _make_handler(self, client: StreamableHTTPClient, remote_name: str) -> Callable:
        async def handler(params: Dict[str, Any], context: Any) -> Any:
            text, is_error = await client.call_tool(remote_name, params)
            if is_error:
                raise RuntimeError(text or f"外部工具 {remote_name} 执行失败")
            # GitHub 等 server 的文本内容是 JSON 字符串：解析成结构化数据，
            # 让标题提取/缓存等既有链路正常工作；解析失败保留原文。
            try:
                return json.loads(text)
            except (TypeError, ValueError):
                return text
        return handler

    async def setup(
        self,
        tool_manager: MCPToolManager,
        tool_whitelist: Optional[set] = None,
    ) -> List[str]:
        """连接、枚举、过滤并注册外部工具；失败返回空列表。

        tool_whitelist：非空时只注册名单内的工具（支持带前缀全名或原始名），
        绕过只读过滤（例如显式放行某个写工具）；None = 默认只读过滤。

        返回注册的工具全名列表（如 ["github_search_repositories"]）。
        """
        client = StreamableHTTPClient(
            self._url,
            token=self._token,
            proxy=self._proxy,
            timeout_s=self._timeout_s,
            transport=self._transport,
        )
        try:
            await client.initialize()
            tools = await client.list_tools()
        except Exception as ex:
            logger.warning(f"外部 MCP 工具源 {self._url} 连接失败，跳过接入: {ex}")
            await client.aclose()
            return []

        registered: List[str] = []
        try:
            for item in tools:
                name = str(item.get("name", "")).strip()
                if not name:
                    continue
                full = f"{self._prefix}_{name}"
                if tool_whitelist is not None:
                    if full not in tool_whitelist and name not in tool_whitelist:
                        continue
                elif not is_read_only_tool(name):
                    logger.info("外部 MCP 跳过写工具: %s", full)
                    continue
                if full in tool_manager._tools:
                    logger.warning("外部 MCP 工具名冲突，跳过: %s", full)
                    continue
                schema = item.get("inputSchema")
                if not isinstance(schema, dict):
                    schema = {"type": "object", "properties": {}}
                tool_manager.register(Tool(
                    name=full,
                    description=str(item.get("description", "") or ""),
                    handler=self._make_handler(client, name),
                    schema=schema,
                    timeout_s=30.0,
                    agent_exposed=False,  # 默认不可见，由编排器显式暴露
                    # 副作用声明：默认只读过滤放行的都是只读命名工具；白名单显式
                    # 放行的写工具同样按只读声明 —— 只读语义宁紧勿松（fail-closed）。
                    effect=ToolEffect.READ,
                ))
                registered.append(full)
        except Exception as ex:
            logger.error(f"外部 MCP 工具注册中断: {ex}")
        logger.info("外部 MCP 工具源 %s 注册 %d 个工具: %s", self._url, len(registered), registered)
        return registered
