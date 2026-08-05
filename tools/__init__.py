"""
业务工具 —— MCP 工具 handler 集合。

每个文件导出一个或多个 async handler，统一签名：
    async def handler(params: Dict, context: Dict) -> Any
其中 context 含 {"agent_type": ..., "user_id": ...}（由 MCPToolManager 注入）。

注册入口在 api/main.py lifespan（与 knowledge_search 相同的 register 模式），
注册后自动暴露给全部 Agent 的 function calling，无需改动 Agent 代码。

依赖注入：handler 需要业务服务（PersonalService / CampusInfoStore）时，
注册侧用 with_service() 包装，把依赖塞进 context —— 不侵入工具框架。
"""

from __future__ import annotations

from typing import Any, Callable, Dict


def with_service(handler: Callable, **deps: Any) -> Callable:
    """
    把依赖注入进工具 context。

    用法（api/main.py 注册工具时）：
        handler=with_service(query_schedule_handler, personal_service=_personal_service)
    运行时 handler 收到 context 中额外的 personal_service 字段，其余行为不变。
    """

    async def wrapped(params: Dict[str, Any], context: Dict) -> Any:
        ctx = dict(context or {})
        ctx.update(deps)
        return await handler(params, ctx)

    return wrapped

