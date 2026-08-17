"""
RuntimeMiddleware / MiddlewareChain —— Agent Runtime 的生命周期钩子。

中间件在真实执行边界触发（run / model / tool / finish），职责正交：
Guard 拦截、Trace 埋点、Skill 解析、预算计数……全部通过同一套钩子接入，
业务执行体（编排器 core）不感知中间件细节。

触发规则：
  - before 钩子按注册顺序执行，GuardRejection / BudgetExceeded 抛出后短路；
  - after 钩子按注册逆序执行且必然触发（单个钩子异常只记日志，不掩盖业务结果）。
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class GuardRejection(Exception):
    """Guard 拦截（输入违规）：业务不执行，返回拒绝结果。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class BudgetExceeded(Exception):
    """执行预算超限：强制中止当前步骤。"""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class RuntimeMiddleware:
    """中间件基类：默认全部钩子为 no-op，子类按需覆盖。"""

    name: str = "base"

    async def before_run(self, ctx: Any) -> None:
        ...

    async def after_run(self, ctx: Any) -> None:
        ...

    async def before_model(self, ctx: Any) -> None:
        ...

    async def after_model(self, ctx: Any, response: Any) -> None:
        ...

    async def before_tool(self, ctx: Any, tool_name: str, tool_input: Any) -> None:
        ...

    async def after_tool(
        self, ctx: Any, tool_name: str, result: Any, error: Optional[str]
    ) -> None:
        ...

    async def before_finish(self, ctx: Any) -> None:
        ...

    async def after_finish(self, ctx: Any, result: Any) -> None:
        ...


class MiddlewareChain:
    """有序中间件链：before 正序（拦截异常短路）、after 逆序（必执行）。"""

    def __init__(self, middlewares: Optional[List[RuntimeMiddleware]] = None):
        self._items: List[RuntimeMiddleware] = list(middlewares or [])

    @property
    def items(self) -> List[RuntimeMiddleware]:
        return list(self._items)

    def add(self, middleware: RuntimeMiddleware) -> "MiddlewareChain":
        self._items.append(middleware)
        return self

    async def _fire(self, hook: str, ctx: Any, **kwargs: Any) -> None:
        """before 钩子：正序；Guard/Budget 异常上抛短路，其余异常记录不阻断。"""
        for mw in self._items:
            fn = getattr(mw, hook, None)
            if fn is None:
                continue
            try:
                await fn(ctx, **kwargs)
            except (GuardRejection, BudgetExceeded):
                raise
            except Exception as ex:
                logger.warning(f"middleware {mw.name}.{hook} 异常: {ex}")
                ctx.state.add_error(f"{mw.name}.{hook}: {ex}")

    async def _fire_reverse(self, hook: str, ctx: Any, **kwargs: Any) -> None:
        """after 钩子：逆序；异常只记日志，绝不掩盖业务结果。"""
        for mw in reversed(self._items):
            fn = getattr(mw, hook, None)
            if fn is None:
                continue
            try:
                await fn(ctx, **kwargs)
            except Exception as ex:
                logger.warning(f"middleware {mw.name}.{hook} 异常: {ex}")
                ctx.state.add_error(f"{mw.name}.{hook}: {ex}")

    async def before_run(self, ctx: Any) -> None:
        await self._fire("before_run", ctx)

    async def after_run(self, ctx: Any) -> None:
        await self._fire_reverse("after_run", ctx)

    async def before_model(self, ctx: Any) -> None:
        await self._fire("before_model", ctx)

    async def after_model(self, ctx: Any, response: Any) -> None:
        await self._fire_reverse("after_model", ctx, response=response)

    async def before_tool(self, ctx: Any, tool_name: str, tool_input: Any) -> None:
        await self._fire("before_tool", ctx, tool_name=tool_name, tool_input=tool_input)

    async def after_tool(
        self, ctx: Any, tool_name: str, result: Any, error: Optional[str]
    ) -> None:
        await self._fire_reverse(
            "after_tool", ctx, tool_name=tool_name, result=result, error=error
        )

    async def before_finish(self, ctx: Any) -> None:
        await self._fire("before_finish", ctx)

    async def after_finish(self, ctx: Any, result: Any) -> None:
        await self._fire_reverse("after_finish", ctx, result=result)
