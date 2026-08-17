"""
AgentRuntime —— EchoGuide 的 Agent Runtime（Harness 核心）。

模型之外的整套控制面：RunState 创建、ExecutionPolicy 预算、中间件链
（Trace/Guard/Budget/Skill）。编排器核心逻辑作为 core 函数在链内执行：
before_run → core → before_finish → after_run。

拦截语义：GuardRejection / BudgetExceeded 时 core 不执行，state.meta
["reject_message"] 携带拒绝文案，run() 返回 None（调用方据此返回拒绝结果）；
after_run 恒执行一次，保证观测闭环不因拦截而中断。

典型用法（编排器）：
    runtime = AgentRuntime()                       # 默认 policy（读环境变量）+ 默认中间件
    state = RunState(request_id=req.request_id, message=req.message, ...)
    result = await runtime.run(state, core, on_event=on_event, services={"req": req})
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from runtime.context import RunContext
from runtime.middleware import (
    BudgetExceeded, GuardRejection, MiddlewareChain, RuntimeMiddleware,
)
from runtime.policy import ExecutionPolicy
from runtime.state import RunState

logger = logging.getLogger(__name__)


def default_middlewares(policy: Optional[ExecutionPolicy] = None) -> List[RuntimeMiddleware]:
    """默认中间件链（注册顺序 = before 执行顺序）：Trace → Guard → Budget → Skill。"""
    from runtime.middlewares import BudgetMiddleware, GuardMiddleware, SkillMiddleware, TraceMiddleware

    policy = policy or ExecutionPolicy()
    return [
        TraceMiddleware(),
        GuardMiddleware(
            enabled=policy.guard_enabled,
            max_message_chars=policy.guard_max_message_chars,
        ),
        BudgetMiddleware(
            max_tool_calls=policy.max_tool_calls,
            max_model_calls=policy.max_model_calls,
        ),
        SkillMiddleware(),
    ]


class AgentRuntime:
    """Agent Runtime：执行策略 + 中间件链 + 模型网关 + 运行入口。"""

    def __init__(
        self,
        policy: Optional[ExecutionPolicy] = None,
        middlewares: Optional[List[RuntimeMiddleware]] = None,
    ):
        self.policy = policy or ExecutionPolicy.from_env()
        self._chain = MiddlewareChain(
            middlewares if middlewares is not None else default_middlewares(self.policy)
        )
        # 统一模型调用入口：所有生产链路 LLM 调用经 gateway 进出
        # （before/after_model 钩子、usage 统计、预算检查、重试、Trace）。
        from runtime.model_gateway import ModelGateway

        self.model_gateway = ModelGateway(self._chain, self.policy)

    @property
    def chain(self) -> MiddlewareChain:
        return self._chain

    def register(self, middleware: RuntimeMiddleware) -> "AgentRuntime":
        """注册自定义中间件（追加到链尾）。"""
        self._chain.add(middleware)
        return self

    def ctx_for(
        self,
        state: RunState,
        on_event: Optional[Callable] = None,
        services: Optional[Dict[str, Any]] = None,
    ) -> RunContext:
        return RunContext(
            state=state,
            policy=self.policy,
            services=services or {},
            on_event=on_event,
        )

    async def run(
        self,
        state: RunState,
        core: Callable[[RunContext], Awaitable[Any]],
        on_event: Optional[Callable] = None,
        services: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        执行一次完整运行：before_run → core → before_finish → after_run。

        拦截（Guard/Budget）：core 不执行，state.meta["reject_message"] 记录原因，
        返回 None；其他异常记录到 state.errors 后向上抛出；after_run 恒执行一次。
        """
        ctx = self.ctx_for(state, on_event=on_event, services=services)

        try:
            await self._chain.before_run(ctx)
        except (GuardRejection, BudgetExceeded) as ex:
            return self._reject(state, ex)

        try:
            from core.tracing import span

            async with span("runtime_run", request_id=state.request_id):
                result = await core(ctx)
            await self._chain.before_finish(ctx)
            await self._chain.after_finish(ctx, result)
            return result
        except (GuardRejection, BudgetExceeded) as ex:
            await self._chain.after_finish(ctx, None)
            return self._reject(state, ex)
        except Exception as ex:
            state.add_error(f"run: {ex}")
            logger.error(f"runtime run 失败: {ex}")
            raise
        finally:
            await self._chain.after_run(ctx)

    @staticmethod
    def _reject(state: RunState, ex: Exception) -> None:
        reason = getattr(ex, "reason", str(ex))
        state.add_error(f"guard: {reason}")
        state.meta["reject_message"] = reason
        logger.warning(f"runtime 拦截请求: {reason}")
        return None

    # ── 真实执行边界钩子（编排器 / Agent 在对应位置触发）──────────────────────

    async def fire_model_before(
        self,
        state: RunState,
        on_event: Optional[Callable] = None,
        services: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._chain.before_model(self.ctx_for(state, on_event=on_event, services=services))

    async def fire_model_after(
        self,
        state: RunState,
        response: Any,
        on_event: Optional[Callable] = None,
        services: Optional[Dict[str, Any]] = None,
    ) -> None:
        await self._chain.after_model(
            self.ctx_for(state, on_event=on_event, services=services), response
        )

    async def fire_tool_before(
        self,
        state: RunState,
        tool_name: str,
        tool_input: Any,
        on_event: Optional[Callable] = None,
    ) -> None:
        await self._chain.before_tool(
            self.ctx_for(state, on_event=on_event), tool_name, tool_input
        )

    async def fire_tool_after(
        self,
        state: RunState,
        tool_name: str,
        result: Any,
        error: Optional[str],
        on_event: Optional[Callable] = None,
    ) -> None:
        await self._chain.after_tool(
            self.ctx_for(state, on_event=on_event), tool_name, result, error
        )
