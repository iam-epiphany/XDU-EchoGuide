"""
ModelGateway —— 统一模型调用入口（Runtime 的「模型总线」）。

所有生产链路 LLM 调用（意图识别、Agent 工具循环、合成器、出口校验、记忆提炼、
查询改写/重排兜底）都经 gateway.call() / call_stream() 进出，真实执行边界为：

    before_model（中间件链：step 计数 + 预算检查 + Skill 解析）
    → provider.messages.create（可重试）
    → usage 统计落 RunState（input/output tokens）
    → after_model

由此保证 Runtime 的计数口径是「真实模型调用次数」而不是「Agent handle 次数」：
一个 handle 内部 LLM → Tool → LLM → Tool → LLM 的 3 次调用，step_count 记 3，
token 逐次累加；一次模型调用 = 一次 before/after_model。

- 预算：BudgetMiddleware.before_model 检查 step 上限（policy.max_model_calls，
  0 = 仅计数不强限）与工具调用上限；
- 重试：retries 参数控制同一调用点内的瞬时失败重试（退避 0.5s/次），默认 0 不重试
  （Fast→Deep 降级是编排器层级的重试，由 max_retries 控制，见 _execute）；
- 拦截：GuardRejection / BudgetExceeded 直接上抛，由 AgentRuntime.run 统一收口
  （core 不执行语义不适用——gateway 在 core 内，拦截表现为本步中止）；
- Trace：每次调用产出 llm_call span（模型 / 耗时 / tokens）；
- state 可为 None（记忆提炼等无请求上下文的路径）：跳过中间件钩子与计数，
  只做 span 与 usage 解析，行为与直接调用一致。

用法（各模块持有自己的 client，gateway 只管边界）：
    result = await gateway.call(
        client=self._client, model=self._model,
        messages=[...], system=..., tools=...,
        state=req.state, services={"skill_manager": ...}, on_event=on_event,
        max_tokens=256, temperature=0.1, thinking={"type": "disabled"},
    )
    raw = result.response.content[0].text
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from runtime.context import RunContext
from runtime.middleware import BudgetExceeded, GuardRejection, MiddlewareChain

logger = logging.getLogger(__name__)


@dataclass
class ModelCallResult:
    """一次模型调用的完整结果（含统计）。"""

    response: Any                # 原始 Message（content/stop_reason/usage）
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    attempts: int                # 实际尝试次数（含重试；1 = 一次成功）
    streamed: bool = False       # 是否流式调用


def _usage_tokens(resp: Any) -> tuple[int, int]:
    """从 Message.usage 解析 (input_tokens, output_tokens)，缺省为 0。"""
    usage = getattr(resp, "usage", None)
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


class ModelGateway:
    """统一模型调用入口：中间件钩子 + 统计 + 预算 + 重试 + Trace。"""

    def __init__(self, chain: MiddlewareChain, policy: Any):
        self._chain = chain
        self._policy = policy

    # ── 非流式调用 ──────────────────────────────────────────────────────────

    async def call(
        self,
        *,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        state: Optional[Any] = None,
        services: Optional[Dict[str, Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
        span_name: str = "llm_call",
        retries: int = 0,
        **kwargs: Any,
    ) -> ModelCallResult:
        """
        非流式模型调用。

        kwargs 透传 provider（max_tokens / temperature / thinking / output_config…）。
        retries > 0 时对瞬时失败做退避重试；Guard/Budget 拦截不重试直接上抛。
        state 为 None 时跳过中间件钩子与计数（仅 span + usage 解析）。
        """
        from core.tracing import span

        t0 = time.monotonic()
        attempts = 0
        while True:
            attempts += 1
            try:
                async with span(span_name, model=model, streamed=False):
                    if state is not None:
                        ctx = RunContext(
                            state=state,
                            policy=self._policy,
                            services=services or {},
                            on_event=on_event,
                        )
                        await self._chain.before_model(ctx)
                    resp = await client.messages.create(
                        model=model,
                        messages=messages,
                        system=system,
                        tools=tools or None,
                        **kwargs,
                    )
                    in_tokens, out_tokens = _usage_tokens(resp)
                    if state is not None:
                        state.input_tokens += in_tokens
                        state.output_tokens += out_tokens
                        await self._chain.after_model(ctx, resp)
                    return ModelCallResult(
                        response=resp,
                        model=model,
                        input_tokens=in_tokens,
                        output_tokens=out_tokens,
                        latency_ms=(time.monotonic() - t0) * 1000,
                        attempts=attempts,
                    )
            except (GuardRejection, BudgetExceeded):
                raise  # 拦截/预算：上抛由 run() 收口，不重试
            except Exception as ex:
                if attempts > retries:
                    raise
                logger.warning(f"模型调用失败（第 {attempts}/{retries + 1} 次），退避重试: {ex}")
                await asyncio.sleep(0.5 * attempts)

    # ── 流式调用（SSE 逐 token 推送 + 最终消息统计）─────────────────────────

    async def call_stream(
        self,
        *,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        state: Optional[Any] = None,
        services: Optional[Dict[str, Any]] = None,
        on_event: Optional[Callable[[Dict[str, Any]], Any]] = None,
        span_name: str = "llm_call",
        **kwargs: Any,
    ) -> ModelCallResult:
        """
        流式调用：逐 token 经 on_event({"type": "delta", "text": ...}) 推送，
        返回带最终 Message 的 ModelCallResult（统计口径与非流式一致）。

        SDK 兼容说明：anthropic 的流式 API 在不同版本有差异，统一按两种形态适配：
          - 0.40（项目锁定版）：messages.stream() 同步返回 manager，__aenter__
            产出 AsyncMessageStream，迭代得到 RawMessageStreamEvent；
          - 0.5x+：messages.stream() 为 async 方法（需 await），事件结构相同。
        """
        import inspect

        from core.tracing import span

        t0 = time.monotonic()
        async with span(span_name, model=model, streamed=True):
            if state is not None:
                ctx = RunContext(
                    state=state,
                    policy=self._policy,
                    services=services or {},
                    on_event=on_event,
                )
                await self._chain.before_model(ctx)
            stream_cm = client.messages.stream(
                model=model,
                messages=messages,
                system=system,
                tools=tools or None,
                **kwargs,
            )
            if inspect.isawaitable(stream_cm):
                stream_cm = await stream_cm  # 新版 SDK：stream() 是 async 方法
            async with stream_cm as stream:
                async for chunk in stream:
                    if getattr(chunk, "type", "") == "content_block_delta":
                        delta = getattr(chunk, "delta", None)
                        text = getattr(delta, "text", None)
                        if text and on_event is not None:
                            await on_event({"type": "delta", "text": text})
                final = await stream.get_final_message()
            in_tokens, out_tokens = _usage_tokens(final)
            if state is not None:
                state.input_tokens += in_tokens
                state.output_tokens += out_tokens
                await self._chain.after_model(ctx, final)
            return ModelCallResult(
                response=final,
                model=model,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                latency_ms=(time.monotonic() - t0) * 1000,
                attempts=1,
                streamed=True,
            )
