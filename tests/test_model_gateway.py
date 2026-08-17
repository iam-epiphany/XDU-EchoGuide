"""
ModelGateway（runtime/model_gateway.py）离线测试。

验证统一模型调用入口的真实边界：
  - step_count 按真实模型调用次数递增（一次 handle 内多次调用 → 多次计数）
  - input/output tokens 逐次累加落 RunState
  - max_model_calls 预算拦截（BudgetExceeded）
  - retries 瞬时失败重试（退避）与 Guard/Budget 不重试
  - state=None 时跳过钩子（仅 span + usage 解析）
  - tool_round_count 按真实工具轮递增
  - Fast→Deep 降级受 policy.max_retries 约束
全部使用 fake client，不触发真实 LLM 与外部服务（项目不依赖 pytest-asyncio，
async 测试统一用 asyncio.run 包裹）。
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentType, ProfileName, QAAgent, Request,
)
from runtime import (
    AgentRuntime, BudgetExceeded, ExecutionPolicy, RunState,
)


class FakeMessage:
    """最小 Message 假对象（content / stop_reason / usage）。"""

    def __init__(self, text: str = "ok", stop_reason: str = "end_turn",
                 input_tokens: int = 10, output_tokens: int = 5):
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.stop_reason = stop_reason
        self.usage = type("U", (), {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
        })()


class FakeClient:
    """fake Anthropic 客户端：可配置调用次数/失败/usage（client.messages.create）。"""

    def __init__(self, fail_times: int = 0, input_tokens: int = 10, output_tokens: int = 5):
        self.calls = 0
        self._fail_times = fail_times
        self._input = input_tokens
        self._output = output_tokens
        self.messages = type("M", (), {"create": self._create})()

    async def _create(self, **kwargs):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise RuntimeError("API 瞬时错误")
        return FakeMessage(input_tokens=self._input, output_tokens=self._output)


def _state(policy: Optional[ExecutionPolicy] = None) -> RunState:
    p = policy or ExecutionPolicy()
    return RunState(request_id="r1", user_id="u1", message="hi", policy=p)


# ── gateway 基础契约 ─────────────────────────────────────────────────────────

def test_gateway_counts_real_model_calls_and_tokens():
    async def run():
        runtime = AgentRuntime(policy=ExecutionPolicy())
        state = _state(runtime.policy)
        client = FakeClient()
        # 一次 handle 内 3 次真实模型调用
        for _ in range(3):
            await runtime.model_gateway.call(
                client=client, model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                state=state, max_tokens=64,
            )
        return state, client

    state, client = asyncio.run(run())
    assert state.step_count == 3        # 模型调用次数（而非 handle 次数）
    assert state.input_tokens == 30
    assert state.output_tokens == 15
    assert client.calls == 3


def test_gateway_without_state_skips_hooks():
    async def run():
        runtime = AgentRuntime(policy=ExecutionPolicy())
        client = FakeClient()
        result = await runtime.model_gateway.call(
            client=client, model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            state=None,  # 记忆提炼等无请求上下文路径
            max_tokens=64,
        )
        return result, client

    result, client = asyncio.run(run())
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.attempts == 1
    assert client.calls == 1


def test_gateway_retries_transient_failures():
    async def run():
        runtime = AgentRuntime(policy=ExecutionPolicy())
        state = _state(runtime.policy)
        client = FakeClient(fail_times=2)  # 前两次失败
        result = await runtime.model_gateway.call(
            client=client, model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            state=state, retries=2, max_tokens=64,
        )
        return result, state, client

    result, state, client = asyncio.run(run())
    assert result.attempts == 3
    assert client.calls == 3
    assert state.step_count == 3  # 每次尝试都是真实模型调用


def test_gateway_no_retry_by_default():
    async def run():
        runtime = AgentRuntime(policy=ExecutionPolicy())
        state = _state(runtime.policy)
        client = FakeClient(fail_times=1)
        with pytest.raises(RuntimeError):
            await runtime.model_gateway.call(
                client=client, model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                state=state,  # retries 默认 0
                max_tokens=64,
            )
        return client

    client = asyncio.run(run())
    assert client.calls == 1


def test_gateway_model_call_budget_enforced():
    async def run():
        # max_model_calls=2：第 3 次模型调用被 BudgetMiddleware 拦截
        runtime = AgentRuntime(policy=ExecutionPolicy(max_model_calls=2))
        state = _state(runtime.policy)
        client = FakeClient()
        for _ in range(2):
            await runtime.model_gateway.call(
                client=client, model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                state=state, max_tokens=64,
            )
        with pytest.raises(BudgetExceeded):
            await runtime.model_gateway.call(
                client=client, model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                state=state, max_tokens=64,
            )
        return state, client

    state, client = asyncio.run(run())
    assert state.step_count == 2  # 被拦截的调用不计数（before_model 抛错后未进 provider）
    assert client.calls == 2


def test_gateway_budget_violation_not_retried():
    async def run():
        # 预算/Guard 拦截不重试：即使 retries>0 也直接上抛，且被拦截的调用不进 provider
        runtime = AgentRuntime(policy=ExecutionPolicy(max_model_calls=1))
        state = _state(runtime.policy)
        client = FakeClient()
        await runtime.model_gateway.call(
            client=client, model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            state=state, retries=3, max_tokens=64,
        )
        with pytest.raises(BudgetExceeded):
            await runtime.model_gateway.call(
                client=client, model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                state=state, retries=3, max_tokens=64,
            )
        return client

    client = asyncio.run(run())
    assert client.calls == 1  # 被拦截的调用未进 provider，也未重试


# ── Agent 工具循环真实计数 ───────────────────────────────────────────────────

def test_agent_loop_steps_and_tool_rounds():
    """一次 handle 内 LLM→Tool→LLM：step=2、tool_round=1、token 累加。"""

    class LoopClient:
        """第一次调用返回 tool_use，第二次返回文本。"""
        def __init__(self):
            self.calls = 0
            self.messages = type("M", (), {"create": self._create})()

        async def _create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                block = type("B", (), {
                    "type": "tool_use", "name": "noop_tool",
                    "input": {}, "id": "tu_1",
                })()
                msg = FakeMessage(stop_reason="tool_use")
                msg.content = [block]
                msg.usage = type("U", (), {"input_tokens": 100, "output_tokens": 20})()
                return msg
            return FakeMessage(text="final answer", input_tokens=50, output_tokens=10)

    class ToolBox:
        _tools = {}
        async def call(self, name, params, context=None, **kw):
            return type("R", (), {"success": True, "data": {"ok": 1}, "error": None})()

    async def run():
        runtime = AgentRuntime(policy=ExecutionPolicy())
        agent = QAAgent(
            LoopClient(), "test-model",
            skill_manager=None, tool_manager=ToolBox(),
        )
        agent._tool_allowlist = ["noop_tool"]
        req = Request(message="hi", user_id="u1", conv_id="c1", action=None, domain=None)
        req.state = _state(runtime.policy)
        req.profile = ProfileName.FAST
        agent._runtime = runtime
        resp = await agent.handle(req)
        return resp, req

    resp, req = asyncio.run(run())
    assert resp.success
    assert req.state.step_count == 2        # 真实模型调用 2 次
    assert req.state.tool_round_count == 1  # 真实工具轮 1 轮
    assert req.state.tool_call_count == 1
    assert req.state.input_tokens == 150
    assert req.state.output_tokens == 30


# ── Fast→Deep 降级受 max_retries 约束 ────────────────────────────────────────

def test_fast_deep_fallback_respects_max_retries():
    """max_retries=0 时 Fast 失败不再降级 Deep；max_retries=1 时降级一次。"""

    class FailClient:
        def __init__(self):
            self.messages = type("M", (), {"create": self._create})()

        async def _create(self, **kwargs):
            raise RuntimeError("API 调用失败")

    async def run_once(max_retries: int) -> int:
        runtime = AgentRuntime(policy=ExecutionPolicy(max_retries=max_retries))
        orchestrator = AgentOrchestrator(
            api_key="sk-test", model="test-model",
            runtime=runtime,
            fast_model="test-fast", deep_model="test-deep",
        )
        # 用失败 client 替换两个 profile 的 client，模拟执行失败
        for agents in orchestrator._pool.values():
            for a in agents:
                a._client = FailClient()
        req = Request(message="hi", user_id="u1", conv_id="c1")
        req.state = _state(runtime.policy)
        req.profile = ProfileName.FAST
        await orchestrator._execute(req, AgentType.QA)
        return req.state.retry_count

    assert asyncio.run(run_once(max_retries=0)) == 0  # 上限 0：不降级
    assert asyncio.run(run_once(max_retries=1)) == 1  # 上限 1：降级一次
