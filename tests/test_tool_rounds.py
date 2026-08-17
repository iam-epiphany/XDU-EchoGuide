"""工具轮次护栏测试：Fast/Deep 分级上限、无进展检测、签名规范化。

离线：AsyncMock 拦截 _stream_llm，不触网（与 test_tool_manager 同风格）；
工具循环走真实 _call_llm，验证提前收尾路径（asyncio.run，无 pytest-asyncio）。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents.agent_orchestrator import (
    QAAgent, ExecutionProfile, ProfileName, Request,
)
from mcp.tool_manager import MCPToolManager, Tool
from runtime.policy import ExecutionPolicy
from runtime.state import RunState

FAKE_KEY = "sk-test-not-used"


def _make_agent(profile_name: ProfileName = ProfileName.FAST) -> QAAgent:
    tm = MCPToolManager(api_key=FAKE_KEY)

    async def echo(params, context):
        return {"pong": params}

    tm.register(Tool(
        name="echo",
        description="回显工具",
        handler=echo,
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
    ))
    profile = ExecutionProfile(
        name=profile_name, model="m", max_tokens=768, thinking=False,
        rag_top_k=3, use_rewrite=False, use_rerank=False,
    )
    agent = QAAgent(
        client=AsyncMock(), model="m", skill_manager=None, tool_manager=tm, profile=profile,
    )
    agent._tool_allowlist = {"echo"}  # 让 echo 通过执行层权限（实例级覆盖公共层）
    return agent


def _req(policy: ExecutionPolicy) -> Request:
    req = Request(message="hi", user_id="u", conv_id="c")
    req.state = RunState(request_id="r", policy=policy)
    return req


def _tool_use_resp(params=None):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", name="echo", input=params or {"text": "x"}, id="t1")],
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def _text_resp(text="完成"):
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def _run(coro):
    return asyncio.run(coro)


async def _noop_ev(event):
    pass


# ── 签名规范化 ────────────────────────────────────────────────────────────────

def test_tool_round_signature_normalizes():
    a = SimpleNamespace(type="tool_use", name="echo", input={"b": 1, "a": 2}, id="t")
    b = SimpleNamespace(type="tool_use", name="echo", input={"a": 2, "b": 1}, id="t")
    assert QAAgent._tool_round_signature([a]) == QAAgent._tool_round_signature([b])
    bad = SimpleNamespace(type="tool_use", name="echo", input={"x": object()}, id="t")
    assert QAAgent._tool_round_signature([bad]) is None  # 不可序列化 → 跳过检测


# ── 分级上限 ──────────────────────────────────────────────────────────────────

def test_fast_profile_respects_tiered_limit():
    agent = _make_agent(ProfileName.FAST)
    policy = ExecutionPolicy(max_tool_rounds_fast=1, max_tool_rounds_deep=6)
    agent._stream_llm = AsyncMock(side_effect=[_tool_use_resp(), _text_resp()])
    text, *_ = _run(agent._call_llm(_req(policy), on_event=_noop_ev))
    assert text == "完成"
    # 1 轮工具 + 1 次收尾普通调用（Fast 上限 1，而非 Deep 的 6）
    assert agent._stream_llm.call_count == 2


def test_deep_profile_respects_tiered_limit():
    agent = _make_agent(ProfileName.DEEP)
    policy = ExecutionPolicy(max_tool_rounds_fast=1, max_tool_rounds_deep=2)
    agent._stream_llm = AsyncMock(side_effect=[_tool_use_resp(), _tool_use_resp(), _text_resp()])
    text, *_ = _run(agent._call_llm(_req(policy), on_event=_noop_ev))
    assert text == "完成"
    # Deep 用自身上限 2（Fast 上限 1 不影响 Deep）
    assert agent._stream_llm.call_count == 3


# ── 无进展检测 ────────────────────────────────────────────────────────────────

def test_stagnant_detection_breaks_loop():
    """连续 2 轮相同工具调用（同名同参）→ 强制收尾，不跑满轮次上限。"""
    agent = _make_agent(ProfileName.FAST)
    policy = ExecutionPolicy(max_tool_rounds_fast=5, stagnant_round_limit=2)
    agent._stream_llm = AsyncMock(side_effect=[
        _tool_use_resp(), _tool_use_resp(), _tool_use_resp(), _text_resp(),
    ])
    text, *_ = _run(agent._call_llm(_req(policy), on_event=_noop_ev))
    assert text == "完成"
    # 3 次工具轮（第 3 轮触发重复检测 break）+ 1 次收尾；无检测时会跑满 5 轮 + 收尾
    assert agent._stream_llm.call_count == 4


def test_different_params_reset_stagnation():
    """参数变化不算无进展：相同工具不同参数 → 不触发重复检测。"""
    agent = _make_agent(ProfileName.FAST)
    policy = ExecutionPolicy(max_tool_rounds_fast=5, stagnant_round_limit=2)
    agent._stream_llm = AsyncMock(side_effect=[
        _tool_use_resp({"text": "a"}), _tool_use_resp({"text": "b"}),
        _tool_use_resp({"text": "c"}), _text_resp(),
    ])
    _run(agent._call_llm(_req(policy), on_event=_noop_ev))
    # 参数各不相同：不触发无进展，4 轮全跑满（3 工具轮 + 收尾）
    assert agent._stream_llm.call_count == 4


# ── 正常路径回归 ──────────────────────────────────────────────────────────────

def test_normal_finish_without_tool_use():
    agent = _make_agent(ProfileName.FAST)
    agent._stream_llm = AsyncMock(side_effect=[_text_resp("你好")])
    text, *_ = _run(agent._call_llm(_req(ExecutionPolicy()), on_event=_noop_ev))
    assert text == "你好"
    assert agent._stream_llm.call_count == 1  # 无工具请求：一次调用直接返回
