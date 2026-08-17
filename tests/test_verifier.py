"""出口校验（Verifier/Grounding）离线测试：规则校验 + LLM 判定 + 编排器集成。

LLM 判定用 AsyncMock 伪客户端，不触网；规则校验为纯函数直测。
"""
from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, Request,
)
from agents.verifier import ResponseVerifier
from core.domains import IntentAction, IntentDomain

FAKE_KEY = "sk-test-not-used"
WRITE_TOOLS = frozenset({"add_todo", "complete_todo"})


def _req(message: str = "x", action=None) -> Request:
    return Request(message=message, user_id="u", conv_id="c", action=action)


def _run(coro):
    return asyncio.run(coro)


# ── 规则校验 ──────────────────────────────────────────────────────────────────

def test_citation_without_evidence_flagged():
    v = ResponseVerifier()
    result = _run(v.verify(_req(), "详见 [1]。", [], [], "fast", WRITE_TOOLS))
    assert "citation_without_evidence" in result.flags
    assert result.grounded is False
    assert result.disclaimer == ""


def test_citation_with_evidence_clean():
    v = ResponseVerifier()
    evidence = [{"title": "t", "content": "c", "source_url": "u"}]
    result = _run(v.verify(_req(), "详见 [1]。", [], evidence, "fast", WRITE_TOOLS))
    assert "citation_without_evidence" not in result.flags
    assert result.grounded is True


def test_write_claim_without_tool_flagged():
    v = ResponseVerifier()
    result = _run(v.verify(
        _req("帮我记个待办", action=IntentAction.REQUEST),
        "已添加待办：买饭卡。", [], [], "fast", WRITE_TOOLS,
    ))
    assert "write_claim_without_tool" in result.flags


def test_write_claim_with_tool_clean():
    v = ResponseVerifier()
    result = _run(v.verify(
        _req(), "已添加待办：买饭卡。", ["add_todo"], [], "fast", WRITE_TOOLS,
    ))
    assert "write_claim_without_tool" not in result.flags


def test_unverified_facts_flagged_and_matching_clean():
    v = ResponseVerifier()
    # 回答里出现证据里没有的电话 → flag
    result = _run(v.verify(_req(), "咨询电话 13812345678。", [], [], "fast", WRITE_TOOLS))
    assert "unverified_facts" in result.flags
    # 电话出现在证据里 → 不 flag
    evidence = [{"title": "t", "content": "咨询电话 13812345678", "source_url": ""}]
    result = _run(v.verify(_req(), "咨询电话 13812345678。", [], evidence, "fast", WRITE_TOOLS))
    assert "unverified_facts" not in result.flags


def test_time_context_dates_not_flagged():
    """回答复述今天的日期（系统注入的时间上下文）不算未验证事实。"""
    from personal.time_context import build_time_context

    v = ResponseVerifier()
    ctx = build_time_context()
    m = re.search(r"\d{4}-\d{1,2}-\d{1,2}", ctx)
    if not m:
        return  # 时间上下文不含具体日期（环境差异）：跳过
    month, day = int(m.group(0)[5:7]), int(m.group(0)[8:10])
    result = _run(v.verify(_req(), f"今天是 {month}月{day}日。", [], [], "fast", WRITE_TOOLS))
    assert "unverified_facts" not in result.flags


def test_plain_answer_grounded():
    v = ResponseVerifier()
    result = _run(v.verify(_req(), "请前往教务处办理。", [], [], "fast", WRITE_TOOLS))
    assert result.flags == []
    assert result.grounded is True
    assert result.source == "rules"


def test_empty_content_skipped():
    v = ResponseVerifier()
    result = _run(v.verify(_req(), "", [], [], "fast", WRITE_TOOLS))
    assert result.source == "skip"


# ── LLM 判定（fake client，不触网）───────────────────────────────────────────

def _make_llm_verifier(text: str):
    client = AsyncMock()
    client.messages.create = AsyncMock(return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
    ))
    return ResponseVerifier(client=client, model="m", llm_enabled=True), client


def test_llm_check_runs_on_deep_and_ungrounded_adds_disclaimer():
    verifier, client = _make_llm_verifier('{"grounded": false, "reason": "引用无出处"}')
    result = _run(verifier.verify(
        _req(), "转专业需绩点 3.9。", [], [], "deep", WRITE_TOOLS,
    ))
    assert client.messages.create.called
    assert "llm_ungrounded" in result.flags
    assert result.disclaimer == ResponseVerifier.LLM_DISCLAIMER
    assert result.source == "rules+llm"


def test_llm_check_grounded_ok():
    verifier, client = _make_llm_verifier('{"grounded": true, "reason": "有据"}')
    result = _run(verifier.verify(_req(), "请前往教务处办理。", [], [], "deep", WRITE_TOOLS))
    assert client.messages.create.called
    assert result.flags == []
    assert result.grounded is True


def test_llm_check_skipped_on_fast_profile():
    verifier, client = _make_llm_verifier('{"grounded": true}')
    result = _run(verifier.verify(_req(), "你好。", [], [], "fast", WRITE_TOOLS))
    assert not client.messages.create.called
    assert result.source == "rules"


def test_llm_check_runs_on_request_action_even_fast():
    """执行路径（REQUEST）即使 Fast profile 也做一次 LLM 判定（写操作风险路径）。"""
    verifier, client = _make_llm_verifier('{"grounded": true}')
    result = _run(verifier.verify(
        _req("帮我记个待办", action=IntentAction.REQUEST),
        "已添加待办：买饭卡。", ["add_todo"], [], "fast", WRITE_TOOLS,
    ))
    assert client.messages.create.called
    assert result.source == "rules+llm"


def test_llm_parse_failure_fails_open():
    """LLM 输出非法 JSON → fail-open：不加免责、不阻断。"""
    verifier, client = _make_llm_verifier("不是JSON")
    result = _run(verifier.verify(_req(), "你好。", [], [], "deep", WRITE_TOOLS))
    assert result.source == "rules+llm"
    assert "llm_ungrounded" not in result.flags
    assert result.grounded is True


# ── 编排器集成 ────────────────────────────────────────────────────────────────

def test_orchestrator_run_attaches_verification_meta():
    """单请求路径：execution.verification 存在、规则校验生效、计数可查。"""
    orch = AgentOrchestrator(api_key=FAKE_KEY)  # 默认 LLM 判定关闭（offline 安全）
    req = Request(
        message="你好", user_id="u1", conv_id="c1",
        domain=IntentDomain.OTHER, action=IntentAction.QUERY,
    )

    async def fake_execute(task_req, agent_type, on_event=None):
        return AgentResponse(
            agent_type=AgentType.QA, content="[1] 已添加待办：买饭卡。", success=True,
        )

    orch._execute = fake_execute
    result = asyncio.run(orch.run(req))
    v = result.execution["verification"]
    assert v["source"] == "rules"
    assert "citation_without_evidence" in v["flags"]
    assert "write_claim_without_tool" in v["flags"]
    assert orch.verification_stats()["citation_without_evidence"] == 1
    assert orch.verification_stats()["write_claim_without_tool"] == 1
