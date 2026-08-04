"""多 Agent 编排器测试：领域路由、领域关键词协作、降级、升级检测、追问路由。

所有测试只测确定性逻辑（路由表、关键词、统计），不触发真实 LLM 调用。
"""
from __future__ import annotations

import asyncio

from core.domains import IntentAction, IntentDomain
from core.intent_recognizer import UrgencyLevel
from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    AgentStats,
    AgentType,
    Request,
)

FAKE_KEY = "sk-test-not-used"


def _orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(api_key=FAKE_KEY)


def _req(message: str, domain=None, action=None, urgency=None) -> Request:
    return Request(
        message=message,
        user_id="u1",
        conv_id="c1",
        domain=domain,
        action=action,
        urgency=urgency,
    )


def test_pool_registers_four_campus_agents():
    orch = _orchestrator()
    pool = orch._pool
    assert set(pool.keys()) == {
        AgentType.ACADEMIC,
        AgentType.CAMPUS_LIFE,
        AgentType.AFFAIRS,
        AgentType.IT_HELP,
    }


def test_campus_domains_route_to_corresponding_agents():
    orch = _orchestrator()
    assert orch._route(IntentDomain.ACADEMIC, None) == AgentType.ACADEMIC
    assert orch._route(IntentDomain.CAMPUS_LIFE, None) == AgentType.CAMPUS_LIFE
    assert orch._route(IntentDomain.AFFAIRS, None) == AgentType.AFFAIRS
    assert orch._route(IntentDomain.IT_HELP, None) == AgentType.IT_HELP


def test_request_form_domain_routes_to_domain_agent():
    """P0 回归：请求句式（帮我/我要）路由到领域 Agent，而不是统一兜底学业 Agent。"""
    orch = _orchestrator()
    assert orch._route(IntentDomain.AFFAIRS, None) == AgentType.AFFAIRS
    assert orch._route(IntentDomain.CAMPUS_LIFE, None) == AgentType.CAMPUS_LIFE


def test_escalation_action_marks_result_escalated():
    orch = _orchestrator()

    async def fake_execute(req, agent_type, on_event=None):
        return AgentResponse(agent_type=agent_type, content="已记录，请联系辅导员。", success=True)

    orch._execute = fake_execute  # type: ignore[method-assign]

    async def scenario():
        return await orch.run(
            _req("我要找辅导员", domain=IntentDomain.OTHER, action=IntentAction.ESCALATION)
        )

    result = asyncio.run(scenario())
    assert result.escalated is True


def test_legacy_intent_category_route_compat():
    """兼容：只传旧版 IntentCategory 也能确定性路由。"""
    from core.intent_recognizer import IntentCategory

    orch = _orchestrator()
    assert orch._route(orch._CATEGORY_TO_DOMAIN[IntentCategory.ACADEMIC], None) == AgentType.ACADEMIC


def test_unclassified_domain_falls_back_to_academic():
    orch = _orchestrator()
    assert orch._route(IntentDomain.OTHER, None) == AgentType.ACADEMIC
    assert orch._route(None, None) == AgentType.ACADEMIC


def test_critical_urgency_routes_to_escalation():
    orch = _orchestrator()
    assert orch._route(IntentDomain.ACADEMIC, UrgencyLevel.CRITICAL) == AgentType.ESCALATION


def test_collaboration_detects_multi_domain_question():
    orch = _orchestrator()
    # 复合问题：教务系统打不开 + 选课问题 → IT_HELP + ACADEMIC 并行
    req = _req("教务系统打不开，没法选课了")
    targets = orch._collaboration_targets(req)
    assert AgentType.IT_HELP in targets
    assert AgentType.ACADEMIC in targets


def test_collaboration_single_domain_single_target():
    orch = _orchestrator()
    req = _req("南校区食堂几点关门？")
    targets = orch._collaboration_targets(req)
    assert targets == [AgentType.CAMPUS_LIFE]


def test_collaboration_deduplicates_targets():
    orch = _orchestrator()
    req = _req("校园卡丢了，想申请奖学金，帮我请假", domain=IntentDomain.AFFAIRS)
    targets = orch._collaboration_targets(req)
    deduped = list(dict.fromkeys(targets))
    assert targets == deduped


def test_collaboration_followup_inherits_domain():
    """追问场景：'那几点开门呢？' 无关键词，但领域由意图识别继承。"""
    orch = _orchestrator()
    req = _req("那几点开门呢？", domain=IntentDomain.CAMPUS_LIFE)
    targets = orch._collaboration_targets(req)
    assert targets == [AgentType.CAMPUS_LIFE]


def test_escalation_keyword_detection_in_response():
    orch = _orchestrator()
    agent = orch._best_agent(AgentType.ACADEMIC)
    assert agent._needs_escalation("这个问题需要转人工，联系辅导员处理")
    assert agent._needs_escalation("建议直接找教务老师确认")
    assert not agent._needs_escalation("选课时间以教务系统通知为准")


def test_execute_falls_back_on_failed_agent():
    orch = _orchestrator()

    async def run_fail(req, on_event=None):
        return AgentResponse(agent_type=AgentType.IT_HELP, content="", success=False, latency_ms=1.0)

    agent = orch._best_agent(AgentType.IT_HELP)
    agent.handle = run_fail  # type: ignore[method-assign]

    async def scenario():
        return await orch._execute(_req("教务系统打不开", domain=IntentDomain.IT_HELP), AgentType.IT_HELP)

    response = asyncio.run(scenario())
    # IT 失败后降级到 AcademicAgent 兜底
    assert response.agent_type == AgentType.ACADEMIC


def test_routing_score_prefers_high_success_low_latency():
    good = AgentStats(total=10, success=10, total_ms=1000.0)
    bad = AgentStats(total=10, success=2, total_ms=5000.0)
    assert good.routing_score() > bad.routing_score()
