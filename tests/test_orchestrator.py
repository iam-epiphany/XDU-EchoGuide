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
    BaseAgent,
    Request,
    TaskPlanner,
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


def test_pool_registers_five_campus_agents():
    """个人助理领域扩展后，Agent 池应包含五大领域。"""
    orch = _orchestrator()
    pool = orch._pool
    assert set(pool.keys()) == {
        AgentType.ACADEMIC,
        AgentType.CAMPUS_LIFE,
        AgentType.AFFAIRS,
        AgentType.IT_HELP,
        AgentType.PERSONAL,
    }


def test_campus_domains_route_to_corresponding_agents():
    orch = _orchestrator()
    assert orch._route(IntentDomain.ACADEMIC, None) == AgentType.ACADEMIC
    assert orch._route(IntentDomain.CAMPUS_LIFE, None) == AgentType.CAMPUS_LIFE
    assert orch._route(IntentDomain.AFFAIRS, None) == AgentType.AFFAIRS
    assert orch._route(IntentDomain.IT_HELP, None) == AgentType.IT_HELP
    assert orch._route(IntentDomain.PERSONAL, None) == AgentType.PERSONAL


def test_personal_schedule_question_routes_to_personal_agent():
    """"今天有什么课"类个人日程问题应路由到 PersonalAgent。"""
    orch = _orchestrator()
    req = _req("今天有什么课？", domain=IntentDomain.PERSONAL)
    targets = orch._collaboration_targets(req)
    assert targets == [AgentType.PERSONAL]


def test_collaboration_personal_prefers_over_academic():
    """"考试安排"同时命中 academic 与 personal 时，personal 优先（避免回答分裂）。"""
    orch = _orchestrator()
    req = _req("我最近的考试安排是什么？", domain=IntentDomain.PERSONAL)
    targets = orch._collaboration_targets(req)
    assert AgentType.PERSONAL in targets
    assert AgentType.ACADEMIC not in targets


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

    # Agent 池每类型有 2 个实例：全部 patch 为失败，验证降级到 AcademicAgent 兜底
    for agent in orch._pool[AgentType.IT_HELP]:
        agent.handle = run_fail  # type: ignore[method-assign]

    async def scenario():
        return await orch._execute(_req("教务系统打不开", domain=IntentDomain.IT_HELP), AgentType.IT_HELP)

    response = asyncio.run(scenario())
    # IT 失败后降级到 AcademicAgent 兜底
    assert response.agent_type == AgentType.ACADEMIC


def test_routing_switches_instance_after_penalty():
    """性能路由闭环：实例 0 被 Monitor 惩罚后，_best_agent 应切换到实例 1。"""
    orch = _orchestrator()
    pool = orch._pool[AgentType.ACADEMIC]
    assert len(pool) == 2  # 每类型双实例

    # 初始同分：max 稳定取第一个实例
    assert orch._best_agent(AgentType.ACADEMIC) is pool[0]

    # Monitor 对实例 0 施加惩罚后，实例 1 接管
    orch.update_routing_penalties({"academic_0": 0.9})
    assert orch._best_agent(AgentType.ACADEMIC) is pool[1]

    # 惩罚清除后回到实例 0
    orch.update_routing_penalties({"academic_0": 0.0})
    assert orch._best_agent(AgentType.ACADEMIC) is pool[0]


def test_routing_score_prefers_high_success_low_latency():
    good = AgentStats(total=10, success=10, total_ms=1000.0)
    bad = AgentStats(total=10, success=2, total_ms=5000.0)
    assert good.routing_score() > bad.routing_score()


# ── 工具调用循环（Agentic RAG）──────────────────────────────────────────────

class _Block:
    """伪 Anthropic content block。"""

    def __init__(self, type: str, **kw):
        self.type = type
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeResp:
    def __init__(self, content, stop_reason):
        self.content = content
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, fake):
        self._fake = fake

    async def create(self, **kwargs):
        return self._fake._create(kwargs)


class _FakeClient:
    """顺序返回预设响应的伪客户端（记录每次调用参数）。"""

    def __init__(self, responses):
        self.messages = _FakeMessages(self)
        self.responses = list(responses)
        self.seen = []

    def _create(self, kwargs):
        self.seen.append(kwargs)
        return self.responses.pop(0)


class _ToolAgent(BaseAgent):
    agent_type = AgentType.CAMPUS_LIFE
    system_prompt = "测试 Agent"


def _tool_agent(responses):
    from mcp.tool_manager import MCPToolManager, Tool

    tm = MCPToolManager(api_key=FAKE_KEY)

    async def echo(params, context):
        return {"echo": params.get("text", "")}

    tm.register(Tool(
        name="echo",
        description="回显工具",
        handler=echo,
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
    ))
    client = _FakeClient(responses)
    agent = _ToolAgent(client, model="test-model", tool_manager=tm)
    # 测试专用：显式授权 echo 工具（默认权限表只放行各领域职责内工具）
    agent._tool_allowlist = {"echo"}
    return agent, client


def test_multi_tool_use_results_merged_into_single_message():
    """P0 回归：一轮多个 tool_use 时，所有 tool_result 必须合并进同一条
    user 消息（逐条分开会触发 Anthropic 兼容端点的 400）。"""
    agent, client = _tool_agent([
        _FakeResp(
            content=[
                _Block("tool_use", id="tu1", name="echo", input={"text": "a"}),
                _Block("tool_use", id="tu2", name="echo", input={"text": "b"}),
            ],
            stop_reason="tool_use",
        ),
        _FakeResp(content=[_Block("text", text="最终答案")], stop_reason="end_turn"),
    ])
    result = asyncio.run(agent.handle(_req("测试", domain=IntentDomain.CAMPUS_LIFE)))
    assert result.success is True
    assert result.content == "最终答案"

    # 第二次调用：assistant（含 2 个 tool_use）后紧跟唯一一条 user 消息
    msgs = client.seen[1]["messages"]
    assert msgs[-1]["role"] == "user"
    assert msgs[-2]["role"] == "assistant"
    assert len([b for b in msgs[-2]["content"] if b.type == "tool_use"]) == 2
    assert len(msgs[-1]["content"]) == 2
    assert all(b["type"] == "tool_result" for b in msgs[-1]["content"])
    assert msgs[-1]["content"][0]["tool_use_id"] == "tu1"
    assert msgs[-1]["content"][1]["tool_use_id"] == "tu2"


def test_tool_round_limit_finishes_with_results_filled():
    """回归：达到轮次上限时，工具结果已全部回填，收尾调用正常完成并流式/普通收尾。"""
    agent, client = _tool_agent([
        _FakeResp(content=[_Block("tool_use", id="tu1", name="echo", input={"text": "a"})], stop_reason="tool_use"),
        _FakeResp(content=[_Block("tool_use", id="tu2", name="echo", input={"text": "b"})], stop_reason="tool_use"),
        _FakeResp(content=[_Block("text", text="收尾答案")], stop_reason="end_turn"),
    ])
    result = asyncio.run(agent.handle(_req("测试", domain=IntentDomain.CAMPUS_LIFE)))
    assert result.success is True
    assert result.content == "收尾答案"
    assert result.tools_used == ["echo", "echo"]

    # 收尾调用（第 3 次）：每条 assistant tool_use 后都紧跟同一条 user 消息的
    # tool_result（tu2 已真实执行回填），不存在孤儿 tool_use 引发 400
    msgs = client.seen[2]["messages"]
    assert msgs[-1]["role"] == "user"
    assert len(msgs[-1]["content"]) == 1
    assert msgs[-1]["content"][0]["type"] == "tool_result"
    assert msgs[-1]["content"][0]["tool_use_id"] == "tu2"


# ── 轻量多 Agent 协作（Planner / Executor / SharedState / Synthesizer）────────

def test_planner_rule_generates_dependency_chain():
    """复合规则命中：t1/t2 无依赖并行，t3 依赖 t1+t2（同一 Agent 类型可多任务）。"""
    orch = _orchestrator()
    req = _req("我明天下午有空，想去办校园卡，帮我记个待办")
    tasks = TaskPlanner().plan(req, [AgentType.PERSONAL, AgentType.CAMPUS_LIFE])
    by_id = {t.task_id: t for t in tasks}
    assert set(by_id) == {"t1", "t2", "t3"}
    assert by_id["t1"].agent_type == AgentType.PERSONAL
    assert by_id["t2"].agent_type == AgentType.CAMPUS_LIFE
    assert by_id["t3"].agent_type == AgentType.PERSONAL      # 同一 Agent 类型多任务
    assert by_id["t3"].depends_on == ["t1", "t2"]            # 依赖前序任务
    assert by_id["t1"].depends_on == [] and by_id["t2"].depends_on == []
    # 任务自包含：message 携带目标与用户请求
    assert "用户请求" in by_id["t1"].message


def test_planner_fallback_generates_parallel_tasks():
    """未命中规则：每个领域一个任务，无依赖（通用降级）。"""
    orch = _orchestrator()
    req = _req("教务系统打不开，选课怎么办")
    tasks = TaskPlanner().plan(req, [AgentType.IT_HELP, AgentType.ACADEMIC])
    assert len(tasks) == 2
    assert all(not t.depends_on for t in tasks)


def test_parallel_executor_runs_waves_and_injects_shared_state():
    """
    执行器分波执行：wave1 = t1/t2 并行（无协作上下文），
    wave2 = t3 依赖完成后执行，context 注入前序 Agent 结果（SharedState 真正生效）。
    Synthesizer LLM 不可用（FAKE_KEY）时降级为规则拼接。
    """
    orch = _orchestrator()
    req = _req("我明天下午有空，想去办校园卡，帮我记个待办")

    calls: list = []  # (agent_type, context)

    async def fake_execute(task_req, agent_type, on_event=None):
        calls.append((agent_type.value, task_req.context or ""))
        return AgentResponse(
            agent_type=agent_type,
            content=f"{agent_type.value} 的结果",
            success=True,
        )

    orch._execute = fake_execute
    result = asyncio.run(orch.run_parallel(
        req, [AgentType.PERSONAL, AgentType.CAMPUS_LIFE],
    ))

    # wave1：t1/t2 并行执行，无协作上下文
    first_wave = [c for c in calls if "协作上下文" not in c[1]]
    assert [a for a, _ in first_wave] == ["personal", "campus_life"]
    # wave2：t3（personal）在依赖完成后执行，并看到前序结果
    dep_calls = [c for c in calls if "协作上下文" in c[1]]
    assert len(dep_calls) == 1
    assert dep_calls[0][0] == "personal"
    assert "personal 的结果" in dep_calls[0][1]
    assert "campus_life 的结果" in dep_calls[0][1]
    # 合成器降级拼接
    assert result.response
    assert result.tools_used == []


def test_parallel_synthesizer_failure_degrades_to_concat():
    """Synthesizer LLM 失败 → 规则拼接（主链路可用）。"""
    orch = _orchestrator()
    req = _req("南校区食堂几点关门，顺便帮我查下图书馆开放时间")

    async def fake_execute(task_req, agent_type, on_event=None):
        return AgentResponse(agent_type=agent_type, content=f"{agent_type.value} 回答", success=True)

    orch._execute = fake_execute
    result = asyncio.run(orch.run_parallel(req, [AgentType.CAMPUS_LIFE, AgentType.AFFAIRS]))
    # 无规则命中 → 2 个领域任务并行 + 合成失败降级拼接
    assert "[campus_life]" in result.response
    assert "[affairs]" in result.response


# ── Agent 工具权限边界 ───────────────────────────────────────────────────────

def _fake_tool_manager():
    from mcp.tool_manager import MCPToolManager, Tool

    tm = MCPToolManager(api_key=FAKE_KEY)

    async def noop(params, context):
        return []

    for name, schema in {
        "knowledge_search": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "query_schedule":   {"type": "object", "properties": {"date": {"type": "string"}}},
        "query_todo":       {"type": "object", "properties": {"status": {"type": "string"}}},
        "add_todo":         {"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
        "complete_todo":    {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]},
        "query_ddl":        {"type": "object", "properties": {"horizon_days": {"type": "integer"}}},
        "query_campus_info": {"type": "object", "properties": {"category": {"type": "string"}}, "required": ["category"]},
        "get_weather":      {"type": "object", "properties": {"place": {"type": "string"}}},
    }.items():
        tm.register(Tool(name=name, description=f"{name} 工具", handler=noop, schema=schema))
    return tm


def test_build_tools_respects_agent_allowlist():
    """工具权限边界：每个 Agent 只暴露职责内工具（最小权限）。"""
    from agents.agent_orchestrator import AGENT_TOOL_ALLOWLIST

    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    for agent_type, agents in orch._pool.items():
        for agent in agents:
            names = {t["name"] for t in agent._build_tools()}
            assert names == AGENT_TOOL_ALLOWLIST[agent_type]

    # 典型断言：Personal 有个人工具；Academic 无个人工具
    personal = orch._pool[AgentType.PERSONAL][0]
    names = {t["name"] for t in personal._build_tools()}
    assert {"query_schedule", "add_todo", "query_ddl"} <= names
    academic = orch._pool[AgentType.ACADEMIC][0]
    names = {t["name"] for t in academic._build_tools()}
    assert "knowledge_search" in names and "query_schedule" not in names


def test_execute_tool_rejects_out_of_scope_tool():
    """防御纵深：Agent 尝试调用权限外工具被直接拒绝，不执行。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    academic = orch._pool[AgentType.ACADEMIC][0]

    data, error = asyncio.run(academic._execute_tool("query_schedule", {"date": "今天"}, _req("hi")))
    assert data is None
    assert "权限" in error

    # 权限内工具正常走执行链（无 handler 结果时返回失败但非权限拒绝）
    data, error = asyncio.run(academic._execute_tool("knowledge_search", {"query": "选课"}, _req("hi")))
    assert "权限" not in (error or "")
