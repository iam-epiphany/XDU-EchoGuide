"""多 Agent 编排器测试：领域路由、领域关键词协作、降级、追问路由。

所有测试只测确定性逻辑（路由表、关键词、统计），不触发真实 LLM 调用。
"""
from __future__ import annotations

import asyncio

from core.domains import IntentAction, IntentDomain
from core.intent_recognizer import ComplexitySignal
from agents.agent_orchestrator import (
    ACTION_GUIDANCE,
    AGENT_TOOL_ALLOWLIST,
    AgentOrchestrator,
    AgentResponse,
    AgentStats,
    AgentType,
    BaseAgent,
    Request,
    SharedState,
    Task,
    TaskPlanner,
    action_allows_tool,
)

FAKE_KEY = "sk-test-not-used"


def _orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(api_key=FAKE_KEY)


def _req(message: str, domain=None, action=None) -> Request:
    return Request(
        message=message,
        user_id="u1",
        conv_id="c1",
        domain=domain,
        action=action,
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
    assert orch._route(IntentDomain.ACADEMIC) == AgentType.ACADEMIC
    assert orch._route(IntentDomain.CAMPUS_LIFE) == AgentType.CAMPUS_LIFE
    assert orch._route(IntentDomain.AFFAIRS) == AgentType.AFFAIRS
    assert orch._route(IntentDomain.IT_HELP) == AgentType.IT_HELP
    assert orch._route(IntentDomain.PERSONAL) == AgentType.PERSONAL


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
    assert orch._route(IntentDomain.AFFAIRS) == AgentType.AFFAIRS
    assert orch._route(IntentDomain.CAMPUS_LIFE) == AgentType.CAMPUS_LIFE


def test_legacy_intent_category_route_compat():
    """兼容：只传旧版 IntentCategory 也能确定性路由。"""
    from core.intent_recognizer import IntentCategory

    orch = _orchestrator()
    assert orch._route(orch._CATEGORY_TO_DOMAIN[IntentCategory.ACADEMIC]) == AgentType.ACADEMIC


def test_unclassified_domain_falls_back_to_academic():
    orch = _orchestrator()
    assert orch._route(IntentDomain.OTHER) == AgentType.ACADEMIC
    assert orch._route(None) == AgentType.ACADEMIC


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


def test_complexity_gate_keeps_implicit_multi_keyword_request_single():
    orch = _orchestrator()
    req = _req("教务系统打不开，没法选课了", domain=IntentDomain.IT_HELP)
    decision = orch._complexity_decision(req)
    assert decision.mode == "single"


def test_complexity_gate_parallel_requires_explicit_connector():
    orch = _orchestrator()
    req = _req("教务系统登录不上，同时我还想了解选课规则", domain=IntentDomain.IT_HELP)
    decision = orch._complexity_decision(req)
    assert decision.mode == "parallel"
    assert {AgentType.IT_HELP, AgentType.ACADEMIC} <= set(decision.targets)


def test_complexity_gate_detects_dependent_errand():
    orch = _orchestrator()
    req = _req("看看我明天下午有没有课，有空就安排补办校园卡并记个待办", domain=IntentDomain.PERSONAL)
    decision = orch._complexity_decision(req)
    assert decision.mode == "dependent"
    assert AgentType.AFFAIRS in decision.targets


def test_profiles_are_real_flash_and_pro_configs():
    orch = AgentOrchestrator(
        api_key=FAKE_KEY,
        fast_model="deepseek-v4-flash",
        deep_model="deepseek-v4-pro",
    )
    profiles = {agent.profile.name.value: agent.profile for agent in orch._pool[AgentType.ACADEMIC]}
    assert profiles["fast"].model == "deepseek-v4-flash"
    assert profiles["fast"].thinking is False
    assert profiles["deep"].model == "deepseek-v4-pro"
    assert profiles["deep"].thinking is True


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
    assert by_id["t2"].agent_type == AgentType.AFFAIRS
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
    assert [a for a, _ in first_wave] == ["personal", "affairs"]
    # wave2：t3（personal）在依赖完成后执行，并看到前序结果
    dep_calls = [c for c in calls if "协作上下文" in c[1]]
    assert len(dep_calls) == 1
    assert dep_calls[0][0] == "personal"
    assert "personal 的结果" in dep_calls[0][1]
    assert "affairs 的结果" in dep_calls[0][1]
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
        "calculate_weighted_score": {"type": "object", "properties": {"courses": {"type": "array"}}, "required": ["courses"]},
        "query_affairs_process": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]},
        "diagnose_it_issue": {"type": "object", "properties": {"system": {"type": "string"}}},
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


# ── Action 层执行策略（domain 决定 Agent，action 决定 How）────────────────────

def test_action_allows_tool_policy_matrix():
    """Action→Tool 策略矩阵：QUERY 禁写、REQUEST 全放行、GREETING/FEEDBACK 全拒、
    COMPLAINT/OTHER 只读、None 不限制（兼容路径）。"""
    assert action_allows_tool(IntentAction.QUERY, "add_todo") is False
    assert action_allows_tool(IntentAction.QUERY, "complete_todo") is False
    assert action_allows_tool(IntentAction.QUERY, "query_todo") is True
    assert action_allows_tool(IntentAction.QUERY, "knowledge_search") is True

    assert action_allows_tool(IntentAction.REQUEST, "add_todo") is True
    assert action_allows_tool(IntentAction.REQUEST, "query_todo") is True

    assert action_allows_tool(IntentAction.GREETING, "knowledge_search") is False
    assert action_allows_tool(IntentAction.FEEDBACK, "query_todo") is False

    assert action_allows_tool(IntentAction.COMPLAINT, "add_todo") is False
    assert action_allows_tool(IntentAction.COMPLAINT, "query_todo") is True
    assert action_allows_tool(IntentAction.OTHER, "add_todo") is False
    assert action_allows_tool(IntentAction.OTHER, "knowledge_search") is True

    assert action_allows_tool(None, "add_todo") is True


def test_build_tools_query_action_readonly():
    """QUERY：PERSONAL Agent 只暴露只读工具，不暴露写工具（add_todo/complete_todo）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    personal = orch._pool[AgentType.PERSONAL][0]

    names = {t["name"] for t in personal._build_tools(_req("看看我的待办", action=IntentAction.QUERY))}
    assert "query_todo" in names and "query_schedule" in names
    assert "add_todo" not in names and "complete_todo" not in names


def test_build_tools_request_action_full_allowlist():
    """REQUEST：PERSONAL Agent 暴露完整 allowlist（含执行类工具）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    personal = orch._pool[AgentType.PERSONAL][0]

    names = {t["name"] for t in personal._build_tools(_req("帮我记个待办", action=IntentAction.REQUEST))}
    assert names == AGENT_TOOL_ALLOWLIST[AgentType.PERSONAL]
    assert "add_todo" in names


def test_build_tools_greeting_feedback_no_tools():
    """GREETING / FEEDBACK：原则上不开放任何工具。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    campus = orch._pool[AgentType.CAMPUS_LIFE][0]

    assert campus._build_tools(_req("你好", action=IntentAction.GREETING)) == []
    assert campus._build_tools(_req("这个建议很好", action=IntentAction.FEEDBACK)) == []


def test_build_tools_complaint_other_readonly():
    """COMPLAINT / OTHER：保守策略，只开放只读工具。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    personal = orch._pool[AgentType.PERSONAL][0]

    for action in (IntentAction.COMPLAINT, IntentAction.OTHER):
        names = {t["name"] for t in personal._build_tools(_req("不满", action=action))}
        assert "query_todo" in names
        assert "add_todo" not in names and "complete_todo" not in names


def test_execute_tool_query_blocks_write_tool():
    """防御纵深：QUERY 动作下 Agent 直接调用写工具 → 拒绝（含"权限"），不执行。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    personal = orch._pool[AgentType.PERSONAL][0]

    data, error = asyncio.run(personal._execute_tool(
        "add_todo", {"content": "买饭卡"}, _req("查一下我的待办", action=IntentAction.QUERY)))
    assert data is None
    assert "权限" in error


def test_execute_tool_request_allows_write_tool():
    """REQUEST 动作下写工具正常走执行链（不被权限拦截）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    personal = orch._pool[AgentType.PERSONAL][0]

    data, error = asyncio.run(personal._execute_tool(
        "add_todo", {"content": "买饭卡"}, _req("帮我记个待办", action=IntentAction.REQUEST)))
    assert error is None  # 放行：fake handler 返回 []
    assert data == []


def test_system_prompt_injects_action_guidance():
    """Action 指引注入 system prompt：各动作注入对应行为指令，None 不注入。"""
    orch = _orchestrator()
    academic = orch._pool[AgentType.ACADEMIC][0]

    prompt = academic._build_system_prompt(_req("hi", action=IntentAction.QUERY))
    assert "[意图指引]" in prompt
    assert ACTION_GUIDANCE[IntentAction.QUERY] in prompt

    prompt = academic._build_system_prompt(_req("hi", action=IntentAction.REQUEST))
    assert "积极调用工具解决问题" in prompt

    prompt = academic._build_system_prompt(_req("hi", action=IntentAction.COMPLAINT))
    assert "识别具体问题点" in prompt

    prompt = academic._build_system_prompt(_req("hi", action=IntentAction.GREETING))
    assert "无需调用工具" in prompt

    # None：不注入意图指引段（保持原有 prompt 结构）
    prompt = academic._build_system_prompt(_req("hi"))
    assert "[意图指引]" not in prompt


def test_run_task_backfill_blocked_on_query_action():
    """Executor 补执行（required_tool）遵守 Action 策略：QUERY 下不补写操作。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())

    async def fake_execute(task_req, agent_type, on_event=None):
        return AgentResponse(agent_type=agent_type, content="查询结果", success=True, tools_used=[])

    orch._execute = fake_execute
    task = Task(
        task_id="t3",
        agent_type=AgentType.PERSONAL,
        goal="记录待办",
        message="帮我记个待办",
        required_tool="add_todo",
        required_tool_args={"content": "补办校园卡"},
    )
    req = _req("帮我记个待办", domain=IntentDomain.PERSONAL, action=IntentAction.QUERY)
    result = asyncio.run(orch._run_task(req, task, SharedState()))
    assert result.success
    assert "已按协作计划记录待办" not in result.content
    assert result.tools_used == []


def test_run_task_backfill_executes_on_request_action():
    """REQUEST 动作下补执行正常落地（写工具被执行并回填）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())

    async def fake_execute(task_req, agent_type, on_event=None):
        return AgentResponse(agent_type=agent_type, content="办理建议", success=True, tools_used=[])

    orch._execute = fake_execute
    task = Task(
        task_id="t3",
        agent_type=AgentType.PERSONAL,
        goal="记录待办",
        message="帮我记个待办",
        required_tool="add_todo",
        required_tool_args={"content": "补办校园卡"},
    )
    req = _req("帮我记个待办", domain=IntentDomain.PERSONAL, action=IntentAction.REQUEST)
    result = asyncio.run(orch._run_task(req, task, SharedState()))
    assert "已按协作计划记录待办" in result.content
    assert "add_todo" in result.tools_used


# ── LLM 复杂度判定（规则筛 + LLM 升级确认）───────────────────────────────────

def _signal(mode, targets=None, reason="", tasks=None) -> ComplexitySignal:
    return ComplexitySignal(mode=mode, targets=targets or [], reason=reason, tasks=tasks)


def test_complexity_from_llm_valid_parallel_maps_targets():
    orch = _orchestrator()
    decision = orch._complexity_from_llm(_signal("parallel", ["campus_life", "personal"]), _req("x"))
    assert decision is not None
    assert decision.mode == "parallel"
    assert set(decision.targets) == {AgentType.CAMPUS_LIFE, AgentType.PERSONAL}


def test_complexity_from_llm_drops_unknown_domains():
    """未知领域丢弃；全部非法 → None（调用方回落关键词规则）。"""
    orch = _orchestrator()
    req = _req("x")
    decision = orch._complexity_from_llm(_signal("parallel", ["campus_life", "unknown_domain"]), req)
    assert decision is not None
    assert decision.targets == [AgentType.CAMPUS_LIFE]
    assert orch._complexity_from_llm(_signal("parallel", ["unknown_domain"]), req) is None
    assert orch._complexity_from_llm(None, req) is None
    assert orch._complexity_from_llm(_signal("weird"), req) is None


def test_complexity_from_llm_single_uses_route_fallback():
    orch = _orchestrator()
    req = _req("随便聊聊", domain=IntentDomain.CAMPUS_LIFE)
    decision = orch._complexity_from_llm(_signal("single", []), req)
    assert decision is not None and decision.mode == "single"
    assert decision.targets == [AgentType.CAMPUS_LIFE]


def test_complexity_from_llm_dependent_builds_tasks():
    """dependent：LLM 任务链通过校验并映射为 Task 列表。"""
    orch = _orchestrator()
    req = _req("我下午有空想办校园卡再记个待办")
    decision = orch._complexity_from_llm(_signal(
        "dependent", ["personal", "affairs"],
        tasks=[
            {"id": "t1", "agent": "personal", "goal": "查空闲", "message": "查我的空闲时间"},
            {"id": "t2", "agent": "affairs", "depends_on": ["t1"]},
        ],
    ), req)
    assert decision is not None
    assert decision.mode == "dependent"
    assert decision.tasks is not None
    assert [t.task_id for t in decision.tasks] == ["t1", "t2"]
    assert decision.tasks[1].depends_on == ["t1"]
    # message 缺失 → 回落自包含格式（含原始用户请求）
    assert "用户请求" in decision.tasks[1].message


def test_complexity_from_llm_dependent_bad_chain_falls_back():
    """dependent 任务链非法（成环 / 缺失）→ 整链作废 → None。"""
    orch = _orchestrator()
    req = _req("我下午有空想办校园卡再记个待办")
    assert orch._complexity_from_llm(_signal(
        "dependent", ["personal", "affairs"],
        tasks=[
            {"id": "t1", "agent": "personal", "depends_on": ["t2"]},
            {"id": "t2", "agent": "affairs", "depends_on": ["t1"]},
        ],
    ), req) is None
    assert orch._complexity_from_llm(_signal("dependent", ["personal"]), req) is None


def test_tasks_from_llm_rejects_invalid_chains():
    orch = _orchestrator()
    req = _req("x")
    # depends_on 引用不存在的 id
    assert orch._tasks_from_llm([{"id": "t1", "agent": "personal", "depends_on": ["t9"]}], req) is None
    # 超过 6 个任务（Executor 上限）
    assert orch._tasks_from_llm(
        [{"id": f"t{i}", "agent": "personal"} for i in range(7)], req,
    ) is None
    # 未知领域
    assert orch._tasks_from_llm([{"id": "t1", "agent": "unknown"}], req) is None
    # id 重复
    assert orch._tasks_from_llm(
        [{"id": "t1", "agent": "personal"}, {"id": "t1", "agent": "affairs"}], req,
    ) is None
    # 空列表 / 非列表
    assert orch._tasks_from_llm([], req) is None
    assert orch._tasks_from_llm("not-a-list", req) is None


def test_needs_llm_complexity_heuristics():
    orch = _orchestrator()
    # 短句单领域 → 不升级（免费路径直答）
    assert orch._needs_llm_complexity(_req("南校区食堂几点关门？")) is False
    # ≥3 从句 → 升级
    assert orch._needs_llm_complexity(
        _req("帮我看看明天有没有课，然后查一下图书馆几点关门，再提醒我交作业")) is True
    # 长句多从句（换说法的复合请求）→ 升级
    assert orch._needs_llm_complexity(
        _req("这周哪天能挤出空，得跑一趟把学费结清，完了帮我定个提醒")) is True
    # 多领域关键词但无连接词（旧规则判 single 的"隐式复合"）→ 升级
    assert orch._needs_llm_complexity(_req("教务系统打不开，没法选课了")) is True


def test_run_parallel_uses_llm_tasks_when_provided():
    """LLM 依赖链传入 run_parallel 时直接采用，不被 Planner 规则覆盖。"""
    orch = _orchestrator()
    req = _req("我下午有空想办校园卡再记个待办")
    plan = [
        Task(task_id="L1", agent_type=AgentType.PERSONAL, goal="查空闲", message="查我的空闲时间"),
        Task(task_id="L2", agent_type=AgentType.AFFAIRS, goal="查办理", message="查办理信息",
             depends_on=["L1"]),
    ]
    calls = []

    async def fake_execute(task_req, agent_type, on_event=None):
        calls.append((task_req.message, agent_type))
        return AgentResponse(agent_type=agent_type, content=f"{agent_type.value} 结果", success=True)

    orch._execute = fake_execute
    result = asyncio.run(orch.run_parallel(
        req, [AgentType.PERSONAL, AgentType.AFFAIRS], tasks=plan,
    ))
    # 两个任务按依赖分波执行（L1 先、L2 后），消息来自 LLM 链而非 Planner
    assert [msg for msg, _ in calls] == ["查我的空闲时间", "查办理信息"]
    assert result.response


def test_run_reuses_llm_complexity_when_stage_llm():
    """意图识别走 LLM 时，run() 复用其 complexity 输出（parallel → run_parallel）。"""
    from core.intent_recognizer import IntentAction, IntentResult

    orch = _orchestrator()
    req = _req("食堂几点关门，顺便查下明天课表")

    async def fake_recognize(message, history=None, force_llm=False):
        return IntentResult(
            domain=IntentDomain.CAMPUS_LIFE, action=IntentAction.QUERY,
            intent=None, confidence=0.6, entities={}, reasoning="llm",
            latency_ms=1.0, classifier_stage="llm",
            complexity=_signal("parallel", ["campus_life", "personal"]),
        )

    calls = []

    async def fake_execute(task_req, agent_type, on_event=None):
        calls.append(agent_type)
        return AgentResponse(agent_type=agent_type, content=f"{agent_type.value} 结果", success=True)

    orch._intent_recognizer.recognize = fake_recognize
    orch._execute = fake_execute
    result = asyncio.run(orch.run(req))
    assert result.execution["mode"] == "parallel"
    assert {AgentType.CAMPUS_LIFE, AgentType.PERSONAL} <= set(calls)


def test_run_upgrade_path_adopts_llm_judgment():
    """规则判 single + 预筛命中 → judge_complexity 升级；LLM 合法结论被采用。"""
    from core.intent_recognizer import IntentAction, IntentResult

    orch = _orchestrator()
    req = _req("教务系统打不开，没法选课了")   # 多领域词但无连接词 → 升级信号

    async def fake_recognize(message, history=None, force_llm=False):
        return IntentResult(
            domain=IntentDomain.IT_HELP, action=IntentAction.QUERY,
            intent=None, confidence=0.95, entities={}, reasoning="pattern",
            latency_ms=1.0, classifier_stage="pattern",
        )

    async def fake_judge(message, history=None):
        return _signal("parallel", ["it_help", "academic"], reason="两个领域")

    async def fake_execute(task_req, agent_type, on_event=None):
        return AgentResponse(agent_type=agent_type, content="ok", success=True)

    orch._intent_recognizer.recognize = fake_recognize
    orch._intent_recognizer.judge_complexity = fake_judge
    orch._execute = fake_execute
    result = asyncio.run(orch.run(req))
    assert result.execution["mode"] == "parallel"
    assert result.execution["complexity_reason"] == "两个领域"


def test_run_upgrade_path_falls_back_when_llm_invalid():
    """升级后 LLM 不可用/输出非法 → 回落规则结论，行为不比现状差。"""
    from core.intent_recognizer import IntentAction, IntentResult

    orch = _orchestrator()
    req = _req("教务系统打不开，没法选课了")

    async def fake_recognize(message, history=None, force_llm=False):
        return IntentResult(
            domain=IntentDomain.IT_HELP, action=IntentAction.QUERY,
            intent=None, confidence=0.95, entities={}, reasoning="pattern",
            latency_ms=1.0, classifier_stage="pattern",
        )

    async def fake_judge(message, history=None):
        return None   # LLM 不可用

    async def fake_execute(task_req, agent_type, on_event=None):
        return AgentResponse(agent_type=agent_type, content="ok", success=True)

    orch._intent_recognizer.recognize = fake_recognize
    orch._intent_recognizer.judge_complexity = fake_judge
    orch._execute = fake_execute
    result = asyncio.run(orch.run(req))
    assert result.execution["mode"] == "single"


def test_run_single_agent_benchmark_forces_single():
    """benchmark single_agent：即使规则判 parallel，也强制压回单 Agent。"""
    orch = _orchestrator()
    req = _req("食堂几点关门，顺便查下明天课表", domain=IntentDomain.CAMPUS_LIFE)
    req.benchmark_strategy = "single_agent"

    async def fake_execute(task_req, agent_type, on_event=None):
        return AgentResponse(agent_type=agent_type, content="ok", success=True)

    orch._execute = fake_execute
    result = asyncio.run(orch.run(req))
    assert result.execution["mode"] == "single"
    assert result.agent_type == AgentType.CAMPUS_LIFE
