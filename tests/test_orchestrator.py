"""多 Agent 编排器测试：Planner/ExecutionPlan、领域挂载、降级、任务执行。

所有测试只测确定性逻辑（Planner 规则、Task DAG、统计、权限），不触发真实 LLM 调用。
"""
from __future__ import annotations

import asyncio

from core.domains import IntentAction, IntentDomain
from agents.agent_orchestrator import (
    AgentOrchestrator,
    AgentResponse,
    Request,
)
from agents.persona import ACTION_GUIDANCE, DOMAIN_PERSONA, action_allows_tool
from agents.profiles import ProfileName
from agents.roles import AgentStats, BaseAgent, TaskAgent, WritePolicy, write_policy_for
from agents.workflow import ExecutionPlan, SharedState, Task, TaskPlanner

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


def _run(coro):
    return asyncio.run(coro)


# ── 执行实例（每 Profile 单实例，无同构实例池）──────────────────────────────

def test_execution_instances_per_profile():
    """每 Profile 一个执行实例（Fast/Deep），Agent 类只有 TaskAgent。

    同构复制多个 TaskAgent 没有能力差异（并发由 asyncio 承担），
    收口为单实例；未来异构 Model/Provider 路由在此扩展。
    """
    orch = _orchestrator()
    agents = orch._agents
    assert set(agents.keys()) == {ProfileName.FAST, ProfileName.DEEP}
    assert isinstance(agents[ProfileName.FAST], TaskAgent)
    assert isinstance(agents[ProfileName.DEEP], TaskAgent)


def test_execution_instances_match_profile():
    """Fast 实例只带 Fast 配置，Deep 实例只带 Deep 配置。"""
    orch = _orchestrator()
    assert orch._agents[ProfileName.FAST].profile.name == ProfileName.FAST
    assert orch._agents[ProfileName.DEEP].profile.name == ProfileName.DEEP


def test_agent_returns_profile_instance_only():
    """_agent(profile) 只可能返回该 Profile 的执行实例（不跨 Profile）。"""
    orch = _orchestrator()
    assert orch._agent(ProfileName.DEEP) is orch._agents[ProfileName.DEEP]
    assert orch._agent(ProfileName.FAST) is orch._agents[ProfileName.FAST]
    assert orch._agent() is orch._agents[ProfileName.FAST]  # 缺省 Fast


def test_write_policy_maps_action_to_execution_policy():
    """执行策略（原 QA/EXECUTOR 角色降级）：REQUEST → WRITE_ALLOWED；
    其余动作与未知动作 → READ_ONLY（防御纵深：非 REQUEST 一律只读）。"""
    assert write_policy_for(IntentAction.REQUEST) == WritePolicy.WRITE_ALLOWED
    assert write_policy_for(IntentAction.QUERY) == WritePolicy.READ_ONLY
    assert write_policy_for(IntentAction.GREETING) == WritePolicy.READ_ONLY
    assert write_policy_for(IntentAction.COMPLAINT) == WritePolicy.READ_ONLY
    assert write_policy_for(None) == WritePolicy.READ_ONLY


def test_domain_persona_covers_all_domains_and_other():
    """领域人格覆盖五大领域 + OTHER（通用接待），这是领域分类的唯一产物。"""
    assert set(DOMAIN_PERSONA.keys()) == {
        IntentDomain.ACADEMIC, IntentDomain.CAMPUS_LIFE, IntentDomain.AFFAIRS,
        IntentDomain.IT_HELP, IntentDomain.PERSONAL, IntentDomain.OTHER,
    }


# ── Planner：ExecutionPlan（Fast Path 本地规则）─────────────────────────────

def test_planner_fast_path_single_task():
    """Fast Path：明显单领域请求 → 1 个 Task（domain/action 来自意图），mode=single。"""
    orch = _orchestrator()
    req = _req("南校区食堂几点关门？", domain=IntentDomain.CAMPUS_LIFE, action=IntentAction.QUERY)
    plan = _run(orch._planner.plan(req, req.domain, req.action))
    assert plan.mode == "single"
    assert len(plan.tasks) == 1
    assert plan.tasks[0].domain == IntentDomain.CAMPUS_LIFE
    assert plan.tasks[0].action == IntentAction.QUERY
    assert plan.tasks[0].message == req.message


def test_planner_rule_generates_dependency_chain():
    """复合规则命中：t1/t2 无依赖并行（QUERY），t3 依赖 t1+t2（REQUEST）。"""
    orch = _orchestrator()
    req = _req("我明天下午有空，想去办校园卡，帮我记个待办")
    plan = _run(orch._planner.plan(req, req.domain, req.action))
    by_id = {t.task_id: t for t in plan.tasks}
    assert plan.mode == "dependent"
    assert set(by_id) == {"t1", "t2", "t3"}
    # 每个任务有自己的 action：t1/t2 查询 → QUERY，t3 写操作 → REQUEST
    assert by_id["t1"].domain == IntentDomain.PERSONAL
    assert by_id["t1"].action == IntentAction.QUERY
    assert by_id["t2"].domain == IntentDomain.AFFAIRS
    assert by_id["t2"].action == IntentAction.QUERY
    assert by_id["t3"].domain == IntentDomain.PERSONAL
    assert by_id["t3"].action == IntentAction.REQUEST
    assert by_id["t3"].depends_on == ["t1", "t2"]            # 依赖前序任务
    assert by_id["t1"].depends_on == [] and by_id["t2"].depends_on == []
    # 任务自包含：message 携带目标与用户请求
    assert "用户请求" in by_id["t1"].message


def test_planner_parallel_for_multi_domain_with_connector():
    """多领域 + 显式连接词 → 并行任务（mode=parallel），无依赖。"""
    orch = _orchestrator()
    req = _req("教务系统登录不上，同时我还想了解选课规则", domain=IntentDomain.IT_HELP)
    plan = _run(orch._planner.plan(req, req.domain, req.action))
    assert plan.mode == "parallel"
    domains = {t.domain for t in plan.tasks}
    assert IntentDomain.IT_HELP in domains and IntentDomain.ACADEMIC in domains
    assert all(not t.depends_on for t in plan.tasks)


def test_planner_dedup_personal_prefers_over_academic():
    """"考试安排"同时命中 academic 与 personal 时，personal 优先（避免回答分裂）。"""
    orch = _orchestrator()
    req = _req("我最近的考试安排是什么？", domain=IntentDomain.PERSONAL)
    targets = orch._planner._collaboration_targets(req, IntentDomain.PERSONAL)
    assert IntentDomain.PERSONAL in targets
    assert IntentDomain.ACADEMIC not in targets


def test_task_domain_label_is_direct_domain():
    """Task.domain 直接使用 IntentDomain（领域只做挂载键，不参与 Agent 类选择）。"""
    task = Task(task_id="t", domain=IntentDomain.AFFAIRS, goal="g", message="m")
    assert task.domain == IntentDomain.AFFAIRS


# ── Planner：LLM 规划升级（校验逻辑，不触发真实 LLM）────────────────────────

def test_tasks_from_llm_rejects_invalid_chains():
    orch = _orchestrator()
    req = _req("x")
    # depends_on 引用不存在的 id
    assert orch._planner._tasks_from_llm(
        [{"id": "t1", "domain": "personal", "depends_on": ["t9"]}], req,
    ) is None
    # 超过 6 个任务（Executor 上限）
    assert orch._planner._tasks_from_llm(
        [{"id": f"t{i}", "domain": "personal"} for i in range(7)], req,
    ) is None
    # 未知领域
    assert orch._planner._tasks_from_llm([{"id": "t1", "domain": "unknown"}], req) is None
    # OTHER 领域不接受（任务角色必须是具体领域）
    assert orch._planner._tasks_from_llm([{"id": "t1", "domain": "other"}], req) is None
    # 非法 action
    assert orch._planner._tasks_from_llm(
        [{"id": "t1", "domain": "personal", "action": "weird"}], req,
    ) is None
    # id 重复
    assert orch._planner._tasks_from_llm(
        [{"id": "t1", "domain": "personal"}, {"id": "t1", "domain": "affairs"}], req,
    ) is None
    # 空列表 / 非列表
    assert orch._planner._tasks_from_llm([], req) is None
    assert orch._planner._tasks_from_llm("not-a-list", req) is None


def test_tasks_from_llm_valid_chain_with_action():
    """合法任务链：action 保留（QUERY/REQUEST 决定执行角色），message 缺失回落自包含。"""
    orch = _orchestrator()
    req = _req("我下午有空想办校园卡再记个待办")
    tasks = orch._planner._tasks_from_llm([
        {"id": "t1", "domain": "personal", "action": "query", "goal": "查空闲", "message": "查我的空闲时间"},
        {"id": "t2", "domain": "affairs", "action": "query", "depends_on": ["t1"]},
        {"id": "t3", "domain": "personal", "action": "request", "depends_on": ["t2"]},
    ], req)
    assert tasks is not None
    assert [t.task_id for t in tasks] == ["t1", "t2", "t3"]
    assert tasks[0].action == IntentAction.QUERY
    assert tasks[2].action == IntentAction.REQUEST
    assert tasks[1].depends_on == ["t1"]
    # message 缺失 → 回落自包含格式（含原始用户请求）
    assert "用户请求" in tasks[1].message


def test_tasks_from_llm_normalizes_write_task_to_personal_domain():
    """写工具只操作个人数据，LLM 不能把创建提醒放进知识领域。"""
    orch = _orchestrator()
    tasks = orch._planner._tasks_from_llm([
        {"id": "t1", "domain": "affairs", "action": "request", "goal": "创建缴费提醒"},
    ], _req("缴费后帮我创建提醒"))
    assert tasks is not None
    assert tasks[0].action == IntentAction.REQUEST
    assert tasks[0].domain == IntentDomain.PERSONAL


def test_tasks_from_llm_bad_chain_falls_back():
    """依赖链非法（成环 / 缺失）→ 整链作废 → None。"""
    orch = _orchestrator()
    req = _req("我下午有空想办校园卡再记个待办")
    assert orch._planner._tasks_from_llm([
        {"id": "t1", "domain": "personal", "depends_on": ["t2"]},
        {"id": "t2", "domain": "affairs", "depends_on": ["t1"]},
    ], req) is None


def test_needs_llm_planning_heuristics():
    orch = _orchestrator()
    planner = orch._planner
    # 短句单领域 → 不升级（免费路径直答）
    assert planner._needs_llm_planning(_req("南校区食堂几点关门？")) is False
    # ≥3 从句 → 升级
    assert planner._needs_llm_planning(
        _req("帮我看看明天有没有课，然后查一下图书馆几点关门，再提醒我交作业")) is True
    # 长句多从句（换说法的复合请求）→ 升级
    assert planner._needs_llm_planning(
        _req("这周哪天能挤出空，得跑一趟把学费结清，完了帮我定个提醒")) is True
    # 多领域关键词但无连接词（"隐式复合"）→ 升级
    assert planner._needs_llm_planning(_req("教务系统打不开，没法选课了")) is True


def test_llm_plan_invalid_falls_back_to_fast():
    """LLM 规划失败/不可用 → 回落本地 Fast Path（行为不比现状差）。"""
    orch = _orchestrator()

    async def fake_llm_plan(req):
        return None  # LLM 不可用

    orch._planner._llm_plan = fake_llm_plan  # type: ignore[method-assign]
    req = _req("教务系统打不开，没法选课了")   # 升级信号命中
    plan = _run(orch._planner.plan(req, IntentDomain.IT_HELP, IntentAction.QUERY))
    assert plan.mode == "single"  # 回落 Fast Path 单任务
    assert plan.tasks[0].domain == IntentDomain.IT_HELP


def test_llm_single_task_domain_or_action_drift_falls_back_to_fast():
    """单任务规划不得覆盖已完成的顶层意图识别，避免执行到错误的工具域。"""
    orch = _orchestrator()

    async def fake_llm_plan(req):
        return ExecutionPlan([
            Task(
                task_id="t0",
                domain=IntentDomain.CAMPUS_LIFE,
                action=IntentAction.QUERY,
                goal="错误的领域",
                message=req.message,
            )
        ])

    orch._planner._llm_plan = fake_llm_plan  # type: ignore[method-assign]
    req = _req("校园卡丢了，补办需要什么材料？")
    plan = _run(orch._planner.plan(req, IntentDomain.AFFAIRS, IntentAction.QUERY))

    assert plan.mode == "single"
    assert plan.tasks[0].domain == IntentDomain.AFFAIRS
    assert plan.tasks[0].action == IntentAction.QUERY


# ── Profile 决策 ─────────────────────────────────────────────────────────────

def test_profiles_are_real_flash_and_pro_configs():
    orch = AgentOrchestrator(
        api_key=FAKE_KEY,
        fast_model="deepseek-v4-flash",
        deep_model="deepseek-v4-pro",
    )
    profiles = {p.value: a.profile for p, a in orch._agents.items()}
    assert profiles["fast"].model == "deepseek-v4-flash"
    assert profiles["fast"].thinking is False
    assert profiles["deep"].model == "deepseek-v4-pro"
    assert profiles["deep"].thinking is True


def test_execute_fast_failure_retries_deep():
    """Fast 执行失败 → Deep 重试（同 Task Run，Profile 间降级）。"""
    orch = _orchestrator()
    fast = orch._agents[ProfileName.FAST]
    deep = orch._agents[ProfileName.DEEP]

    async def run_fail(req, on_event=None):
        return AgentResponse(content="", success=False, latency_ms=1.0)

    async def run_ok(req, on_event=None):
        return AgentResponse(content="deep 结果", success=True, latency_ms=1.0)

    fast.handle = run_fail  # type: ignore[method-assign]
    deep.handle = run_ok  # type: ignore[method-assign]

    req = _req("教务系统打不开", domain=IntentDomain.IT_HELP)
    req.profile = ProfileName.FAST
    response = _run(orch._execute(req))
    assert response.success
    assert response.content == "deep 结果"


def test_stats_profile_level_with_percentiles():
    """Profile 统计：成功率/延迟/P50/P95（无实例级路由分）。"""
    orch = _orchestrator()
    stats = orch.get_stats()
    assert set(stats.keys()) == {"fast", "deep"}
    assert all("profile" in s and "p50_ms" in s and "p95_ms" in s for s in stats.values())

    # AgentStats 记录延迟后给出 P50/P95（排序后取分位）
    s = AgentStats()
    for ms in (100, 200, 300, 400, 500):
        s.record(ms)
    assert s.p50_ms == 300.0
    assert s.p95_ms == 500.0


def test_fast_unhealthy_upgrades_to_deep():
    """Monitor 反馈：Fast 不健康 → 本应 Fast 的请求临时升级 Deep（有限反馈）。"""
    orch = _orchestrator()
    req = _req("南校区食堂几点关门？", domain=IntentDomain.CAMPUS_LIFE)
    req.classifier_stage = "pattern"
    # 默认健康：Fast 路径
    assert orch._select_profile(req, "single") == ProfileName.FAST
    # Monitor 标记不健康：临时升级 Deep
    orch.set_fast_health(False)
    assert orch._select_profile(req, "single") == ProfileName.DEEP
    # 恢复健康：回落 Fast
    orch.set_fast_health(True)
    assert orch._select_profile(req, "single") == ProfileName.FAST


# ── Monitor Fast 健康判定（_fast_health，按 profile 字段识别）──────────────────

def _monitor_fast_health(stats):
    from monitor.performance_monitor import PerformanceMonitor
    return PerformanceMonitor._fast_health(stats)


def test_fast_health_detects_unhealthy_fast_via_profile_field():
    """Profile 级统计（key=fast/deep）：Fast 样本足且成功率 <0.85 → 不健康。"""
    stats = {
        "fast": {"total": 10, "success_rate": 0.80, "profile": "fast"},
        "deep": {"total": 10, "success_rate": 0.99, "profile": "deep"},
    }
    assert _monitor_fast_health(stats) is False


def test_fast_health_healthy_when_samples_insufficient_or_ok():
    """样本不足视为健康；成功率达标视为健康；无 Fast 记录视为健康。"""
    # 样本不足（total < 10）→ 不干预
    assert _monitor_fast_health({
        "fast": {"total": 9, "success_rate": 0.5, "profile": "fast"},
    }) is True
    # 成功率达标 → 健康
    assert _monitor_fast_health({
        "fast": {"total": 10, "success_rate": 0.90, "profile": "fast"},
    }) is True
    # 只有 Deep → 健康
    assert _monitor_fast_health({
        "deep": {"total": 10, "success_rate": 0.5, "profile": "deep"},
    }) is True
    assert _monitor_fast_health({}) is True


def test_fast_health_ignores_key_format():
    """不依赖 key 名称：只要 profile 字段是 fast 就能识别（兼容旧 key 形状）。"""
    assert _monitor_fast_health({
        "instance_a": {"total": 10, "success_rate": 0.70, "profile": "fast"},
        "instance_b": {"total": 10, "success_rate": 0.99, "profile": "deep"},
    }) is False


def test_monitor_feedback_marks_fast_unhealthy_end_to_end():
    """Monitor 反馈闭环：真实 get_stats() 形状 → _fast_health 判定 → set_fast_health(False)
    → 本应走 Fast 的请求被升级 Deep。"""
    orch = _orchestrator()
    stats = orch.get_stats()
    assert stats and all("profile" in s for s in stats.values())
    # 模拟采集数据：所有 Fast 实例样本足 + 低成功率
    for s in stats.values():
        if s["profile"] == "fast":
            s["total"] = 10
            s["success_rate"] = 0.50
    orch.set_fast_health(_monitor_fast_health(stats))
    assert orch.fast_unhealthy is True
    req = _req("南校区食堂几点关门？", domain=IntentDomain.CAMPUS_LIFE)
    req.classifier_stage = "pattern"
    assert orch._select_profile(req, "single") == ProfileName.DEEP


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
    system_prompt = "测试 Agent"


def _tool_agent(responses):
    from mcp.tool_manager import MCPToolManager, Tool, ToolEffect

    tm = MCPToolManager(api_key=FAKE_KEY)

    async def echo(params, context):
        return {"echo": params.get("text", "")}

    tm.register(Tool(
        name="echo",
        description="回显工具",
        handler=echo,
        schema={"type": "object", "properties": {"text": {"type": "string"}}},
        effect=ToolEffect.READ,  # 显式副作用声明（fail-closed）
    ))
    client = _FakeClient(responses)
    agent = _ToolAgent(client, model="test-model", tool_manager=tm)
    # 测试专用：显式授权 echo 工具
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
    result = _run(agent.handle(_req("测试", domain=IntentDomain.CAMPUS_LIFE)))
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
    result = _run(agent.handle(_req("测试", domain=IntentDomain.CAMPUS_LIFE)))
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


# ── 轻量多 Agent 协作（Executor / SharedState / Synthesizer）────────────────

def test_parallel_executor_runs_waves_and_injects_shared_state():
    """
    执行器分波执行：wave1 = t1/t2 并行（无协作上下文），
    wave2 = t3 依赖完成后执行，context 注入前序 Agent 结果（SharedState 真正生效）。
    Synthesizer LLM 不可用（FAKE_KEY）时降级为规则拼接。
    """
    orch = _orchestrator()
    req = _req("我明天下午有空，想去办校园卡，帮我记个待办")

    calls: list = []  # (任务领域, context)

    async def fake_execute(task_req, on_event=None):
        calls.append((task_req.domain.value, task_req.context or ""))
        return AgentResponse(
            agent_type=task_req.domain.value if task_req.domain else "task_agent",
            content=f"{task_req.domain.value} 的结果",
            success=True,
        )

    orch._execute = fake_execute
    plan = _run(orch._planner.plan(req, req.domain, req.action))
    result = _run(orch.run_parallel(req, plan))

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
    req = _req("食堂几点关门，顺便帮我查下奖学金的申请流程")

    async def fake_execute(task_req, on_event=None):
        label = task_req.domain.value if task_req.domain else "task_agent"
        return AgentResponse(agent_type=label, content=f"{label} 回答", success=True)

    orch._execute = fake_execute
    plan = _run(orch._planner.plan(req, IntentDomain.CAMPUS_LIFE, IntentAction.QUERY))
    result = _run(orch.run_parallel(req, plan))
    # 并行任务 + 合成失败降级拼接
    assert "[campus_life]" in result.response
    assert "[affairs]" in result.response


def test_dag_failure_propagates_to_blocked():
    """DAG 失败传播：t1 失败 → 依赖它的 t3 BLOCKED（不执行、不注入失败上下文）。"""
    orch = _orchestrator()
    req = _req("我明天下午有空，想去办校园卡，帮我记个待办")

    calls = []

    async def fake_execute(task_req, on_event=None):
        calls.append(task_req.domain.value)
        # t1（personal 查课表）失败；t2（affairs）成功
        if task_req.domain == IntentDomain.PERSONAL and "空闲" in task_req.message:
            return AgentResponse(content="查询失败", success=False)
        return AgentResponse(content="affairs 结果", success=True)

    orch._execute = fake_execute
    plan = _run(orch._planner.plan(req, req.domain, req.action))
    shared = _run(orch._executor.execute(req, plan.tasks, max_tasks=6))

    # t1 FAILED → t3 BLOCKED（不执行）；t2 正常 SUCCESS
    assert shared.status("t1") == "failed"
    assert shared.status("t2") == "success"
    assert shared.status("t3") == "blocked"
    # 被 BLOCKED 的任务不产生工具调用（calls 只有 t1/t2）
    assert "personal" in calls and "affairs" in calls
    # 失败/阻塞结果不注入协作上下文快照（只读取声明的依赖）
    t3 = next(t for t in plan.tasks if t.task_id == "t3")
    snapshot = shared.snapshot_for(t3)
    assert "查询失败" not in snapshot
    assert "affairs 结果" in snapshot


def test_dag_success_chain_allows_dependent():
    """依赖链全部成功 → 依赖任务正常执行（BLOCKED 不误伤）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())  # t3 的 required_tool 补执行需要工具管理器
    req = _req("我明天下午有空，想去办校园卡，帮我记个待办")

    async def fake_execute(task_req, on_event=None):
        return AgentResponse(content=f"{task_req.domain.value} 结果", success=True)

    orch._execute = fake_execute
    plan = _run(orch._planner.plan(req, req.domain, req.action))
    shared = _run(orch._executor.execute(req, plan.tasks, max_tasks=6))

    assert shared.status("t1") == "success"
    assert shared.status("t2") == "success"
    assert shared.status("t3") == "success"
    # 依赖任务只读取自己声明的依赖结果（t1/t2）
    t3 = next(t for t in plan.tasks if t.task_id == "t3")
    snapshot = shared.snapshot_for(t3)
    assert "personal 结果" in snapshot and "affairs 结果" in snapshot


def test_task_duration_measures_each_task_independently():
    """同 wave 并行任务的 duration_ms 各自独立计时（修复：统一用 wave 耗时导致
    同波任务都接近最长任务耗时）。"""
    orch = _orchestrator()
    req = _req("食堂几点关门，顺便查下明天课表", domain=IntentDomain.CAMPUS_LIFE)

    async def fake_execute(task_req, on_event=None):
        if task_req.task_id == "t1":
            await asyncio.sleep(0.15)
        else:
            await asyncio.sleep(0.02)
        return AgentResponse(content="ok", success=True)

    orch._execute = fake_execute
    tasks = [
        Task(task_id="t1", domain=IntentDomain.CAMPUS_LIFE, goal="g", message="m1"),
        Task(task_id="t2", domain=IntentDomain.IT_HELP, goal="g", message="m2"),
    ]
    shared = _run(orch._executor.execute(req, tasks, max_tasks=6))
    meta = {m["id"]: m for m in shared.task_meta()}
    assert meta["t1"]["duration_ms"] > meta["t2"]["duration_ms"] + 80
    assert 10 < meta["t2"]["duration_ms"] < 200  # 快任务没有被慢任务拖到 wave 总耗时


def test_task_execute_span_recorded_per_task():
    """Task 级 Trace：每个 TaskAgent Run 产生独立 task_execute span（task_id/domain/action/depends_on），
    与既有 agent_handle/llm/tool span 并存（t1 失败传播到 t3 BLOCKED 时 t3 不产生 span）。"""
    from core.tracing import begin_trace, end_trace

    orch = _orchestrator()
    req = _req("我明天下午有空，想去办校园卡，帮我记个待办")

    async def fake_execute(task_req, on_event=None):
        if task_req.domain == IntentDomain.PERSONAL and "空闲" in task_req.message:
            return AgentResponse(content="查询失败", success=False)
        return AgentResponse(content="affairs 结果", success=True)

    orch._execute = fake_execute
    plan = _run(orch._planner.plan(req, req.domain, req.action))

    trace = begin_trace()
    try:
        _run(orch._executor.execute(req, plan.tasks, max_tasks=6))
    finally:
        ended = end_trace()
    assert ended is not None
    spans = ended.spans

    task_spans = [s for s in spans if s.name == "task_execute"]
    assert len(task_spans) == 2  # t1/t2 执行；t3 因依赖失败 BLOCKED 不执行
    by_id = {s.meta.get("task_id"): s for s in task_spans}
    assert set(by_id) == {"t1", "t2"}
    assert by_id["t1"].meta["domain"] == "personal"
    assert by_id["t2"].meta["action"] == "query"
    assert by_id["t1"].meta["depends_on"] == ""


# ── Agent 工具权限边界 ───────────────────────────────────────────────────────

def _fake_tool_manager():
    from mcp.tool_manager import MCPToolManager, Tool, ToolEffect

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
        # 写工具由副作用声明推导（effect=WRITE）：add_todo / complete_todo 进入写集合
        tm.register(Tool(
            name=name, description=f"{name} 工具", handler=noop, schema=schema,
            effect=ToolEffect.WRITE if name in ("add_todo", "complete_todo") else ToolEffect.READ,
        ))
    return tm


def test_build_tools_read_only_policy_surface():
    """QUERY（READ_ONLY）：公共工具层 − 写工具（Run 级写策略，与领域无关）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]
    names = {t["name"] for t in agent._build_tools(
        _req("看看我的待办", action=IntentAction.QUERY))}
    assert names == {
        "knowledge_search", "query_schedule", "query_todo", "query_ddl",
        "query_campus_info", "get_weather", "calculate_weighted_score",
        "query_affairs_process", "diagnose_it_issue",
    }
    assert "add_todo" not in names and "complete_todo" not in names


def test_build_tools_write_allowed_policy_full_surface():
    """REQUEST（WRITE_ALLOWED）：全量公共工具层（含写工具）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]
    names = {t["name"] for t in agent._build_tools(
        _req("帮我记个待办", action=IntentAction.REQUEST))}
    assert {"add_todo", "complete_todo"} <= names
    assert "diagnose_it_issue" in names


def test_execute_tool_respects_instance_allowlist_override():
    """实例级 _tool_allowlist 覆盖公共层（测试/定制）：范围外工具直接拒绝。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]
    agent._tool_allowlist = {"knowledge_search"}

    data, error = _run(agent._execute_tool("query_schedule", {"date": "今天"}, _req("hi")))
    assert data is None
    assert "权限" in error

    # 覆盖范围内工具正常走执行链（无 handler 结果时返回失败但非权限拒绝）
    data, error = _run(agent._execute_tool("knowledge_search", {"query": "选课"}, _req("hi")))
    assert "权限" not in (error or "")


# ── Skill 工具（渐进披露：完整 SKILL.md 按需加载）─────────────────────────────

def _fake_skill_manager():
    """临时目录 SkillManager：academic / campus_life 两个 Skill，构造即加载。"""
    import tempfile
    from pathlib import Path

    from core.skill_loader import SkillManager

    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    for slug, name, desc, kws in [
        ("academic", "学业咨询规范", "选课等学业问题答复规范", "选课,课表,考试"),
        ("campus_life", "校园生活向导规范", "食堂校车等生活问题答复规范", "食堂,校车"),
    ]:
        d = root / slug
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"""---
name: {name}
description: {desc}
keywords: {kws}
enabled: true
---

# {name}
{name}的完整正文。""",
            encoding="utf-8",
        )
    mgr = SkillManager(root_dir=str(root))
    mgr.load()
    return tmp, mgr


def test_build_tools_exposes_single_skill_loader():
    """唯一 load_skill 工具追加进公共工具层，与 MCP 工具并存。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    tmp, mgr = _fake_skill_manager()
    try:
        orch.set_skill_manager(mgr)
        agent = orch._agents[ProfileName.FAST]
        names = {t["name"] for t in agent._build_tools(_req("选课什么时候开始", action=IntentAction.QUERY))}
        assert "load_skill" in names
        assert "knowledge_search" in names  # MCP 工具不受影响
    finally:
        tmp.cleanup()


def test_build_tools_skill_tools_hidden_on_greeting():
    """GREETING/FEEDBACK 动作下 Skill 工具与普通工具一起隐藏（Action 门禁一致）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    tmp, mgr = _fake_skill_manager()
    try:
        orch.set_skill_manager(mgr)
        agent = orch._agents[ProfileName.FAST]
        assert agent._build_tools(_req("你好", action=IntentAction.GREETING)) == []
        assert agent._build_tools(_req("这个建议很好", action=IntentAction.FEEDBACK)) == []
    finally:
        tmp.cleanup()


def test_execute_tool_serves_skill_content_locally():
    """load_skill 被 _execute_tool 拦截：完整正文本地返回，不经过 MCPToolManager。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    tmp, mgr = _fake_skill_manager()
    try:
        orch.set_skill_manager(mgr)
        agent = orch._agents[ProfileName.FAST]
        data, error = _run(agent._execute_tool("load_skill", {"skill_name": "academic"}, _req("选课")))
        assert error is None
        assert "学业咨询规范的完整正文" in data
        # 未知 Skill id 返回错误文本
        data, error = _run(agent._execute_tool("load_skill", {"skill_name": "unknown"}, _req("选课")))
        assert data is None
        assert "不存在" in error
        # 普通工具路径不受影响（无 handler → 执行失败但非权限拒绝）
        data, error = _run(agent._execute_tool("query_schedule", {"date": "今天"}, _req("hi")))
        assert "权限" not in (error or "")
    finally:
        tmp.cleanup()


def test_execute_tool_skill_loader_respects_allowlist():
    """实例级 _tool_allowlist 同样约束 Skill 加载（防御纵深）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    tmp, mgr = _fake_skill_manager()
    try:
        orch.set_skill_manager(mgr)
        agent = orch._agents[ProfileName.FAST]
        agent._tool_allowlist = {"load_skill"}
        data, error = _run(agent._execute_tool("query_schedule", {}, _req("食堂")))
        assert data is None
        assert "权限" in error
        data, error = _run(agent._execute_tool("load_skill", {"skill_name": "academic"}, _req("选课")))
        assert error is None
        assert "学业咨询规范的完整正文" in data
    finally:
        tmp.cleanup()


# ── Action 层执行策略（domain 决定 Agent，action 决定 How）────────────────────

def test_action_allows_tool_policy_matrix():
    """Action→Tool 策略矩阵：QUERY 禁写、REQUEST 全放行、GREETING/FEEDBACK 全拒、
    COMPLAINT/OTHER 只读、None 不限制（兼容路径）。写工具集合来自 Tool.write 声明。"""
    write_tools = frozenset({"add_todo", "complete_todo"})
    assert action_allows_tool(IntentAction.QUERY, "add_todo", write_tools) is False
    assert action_allows_tool(IntentAction.QUERY, "complete_todo", write_tools) is False
    assert action_allows_tool(IntentAction.QUERY, "query_todo", write_tools) is True
    assert action_allows_tool(IntentAction.QUERY, "knowledge_search", write_tools) is True

    assert action_allows_tool(IntentAction.REQUEST, "add_todo", write_tools) is True
    assert action_allows_tool(IntentAction.REQUEST, "query_todo", write_tools) is True

    assert action_allows_tool(IntentAction.GREETING, "knowledge_search", write_tools) is False
    assert action_allows_tool(IntentAction.FEEDBACK, "query_todo", write_tools) is False

    assert action_allows_tool(IntentAction.COMPLAINT, "add_todo", write_tools) is False
    assert action_allows_tool(IntentAction.COMPLAINT, "query_todo", write_tools) is True
    assert action_allows_tool(IntentAction.OTHER, "add_todo", write_tools) is False
    assert action_allows_tool(IntentAction.OTHER, "knowledge_search", write_tools) is True

    assert action_allows_tool(None, "add_todo", write_tools) is True


def test_write_tools_derived_from_tool_declaration():
    """写工具集合从 Tool.effect 声明推导（WRITE / EXTERNAL_SIDE_EFFECT 入写集合）。"""
    tm = _fake_tool_manager()
    assert tm.write_tools() == frozenset({"add_todo", "complete_todo"})


def test_undeclared_effect_tool_fail_closed():
    """未声明 effect 的工具 fail-closed：不暴露给 Agent、不在写集合、不可执行。"""
    from mcp.tool_manager import MCPToolManager, Tool

    tm = MCPToolManager(api_key=FAKE_KEY)

    async def side_effect(params, context):
        return {"done": True}  # 实际有副作用但忘记声明 effect

    tm.register(Tool(
        name="forgot_effect",
        description="忘记声明副作用的工具",
        handler=side_effect,
        schema={"type": "object", "properties": {}},
        agent_exposed=True,
    ))
    assert "forgot_effect" not in tm.write_tools()  # 不在写集合

    orch = _orchestrator()
    orch.set_tool_manager(tm)
    agent = orch._agents[ProfileName.FAST]
    # 暴露层：不可见（fail-closed，不会被只读动作误当只读工具开放）
    names = {t["name"] for t in agent._build_tools(_req("查一下", action=IntentAction.QUERY))}
    assert "forgot_effect" not in names
    # 执行层：直接拒绝
    data, error = _run(agent._execute_tool("forgot_effect", {}, _req("执行", action=IntentAction.REQUEST)))
    assert data is None
    assert "权限" in error


def test_build_tools_query_action_readonly():
    """QUERY：只暴露只读工具，不暴露写工具（Action 层门禁）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]

    names = {t["name"] for t in agent._build_tools(_req("看看我的待办", action=IntentAction.QUERY))}
    assert "query_todo" in names and "query_schedule" in names
    assert "add_todo" not in names and "complete_todo" not in names


def test_build_tools_request_action_full_surface():
    """REQUEST：暴露完整公共工具层（含执行类工具）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]

    names = {t["name"] for t in agent._build_tools(_req("帮我记个待办", action=IntentAction.REQUEST))}
    assert "add_todo" in names
    assert "query_schedule" in names and "diagnose_it_issue" in names


def test_build_tools_default_read_only_blocks_writes():
    """Run 级写策略缺省（动作未知/None）：不暴露写工具（防御纵深）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]

    names = {t["name"] for t in agent._build_tools(_req("帮我记个待办"))}
    assert "query_schedule" in names
    assert "add_todo" not in names and "complete_todo" not in names


def test_build_tools_greeting_feedback_no_tools():
    """GREETING / FEEDBACK：原则上不开放任何工具。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]

    assert agent._build_tools(_req("你好", action=IntentAction.GREETING)) == []
    assert agent._build_tools(_req("这个建议很好", action=IntentAction.FEEDBACK)) == []


def test_build_tools_complaint_other_readonly():
    """COMPLAINT / OTHER：保守策略，只开放只读工具。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]

    for action in (IntentAction.COMPLAINT, IntentAction.OTHER):
        names = {t["name"] for t in agent._build_tools(_req("不满", action=action))}
        assert "query_todo" in names
        assert "add_todo" not in names and "complete_todo" not in names


def test_execute_tool_query_blocks_write_tool():
    """防御纵深：QUERY 动作下写权限角色调用写工具 → 拒绝（Action 门禁），不执行。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]

    data, error = _run(agent._execute_tool(
        "add_todo", {"content": "买饭卡"}, _req("查一下我的待办", action=IntentAction.QUERY)))
    assert data is None
    assert "权限" in error


def test_execute_tool_default_read_only_blocks_write_tool():
    """Run 级写策略缺省（动作未知）：拒绝写工具（防御纵深）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]

    data, error = _run(agent._execute_tool(
        "add_todo", {"content": "买饭卡"}, _req("帮我记个待办")))
    assert data is None
    assert "权限" in error


def test_execute_tool_request_allows_write_tool():
    """REQUEST（WRITE_ALLOWED）写工具正常走执行链（不被权限拦截）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]

    data, error = _run(agent._execute_tool(
        "add_todo", {"content": "买饭卡"}, _req("帮我记个待办", action=IntentAction.REQUEST)))
    assert error is None  # 放行：fake handler 返回 []
    assert data == []


def test_system_prompt_injects_action_guidance():
    """Action 指引注入 system prompt：各动作注入对应行为指令，None 不注入。"""
    orch = _orchestrator()
    agent = orch._agents[ProfileName.FAST]

    prompt = agent._build_system_prompt(_req("hi", action=IntentAction.QUERY))
    assert "[意图指引]" in prompt
    assert ACTION_GUIDANCE[IntentAction.QUERY] in prompt

    prompt = agent._build_system_prompt(_req("hi", action=IntentAction.REQUEST))
    assert "积极调用工具解决问题" in prompt

    prompt = agent._build_system_prompt(_req("hi", action=IntentAction.COMPLAINT))
    assert "识别具体问题点" in prompt

    prompt = agent._build_system_prompt(_req("hi", action=IntentAction.GREETING))
    assert "无需调用工具" in prompt

    # None：不注入意图指引段（保持原有 prompt 结构）
    prompt = agent._build_system_prompt(_req("hi"))
    assert "[意图指引]" not in prompt


def test_system_prompt_mounts_domain_persona():
    """领域人格按 domain 挂载（只提供语境）；Task goal 注入 [任务目标]。"""
    orch = _orchestrator()
    agent = orch._agents[ProfileName.FAST]

    prompt = agent._build_system_prompt(_req("hi", domain=IntentDomain.IT_HELP))
    assert "[领域人格]" in prompt
    assert "校园 IT 支持语境" in prompt
    assert "diagnose_it_issue" not in prompt

    prompt = agent._build_system_prompt(_req("hi", domain=IntentDomain.OTHER))
    assert DOMAIN_PERSONA[IntentDomain.OTHER] in prompt

    prompt = agent._build_system_prompt(_req("hi", domain=None))
    assert "校园 IT 支持语境" not in prompt  # 无领域：不注入任何领域人格内容

    # Task-scoped：goal 注入 [任务目标]（独立于其他 Task 的指令）
    goal_req = _req("hi", domain=IntentDomain.IT_HELP)
    goal_req.goal = "诊断教务系统登录问题"
    prompt = agent._build_system_prompt(goal_req)
    assert "[任务目标]" in prompt
    assert "诊断教务系统登录问题" in prompt


def test_run_task_backfill_blocked_on_query_action():
    """Executor 补执行（required_tool）遵守 Action 策略：QUERY 下不补写操作。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())

    async def fake_execute(task_req, on_event=None):
        return AgentResponse(content="查询结果", success=True, tools_used=[])

    orch._execute = fake_execute
    task = Task(
        task_id="t3",
        domain=IntentDomain.PERSONAL,
        action=IntentAction.QUERY,
        goal="记录待办",
        message="帮我记个待办",
        required_tool="add_todo",
        required_tool_args={"content": "补办校园卡"},
    )
    req = _req("帮我记个待办", domain=IntentDomain.PERSONAL, action=IntentAction.QUERY)
    result = _run(orch._run_task(req, task, SharedState()))
    assert result.success
    assert "已按协作计划记录待办" not in result.content
    assert result.tools_used == []


def test_run_task_backfill_executes_on_request_action():
    """REQUEST 动作（任务自己的 action）下补执行正常落地（写工具被执行并回填）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())

    async def fake_execute(task_req, on_event=None):
        return AgentResponse(content="办理建议", success=True, tools_used=[])

    orch._execute = fake_execute
    task = Task(
        task_id="t3",
        domain=IntentDomain.PERSONAL,
        action=IntentAction.REQUEST,
        goal="记录待办",
        message="帮我记个待办",
        required_tool="add_todo",
        required_tool_args={"content": "补办校园卡"},
    )
    req = _req("帮我记个待办", domain=IntentDomain.PERSONAL, action=IntentAction.REQUEST)
    result = _run(orch._run_task(req, task, SharedState()))
    assert "已按协作计划记录待办" in result.content
    assert "add_todo" in result.tools_used
def test_run_task_backfill_goes_through_runtime_tool_boundary():
    """补执行必须走统一 Runtime Tool 边界：before/after 钩子触发，
    工具调用计数进 RunState（与 Agent 工具循环口径一致，不绕过 Runtime）。"""
    from runtime.policy import ExecutionPolicy
    from runtime.runtime import AgentRuntime
    from runtime.state import RunState

    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    runtime = AgentRuntime(policy=ExecutionPolicy())
    orch._runtime = runtime
    for agent in orch._agents.values():
        agent._runtime = runtime

    async def fake_execute(task_req, on_event=None):
        return AgentResponse(content="办理建议", success=True, tools_used=[])

    orch._execute = fake_execute
    task = Task(
        task_id="t3",
        domain=IntentDomain.PERSONAL,
        action=IntentAction.REQUEST,
        goal="记录待办",
        message="帮我记个待办",
        required_tool="add_todo",
        required_tool_args={"content": "补办校园卡"},
    )
    req = _req("帮我记个待办", domain=IntentDomain.PERSONAL, action=IntentAction.REQUEST)
    req.state = RunState(
        request_id="r1", user_id="u1", conv_id="c1",
        message=req.message, policy=runtime.policy,
    )
    result = _run(orch._run_task(req, task, SharedState()))
    assert "已按协作计划记录待办" in result.content
    # before_tool 计数：补执行的写工具调用计入 RunState（不绕过 Runtime 边界）
    assert req.state.tool_call_count == 1




# ── 编排器主链路（run）──────────────────────────────────────────────────────

def test_run_single_executes_one_task_agent_run():
    """单任务请求只产生一次 TaskAgent Run；执行策略由 task.action 决定。"""
    orch = _orchestrator()
    req = _req("帮我添加一个补办校园卡的待办", domain=IntentDomain.PERSONAL, action=IntentAction.REQUEST)
    calls = []

    async def fake_execute(task_req, on_event=None):
        calls.append((task_req.action.value, task_req.goal))
        return AgentResponse(agent_type=task_req.domain.value, content="ok", success=True)

    orch._execute = fake_execute
    result = _run(orch.run(req))
    assert result.execution["mode"] == "single"
    # 1 次 TaskAgent Run，action 决定 Run 执行策略（REQUEST → WRITE_ALLOWED）
    assert calls == [("request", "从个人助理角度回答用户的请求（我的课表/待办/考试安排等）")]


def test_run_parallel_executes_via_planner():
    """run() 完整闭环：意图（已给）→ Planner 判 parallel → 多任务执行。"""
    orch = _orchestrator()
    req = _req("食堂几点关门，顺便查下明天课表", domain=IntentDomain.CAMPUS_LIFE, action=IntentAction.QUERY)

    async def fake_execute(task_req, on_event=None):
        return AgentResponse(agent_type=task_req.domain.value, content="ok", success=True)

    orch._execute = fake_execute
    result = _run(orch.run(req))
    assert result.execution["mode"] == "parallel"
    assert result.agent_type == "campus_life"


def test_run_upgrade_path_falls_back_when_llm_invalid():
    """升级后 LLM 不可用/输出非法 → 回落本地规则，行为不比现状差。"""
    orch = _orchestrator()
    req = _req("教务系统打不开，没法选课了")

    async def fake_llm_plan(req):
        return None  # LLM 不可用

    orch._planner._llm_plan = fake_llm_plan  # type: ignore[method-assign]

    async def fake_execute(task_req, on_event=None):
        return AgentResponse(agent_type=task_req.domain.value, content="ok", success=True)

    orch._execute = fake_execute
    result = _run(orch.run(req))
    assert result.execution["mode"] == "single"


def test_run_single_agent_benchmark_forces_single():
    """benchmark single_agent：即使 Planner 判 parallel，也强制压回单 Agent。"""
    orch = _orchestrator()
    req = _req("食堂几点关门，顺便查下明天课表", domain=IntentDomain.CAMPUS_LIFE)
    req.benchmark_strategy = "single_agent"

    async def fake_execute(task_req, on_event=None):
        return AgentResponse(agent_type=task_req.domain.value, content="ok", success=True)

    orch._execute = fake_execute
    result = _run(orch.run(req))
    assert result.execution["mode"] == "single"
    assert result.agent_type == "campus_life"


def test_needs_knowledge_consumed_by_verifier():
    """needs_knowledge=true 且无检索证据 → execution.verification 标记 expected_retrieval_missing。"""
    orch = _orchestrator()
    req = _req("转专业有什么条件？", domain=IntentDomain.ACADEMIC, action=IntentAction.QUERY)
    req.state_query = {"needs_knowledge": True}

    async def fake_execute(task_req, on_event=None):
        return AgentResponse(agent_type=task_req.domain.value, content="转专业需要绩点达标。", success=True)

    orch._execute = fake_execute
    result = _run(orch.run(req))
    v = result.execution["verification"]
    assert "expected_retrieval_missing" in v["flags"]
    assert orch.verification_stats()["expected_retrieval_missing"] == 1


# ── Task 级写能力（allowed_write_tools，最小权限）──────────────────────────

def test_build_tools_respects_task_write_capability():
    """任务级写能力：REQUEST 任务只能看到 allowed_write_tools 内的写工具；
    读工具不受限（与 Action 策略取交集）。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]
    req = _req("帮我记个待办", action=IntentAction.REQUEST)
    req.allowed_write_tools = ["add_todo"]

    names = {t["name"] for t in agent._build_tools(req)}
    assert "add_todo" in names              # 白名单内写工具可见
    assert "complete_todo" not in names     # 白名单外写工具不可见
    assert "knowledge_search" in names      # 读工具不受限
    assert "query_schedule" in names


def test_execute_tool_denies_outside_task_write_capability():
    """执行前再次校验：allowed_write_tools 之外的写工具被硬拒绝（双重门禁）；
    读工具不受任务白名单影响。"""
    orch = _orchestrator()
    orch.set_tool_manager(_fake_tool_manager())
    agent = orch._agents[ProfileName.FAST]
    req = _req("帮我记个待办", action=IntentAction.REQUEST)
    req.allowed_write_tools = ["add_todo"]

    # 白名单内写工具放行
    data, error = _run(agent._execute_tool("add_todo", {"content": "x"}, req))
    assert error is None
    # 白名单外写工具拒绝（即使 action=REQUEST 且工具本身允许）
    data, error = _run(agent._execute_tool("complete_todo", {"id": 1}, req))
    assert data is None
    assert "写能力" in error
    # 读工具不受任务白名单影响（正常执行链，非权限拒绝）
    data, error = _run(agent._execute_tool("knowledge_search", {"query": "选课"}, req))
    assert "权限" not in (error or "")


def test_planner_rule_task_carries_write_capability():
    """规则链 t3（REQUEST）只声明 add_todo 写能力（最小权限，与 required_tool 一致）。"""
    orch = _orchestrator()
    req = _req("我明天下午有空，想去办校园卡，帮我记个待办")
    plan = _run(orch._planner.plan(req, req.domain, req.action))
    t3 = next(t for t in plan.tasks if t.task_id == "t3")
    assert t3.action == IntentAction.REQUEST
    assert t3.allowed_write_tools == ["add_todo"]


def test_planner_single_request_task_gets_write_hint():
    """单任务 REQUEST 最小权限：'帮我添加一个待办' → 只给 add_todo 写能力。"""
    orch = _orchestrator()
    req = _req("帮我添加一个补办校园卡的待办", domain=IntentDomain.PERSONAL, action=IntentAction.REQUEST)
    plan = _run(orch._planner.plan(req, req.domain, req.action))
    assert plan.mode == "single"
    assert plan.tasks[0].allowed_write_tools == ["add_todo"]


def test_planner_single_query_task_no_capability():
    """单任务 QUERY 不做任务级写限制（allowed_write_tools=None，Action 策略已约束）。"""
    orch = _orchestrator()
    req = _req("南校区食堂几点关门？", domain=IntentDomain.CAMPUS_LIFE, action=IntentAction.QUERY)
    plan = _run(orch._planner.plan(req, req.domain, req.action))
    assert plan.tasks[0].allowed_write_tools is None


def test_tasks_from_llm_validates_tools_field():
    """LLM 规划的 tools 字段硬校验：合法保留；未注册工具 → 整链作废（fail-closed）。"""
    orch = _orchestrator()
    req = _req("帮我记个待办")
    orch._planner.set_tool_names({"add_todo", "complete_todo", "knowledge_search"})

    # 合法 tools：保留为任务写能力
    tasks = orch._planner._tasks_from_llm([
        {"id": "t1", "domain": "personal", "action": "request", "tools": ["add_todo"]},
    ], req)
    assert tasks is not None
    assert tasks[0].allowed_write_tools == ["add_todo"]

    # 引用未注册工具 → None（整链作废，回落本地 Fast Path）
    assert orch._planner._tasks_from_llm([
        {"id": "t1", "domain": "personal", "action": "request", "tools": ["delete_todo"]},
    ], req) is None
    # 非字符串数组 → None
    assert orch._planner._tasks_from_llm([
        {"id": "t1", "domain": "personal", "action": "request", "tools": "add_todo"},
    ], req) is None
    # 工具名集合未注入时不校验名称（运行期 Agent 门禁按实际注册表拦截）
    orch._planner.set_tool_names(None)
    tasks = orch._planner._tasks_from_llm([
        {"id": "t1", "domain": "personal", "action": "request", "tools": ["anything"]},
    ], req)
    assert tasks is not None and tasks[0].allowed_write_tools == ["anything"]


def test_capability_flows_to_task_request():
    """allowed_write_tools 从 Task 流入 Task-scoped Request（Agent 门禁消费同一事实来源）。"""
    orch = _orchestrator()
    req = _req("我明天下午有空，想去办校园卡，帮我记个待办")
    seen = []

    async def fake_execute(task_req, on_event=None):
        seen.append((task_req.task_id, task_req.allowed_write_tools))
        return AgentResponse(agent_type=task_req.domain.value, content="ok", success=True)

    orch._execute = fake_execute
    plan = _run(orch._planner.plan(req, req.domain, req.action))
    _run(orch.run_parallel(req, plan))
    by_id = dict(seen)
    assert by_id["t1"] is None and by_id["t2"] is None  # QUERY 任务不限制
    assert by_id["t3"] == ["add_todo"]                  # REQUEST 任务只给 add_todo


# ── 复合请求 Task 级 action（不继承顶层 action）──────────────────────────────

def test_parallel_tasks_get_own_action():
    """并行兜底：'查校车 + 添加待办' 拆出不同 action（campus QUERY / personal REQUEST）。"""
    orch = _orchestrator()

    async def fake_llm_plan(req):
        return None  # LLM 不可用 → 回落本地 Fast Path

    orch._planner._llm_plan = fake_llm_plan  # type: ignore[method-assign]
    req = _req("帮我查一下明天校车，同时帮我添加一个下午三点的待办")
    plan = _run(orch._planner.plan(req, IntentDomain.CAMPUS_LIFE, IntentAction.QUERY))
    assert plan.mode == "parallel"
    by_domain = {t.domain: t.action for t in plan.tasks}
    assert by_domain[IntentDomain.CAMPUS_LIFE] == IntentAction.QUERY
    assert by_domain[IntentDomain.PERSONAL] == IntentAction.REQUEST


def test_mixed_query_request_upgrades_to_llm_planning():
    """QUERY + REQUEST 混合（复合形态 + 写操作词）→ 升级 LLM 规划。

    该消息单领域、短句（不触发从句/长度/多领域信号），
    仅因"复合形态 + 写操作词"升级 —— 让 LLM 显式给出各任务 action。
    """
    orch = _orchestrator()
    req = _req("帮我添加一个待办，然后设置提醒")
    assert orch._planner._needs_llm_planning(req) is True
    # 对照：同样短但无写操作词的复合查询不因此升级（纯查询走本地 Fast Path）
    req2 = _req("校车几点，然后图书馆几点关")
    assert orch._planner._needs_llm_planning(req2) is False
