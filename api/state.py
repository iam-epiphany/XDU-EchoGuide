"""
API 层共享状态与运行时构建 —— 全局组件与初始化逻辑的单一归属。

lifespan 与交互式 CLI 共用 _build_runtime()，避免两处各自初始化导致配置漂移
（此前 CLI 缺少工具注册/缓存/monitor，且默认值不一致）。各 router 从本模块
读取全局组件（state._orchestrator 等），保证测试打桩与真实运行读写同一对象。
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import HTTPException, Request

load_dotenv()

logger = logging.getLogger(__name__)

# 将项目根目录加入 sys.path，确保无论从哪里执行都能找到 agents/core/memory 等模块
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── 全局组件（_build_runtime 中初始化）──────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_semantic_cache = None
_personal_service = None
_campus_store = None
_kb           = None

# 后台任务跟踪：防止 fire-and-forget 任务被 GC 回收或异常无人检索
_bg_tasks: set = set()


def _spawn_background(coro: Any) -> None:
    """启动后台任务并跟踪生命周期，异常记录到日志。"""
    task = asyncio.create_task(coro)

    def _done(t: asyncio.Task) -> None:
        _bg_tasks.discard(t)
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.warning("后台任务异常: %s", exc)

    _bg_tasks.add(task)
    task.add_done_callback(_done)


def _allowed_origins() -> List[str]:
    """浏览器来源白名单。未配置时仅允许本地开发和 Compose 入口。"""
    default = "http://localhost:5175,http://127.0.0.1:5175,http://localhost:8088,http://127.0.0.1:8088"
    raw = os.getenv("ECHOGUIDE_ALLOWED_ORIGINS", default)
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]


def _validate_mcp_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") not in _allowed_origins():
        raise HTTPException(403, "MCP Origin 不在 ECHOGUIDE_ALLOWED_ORIGINS 白名单中")


def _validate_mcp_accept(request: Request) -> None:
    accept = request.headers.get("accept", "")
    required = ("application/json", "text/event-stream")
    if not all(media in accept or "*/*" in accept for media in required):
        raise HTTPException(406, "MCP Accept 必须同时包含 application/json 和 text/event-stream")


async def _cache_get(cache: Any, query: str, *, user_id: str, dependence: str) -> Any:
    """兼容旧测试替身；真实 SemanticCache 使用异步接口。"""
    async_get = getattr(cache, "aget", None)
    if async_get:
        return await async_get(query, user_id=user_id, dependence=dependence)
    return cache.get(query, user_id=user_id, dependence=dependence)


def _cache_put(cache: Any, query: str, response: str, **kwargs: Any) -> None:
    """真实实现异步落库；同步替身/兼容实现保持原调用语义。"""
    async_put = getattr(cache, "aput", None)
    if async_put:
        _spawn_background(async_put(query, response, **kwargs))
    else:
        # 旧测试/第三方 cache adapter 尚未接受 provenance 字段时保持兼容；
        # 真实 SemanticCache 的 aput 会持久化 knowledge_used。
        try:
            cache.put(query, response, **kwargs)
        except TypeError as ex:
            if "knowledge_used" not in str(ex):
                raise
            kwargs.pop("knowledge_used", None)
            cache.put(query, response, **kwargs)


def _anthropic_cfg() -> Dict[str, Any]:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 ANTHROPIC_API_KEY")
    cfg: Dict[str, Any] = {
        "api_key":  key,
        "model":    os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    }
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip()
    if base_url:
        cfg["base_url"] = base_url
    cfg["fast_api_key"] = os.getenv("ECHOGUIDE_FAST_API_KEY", "").strip() or key
    cfg["fast_base_url"] = os.getenv("ECHOGUIDE_FAST_BASE_URL", "").strip() or base_url or None
    cfg["fast_model"] = os.getenv("ECHOGUIDE_FAST_MODEL", "deepseek-v4-flash").strip()
    cfg["deep_api_key"] = os.getenv("ECHOGUIDE_DEEP_API_KEY", "").strip() or key
    cfg["deep_base_url"] = os.getenv("ECHOGUIDE_DEEP_BASE_URL", "").strip() or base_url or None
    cfg["deep_model"] = os.getenv("ECHOGUIDE_DEEP_MODEL", "deepseek-v4-pro").strip()
    return cfg


def _build_runtime() -> None:
    """构造运行期组件并注册工具。"""
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager
    global _personal_service, _campus_store, _kb, _semantic_cache

    from agents.agent_orchestrator import AgentOrchestrator
    from campus.store import CampusInfoStore
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.semantic_cache import SemanticCache
    from mcp.tool_manager import MCPToolManager, Tool, ToolEffect
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager
    from personal.service import PersonalService
    from personal.store import PersonalStore
    from runtime.policy import ExecutionPolicy
    from runtime.runtime import AgentRuntime
    from tools import with_service
    from tools.academic_tool import calculate_weighted_score_handler
    from tools.affairs_tool import query_affairs_process_handler
    from tools.campus_tool import campus_info_handler
    from tools.ddl_tool import query_ddl_handler
    from tools.schedule_tool import query_schedule_handler
    from tools.it_tool import diagnose_it_issue_handler
    from tools.todo_tool import add_todo_handler, complete_todo_handler, query_todo_handler
    from tools.weather import weather_handler

    cfg = _anthropic_cfg()
    logger.info(
        "执行配置: fast=%s deep=%s base_url=%s",
        cfg["fast_model"], cfg["deep_model"], cfg.get("base_url", "(官方)"),
    )

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("ECHOGUIDE_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("ECHOGUIDE_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # Agent Runtime + 统一模型调用入口（ModelGateway）：意图识别 / Agent 工具循环 /
    # 合成 / 出口校验 / 记忆提炼 / 查询改写重排的 LLM 调用统一经 gateway 进出，
    # 计数、token 统计、预算与 Trace 口径一致。
    _runtime = AgentRuntime(policy=ExecutionPolicy.from_env())
    _gateway = _runtime.model_gateway

    # Agent 编排器（内部持有意图识别器，供评测器复用，避免双实例缓存分家）
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
        fast_api_key=cfg["fast_api_key"],
        fast_base_url=cfg["fast_base_url"],
        fast_model=cfg["fast_model"],
        deep_api_key=cfg["deep_api_key"],
        deep_base_url=cfg["deep_base_url"],
        deep_model=cfg["deep_model"],
        runtime=_runtime,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像 + SQLite 分层存储）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        gateway=_gateway,
    )
    # 分层记忆存储注入编排器（上下文卸载落盘与 MemoryManager 共享同一实例）
    _orchestrator.set_memory_store(_memory.layered_store)

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        # 结果重排后端：local=本地 bge-reranker（默认，不可用自动降级 LLM）
        rerank_backend=os.getenv("ECHOGUIDE_RERANK_BACKEND", "local"),
        gateway=_gateway,
    )
    _kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    logger.info("知识库已加载: %s 个文档片段", _kb.doc_count)

    def knowledge_fallback(params: Dict[str, Any], context: Optional[Dict[str, Any]], error: str):
        query = params.get("query", "")
        return [{
            "title": "知识库降级结果",
            "content": f"知识库暂时不可用，未能完成对“{query}”的语义检索。请稍后重试，或联系辅导员/教务老师确认。",
            "score": 0.0,
            "fallback": True,
            "error": error,
        }]

    _tool_manager.register(Tool(
        name="knowledge_search",
        description="搜索西电校园知识库（基于 ChromaDB 向量检索），返回相关文档片段",
        handler=_kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询（可带领域词）"},
                "top_k": {"type": "integer", "description": "返回条数，默认 5"},
                "min_score": {"type": "number", "description": "相关性阈值，默认 0.25"},
                "domain": {"type": "string", "description": "领域过滤：academic/campus_life/affairs/it_help"},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
        use_rewrite=True,  # Agent 调用 knowledge_search 也走「改写→并行召回→去重→重排」链路（与 /search 一致）
        fallback=knowledge_fallback,
        effect=ToolEffect.READ,
    ))

    _tool_manager.register(Tool(
        name="calculate_weighted_score",
        description="按 Σ(成绩×学分)/Σ学分 计算加权学分成绩；这不是官方 GPA 换算",
        handler=calculate_weighted_score_handler,
        schema={
            "type": "object",
            "properties": {
                "courses": {
                    "type": "array",
                    "description": "课程数组，每项包含 name、credits、score",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "credits": {"type": "number"},
                            "score": {"type": "number"},
                        },
                        "required": ["credits", "score"],
                    },
                },
            },
            "required": ["courses"],
        },
        cache_ttl=0.0,
        effect=ToolEffect.READ,
    ))
    _tool_manager.register(Tool(
        name="query_affairs_process",
        description="查询版本化校园办事流程，包括材料、步骤、部门、来源与更新时间",
        handler=query_affairs_process_handler,
        schema={
            "type": "object",
            "properties": {
                "service": {"type": "string", "description": "事项名称，如校园卡补办/请假/在读证明/缓考"},
            },
            "required": ["service"],
        },
        cache_ttl=300.0,
        effect=ToolEffect.READ,
    ))
    _tool_manager.register(Tool(
        name="diagnose_it_issue",
        description="使用确定性诊断树排查校园网、VPN、统一身份认证和教务系统故障",
        handler=diagnose_it_issue_handler,
        schema={
            "type": "object",
            "properties": {
                "system": {"type": "string", "description": "校园网/VPN/统一身份认证/教务系统"},
                "symptom": {"type": "string", "description": "故障现象"},
                "error_code": {"type": "string", "description": "可选错误码"},
                "network": {"type": "string", "description": "可选网络环境"},
            },
        },
        cache_ttl=0.0,
        effect=ToolEffect.READ,
    ))

    # ── 个人数据中心（课表 / 待办 / DDL，按 user_id 隔离，SQLite 持久化）──
    _personal_service = PersonalService(PersonalStore())
    logger.info("个人数据中心已就绪: %s", _personal_service.store.db_path)

    _tool_manager.register(Tool(
        name="query_schedule",
        description="查询用户个人课程表（按 user_id 隔离）。date 支持：今天/明天/后天/周X/星期X/YYYY-MM-DD，返回当天课程列表（含时间与地点）",
        handler=with_service(query_schedule_handler, personal_service=_personal_service),
        schema={
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "日期表达式，默认今天，如：今天/明天/周三/2026-09-14"},
            },
        },
        cache_ttl=0.0,  # 个人数据实时查询，不缓存（缓存 key 不含 user_id）
        effect=ToolEffect.READ,
    ))
    _tool_manager.register(Tool(
        name="query_todo",
        description="查询用户的待办清单（按 user_id 隔离）。status 支持 open/done/all",
        handler=with_service(query_todo_handler, personal_service=_personal_service),
        schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "description": "open（未完成，默认）/ done / all"},
                "kinds": {"type": "array", "items": {"type": "string"}, "description": "过滤类型：todo/ddl/exam"},
            },
        },
        cache_ttl=0.0,
        effect=ToolEffect.READ,
    ))
    _tool_manager.register(Tool(
        name="add_todo",
        description="新增待办/DDL/考试安排。kind: todo（待办，默认）/ ddl（截止任务）/ exam（考试）；due_at 为 YYYY-MM-DD 或 YYYY-MM-DD HH:MM",
        handler=with_service(add_todo_handler, personal_service=_personal_service),
        schema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "事项内容（必填）"},
                "kind": {"type": "string", "description": "todo/ddl/exam，默认 todo"},
                "due_at": {"type": "string", "description": "截止/考试时间，如 2026-09-14 或 2026-09-14 09:00"},
            },
            "required": ["content"],
        },
        cache_ttl=0.0,
        effect=ToolEffect.WRITE,  # 状态修改类工具：显式副作用声明（fail-closed）
    ))
    _tool_manager.register(Tool(
        name="complete_todo",
        description="把待办标记为完成（done=true）或恢复未完成（done=false），id 为待办编号",
        handler=with_service(complete_todo_handler, personal_service=_personal_service),
        schema={
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "待办 id（必填）"},
                "done": {"type": "boolean", "description": "true=完成（默认）/ false=恢复"},
            },
            "required": ["id"],
        },
        cache_ttl=0.0,
        effect=ToolEffect.WRITE,  # 状态修改类工具：显式副作用声明（fail-closed）
    ))
    _tool_manager.register(Tool(
        name="query_ddl",
        description="查询用户的考试与 DDL 安排（按 user_id 隔离），返回未来 horizon_days 天内的倒计时列表（含今天到期与已过期未完成）",
        handler=with_service(query_ddl_handler, personal_service=_personal_service),
        schema={
            "type": "object",
            "properties": {
                "horizon_days": {"type": "integer", "description": "查询范围天数，默认 30"},
            },
        },
        cache_ttl=0.0,
        effect=ToolEffect.READ,
    ))

    # ── 结构化公开信息（校车/楼宇/场馆/图书馆，data/public/*.json）──
    _campus_store = CampusInfoStore()
    logger.info("校园公开信息已就绪: %s", _campus_store.load_status)
    _tool_manager.register(Tool(
        name="query_campus_info",
        description=(
            "查询西电校园公开信息（结构化数据）。category: auto（汇总全部公开数据）/ shuttle（校车，返回下一班及剩余分钟，"
            "keyword 可传方向如'南→北'）/ buildings（楼宇，keyword 传楼名如'信远楼'）/ "
            "venues（运动场馆，keyword 可传场馆名）/ library（图书馆开放时间）。数据暂未录入时返回提示"
        ),
        handler=with_service(campus_info_handler, campus_store=_campus_store),
        schema={
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "auto/shuttle/buildings/venues/library"},
                "keyword": {"type": "string", "description": "校车方向或楼名/场馆名"},
            },
            "required": ["category"],
        },
        cache_ttl=0.0,  # 校车查询依赖当前时间，不缓存
        effect=ToolEffect.READ,
    ))
    _tool_manager.register(Tool(
        name="get_weather",
        description="查询天气（Open-Meteo 免费数据源）。place: 南校区/北校区/西安，默认南校区；days: 预报天数 1-7，默认 3",
        handler=weather_handler,
        schema={
            "type": "object",
            "properties": {
                "place": {"type": "string", "description": "南校区（默认）/北校区/西安"},
                "days": {"type": "integer", "description": "预报天数 1-7，默认 3"},
            },
        },
        cache_ttl=300.0,
        timeout_s=15.0,
        effect=ToolEffect.READ,
    ))

    # Agentic RAG：把工具管理器注入执行实例，让 Agent 自主决定何时检索知识库
    _orchestrator.set_tool_manager(_tool_manager)

    # 语义缓存（默认关闭）：上下文相关回答做 Semantic Cache 容易错误复用，
    # 已不属于核心能力；保留代码，由 SEMANTIC_CACHE_ENABLED=1 显式开启。
    # 确定性的 Tool 结果仍走 MCPToolManager 的 TTL 精确缓存（主链路不受影响）。
    _semantic_cache = SemanticCache(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        threshold=float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.85")),
        enabled=os.getenv("SEMANTIC_CACHE_ENABLED", "0") == "1",
    )

    # 性能监控（可选启动 Prometheus）
    prom_port = int(os.getenv("PROMETHEUS_PORT", "0")) or None
    _monitor = PerformanceMonitor(
        orchestrator=_orchestrator,
        tool_manager=_tool_manager,
        interval_s=float(os.getenv("MONITOR_INTERVAL", "10")),
        webhook_url=os.getenv("ALERT_WEBHOOK_URL") or None,
        prometheus_port=prom_port,
    )

    # 评测器（复用编排器内部的意图识别器，避免双实例缓存/统计分家）
    # LLM-as-Judge 可与生成模型分离；独立 Judge 只能降低自评偏差，不能取代人工抽检。
    # 传入知识库 → 额外产出 RAG 检索硬指标（HitRate@K/Recall@K/MRR）与生成端引用/忠实性评测
    _evaluator = EndToEndEvaluator(
        orchestrator=_orchestrator,
        recognizer=_orchestrator.intent_recognizer,
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        judge_api_key=os.getenv("EVAL_JUDGE_API_KEY") or cfg["api_key"],
        judge_base_url=os.getenv("EVAL_JUDGE_BASE_URL") or cfg.get("base_url"),
        judge_model=os.getenv("EVAL_JUDGE_MODEL") or cfg["model"],
        baseline_path=os.getenv("EVAL_BASELINE_PATH", "/app/data/eval/baseline.json"),
        knowledge_base=_kb,
    )


async def _setup_external_mcp() -> None:
    """接入外部 MCP 工具源（默认关闭，optional integration / example）。

    把远程 MCP server（如 GitHub 官方 remote server）的只读工具注册进
    工具管理器；连接失败只记日志，服务照常启动（全链路降级哲学）。
    ECHOGUIDE_EXTERNAL_MCP_EXPOSE_AGENTS 非空时把注册的工具加入公共工具层
    （任何请求可见，仍受 Action 读写门禁）；空 = 只注册不暴露
    （agent_exposed=False，对 LLM 不可见，仅注册在工具管理器）。
    """
    if os.getenv("ECHOGUIDE_EXTERNAL_MCP_ENABLED", "0") != "1":
        return
    try:
        from mcp.external_client import ExternalMCPSource

        if _tool_manager is None:
            logger.warning("外部 MCP 工具源跳过：工具管理器未初始化")
            return
        source = ExternalMCPSource(
            url=os.getenv("ECHOGUIDE_EXTERNAL_MCP_URL", "https://api.githubcopilot.com/mcp/"),
            token=os.getenv("ECHOGUIDE_EXTERNAL_MCP_TOKEN") or None,
            proxy=os.getenv("ECHOGUIDE_EXTERNAL_MCP_PROXY") or None,
            prefix=os.getenv("ECHOGUIDE_EXTERNAL_MCP_PREFIX", "github").strip() or "github",
        )
        whitelist_raw = os.getenv("ECHOGUIDE_EXTERNAL_MCP_TOOL_WHITELIST", "").strip()
        whitelist = {t.strip() for t in whitelist_raw.split(",") if t.strip()} or None
        registered = await source.setup(_tool_manager, tool_whitelist=whitelist)
        if not registered:
            logger.warning("外部 MCP 工具源未注册任何工具（检查 URL/Token 或工具过滤）")
            return
        expose_raw = os.getenv("ECHOGUIDE_EXTERNAL_MCP_EXPOSE_AGENTS", "").strip()
        if expose_raw:
            # 外部工具进公共工具层（领域不再构成工具门禁，兼容旧值语义）
            _orchestrator.expose_external_tools(registered)
    except Exception as ex:
        logger.error("外部 MCP 工具源接入失败（不影响服务启动）: %s", ex)
