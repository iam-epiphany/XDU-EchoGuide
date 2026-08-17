"""
西电校园智慧助手（EchoGuide）— FastAPI 入口

启动时打印校徽风格图案。
所有核心组件在 lifespan 中初始化，通过环境变量配置。
"""
import asyncio
import json
import logging
import os
import pathlib
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

# 将项目根目录加入 sys.path，确保无论从哪里执行都能找到 agents/core/memory 等模块
# 这一行必须在所有项目内部 import 之前执行
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field, model_validator

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║  EchoGuide  v2.0     ║
   ║  西电校园智慧助手     ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""

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
    """构造运行期组件并注册工具。

    lifespan 与交互式 CLI 共用，避免两处各自初始化导致配置漂移
    （此前 CLI 缺少工具注册/缓存/monitor，且默认值不一致）。
    Monitor 的异步启动由调用方负责。
    """
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager
    global _personal_service, _campus_store, _kb, _semantic_cache

    from agents.agent_orchestrator import AgentOrchestrator
    from campus.store import CampusInfoStore
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.semantic_cache import SemanticCache
    from mcp.tool_manager import MCPToolManager, Tool
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
    ))

    # ── 结构化公开信息（校车/楼宇/场馆/图书馆，data/public/*.json）──
    _campus_store = CampusInfoStore()
    logger.info("校园公开信息已就绪: %s", _campus_store.load_status)
    _tool_manager.register(Tool(
        name="query_campus_info",
        description=(
            "查询西电校园公开信息（结构化数据）。category: shuttle（校车，返回下一班及剩余分钟，"
            "keyword 可传方向如'南→北'）/ buildings（楼宇，keyword 传楼名如'信远楼'）/ "
            "venues（运动场馆，keyword 可传场馆名）/ library（图书馆开放时间）。数据暂未录入时返回提示"
        ),
        handler=with_service(campus_info_handler, campus_store=_campus_store),
        schema={
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "shuttle/buildings/venues/library"},
                "keyword": {"type": "string", "description": "校车方向或楼名/场馆名"},
            },
            "required": ["category"],
        },
        cache_ttl=0.0,  # 校车查询依赖当前时间，不缓存
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
    ))

    # Agentic RAG：把工具管理器注入 Agent 池，让 Agent 自主决定何时检索知识库
    _orchestrator.set_tool_manager(_tool_manager)

    # 语义缓存（GPTCache 思路）：相似问题直接复用答案，成本/延迟趋近于 0
    _semantic_cache = SemanticCache(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        threshold=float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.85")),
        enabled=os.getenv("SEMANTIC_CACHE_ENABLED", "1") == "1",
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
    """接入外部 MCP 工具源（默认关闭）。

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
            # v2：外部工具进公共工具层（领域不再构成工具门禁，兼容旧值语义）
            _orchestrator.expose_external_tools(registered)
    except Exception as ex:
        logger.error("外部 MCP 工具源接入失败（不影响服务启动）: %s", ex)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _monitor, _memory

    print(BANNER, flush=True)
    environment = os.getenv("ECHOGUIDE_ENV", os.getenv("APP_ENV", "development")).lower()
    if environment == "production" and not os.getenv("ECHOGUIDE_GUARD_TOKEN"):
        logger.warning("生产环境未配置 ECHOGUIDE_GUARD_TOKEN；浏览器登录仍可用，机器调用不具备服务令牌")

    _build_runtime()
    await _setup_external_mcp()
    await _monitor.start()

    logger.info("EchoGuide 西电校园智慧助手已就绪")
    yield

    await _monitor.stop()
    if _memory is not None:
        await _memory.close()
    logger.info("EchoGuide 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="西电校园智慧助手 EchoGuide",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENABLE_SWAGGER_UI", "true").lower() == "true" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins(),
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "MCP-Protocol-Version"],
    allow_credentials=True,
)

# EchoGuard 真实接入：中间件必须在应用启动前挂载（lifespan 中挂载会报错）。
# 默认启用，保护 /chat、/personal、/mcp 等 POST 端点：
# 注入检测 / 限流 / 脱敏审计开箱即用；配置 ECHOGUIDE_GUARD_TOKEN 后开启认证。
if os.getenv("ECHOGUIDE_GUARD_ENABLED", "1") == "1":
    from echoguide_guard.integration import EchoGuardMiddleware, GuardSettings

    app.add_middleware(EchoGuardMiddleware, settings=GuardSettings())
    logger.warning("[EchoGuard] 中间件已接入真实请求链（注入检测/限流/脱敏审计，认证按需启用）")


# ── 轻量登录认证 ──────────────────────────────────────────────────────────────
from auth.service import (
    SESSION_COOKIE,
    AuthUser,
    create_session_token,
    get_auth_store,
    user_from_scope,
)


def optional_user(request: Request) -> Optional[AuthUser]:
    return user_from_scope(request.scope)


def require_user(request: Request) -> AuthUser:
    user = optional_user(request)
    if user is None:
        raise HTTPException(401, "请先登录")
    return user


def require_admin(user: AuthUser = Depends(require_user)) -> AuthUser:
    if user.role != "admin":
        raise HTTPException(403, "需要管理员权限")
    return user


def require_observability(user: AuthUser = Depends(require_user)) -> AuthUser:
    """
    观测接口权限：管理员始终可看；演示环境（ECHOGUIDE_OBSERVABILITY_PUBLIC=1）
    下登录用户也可看。

    权衡：trace 含用户消息内容，生产必须保持 admin-only（默认 fail-closed）。
    该开关只应在本地演示/开发环境开启，与 ECHOGUIDE_BENCHMARK_ENABLED 同类。
    """
    if user.role == "admin" or os.getenv("ECHOGUIDE_OBSERVABILITY_PUBLIC", "0") == "1":
        return user
    raise HTTPException(403, "需要管理员权限")


def _cookie_secure() -> bool:
    # 本项目默认也支持本地 HTTP/Compose；正式 HTTPS 部署显式设为 1。
    return os.getenv("ECHOGUIDE_COOKIE_SECURE", "0") == "1"


class AuthCredentials(BaseModel):
    username: str = Field(min_length=3, max_length=32, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=6, max_length=128)


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=6, max_length=128)


@app.post("/auth/register", tags=["认证"], status_code=201)
async def register(body: AuthCredentials, response: Response):
    if os.getenv("ECHOGUIDE_ALLOW_REGISTRATION", "1") != "1":
        raise HTTPException(403, "当前未开放注册")
    from auth.service import UsernameExistsError

    try:
        user = await asyncio.to_thread(get_auth_store().create_user, body.username, body.password)
    except UsernameExistsError as ex:
        raise HTTPException(409, str(ex)) from ex
    except ValueError as ex:
        raise HTTPException(400, str(ex)) from ex
    response.set_cookie(
        SESSION_COOKIE, create_session_token(user), httponly=True, secure=_cookie_secure(),
        samesite="lax", max_age=7 * 24 * 3600, path="/",
    )
    return {"authenticated": True, "user": user.public()}


@app.post("/auth/login", tags=["认证"])
async def login(body: AuthCredentials, response: Response):
    user = await asyncio.to_thread(get_auth_store().authenticate, body.username, body.password)
    if user is None:
        raise HTTPException(401, "用户名或密码错误")
    response.set_cookie(
        SESSION_COOKIE, create_session_token(user), httponly=True, secure=_cookie_secure(),
        samesite="lax", max_age=7 * 24 * 3600, path="/",
    )
    return {"authenticated": True, "user": user.public()}


@app.post("/auth/logout", tags=["认证"])
async def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/", samesite="lax", secure=_cookie_secure(), httponly=True)
    return {"authenticated": False}


@app.get("/auth/me", tags=["认证"])
async def auth_me(user: Optional[AuthUser] = Depends(optional_user)):
    return {"authenticated": user is not None, "user": user.public() if user else None}


@app.post("/auth/password", tags=["认证"])
async def change_password(body: PasswordChange, user: AuthUser = Depends(require_user)):
    try:
        changed = await asyncio.to_thread(
            get_auth_store().change_password, user.id, body.current_password, body.new_password
        )
    except ValueError as ex:
        raise HTTPException(400, str(ex)) from ex
    if not changed:
        raise HTTPException(400, "当前密码错误")
    return {"message": "密码已修改"}


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(default="anonymous", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    conv_id: Optional[str] = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")


class ChatResponse(BaseModel):
    conv_id:     str
    response:    str
    intent:      str
    domain:      str = "other"
    action:      str = "other"
    agent_type:  str
    latency_ms:  float
    knowledge_used: bool = False
    cached: bool = False
    execution: Dict[str, Any] = Field(default_factory=dict)


def _benchmark_strategy(request: Optional[Request]) -> str:
    """仅在显式启用的本地演示环境接受基准策略覆盖。"""
    if request is None or os.getenv("ECHOGUIDE_BENCHMARK_ENABLED", "0") != "1":
        return "adaptive"
    value = request.headers.get("X-EchoGuide-Benchmark-Strategy", "adaptive").strip().lower()
    allowed = {"adaptive", "always_deep", "always_llm_deep", "single_agent", "generic_rag"}
    if value not in allowed:
        raise HTTPException(400, "不支持的 Benchmark 策略")
    return value


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {
        "status": "ok",
        "agents": _orchestrator.get_stats(),
        "verification": _orchestrator.verification_stats(),
    }


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills(_admin: AuthUser = Depends(require_admin)):
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, response: Response, request: Request = None):
    """
    主对话接口。完整流程：
      语义缓存 → 记忆读取 → 意图识别（领域×动作）→ Agent 路由 →
      Agentic RAG 执行 → 记忆写入 → 语义缓存写入

    request 默认 None：HTTP 场景由 FastAPI 注入真实请求，
    离线单测直接传 None 跳过身份覆盖（保留旧测试路径）。
    注意：注解不能写成 Optional[Request]（新版 FastAPI 会当成响应字段报错）。
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    # HTTP 请求只信任签名会话中的身份；request=None 保留离线单测兼容性。
    if request is not None:
        user = optional_user(request)
        req.user_id = user.id if user else "anonymous"

    from agents.agent_orchestrator import Request as OrcReq
    from core.tracing import begin_trace, end_trace, span
    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())
    request_started = time.perf_counter()

    # 全链路 trace（X-Trace-Id 响应头，/traces/{id} 可查）
    trace = begin_trace("chat")
    trace.tags.update({"user_id": req.user_id, "conv_id": conv_id})
    response.headers["X-Trace-Id"] = trace.trace_id

    try:
        # 1. 先读取记忆上下文 —— 语义缓存必须在其后：
        #    需要据此判断请求是否依赖历史上下文（追问/省略句/个人数据等），
        #    决定走 Global / User / 直接 bypass。
        async with span("memory_read"):
            mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)
        ctx_text = mem_ctx.to_prompt_text()
        from mcp.semantic_cache import classify_context_dependence

        dependence = classify_context_dependence(req.message, ctx_text)

        # 2. 双层语义缓存读取（读取层由上下文依赖性决定，见 cache_read_tier）：
        #    - 公共事实查询（global）→ 只查 Global（语义匹配，容忍近义改写）；
        #    - 依赖用户画像（user）+ 有效身份 → 只查 User（仅 user_id 分区，
        #      miss 不回退 Global，防止公共答案绕过个性化 Agent 推理）；
        #    - 强上下文依赖（skip：追问/省略句/指代/个人数据）→ 直接 bypass。
        cached = await _cache_get(_semantic_cache, req.message, user_id=req.user_id, dependence=dependence) if _semantic_cache else None
        if cached and cached.get("domain") == "personal":
            logger.warning("命中 personal 领域缓存，丢弃（防跨用户串扰）")
            cached = None
        if cached:
            logger.info("语义缓存命中 %r", req.message[:30])
            await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
            await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, cached["response"])
            # 缓存未持久化独立 intent 字段；intent 与 domain 同值（旧版单维兼容）
            cached_intent = cached.get("intent") or cached["domain"]
            return ChatResponse(
                conv_id=conv_id,
                response=cached["response"],
                intent=cached_intent,
                domain=cached["domain"],
                action="query",
                agent_type=cached["agent_type"],
                latency_ms=round((time.perf_counter() - request_started) * 1000, 1),
                knowledge_used=bool(cached.get("knowledge_used", False)),
                cached=True,
                execution={
                    "mode": "cache", "profile": "cache", "classifier_stage": "cache",
                    "complexity_reason": "语义缓存命中", "agents": [cached["agent_type"]],
                    "tools": [], "tasks": [], "model": "", "trace_id": trace.trace_id,
                    "input_tokens": 0, "output_tokens": 0,
                },
            )

        # 3. 构建编排请求（含对话历史，用于意图识别上下文与追问继承）
        history = [
            {"role": m.role.value, "content": m.content}
            for m in mem_ctx.recent_messages[-5:]
        ] if mem_ctx.recent_messages else None

        orch_req = OrcReq(
            message=req.message,
            user_id=req.user_id,
            conv_id=conv_id,
            context=ctx_text,
            history=history,
            benchmark_strategy=_benchmark_strategy(request),
        )

        # 4. 执行（RAG 检索由 Agent 通过工具调用自主完成 —— Agentic RAG）
        async with span("orchestrator_run"):
            result = await _orchestrator.run(orch_req)
        result.execution["trace_id"] = trace.trace_id
        # 记忆 trace（四层命中统计，透出给前端 debug 面板 / 评测统计）
        result.execution["memory_trace"] = getattr(mem_ctx, "memory_trace", {})

        # 5. 写入记忆
        async with span("memory_write"):
            await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
            await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)

        # 6. 异步更新用户画像 + 双层语义缓存（不阻塞响应）
        _spawn_background(_memory.update_profile(req.user_id, conv_id))
        if _semantic_cache:
            # 写入层决策：与读取侧同一规则（叠加编排信号：personal/request
            # → skip 不落库），classify 决定 global/user/skip，cache_tier 只做映射
            from mcp.semantic_cache import cache_tier, classify_context_dependence

            dep_write = classify_context_dependence(
                req.message, ctx_text,
                domain=result.domain.value if result.domain else None,
                action=result.action.value if result.action else None,
            )
            tier = cache_tier(dep_write, req.user_id)
            if tier == "user":
                _cache_put(_semantic_cache,
                    req.message, result.response,
                    domain=result.domain.value,
                    agent_type=result.agent_type.value,
                    user_id=req.user_id,
                    dependence=dep_write,
                    knowledge_used="knowledge_search" in result.tools_used,
                )
            elif tier == "global":
                _cache_put(_semantic_cache,
                    req.message, result.response,
                    domain=result.domain.value,
                    agent_type=result.agent_type.value,
                    dependence=dep_write,
                    knowledge_used="knowledge_search" in result.tools_used,
                )

        return ChatResponse(
            conv_id=conv_id,
            response=result.response,
            intent=result.domain.value if result.domain else "other",
            domain=result.domain.value if result.domain else "other",
            action=result.action.value if result.action else "other",
            agent_type=result.agent_type.value,
            latency_ms=round((time.perf_counter() - request_started) * 1000, 1),
            knowledge_used="knowledge_search" in result.tools_used,
            cached=False,
            execution=result.execution,
        )
    finally:
        end_trace()


# （已移除）RAG 上下文改由 Agent 通过工具调用自主获取 —— Agentic RAG。


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request = None):
    """
    流式对话接口（SSE / Server-Sent Events）。

    事件序列：
      event: meta   意图/Agent 识别结果（含置信度）
      event: tool   Agent 工具调用过程（如 RAG 检索中/完成）
      event: delta  生成内容的增量文本（逐 token）
      event: done   最终汇总（完整回答、耗时、是否用 RAG）
      event: error  出错信息

    request 默认 None：HTTP 场景由 FastAPI 注入真实请求，
    离线单测直接传 None 跳过身份覆盖。
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    if request is not None:
        user = optional_user(request)
        req.user_id = user.id if user else "anonymous"

    from agents.agent_orchestrator import Request as OrcReq
    from core.tracing import begin_trace, end_trace, span
    from memory.conversation_memory import MsgRole
    from fastapi.responses import StreamingResponse

    conv_id = req.conv_id or str(uuid.uuid4())

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(ev: dict) -> None:
            await queue.put(ev)

        async def run_and_finish() -> None:
            trace = begin_trace("chat_stream")
            request_started = time.perf_counter()
            try:
                # 1. 先读记忆上下文 —— 语义缓存必须在其后（同 /chat）：
                #    据此判断请求是否依赖历史上下文（追问/省略句/个人数据等），
                #    决定走 Global / User / 直接 bypass。
                async with span("memory_read"):
                    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)
                ctx_text = mem_ctx.to_prompt_text()
                from mcp.semantic_cache import classify_context_dependence

                dependence = classify_context_dependence(req.message, ctx_text)

                # 2. 双层语义缓存读取（读取层由上下文依赖性决定，与 /chat 完全一致）：
                #    - 公共事实查询（global）→ 只查 Global；依赖用户画像（user）+
                #      有效身份 → 只查 User（仅 user_id 分区，不回退 Global）；
                #    - 强上下文依赖（skip）→ 直接 bypass。
                cached = await _cache_get(_semantic_cache, req.message, user_id=req.user_id, dependence=dependence) if _semantic_cache else None
                if cached and cached.get("domain") == "personal":
                    logger.warning("命中 personal 领域缓存，丢弃（防跨用户串扰）")
                    cached = None
                if cached:
                    # 缓存命中：写记忆 + 推送 meta/delta/done（hello 由外层统一输出）
                    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
                    await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, cached["response"])
                    cached_intent = cached.get("intent") or cached["domain"]
                    await queue.put({
                        "type": "meta", "domain": cached["domain"], "action": "query",
                        "agent": cached["agent_type"], "cached": True,
                    })
                    await queue.put({"type": "delta", "text": cached["response"]})
                    await queue.put({
                        "type": "done", "conv_id": conv_id, "response": cached["response"],
                        "intent": cached_intent, "agent_type": cached["agent_type"],
                        "latency_ms": round((time.perf_counter() - request_started) * 1000, 1),
                        "knowledge_used": bool(cached.get("knowledge_used", False)), "cached": True,
                        "execution": {
                            "mode": "cache", "profile": "cache", "classifier_stage": "cache",
                            "complexity_reason": "语义缓存命中", "agents": [cached["agent_type"]],
                            "tools": [], "tasks": [], "model": "", "trace_id": trace.trace_id,
                            "input_tokens": 0, "output_tokens": 0,
                        },
                    })
                    return

                history = [
                    {"role": m.role.value, "content": m.content}
                    for m in mem_ctx.recent_messages[-5:]
                ] if mem_ctx.recent_messages else None

                orch_req = OrcReq(
                    message=req.message,
                    user_id=req.user_id,
                    conv_id=conv_id,
                    context=ctx_text,
                    history=history,
                    benchmark_strategy=_benchmark_strategy(request),
                )
                async with span("orchestrator_run"):
                    result = await _orchestrator.run(orch_req, on_event=on_event)
                result.execution["trace_id"] = trace.trace_id
                # 记忆 trace（四层命中统计，透出给前端 debug 面板 / 评测统计）
                result.execution["memory_trace"] = getattr(mem_ctx, "memory_trace", {})

                async with span("memory_write"):
                    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
                    await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)
                _spawn_background(_memory.update_profile(req.user_id, conv_id))

                if _semantic_cache:
                    # 写入层决策：与读取侧同一规则（叠加编排信号），同 /chat
                    from mcp.semantic_cache import cache_tier, classify_context_dependence

                    dep_write = classify_context_dependence(
                        req.message, ctx_text,
                        domain=result.domain.value if result.domain else None,
                        action=result.action.value if result.action else None,
                    )
                    tier = cache_tier(dep_write, req.user_id)
                    if tier == "user":
                        _cache_put(_semantic_cache,
                            req.message, result.response,
                            domain=result.domain.value,
                            agent_type=result.agent_type.value,
                            user_id=req.user_id,
                            dependence=dep_write,
                            knowledge_used="knowledge_search" in result.tools_used,
                        )
                    elif tier == "global":
                        _cache_put(_semantic_cache,
                            req.message, result.response,
                            domain=result.domain.value,
                            agent_type=result.agent_type.value,
                            dependence=dep_write,
                            knowledge_used="knowledge_search" in result.tools_used,
                        )

                await queue.put({
                    "type": "done",
                    "conv_id": conv_id,
                    "response": result.response,
                    "intent": result.domain.value if result.domain else "other",
                    "agent_type": result.agent_type.value,
                    "latency_ms": round((time.perf_counter() - request_started) * 1000, 1),
                    "knowledge_used": "knowledge_search" in result.tools_used,
                    "cached": False,
                    "execution": result.execution,
                })
            except Exception as ex:
                logger.exception("流式对话失败")
                await queue.put({"type": "error", "message": str(ex)})
            finally:
                end_trace()

        task = asyncio.create_task(run_and_finish())

        yield "data: " + json.dumps({"type": "hello", "conv_id": conv_id}, ensure_ascii=False) + "\n\n"
        while True:
            ev = await queue.get()
            yield "data: " + json.dumps(ev, ensure_ascii=False) + "\n\n"
            if ev.get("type") in ("done", "error"):
                break
        await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲，保证逐 token 透传
        },
    )


@app.post("/mcp", tags=["MCP"])
async def mcp_endpoint(request: Request):
    """
    标准 MCP 协议端点（JSON-RPC 2.0 / Streamable HTTP transport）。

    支持 MCP 2025-11-25 Streamable HTTP 的 tools 子集。请求必须声明 JSON 与
    SSE Accept；通知按规范返回 202。它不是完整的账号或 RBAC 服务。
    用户身份来自签名登录 Cookie；未登录的 MCP 调用仍可使用公开工具，但个人工具拒绝访问。
    """
    if _tool_manager is None:
        raise HTTPException(503, "工具管理器未初始化")
    from mcp.protocol import MCPServer, SUPPORTED_PROTOCOL_VERSIONS

    _validate_mcp_origin(request)
    _validate_mcp_accept(request)
    version = request.headers.get("MCP-Protocol-Version")
    if version and version not in SUPPORTED_PROTOCOL_VERSIONS:
        raise HTTPException(400, "不支持的 MCP-Protocol-Version")
    user = optional_user(request)
    user_id = user.id if user else "anonymous"
    server = MCPServer(_tool_manager, user_id=user_id)
    try:
        raw = (await request.body()).decode("utf-8")
    except UnicodeDecodeError:
        # 非法 UTF-8：返回标准 JSON-RPC PARSE_ERROR（-32700），
        # 而不是静默丢弃字节后让协议层给出通用解析错误。
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": "Parse error: 请求体不是合法 UTF-8"},
        }, status_code=400)
    result = await server.handle(raw)
    if result is None:
        return Response(status_code=202)
    return JSONResponse(result)


@app.get("/mcp", tags=["MCP"])
async def mcp_method_not_allowed():
    """Streamable HTTP 在本服务不提供 SSE GET 流；信息改由 /mcp/info 提供。"""
    return Response(status_code=405, headers={"Allow": "POST"})


@app.get("/mcp/info", tags=["MCP"])
async def mcp_info():
    """EchoGuide 支持的 MCP transport 与工具说明（非协议握手端点）。"""
    if _tool_manager is None:
        raise HTTPException(503, "工具管理器未初始化")
    from mcp.protocol import PROTOCOL_VERSION

    tools = [
        {"name": name, "description": t.description, "inputSchema": t.schema}
        for name, t in _tool_manager._tools.items()
    ]
    return {
        "server": "echoguide-mcp",
        "protocolVersion": PROTOCOL_VERSION,
        "tools": tools,
        "note": "POST /mcp 为 MCP Streamable HTTP tools 子集；GET /mcp 明确返回 405。",
    }


@app.get("/traces", tags=["观测"])
async def traces_list(limit: int = 20, _admin: AuthUser = Depends(require_observability)):
    """最近的全链路 trace（排障/演示用）。"""
    from core.tracing import list_traces

    return {"traces": list_traces(limit=max(1, min(limit, 200)))}


@app.get("/traces/{trace_id}", tags=["观测"])
async def trace_detail(trace_id: str, _admin: AuthUser = Depends(require_observability)):
    """单条 trace 详情：request → intent → agent → tool → LLM 逐跳耗时。"""
    from core.tracing import get_trace

    record = get_trace(trace_id)
    if record is None:
        raise HTTPException(404, f"trace 不存在: {trace_id}")
    return record


@app.get("/monitor")
async def monitor_summary(_admin: AuthUser = Depends(require_observability)):
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/metrics")
async def prometheus_metrics():
    """
    Prometheus 指标入口（只读，无认证）。

    安全权衡：指标不含敏感数据，供 Prometheus 无认证抓取（config/prometheus.yml
    已按此配置）；生产环境应通过网络层（防火墙/内网）限制 /metrics 的暴露面。
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(
    query: str = Query(min_length=1, max_length=500),
    top_k: int = Query(default=5, ge=1, le=20),
):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if _tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await _tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k)
    return {"query": query, "results": result.data, "reranked": result.reranked}


class DocInput(BaseModel):
    """单篇文档输入。"""
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1, max_length=100_000)
    domain: Optional[Literal["academic", "campus_life", "affairs", "it_help", "general"]] = None
    source_url: Optional[str] = Field(default=None, max_length=1000)
    updated_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    source_status: Literal["official", "unverified", "sample", "stale"] = "unverified"


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput] = Field(min_length=1, max_length=100)


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str = Field(min_length=1, max_length=2000)
    expected_intent: str = Field(min_length=1, max_length=80)
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮；可选 golden_answer（Answer Correctness）。"""
    question: Optional[str] = Field(default=None, min_length=1, max_length=2000)
    turns: Optional[List[str]] = Field(default=None, min_length=1, max_length=12)
    user_id: Optional[str] = Field(default=None, max_length=64)
    conv_id: Optional[str] = Field(default=None, max_length=80)
    golden_answer: Optional[str] = Field(default=None, max_length=10000)

    @model_validator(mode="after")
    def require_question_or_turns(self):
        if not self.question and not self.turns:
            raise ValueError("question 或 turns 至少提供一个")
        return self


class RetrievalCaseInput(BaseModel):
    """RAG 检索硬指标评测用例。"""
    query: str = Field(min_length=1, max_length=500)
    relevant_titles: List[str] = Field(min_length=1, max_length=20)


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None
    routing_cases: Optional[List[Dict[str, Any]]] = None
    retrieval_cases: Optional[List[RetrievalCaseInput]] = None
    promote_baseline: bool = False


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput, _admin: AuthUser = Depends(require_admin)):
    """
    批量导入文档到知识库。

    文档会自动切片（每片 500 字）并存入 ChromaDB，ChromaDB 内置 Embedding 模型自动向量化。

    示例请求体：
    ```json
    {
      "documents": [
        {"title": "选课指南", "content": "西电选课通过教务系统进行，分预选、正选、退改选阶段..."},
        {"title": "校园穿梭车", "content": "校园穿梭车连接南校区与北校区，工作日班次较多..."}
      ]
    }
    ```
    """
    if _kb is None:
        raise HTTPException(503, "知识库未初始化")
    kb = _kb
    count = await asyncio.to_thread(kb.add_documents, [
        {
            "title": d.title, "content": d.content, "domain": d.domain,
            "source_url": d.source_url, "updated_at": d.updated_at.isoformat() if d.updated_at else "",
            "valid_from": d.valid_from.isoformat() if d.valid_from else "", "source_status": d.source_status,
        } for d in body.documents
    ])
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": kb.doc_count}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...), _admin: AuthUser = Depends(require_admin)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json` / `.jsonl`：JSON 数组 `[{"title": "...", "content": "..."}, ...]` / 每行一个对象
    - 其余格式（pdf/doc/docx/ppt/pptx/xls/xlsx/odt/odp/rtf/epub/csv 等）由
      Firecrawl anydoc 统一转为 GFM Markdown，标题/表格结构完整保留；
      扫描件（无文本层的 PDF）会明确报错

    文件大小限制：10MB
    """
    if _kb is None:
        raise HTTPException(503, "知识库未初始化")
    kb = _kb

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    from mcp.document_parser import parse_document
    try:
        docs = parse_document(file.filename or "unknown", content)
    except ValueError as e:
        raise HTTPException(400, str(e))

    count = await asyncio.to_thread(kb.add_documents, docs)
    return {
        "message": f"文件 {file.filename} 导入成功",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    if _kb is None:
        raise HTTPException(503, "知识库未初始化")
    return {"total_chunks": _kb.doc_count}


# ── 个人数据中心 ──────────────────────────────────────────────────────────────

class ScheduleImportBody(BaseModel):
    """课表导入请求体：courses（JSON 课表）与 ics_text（ICS 文本）二选一。"""
    user_id: str = Field(default="anonymous", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    courses: Optional[List[Dict[str, Any]]] = Field(default=None, max_length=200)
    ics_text: Optional[str] = Field(default=None, max_length=500_000)


def _require_personal_service():
    if _personal_service is None:
        raise HTTPException(503, "个人数据中心未初始化")
    return _personal_service


@app.post("/personal/schedule/import", tags=["个人数据"])
async def import_schedule(body: ScheduleImportBody, user: AuthUser = Depends(require_user)):
    """
    导入课程表（整表替换）。支持两种格式：
      1. JSON 课表：{"user_id": "...", "courses": [{"course", "day_of_week", "start_time", "end_time", "location", "weeks"}]}
      2. ICS 文本：{"user_id": "...", "ics_text": "BEGIN:VCALENDAR..."}（教务系统导出）
    返回导入的课程数量。
    """
    personal = _require_personal_service()
    if body.courses is not None:
        count = await personal.import_courses(user.id, body.courses)
    elif body.ics_text:
        from personal.ics_parser import parse_ics
        from personal.time_context import SEMESTER_START, SEMESTER_WEEKS

        courses = parse_ics(body.ics_text, SEMESTER_START, SEMESTER_WEEKS)
        count = await personal.import_courses(
            user.id, [c.to_dict() for c in courses]
        )
    else:
        raise HTTPException(400, "请提供 courses（JSON 课表）或 ics_text（ICS 文本）")
    return {"message": f"课表导入成功，共 {count} 门课程", "courses": count}


@app.post("/personal/schedule/import/file", tags=["个人数据"])
async def import_schedule_file(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
    user: AuthUser = Depends(require_user),
):
    """
    上传 .ics（教务系统导出）或 .json 课表文件导入。
    文件大小限制 5MB。
    """
    personal = _require_personal_service()
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 5MB 限制")
    text = content.decode("utf-8", errors="ignore")
    filename = (file.filename or "").lower()

    from personal.ics_parser import parse_ics
    from personal.time_context import SEMESTER_START, SEMESTER_WEEKS

    if filename.endswith(".json"):
        try:
            docs = json.loads(text)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
        if isinstance(docs, dict):
            docs = docs.get("courses", [])
        if not isinstance(docs, list):
            raise HTTPException(400, "JSON 课表应为数组: [{course, day_of_week, start_time, end_time, location, weeks}]")
        count = await personal.import_courses(user.id, docs)
    elif filename.endswith(".ics"):
        courses = parse_ics(text, SEMESTER_START, SEMESTER_WEEKS)
        count = await personal.import_courses(user.id, [c.to_dict() for c in courses])
    else:
        raise HTTPException(400, "仅支持 .ics 或 .json 文件")
    return {"message": f"文件 {file.filename} 导入成功，共 {count} 门课程", "courses": count}


@app.get("/personal/schedule", tags=["个人数据"])
async def get_schedule(user: AuthUser = Depends(require_user)):
    """查看用户课表（本周周视图 + 全部课程）。"""
    personal = _require_personal_service()
    weekly = await personal.weekly_overview(user.id)
    return {
        "user_id": user.id,
        "week_num": weekly["week_num"],
        "monday": weekly["monday"],
        "courses": weekly["courses"],
        "total": len(weekly["courses"]),
    }


@app.delete("/personal/schedule", tags=["个人数据"])
async def clear_schedule(user: AuthUser = Depends(require_user)):
    """清空用户课表（重新导入前使用）。"""
    personal = _require_personal_service()
    await personal.store.clear_schedule(user.id)
    return {"message": "课表已清空"}


class TodoBody(BaseModel):
    user_id: str = Field(default="anonymous", min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    content: str = Field(min_length=1, max_length=500)
    kind: Literal["todo", "ddl", "exam"] = "todo"
    due_at: Optional[str] = Field(default=None, max_length=32, pattern=r"^\d{4}-\d{2}-\d{2}( \d{2}:\d{2})?$")


@app.post("/personal/todo", tags=["个人数据"])
async def add_todo(body: TodoBody, user: AuthUser = Depends(require_user)):
    """新增待办 / DDL / 考试安排。"""
    personal = _require_personal_service()
    if not body.content.strip():
        raise HTTPException(400, "content 不能为空")
    todo = await personal.add_todo(
        user.id, body.content.strip(),
        kind=body.kind,  # Literal["todo","ddl","exam"] 已校验，无需再判
        due_at=body.due_at,
    )
    return {"message": "已记录", "todo": todo}


@app.get("/personal/todo", tags=["个人数据"])
async def list_todos(status: str = "open", user: AuthUser = Depends(require_user)):
    """查看待办清单（open/done/all）。"""
    personal = _require_personal_service()
    todos = await personal.list_todos(user.id, status=status)
    return {"user_id": user.id, "status": status, "todos": todos, "total": len(todos)}


@app.post("/personal/todo/{todo_id}/complete", tags=["个人数据"])
async def complete_todo(todo_id: int, done: bool = True, user: AuthUser = Depends(require_user)):
    """标记完成 / 恢复待办。"""
    personal = _require_personal_service()
    todo = await personal.complete_todo(user.id, todo_id, done=done)
    if todo is None:
        raise HTTPException(404, f"待办 {todo_id} 不存在或不属于该用户")
    return {"message": "已标记完成" if done else "已恢复未完成", "todo": todo}


@app.delete("/personal/todo/{todo_id}", tags=["个人数据"])
async def delete_todo(todo_id: int, user: AuthUser = Depends(require_user)):
    """删除待办。"""
    personal = _require_personal_service()
    ok = await personal.delete_todo(user.id, todo_id)
    if not ok:
        raise HTTPException(404, f"待办 {todo_id} 不存在或不属于该用户")
    return {"message": "已删除"}


@app.get("/personal/overview", tags=["个人数据"])
async def personal_overview(user: AuthUser = Depends(require_user)):
    """当日汇总：课程 + 待办 + 未来 7 天 DDL/考试倒计时（对话工具与前端共用）。"""
    personal = _require_personal_service()
    return await personal.overview(user.id)


# ── 结构化公开信息 ────────────────────────────────────────────────────────────

@app.get("/campus/info", tags=["公开信息"])
async def campus_info(category: str = "shuttle", keyword: str = ""):
    """
    查询校园公开信息（结构化数据）。
    category: shuttle（校车下一班，keyword 传方向）/ buildings（楼宇）/ venues（场馆）/ library（图书馆）。
    """
    if _campus_store is None:
        raise HTTPException(503, "公开信息数据源未初始化")
    return _campus_store.search(category, keyword)


@app.post("/campus/reload", tags=["公开信息"])
async def campus_reload(_admin: AuthUser = Depends(require_admin)):
    """热加载 data/public/*.json（填充真实数据后无需重启）。"""
    if _campus_store is None:
        raise HTTPException(503, "公开信息数据源未初始化")
    return {"status": _campus_store.reload()}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None, _admin: AuthUser = Depends(require_admin)):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import (
        DEFAULT_DIALOG_CASES,
        DEFAULT_INTENT_CASES,
        DEFAULT_RETRIEVAL_CASES,
        DEFAULT_ROUTING_CASES,
        IntentTestCase,
        RetrievalTestCase,
    )

    if body and body.intent_cases is not None:
        intent_cases = [
            IntentTestCase(
                message=c.message,
                expected_intent=c.expected_intent,
                context=c.context,
            )
            for c in body.intent_cases
        ]
    else:
        intent_cases = DEFAULT_INTENT_CASES

    if body and body.dialog_cases is not None:
        dialog_cases = [
            c.model_dump(exclude_none=True)
            for c in body.dialog_cases
        ]
    else:
        dialog_cases = DEFAULT_DIALOG_CASES

    routing_cases = body.routing_cases if body and body.routing_cases is not None else DEFAULT_ROUTING_CASES

    if body and body.retrieval_cases is not None:
        retrieval_cases = [
            RetrievalTestCase(query=c.query, relevant_titles=c.relevant_titles)
            for c in body.retrieval_cases
        ]
    else:
        retrieval_cases = DEFAULT_RETRIEVAL_CASES

    custom_cases = bool(
        body
        and (
            body.intent_cases is not None
            or body.dialog_cases is not None
            or body.routing_cases is not None
            or body.retrieval_cases is not None
        )
    )
    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
        routing_cases=routing_cases,
        retrieval_cases=retrieval_cases,
        promote_baseline=bool(body and body.promote_baseline),
        dataset="custom_cases" if custom_cases else "built_in_cases_v1",
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
        "retrieval":       report.retrieval,
        "provenance":      report.provenance,
        "judge":           report.judge,
        "results": [
            {
                "test_id": r.test_id,
                "passed": r.passed,
                "scores": r.scores,
                "detail": r.detail,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
    }


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("EchoGuide CLI — 输入 quit 退出\n")

    # 复用与 lifespan 相同的组件构造（含工具注册/记忆/语义缓存），避免配置漂移
    from agents.agent_orchestrator import Request
    from memory.conversation_memory import MsgRole

    _build_runtime()
    await _setup_external_mcp()
    orch = _orchestrator
    mem  = _memory

    user_id, conv_id = "cli_user", str(uuid.uuid4())

    while True:
        try:
            msg = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见 ʕ•ᴥ•ʔ")
            break
        if not msg or msg.lower() in ("quit", "exit", "退出"):
            print("再见 ʕ•ᴥ•ʔ")
            break

        ctx = await mem.get_context(user_id, conv_id, query=msg)
        history = [
            {"role": m.role.value, "content": m.content}
            for m in ctx.recent_messages[-5:]
        ] if ctx.recent_messages else None
        req = Request(message=msg, user_id=user_id, conv_id=conv_id, context=ctx.to_prompt_text(), history=history)
        result = await orch.run(req)

        await mem.add_message(user_id, conv_id, MsgRole.USER, msg)
        await mem.add_message(user_id, conv_id, MsgRole.ASSISTANT, result.response)

        print(f"\nEchoGuide [{result.agent_type.value}]: {result.response}\n")


# ── 同源托管：单端口同时提供前端页面与 API（本地/单进程模式）──────────────────
# 前端 dist 存在时自动启用：/api/* 剥离前缀转给真实路由（与 Vite/nginx 代理
# 语义一致），其余路径走 SPA 回退到 index.html。ECHOGUIDE_SERVE_STATIC=0 关闭。
# 中间件在文件末尾注册（晚于 EchoGuard）：Starlette 后注册者先执行，保证
# /api 前缀在 Guard 看到请求前剥离（Guard 路径白名单基于真实路由）。
_FRONTEND_DIST = pathlib.Path(_ROOT) / "frontend" / "dist"


@app.middleware("http")
async def _strip_api_prefix(request: Request, call_next):
    if request.url.path.startswith("/api/"):
        scope = request.scope
        scope["path"] = scope["path"][4:]
        raw = scope.get("raw_path")
        if raw:
            scope["raw_path"] = raw[4:]
    return await call_next(request)


if os.getenv("ECHOGUIDE_SERVE_STATIC", "1") == "1" and _FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="frontend_assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa_fallback(full_path: str):
        """SPA 回退：存在的静态文件直接返回，其余一律 index.html（前端路由接管）。"""
        root = _FRONTEND_DIST.resolve()
        target = (root / full_path).resolve()
        if full_path and target.is_file() and target.is_relative_to(root):
            return FileResponse(target)
        return FileResponse(root / "index.html")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
