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
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

# 将项目根目录加入 sys.path，确保无论从哪里执行都能找到 agents/core/memory 等模块
# 这一行必须在所有项目内部 import 之前执行
_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

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

# ── 全局组件（lifespan 中初始化）─────────────────────────────────────────────
_orchestrator = None
_memory       = None
_tool_manager = None
_monitor      = None
_evaluator    = None
_skill_manager = None
_semantic_cache = None
_personal_service = None
_campus_store = None


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
    return cfg


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _orchestrator, _memory, _tool_manager, _monitor, _evaluator, _skill_manager
    global _personal_service, _campus_store

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request
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
    from tools import with_service
    from tools.campus_tool import campus_info_handler
    from tools.ddl_tool import query_ddl_handler
    from tools.schedule_tool import query_schedule_handler
    from tools.todo_tool import add_todo_handler, complete_todo_handler, query_todo_handler
    from tools.weather import weather_handler

    cfg = _anthropic_cfg()
    logger.info(f"模型: {cfg['model']}  base_url: {cfg.get('base_url', '(官方)')}")

    # Skills：启动时从目录加载业务能力说明，并在 Agent 调用 LLM 时动态注入。
    skills_dir = os.getenv("ECHOGUIDE_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills"))
    _skill_manager = SkillManager(
        root_dir=skills_dir,
        max_prompt_chars=int(os.getenv("ECHOGUIDE_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    _skill_manager.load()

    # Agent 编排器（内部持有意图识别器，供评测器复用，避免双实例缓存分家）
    _orchestrator = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=_skill_manager,
    )

    # 记忆管理器（Redis 工作记忆 + ChromaDB 情景记忆/用户画像）
    _memory = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

    # MCP 工具管理器 + RAG 知识库（基于 ChromaDB 的真实检索）
    _tool_manager = MCPToolManager(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )
    kb = KnowledgeBase(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
    )
    logger.info(f"知识库已加载: {kb.doc_count} 个文档片段")

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
        handler=kb.search_handler,
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
        fallback=knowledge_fallback,
    ))

    # ── 个人数据中心（课表 / 待办 / DDL，按 user_id 隔离，SQLite 持久化）──
    _personal_service = PersonalService(PersonalStore())
    logger.info(f"个人数据中心已就绪: {_personal_service.store.db_path}")

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
    logger.info(f"校园公开信息已就绪: {_campus_store.load_status}")
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
    await _monitor.start()

    # 评测器（复用编排器内部的意图识别器，避免双实例缓存/统计分家）
    # 双模型 LLM-as-Judge：评判模型可与生成模型分离，消除自评偏差
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
    )

    logger.info("EchoGuide 西电校园智慧助手已就绪")
    yield

    await _monitor.stop()
    logger.info("EchoGuide 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="西电校园智慧助手 EchoGuide",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# EchoGuard 真实接入：中间件必须在应用启动前挂载（lifespan 中挂载会报错）。
# 默认启用，保护 /chat、/personal、/mcp 等 POST 端点：
# 注入检测 / 限流 / 脱敏审计开箱即用；配置 ECHOGUIDE_GUARD_TOKEN 后开启认证。
if os.getenv("ECHOGUIDE_GUARD_ENABLED", "1") == "1":
    from echoguide_guard.integration import EchoGuardMiddleware, GuardSettings

    app.add_middleware(EchoGuardMiddleware, settings=GuardSettings())
    logger.warning("[EchoGuard] 中间件已接入真实请求链（注入检测/限流/脱敏审计，认证按需启用）")


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:     str
    user_id:     str = "anonymous"
    conv_id:     Optional[str] = None


class ChatResponse(BaseModel):
    conv_id:     str
    response:    str
    intent:      str
    domain:      str = "other"
    action:      str = "other"
    agent_type:  str
    escalated:   bool
    latency_ms:  float
    knowledge_used: bool = False


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    if _orchestrator is None:
        raise HTTPException(503, "服务未就绪")
    return {"status": "ok", "agents": _orchestrator.get_stats()}


@app.get("/skills", tags=["Skills"])
async def skills_summary():
    """查看当前已加载的 Skills，便于确认热加载结果和排查解析错误。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    return _skill_manager.summary()


@app.post("/skills/reload", tags=["Skills"])
async def reload_skills():
    """运行时重新扫描 Skill 目录，不需要重启服务。"""
    if _skill_manager is None:
        raise HTTPException(503, "Skills 未初始化")
    _skill_manager.reload()
    if _orchestrator is not None:
        _orchestrator.set_skill_manager(_skill_manager)
    return _skill_manager.summary()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, response: Response):
    """
    主对话接口。完整流程：
      语义缓存 → 记忆读取 → 意图识别（领域×动作）→ Agent 路由 →
      Agentic RAG 执行 → 记忆写入 → 语义缓存写入
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from core.tracing import begin_trace, end_trace, span
    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())

    # 全链路 trace（X-Trace-Id 响应头，/traces/{id} 可查）
    trace = begin_trace("chat")
    trace.tags.update({"user_id": req.user_id, "conv_id": conv_id})
    response.headers["X-Trace-Id"] = trace.trace_id

    try:
        # 0. 语义缓存：相似问题直接复用答案（GPTCache 思路）
        #    注意：缓存 key 不区分 user_id，personal 领域（课表/待办等个人数据）
        #    的回答绝不缓存也不复用，否则会造成用户间数据串扰。
        cached = _semantic_cache.get(req.message) if _semantic_cache else None
        if cached and cached.get("domain") == "personal":
            logger.warning("命中 personal 领域缓存，丢弃（防跨用户串扰）")
            cached = None
        if cached:
            logger.info(f"语义缓存命中 {req.message[:30]!r}")
            await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
            await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, cached["response"])
            return ChatResponse(
                conv_id=conv_id,
                response=cached["response"],
                intent=cached["domain"],
                domain=cached["domain"],
                action="query",
                agent_type=cached["agent_type"],
                escalated=False,
                latency_ms=0.0,
                knowledge_used=True,
            )

        # 1. 读取记忆上下文
        async with span("memory_read"):
            mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)

        # 2. 构建编排请求（含对话历史，用于意图识别上下文与追问继承）
        history = [
            {"role": m.role.value, "content": m.content}
            for m in mem_ctx.recent_messages[-5:]
        ] if mem_ctx.recent_messages else None

        orch_req = OrcReq(
            message=req.message,
            user_id=req.user_id,
            conv_id=conv_id,
            context=mem_ctx.to_prompt_text(),
            history=history,
        )

        # 3. 执行（RAG 检索由 Agent 通过工具调用自主完成 —— Agentic RAG）
        async with span("orchestrator_run"):
            result = await _orchestrator.run(orch_req)

        # 4. 写入记忆
        async with span("memory_write"):
            await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
            await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)

        # 5. 异步更新用户画像 + 语义缓存（不阻塞响应）
        asyncio.create_task(_memory.update_profile(req.user_id, conv_id))
        if _semantic_cache and result.domain and result.domain.value not in ("other", "personal"):
            # personal 领域含个人数据，禁止进共享缓存（防跨用户串扰）
            _semantic_cache.put(
                req.message, result.response,
                domain=result.domain.value,
                agent_type=result.agent_type.value,
            )

        return ChatResponse(
            conv_id=conv_id,
            response=result.response,
            intent=result.domain.value if result.domain else "other",
            domain=result.domain.value if result.domain else "other",
            action=result.action.value if result.action else "other",
            agent_type=result.agent_type.value,
            escalated=result.escalated,
            latency_ms=round(result.latency_ms, 1),
            knowledge_used="knowledge_search" in result.tools_used,
        )
    finally:
        end_trace()


# （已移除）RAG 上下文改由 Agent 通过工具调用自主获取 —— Agentic RAG。


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    流式对话接口（SSE / Server-Sent Events）。

    事件序列：
      event: meta   意图/Agent 识别结果（含置信度）
      event: tool   Agent 工具调用过程（如 RAG 检索中/完成）
      event: delta  生成内容的增量文本（逐 token）
      event: done   最终汇总（完整回答、耗时、是否用 RAG）
      event: error  出错信息
    """
    if _orchestrator is None or _memory is None:
        raise HTTPException(503, "服务未就绪")

    from agents.agent_orchestrator import Request as OrcReq
    from core.tracing import begin_trace, end_trace, span
    from memory.conversation_memory import MsgRole
    from fastapi.responses import StreamingResponse

    conv_id = req.conv_id or str(uuid.uuid4())

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(ev: dict) -> None:
            await queue.put(ev)

        # 语义缓存命中：直接输出完整答案，跳过 LLM 链路。
        # personal 领域条目丢弃（缓存 key 不区分 user_id，防跨用户串扰）
        cached = _semantic_cache.get(req.message) if _semantic_cache else None
        if cached and cached.get("domain") == "personal":
            logger.warning("命中 personal 领域缓存，丢弃（防跨用户串扰）")
            cached = None
        if cached:
            await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
            await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, cached["response"])
            yield "data: " + json.dumps({"type": "hello", "conv_id": conv_id}, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({
                "type": "meta", "domain": cached["domain"], "action": "query",
                "agent": cached["agent_type"], "cached": True,
            }, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({"type": "delta", "text": cached["response"]}, ensure_ascii=False) + "\n\n"
            yield "data: " + json.dumps({
                "type": "done", "conv_id": conv_id, "response": cached["response"],
                "intent": cached["domain"], "agent_type": cached["agent_type"],
                "escalated": False, "latency_ms": 0.0, "knowledge_used": True, "cached": True,
            }, ensure_ascii=False) + "\n\n"
            return

        async def run_and_finish() -> None:
            trace = begin_trace("chat_stream")
            try:
                async with span("memory_read"):
                    mem_ctx = await _memory.get_context(req.user_id, conv_id, query=req.message)
                history = [
                    {"role": m.role.value, "content": m.content}
                    for m in mem_ctx.recent_messages[-5:]
                ] if mem_ctx.recent_messages else None

                orch_req = OrcReq(
                    message=req.message,
                    user_id=req.user_id,
                    conv_id=conv_id,
                    context=mem_ctx.to_prompt_text(),
                    history=history,
                )
                async with span("orchestrator_run"):
                    result = await _orchestrator.run(orch_req, on_event=on_event)

                async with span("memory_write"):
                    await _memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
                    await _memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)
                asyncio.create_task(_memory.update_profile(req.user_id, conv_id))

                if _semantic_cache and result.domain and result.domain.value not in ("other", "personal"):
                    # personal 领域含个人数据，禁止进共享缓存（防跨用户串扰）
                    _semantic_cache.put(
                        req.message, result.response,
                        domain=result.domain.value,
                        agent_type=result.agent_type.value,
                    )

                await queue.put({
                    "type": "done",
                    "conv_id": conv_id,
                    "response": result.response,
                    "intent": result.domain.value if result.domain else "other",
                    "agent_type": result.agent_type.value,
                    "escalated": result.escalated,
                    "latency_ms": round(result.latency_ms, 1),
                    "knowledge_used": "knowledge_search" in result.tools_used,
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

    支持 initialize / tools/list / tools/call / ping，任何 MCP 客户端可即插即用。
    用户身份：通过 X-User-Id 请求头传入（与前端 user_id 相同的软身份信任模型），
    个人工具（课表/待办/DDL）按该身份生效；不传则按 anonymous。
    """
    if _tool_manager is None:
        raise HTTPException(503, "工具管理器未初始化")
    from mcp.protocol import MCPServer

    user_id = (request.headers.get("X-User-Id") or "anonymous").strip()[:64]
    server = MCPServer(_tool_manager, user_id=user_id)
    raw = (await request.body()).decode("utf-8", errors="ignore")
    return await server.handle(raw)


@app.get("/mcp", tags=["MCP"])
async def mcp_info():
    """MCP 服务信息。"""
    if _tool_manager is None:
        raise HTTPException(503, "工具管理器未初始化")
    tools = [
        {"name": name, "description": t.description, "inputSchema": t.schema}
        for name, t in _tool_manager._tools.items()
    ]
    return {
        "server": "echoguide-mcp",
        "protocolVersion": "2025-03-26",
        "tools": tools,
        "note": "POST /mcp 为 JSON-RPC 2.0 端点（initialize/tools/list/tools/call）",
    }


@app.get("/traces", tags=["观测"])
async def traces_list(limit: int = 20):
    """最近的全链路 trace（排障/演示用）。"""
    from core.tracing import list_traces

    return {"traces": list_traces(limit=limit)}


@app.get("/traces/{trace_id}", tags=["观测"])
async def trace_detail(trace_id: str):
    """单条 trace 详情：request → intent → agent → tool → LLM 逐跳耗时。"""
    from core.tracing import get_trace

    record = get_trace(trace_id)
    if record is None:
        raise HTTPException(404, f"trace 不存在: {trace_id}")
    return record


@app.get("/monitor")
async def monitor_summary():
    """实时监控摘要：Agent 成功率、工具统计、告警、优化建议。"""
    if _monitor is None:
        raise HTTPException(503, "服务未就绪")
    return _monitor.summary()


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus 指标入口。"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/search")
async def search(query: str, top_k: int = 5):
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
    title:   str
    content: str


class BatchDocInput(BaseModel):
    """批量文档导入请求体。"""
    documents: List[DocInput]


class EvalIntentInput(BaseModel):
    """意图识别评测用例。"""
    message: str
    expected_intent: str
    context: Optional[Dict[str, Any]] = None


class EvalDialogInput(BaseModel):
    """对话质量评测用例。question 单轮，turns 多轮。"""
    question: Optional[str] = None
    turns: Optional[List[str]] = None
    user_id: Optional[str] = None
    conv_id: Optional[str] = None


class EvalRunInput(BaseModel):
    """评测请求。为空时使用内置默认用例。"""
    intent_cases: Optional[List[EvalIntentInput]] = None
    dialog_cases: Optional[List[EvalDialogInput]] = None
    routing_cases: Optional[List[Dict[str, Any]]] = None


@app.post("/knowledge/add", tags=["知识库"])
async def add_knowledge(body: BatchDocInput):
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
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    count = kb.add_documents([{"title": d.title, "content": d.content} for d in body.documents])
    return {"message": f"成功导入 {count} 个文档片段", "added_chunks": count, "total_chunks": kb.doc_count}


@app.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(file: UploadFile = File(...)):
    """
    上传文件导入知识库。

    支持格式：
    - `.txt` / `.md`：整个文件作为一篇文档，文件名作为标题
    - `.json`：JSON 数组格式 `[{"title": "...", "content": "..."}, ...]`

    文件大小限制：10MB
    """
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "文件大小超过 10MB 限制")

    text = content.decode("utf-8", errors="ignore")
    filename = file.filename or "unknown"

    if filename.endswith(".json"):
        import json as _json
        try:
            docs = _json.loads(text)
            if not isinstance(docs, list):
                raise HTTPException(400, "JSON 文件应为数组格式: [{title, content}, ...]")
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
    else:
        # txt / md：整个文件作为一篇文档
        title = filename.rsplit(".", 1)[0] if "." in filename else filename
        docs = [{"title": title, "content": text}]

    count = kb.add_documents(docs)
    return {
        "message": f"文件 {filename} 导入成功",
        "added_chunks": count,
        "total_chunks": kb.doc_count,
    }


@app.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """查看知识库统计信息（文档片段总数）。"""
    tool = _tool_manager._tools.get("knowledge_search") if _tool_manager else None
    if tool is None:
        raise HTTPException(503, "知识库未初始化")
    kb = tool.handler.__self__
    return {"total_chunks": kb.doc_count}


# ── 个人数据中心 ──────────────────────────────────────────────────────────────

class ScheduleImportBody(BaseModel):
    """课表导入请求体：courses（JSON 课表）与 ics_text（ICS 文本）二选一。"""
    user_id:  str = "anonymous"
    courses:  Optional[List[Dict[str, Any]]] = None
    ics_text: Optional[str] = None


def _require_personal_service():
    if _personal_service is None:
        raise HTTPException(503, "个人数据中心未初始化")
    return _personal_service


@app.post("/personal/schedule/import", tags=["个人数据"])
async def import_schedule(body: ScheduleImportBody):
    """
    导入课程表（整表替换）。支持两种格式：
      1. JSON 课表：{"user_id": "...", "courses": [{"course", "day_of_week", "start_time", "end_time", "location", "weeks"}]}
      2. ICS 文本：{"user_id": "...", "ics_text": "BEGIN:VCALENDAR..."}（教务系统导出）
    返回导入的课程数量。
    """
    personal = _require_personal_service()
    if body.courses is not None:
        count = await personal.import_courses(body.user_id, body.courses)
    elif body.ics_text:
        from personal.ics_parser import parse_ics
        from personal.time_context import SEMESTER_START, SEMESTER_WEEKS

        courses = parse_ics(body.ics_text, SEMESTER_START, SEMESTER_WEEKS)
        count = await personal.import_courses(
            body.user_id, [c.to_dict() for c in courses]
        )
    else:
        raise HTTPException(400, "请提供 courses（JSON 课表）或 ics_text（ICS 文本）")
    return {"message": f"课表导入成功，共 {count} 门课程", "courses": count}


@app.post("/personal/schedule/import/file", tags=["个人数据"])
async def import_schedule_file(
    file: UploadFile = File(...),
    user_id: str = Form("anonymous"),
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
        import json as _json
        try:
            docs = _json.loads(text)
        except _json.JSONDecodeError as e:
            raise HTTPException(400, f"JSON 解析失败: {e}")
        if isinstance(docs, dict):
            docs = docs.get("courses", [])
        if not isinstance(docs, list):
            raise HTTPException(400, "JSON 课表应为数组: [{course, day_of_week, start_time, end_time, location, weeks}]")
        count = await personal.import_courses(user_id, docs)
    elif filename.endswith(".ics"):
        courses = parse_ics(text, SEMESTER_START, SEMESTER_WEEKS)
        count = await personal.import_courses(user_id, [c.to_dict() for c in courses])
    else:
        raise HTTPException(400, "仅支持 .ics 或 .json 文件")
    return {"message": f"文件 {file.filename} 导入成功，共 {count} 门课程", "courses": count}


@app.get("/personal/schedule", tags=["个人数据"])
async def get_schedule(user_id: str = "anonymous"):
    """查看用户课表（本周周视图 + 全部课程）。"""
    personal = _require_personal_service()
    weekly = await personal.weekly_overview(user_id)
    return {
        "user_id": user_id,
        "week_num": weekly["week_num"],
        "monday": weekly["monday"],
        "courses": weekly["courses"],
        "total": len(weekly["courses"]),
    }


@app.delete("/personal/schedule", tags=["个人数据"])
async def clear_schedule(user_id: str = "anonymous"):
    """清空用户课表（重新导入前使用）。"""
    personal = _require_personal_service()
    await personal.store.clear_schedule(user_id)
    return {"message": "课表已清空"}


class TodoBody(BaseModel):
    user_id: str = "anonymous"
    content: str
    kind:    str = "todo"   # todo / ddl / exam
    due_at:  Optional[str] = None


@app.post("/personal/todo", tags=["个人数据"])
async def add_todo(body: TodoBody):
    """新增待办 / DDL / 考试安排。"""
    personal = _require_personal_service()
    if not body.content.strip():
        raise HTTPException(400, "content 不能为空")
    todo = await personal.add_todo(
        body.user_id, body.content.strip(),
        kind=body.kind if body.kind in ("todo", "ddl", "exam") else "todo",
        due_at=body.due_at,
    )
    return {"message": "已记录", "todo": todo}


@app.get("/personal/todo", tags=["个人数据"])
async def list_todos(user_id: str = "anonymous", status: str = "open"):
    """查看待办清单（open/done/all）。"""
    personal = _require_personal_service()
    todos = await personal.list_todos(user_id, status=status)
    return {"user_id": user_id, "status": status, "todos": todos, "total": len(todos)}


@app.post("/personal/todo/{todo_id}/complete", tags=["个人数据"])
async def complete_todo(todo_id: int, user_id: str = "anonymous", done: bool = True):
    """标记完成 / 恢复待办。"""
    personal = _require_personal_service()
    todo = await personal.complete_todo(user_id, todo_id, done=done)
    if todo is None:
        raise HTTPException(404, f"待办 {todo_id} 不存在或不属于该用户")
    return {"message": "已标记完成" if done else "已恢复未完成", "todo": todo}


@app.delete("/personal/todo/{todo_id}", tags=["个人数据"])
async def delete_todo(todo_id: int, user_id: str = "anonymous"):
    """删除待办。"""
    personal = _require_personal_service()
    ok = await personal.delete_todo(user_id, todo_id)
    if not ok:
        raise HTTPException(404, f"待办 {todo_id} 不存在或不属于该用户")
    return {"message": "已删除"}


@app.get("/personal/overview", tags=["个人数据"])
async def personal_overview(user_id: str = "anonymous"):
    """当日汇总：课程 + 待办 + 未来 7 天 DDL/考试倒计时（对话工具与前端共用）。"""
    personal = _require_personal_service()
    return await personal.overview(user_id)


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
async def campus_reload():
    """热加载 data/public/*.json（填充真实数据后无需重启）。"""
    if _campus_store is None:
        raise HTTPException(503, "公开信息数据源未初始化")
    return {"status": _campus_store.reload()}


@app.post("/eval/run")
async def run_eval(body: Optional[EvalRunInput] = None):
    """运行内置评测用例，返回评测报告。"""
    if _evaluator is None:
        raise HTTPException(503, "服务未就绪")
    from evaluation.evaluator import (
        DEFAULT_DIALOG_CASES,
        DEFAULT_INTENT_CASES,
        DEFAULT_ROUTING_CASES,
        IntentTestCase,
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

    report = await _evaluator.run(
        intent_cases=intent_cases,
        dialog_cases=dialog_cases,
        routing_cases=routing_cases,
    )
    return {
        "pass_rate":       report.pass_rate,
        "total":           report.total,
        "passed":          report.passed,
        "avg_scores":      report.avg_scores,
        "regressions":     report.regressions,
        "recommendations": report.recommendations,
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

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from memory.conversation_memory import MemoryManager, MsgRole
    from core.skill_loader import SkillManager

    cfg = _anthropic_cfg()
    skill_manager = SkillManager(
        root_dir=os.getenv("ECHOGUIDE_SKILLS_DIR", str(pathlib.Path(_ROOT) / "skills")),
        max_prompt_chars=int(os.getenv("ECHOGUIDE_SKILLS_MAX_PROMPT_CHARS", "5000")),
    )
    skill_manager.load()
    orch = AgentOrchestrator(
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
        skill_manager=skill_manager,
    )
    mem  = MemoryManager(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        chroma_host=os.getenv("CHROMA_HOST", "localhost"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/tmp/chroma"),
        api_key=cfg["api_key"],
        base_url=cfg.get("base_url"),
        model=cfg["model"],
    )

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
