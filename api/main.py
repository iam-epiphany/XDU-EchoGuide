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
from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File
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

    print(BANNER, flush=True)

    from agents.agent_orchestrator import AgentOrchestrator, Request
    from evaluation.evaluator import EndToEndEvaluator
    from mcp.knowledge_base import KnowledgeBase
    from mcp.semantic_cache import SemanticCache
    from mcp.tool_manager import MCPToolManager, Tool
    from memory.conversation_memory import MemoryManager
    from monitor.performance_monitor import PerformanceMonitor
    from core.skill_loader import SkillManager

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

    # Agentic RAG：把工具管理器注入 Agent 池，让 Agent 自主决定何时检索知识库
    _orchestrator.set_tool_manager(_tool_manager)

    # 语义缓存（GPTCache 思路）：相似问题直接复用答案，成本/延迟趋近于 0
    _semantic_cache = SemanticCache(
        chroma_host=os.getenv("CHROMA_HOST", "chromadb"),
        chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        chroma_path=os.getenv("CHROMA_PERSIST_DIRECTORY", "/app/data/chroma"),
        threshold=float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90")),
        enabled=os.getenv("SEMANTIC_CACHE_ENABLED", "1") == "1",
    )

    # EchoGuard 真实接入：中间件形式保护 /chat 等敏感端点（默认关闭）
    if os.getenv("ECHOGUIDE_GUARD_ENABLED", "0") == "1":
        from echoguide_guard.integration import EchoGuardMiddleware, GuardSettings

        app.add_middleware(EchoGuardMiddleware, settings=GuardSettings())
        logger.warning("[EchoGuard] 中间件已接入真实请求链（认证/注入检测/限流/脱敏审计）")

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
        cached = _semantic_cache.get(req.message) if _semantic_cache else None
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
        if _semantic_cache and result.domain and result.domain.value != "other":
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

        # 语义缓存命中：直接输出完整答案，跳过 LLM 链路
        cached = _semantic_cache.get(req.message) if _semantic_cache else None
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

                if _semantic_cache and result.domain and result.domain.value != "other":
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
    """
    if _tool_manager is None:
        raise HTTPException(503, "工具管理器未初始化")
    from mcp.protocol import MCPServer

    server = MCPServer(_tool_manager)
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
