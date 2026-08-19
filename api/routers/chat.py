"""对话路由：/chat、/chat/stream（SSE）、/search（RAG 链路演示）。"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api import state
from api.deps import optional_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["对话"])


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


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, response: Response, request: Request = None):
    """
    主对话接口。完整流程：
      语义缓存 → 记忆读取 → 意图识别（领域×动作）→ Planner（Task DAG）→ Profile 选择 →
      Agentic RAG 执行 → 记忆写入 → 语义缓存写入

    request 默认 None：HTTP 场景由 FastAPI 注入真实请求，
    离线单测直接传 None 跳过身份覆盖（保留旧测试路径）。
    注意：注解不能写成 Optional[Request]（新版 FastAPI 会当成响应字段报错）。
    """
    if state._orchestrator is None or state._memory is None:
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
            mem_ctx = await state._memory.get_context(req.user_id, conv_id, query=req.message)
        ctx_text = mem_ctx.to_prompt_text()
        from mcp.semantic_cache import classify_context_dependence

        dependence = classify_context_dependence(req.message, ctx_text)

        # 2. 双层语义缓存读取（默认关闭，仅 SEMANTIC_CACHE_ENABLED=1 时生效）：
        #    - 公共事实查询（global）→ 只查 Global（语义匹配，容忍近义改写）；
        #    - 依赖用户画像（user）+ 有效身份 → 只查 User（仅 user_id 分区，
        #      miss 不回退 Global，防止公共答案绕过个性化 Agent 推理）；
        #    - 强上下文依赖（skip：追问/省略句/指代/个人数据）→ 直接 bypass。
        cached = await state._cache_get(
            state._semantic_cache, req.message, user_id=req.user_id, dependence=dependence
        ) if state._semantic_cache else None
        if cached and cached.get("domain") == "personal":
            logger.warning("命中 personal 领域缓存，丢弃（防跨用户串扰）")
            cached = None
        if cached:
            logger.info("语义缓存命中 %r", req.message[:30])
            await state._memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
            await state._memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, cached["response"])
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
            result = await state._orchestrator.run(orch_req)
        result.execution["trace_id"] = trace.trace_id
        # 记忆 trace（分层命中统计，透出给前端 debug 面板 / 评测统计）
        result.execution["memory_trace"] = getattr(mem_ctx, "memory_trace", {})

        # 5. 写入记忆
        async with span("memory_write"):
            await state._memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
            await state._memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)

        # 6. 异步更新用户画像 + 双层语义缓存（不阻塞响应）
        state._spawn_background(state._memory.update_profile(req.user_id, conv_id))
        if state._semantic_cache:
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
                state._cache_put(state._semantic_cache,
                    req.message, result.response,
                    domain=result.domain.value,
                    agent_type=result.agent_type,
                    user_id=req.user_id,
                    dependence=dep_write,
                    knowledge_used="knowledge_search" in result.tools_used,
                )
            elif tier == "global":
                state._cache_put(state._semantic_cache,
                    req.message, result.response,
                    domain=result.domain.value,
                    agent_type=result.agent_type,
                    dependence=dep_write,
                    knowledge_used="knowledge_search" in result.tools_used,
                )

        return ChatResponse(
            conv_id=conv_id,
            response=result.response,
            intent=result.domain.value if result.domain else "other",
            domain=result.domain.value if result.domain else "other",
            action=result.action.value if result.action else "other",
            agent_type=result.agent_type,
            latency_ms=round((time.perf_counter() - request_started) * 1000, 1),
            knowledge_used="knowledge_search" in result.tools_used,
            cached=False,
            execution=result.execution,
        )
    finally:
        end_trace()


@router.post("/chat/stream")
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
    if state._orchestrator is None or state._memory is None:
        raise HTTPException(503, "服务未就绪")

    if request is not None:
        user = optional_user(request)
        req.user_id = user.id if user else "anonymous"

    from agents.agent_orchestrator import Request as OrcReq
    from core.tracing import begin_trace, end_trace, span
    from memory.conversation_memory import MsgRole

    conv_id = req.conv_id or str(uuid.uuid4())

    async def event_stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def on_event(ev: dict) -> None:
            await queue.put(ev)

        async def run_and_finish() -> None:
            trace = begin_trace("chat_stream")
            request_started = time.perf_counter()
            try:
                # 1. 先读记忆上下文 —— 语义缓存必须在其后（同 /chat）
                async with span("memory_read"):
                    mem_ctx = await state._memory.get_context(req.user_id, conv_id, query=req.message)
                ctx_text = mem_ctx.to_prompt_text()
                from mcp.semantic_cache import classify_context_dependence

                dependence = classify_context_dependence(req.message, ctx_text)

                # 2. 双层语义缓存读取（默认关闭；读取层由上下文依赖性决定，与 /chat 完全一致）
                cached = await state._cache_get(
                    state._semantic_cache, req.message, user_id=req.user_id, dependence=dependence
                ) if state._semantic_cache else None
                if cached and cached.get("domain") == "personal":
                    logger.warning("命中 personal 领域缓存，丢弃（防跨用户串扰）")
                    cached = None
                if cached:
                    # 缓存命中：写记忆 + 推送 meta/delta/done（hello 由外层统一输出）
                    await state._memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
                    await state._memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, cached["response"])
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
                    result = await state._orchestrator.run(orch_req, on_event=on_event)
                result.execution["trace_id"] = trace.trace_id
                # 记忆 trace（分层命中统计，透出给前端 debug 面板 / 评测统计）
                result.execution["memory_trace"] = getattr(mem_ctx, "memory_trace", {})

                async with span("memory_write"):
                    await state._memory.add_message(req.user_id, conv_id, MsgRole.USER, req.message)
                    await state._memory.add_message(req.user_id, conv_id, MsgRole.ASSISTANT, result.response)
                state._spawn_background(state._memory.update_profile(req.user_id, conv_id))

                if state._semantic_cache:
                    # 写入层决策：与读取侧同一规则（叠加编排信号），同 /chat
                    from mcp.semantic_cache import cache_tier, classify_context_dependence

                    dep_write = classify_context_dependence(
                        req.message, ctx_text,
                        domain=result.domain.value if result.domain else None,
                        action=result.action.value if result.action else None,
                    )
                    tier = cache_tier(dep_write, req.user_id)
                    if tier == "user":
                        state._cache_put(state._semantic_cache,
                            req.message, result.response,
                            domain=result.domain.value,
                            agent_type=result.agent_type,
                            user_id=req.user_id,
                            dependence=dep_write,
                            knowledge_used="knowledge_search" in result.tools_used,
                        )
                    elif tier == "global":
                        state._cache_put(state._semantic_cache,
                            req.message, result.response,
                            domain=result.domain.value,
                            agent_type=result.agent_type,
                            dependence=dep_write,
                            knowledge_used="knowledge_search" in result.tools_used,
                        )

                await queue.put({
                    "type": "done",
                    "conv_id": conv_id,
                    "response": result.response,
                    "intent": result.domain.value if result.domain else "other",
                    "agent_type": result.agent_type,
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


@router.post("/search")
async def search(
    query: str = Query(min_length=1, max_length=500),
    top_k: int = Query(default=5, ge=1, le=20),
):
    """
    演示检索优化链路：查询改写 → 并行召回 → 重排 → Top-K。
    展示 MCP 工具调用的核心亮点。
    """
    if state._tool_manager is None:
        raise HTTPException(503, "服务未就绪")
    result = await state._tool_manager.search_with_rewrite("knowledge_search", query, top_k=top_k)
    return {"query": query, "results": result.data, "reranked": result.reranked}
