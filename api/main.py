"""
西电校园智慧助手（EchoGuide）— FastAPI 入口（收口）

职责只保留四件事：
  1. app 组装（CORS / EchoGuard / 同源托管中间件）
  2. lifespan（构建运行时组件 → 启动 Monitor）
  3. 轻量登录认证（/auth/*）
  4. 交互式 CLI 与进程启动

业务路由按模块拆分（api/routers/）：chat（对话/SSE/检索演示）、
memory（个人数据）、knowledge（知识库）、monitor（观测/评测）、
mcp（MCP 协议）、system（健康/Skills/校园公开信息）。
全局组件与运行时构建统一在 api/state.py，避免多处初始化漂移。
"""
import asyncio
import logging
import os
import pathlib
import sys
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

from api import state  # noqa: E402  （全局组件：_orchestrator/_memory/_monitor…）
from api.deps import cookie_secure  # noqa: E402
from api.routers import chat as chat_router  # noqa: E402
from api.routers import knowledge as knowledge_router  # noqa: E402
from api.routers import mcp as mcp_router  # noqa: E402
from api.routers import memory as memory_router  # noqa: E402
from api.routers import monitor as monitor_router  # noqa: E402
from api.routers import system as system_router  # noqa: E402

# 向后兼容：旧测试/脚本直接访问 api.main 的组件与请求模型
_orchestrator = state._orchestrator
_memory = state._memory
_tool_manager = state._tool_manager
_monitor = state._monitor
_evaluator = state._evaluator
_skill_manager = state._skill_manager
_semantic_cache = state._semantic_cache
_personal_service = state._personal_service
_campus_store = state._campus_store
_kb = state._kb
_spawn_background = state._spawn_background
_cache_get = state._cache_get
_cache_put = state._cache_put
ChatRequest = chat_router.ChatRequest
ChatResponse = chat_router.ChatResponse
chat = chat_router.chat
chat_stream = chat_router.chat_stream
search = chat_router.search

BANNER = r"""
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
   ╔══════════════════════╗
   ║  EchoGuide  v4       ║
   ║  西电校园智慧助手     ║
   ╚══════════════════════╝
    ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ  ʕ•ᴥ•ʔ
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _monitor, _memory

    print(BANNER, flush=True)
    environment = os.getenv("ECHOGUIDE_ENV", os.getenv("APP_ENV", "development")).lower()
    if environment == "production" and not os.getenv("ECHOGUIDE_GUARD_TOKEN"):
        logger.warning("生产环境未配置 ECHOGUIDE_GUARD_TOKEN；浏览器登录仍可用，机器调用不具备服务令牌")

    state._build_runtime()
    await state._setup_external_mcp()
    await state._monitor.start()

    # main 命名空间同步（router 读 state.*，这里保持向后兼容别名）
    _monitor = state._monitor
    _memory = state._memory

    logger.info("EchoGuide 西电校园智慧助手已就绪")
    yield

    await state._monitor.stop()
    if state._memory is not None:
        await state._memory.close()
    logger.info("EchoGuide 已关闭")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="西电校园智慧助手 EchoGuide",
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENABLE_SWAGGER_UI", "true").lower() == "true" else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=state._allowed_origins(),
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

# 业务路由（按模块拆分，见 api/routers/）
for _router in (
    system_router.router,   # /health /skills /campus
    chat_router.router,     # /chat /chat/stream /search
    mcp_router.router,      # /mcp /mcp/info
    memory_router.router,   # /personal/*
    knowledge_router.router,  # /knowledge/*
    monitor_router.router,  # /monitor /metrics /traces /eval/run
):
    app.include_router(_router)


# ── 轻量登录认证 ──────────────────────────────────────────────────────────────
from auth.service import (  # noqa: E402
    SESSION_COOKIE,
    create_session_token,
    get_auth_store,
)
from api.deps import optional_user, require_user  # noqa: E402


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
        SESSION_COOKIE, create_session_token(user), httponly=True, secure=cookie_secure(),
        samesite="lax", max_age=7 * 24 * 3600, path="/",
    )
    return {"authenticated": True, "user": user.public()}


@app.post("/auth/login", tags=["认证"])
async def login(body: AuthCredentials, response: Response):
    user = await asyncio.to_thread(get_auth_store().authenticate, body.username, body.password)
    if user is None:
        raise HTTPException(401, "用户名或密码错误")
    response.set_cookie(
        SESSION_COOKIE, create_session_token(user), httponly=True, secure=cookie_secure(),
        samesite="lax", max_age=7 * 24 * 3600, path="/",
    )
    return {"authenticated": True, "user": user.public()}


@app.post("/auth/logout", tags=["认证"])
async def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/", samesite="lax", secure=cookie_secure(), httponly=True)
    return {"authenticated": False}


@app.get("/auth/me", tags=["认证"])
async def auth_me(user=Depends(optional_user)):
    return {"authenticated": user is not None, "user": user.public() if user else None}


@app.post("/auth/password", tags=["认证"])
async def change_password(body: PasswordChange, user=Depends(require_user)):
    try:
        changed = await asyncio.to_thread(
            get_auth_store().change_password, user.id, body.current_password, body.new_password
        )
    except ValueError as ex:
        raise HTTPException(400, str(ex)) from ex
    if not changed:
        raise HTTPException(400, "当前密码错误")
    return {"message": "密码已修改"}


# ── 交互式 CLI ────────────────────────────────────────────────────────────────
async def _cli():
    print(BANNER)
    print("EchoGuide CLI — 输入 quit 退出\n")

    from agents.agent_orchestrator import Request
    from memory.conversation_memory import MsgRole

    state._build_runtime()
    await state._setup_external_mcp()
    orch = state._orchestrator
    mem  = state._memory

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

        print(f"\nEchoGuide [{result.agent_type}]: {result.response}\n")


# ── 同源托管：单端口同时提供前端页面与 API（本地/单进程模式）──────────────────
# 前端 dist 存在时自动启用：/api/* 剥离前缀转给真实路由（与 Vite/nginx 代理
# 语义一致），其余路径走 SPA 回退到 index.html。ECHOGUIDE_SERVE_STATIC=0 关闭。
# 中间件在文件末尾注册（晚于 EchoGuard）：Starlette 后注册者先执行，保证
# /api 前缀在 Guard 看到请求前剥离（Guard 路径白名单基于真实路由）。
_FRONTEND_DIST = pathlib.Path(state._ROOT) / "frontend" / "dist"


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
    import uvicorn

    if "--cli" in sys.argv:
        asyncio.run(_cli())
    else:
        uvicorn.run(
            "api.main:app",
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            reload=os.getenv("APP_ENV") == "development",
        )
