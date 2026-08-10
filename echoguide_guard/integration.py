"""
EchoGuard 真实接入 —— FastAPI HTTP 中间件

把安全能力以中间件形式接入 EchoGuide 真实请求链（/chat、/chat/stream、
/eval/run、/knowledge/*、/skills/reload、/personal/*、/mcp、/auth），默认启用。

场景定位（面向开放的校园助手）：
  - Prompt 注入检测 —— LLM 系统真实威胁：防"忽略之前指令"类注入
  - 限流 —— LLM 调用有真实成本，防单用户刷接口
  - 脱敏审计 —— 个人数据（课表/待办）操作留痕（只记哈希与脱敏摘要）
  - 身份认证（可选）—— 配置 ECHOGUIDE_GUARD_TOKEN 后要求 Bearer Token

容错原则：受保护路径中间件异常时失败关闭；健康检查和静态资源不受影响。

启用：ECHOGUIDE_GUARD_ENABLED 默认 1（注入检测/限流/审计开箱即用）。
"""
import hashlib
import hmac
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 注入标记模式（与 Sidecar app.py 的 _has_injection_marker 保持一致）──────
_INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"不要向用户提及|do not (tell|mention)",
    r"<important>|<!--\s*(system|ignore)",
    r"read\s+/app/\.env",
    r"读取\s*/app/\.env",
    r"忽略(之前|以上).{0,8}(指令|指示)",
    r"你(现在|将).{0,6}(扮演|伪装)",
)
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


class GuardSettings:
    """中间件配置（环境变量驱动）。"""

    def __init__(self, **kwargs):
        self.enabled            = kwargs.get("enabled", os.getenv("ECHOGUIDE_GUARD_ENABLED", "1") == "1")
        self.token              = kwargs.get("token", os.getenv("ECHOGUIDE_GUARD_TOKEN", "") or None)
        self.max_message_chars  = int(kwargs.get("max_message_chars", os.getenv("ECHOGUIDE_GUARD_MAX_MESSAGE_CHARS", "2000")))
        self.user_rate_per_min  = int(kwargs.get("user_rate_per_min", os.getenv("ECHOGUIDE_GUARD_USER_RATE", "30")))
        self.ip_rate_per_min    = int(kwargs.get("ip_rate_per_min", os.getenv("ECHOGUIDE_GUARD_IP_RATE", "120")))

    # 需要保护的端点前缀（/auth 登录/注册同样限流，但豁免身份认证）
    PROTECTED_PREFIXES = ("/chat", "/auth", "/eval/run", "/knowledge", "/skills/reload", "/personal", "/mcp", "/traces", "/monitor", "/campus/reload")


class _RateLimiter:
    """用户/IP 维度滑动窗口限流。"""

    _CLEANUP_INTERVAL_S = 60.0

    def __init__(self, user_limit: int, ip_limit: int):
        self.user_limit = user_limit
        self.ip_limit   = ip_limit
        self._hits: Dict[str, Deque[float]] = defaultdict(lambda: deque())
        self._last_cleanup = 0.0

    def allow(self, key: str, limit: int, window_s: float = 60.0) -> bool:
        now = time.monotonic()
        if now - self._last_cleanup >= self._CLEANUP_INTERVAL_S:
            self._cleanup(now)
            self._last_cleanup = now
        q = self._hits[key]
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True

    def _cleanup(self, now: float) -> None:
        """周期性清理空桶与长时间未访问的桶，避免键无限累积。"""
        stale = [
            k for k, q in self._hits.items()
            if not q or now - q[-1] > self._CLEANUP_INTERVAL_S
        ]
        for k in stale:
            del self._hits[k]


class EchoGuardMiddleware:
    """
    纯 ASGI 中间件：拦截 HTTP 请求 → 认证/注入检测/限流/输入约束 →
    通过后放行并输出脱敏审计日志。
    """

    def __init__(self, app: Any, settings: Optional[GuardSettings] = None):
        self.app = app
        self.settings = settings or GuardSettings()
        self._limiter = _RateLimiter(self.settings.user_rate_per_min, self.settings.ip_rate_per_min)

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http" or not self.settings.enabled:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        protected = path.startswith(self.settings.PROTECTED_PREFIXES)
        try:
            await self._guard(scope, receive, send, protected)
        except Exception as ex:
            logger.exception(f"[EchoGuard] 中间件异常: {ex}")
            if protected:
                await self._reject(send, 503, "安全检查暂不可用，请稍后重试")
            else:
                await self.app(scope, receive, send)

    async def _guard(self, scope: Dict[str, Any], receive: Any, send: Any, protected: bool) -> None:
        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # 只保护敏感端点；静态/健康检查放行
        if not protected:
            await self.app(scope, receive, send)
            return

        # /auth/*（登录/注册）是未登录状态的必经入口，豁免身份认证，
        # 但照常执行限流、注入检测与长度约束（防暴破与注入投递）。
        needs_auth = not path.startswith("/auth")

        # 1. 解析可信登录身份。配置服务 token 时，浏览器会话或 Bearer Token
        # 任一有效即可；服务 token 无需进入前端代码。
        from auth.service import user_from_scope

        auth_user = user_from_scope(scope) if needs_auth else None
        bearer_ok = False
        auth_header = b""
        if self.settings.token:
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"")
            expected = f"Bearer {self.settings.token}".encode()
            bearer_ok = hmac.compare_digest(auth_header, expected)
            if needs_auth and auth_user is None and not bearer_ok:
                await self._reject(send, 401, "未授权：请先登录或提供有效访问令牌")
                return

        # 2. 仅为带请求体的方法读取并缓存，再将同一 body 重放给下游。
        # GET/DELETE 也接受服务级 token 和限流，但不因无 body 消耗 receive。
        body = b""
        if method in {"POST", "PUT", "PATCH"}:
            body = await self._read_body(receive)
            if body is None:
                return

        client = scope.get("client", ("", 0))[0] or "unknown"

        # 3. 限流：登录用户按 user 桶；有效服务 token 独立 token 桶；
        # 匿名调用按 IP 隔离（避免全体匿名共享一个桶被集体限流）。
        if auth_user is not None:
            rate_key = f"user:{auth_user.id}"
        elif bearer_ok and self.settings.token:
            rate_key = f"token:{hashlib.sha256(auth_header).hexdigest()[:32]}"
        else:
            rate_key = f"anon:{client}"
        if not self._limiter.allow(rate_key, self.settings.user_rate_per_min):
            await self._reject(send, 429, "请求过于频繁，请稍后再试")
            return
        if not self._limiter.allow(f"ip:{client}", self.settings.ip_rate_per_min):
            await self._reject(send, 429, "请求过于频繁，请稍后再试")
            return

        # 4. 注入检测 + 输入约束（递归扫描所有字符串字段，覆盖
        # /chat 的 message、/mcp 的 params、/knowledge 的 documents 等）
        texts = self._collect_strings(body)
        if any(len(t) > self.settings.max_message_chars for t in texts):
            await self._reject(send, 413, f"请求内容过长：上限 {self.settings.max_message_chars} 字")
            return
        if any(_INJECTION_RE.search(t or "") for t in texts):
            await self._reject(send, 403, "检测到疑似注入内容，请求已拦截")
            return

        # 5. 放行：重放请求体给下游，并输出脱敏审计日志
        await self._audit(path, rate_key, texts)
        if method in {"POST", "PUT", "PATCH"}:
            await self.app(scope, self._replay_body(body, receive), send)
        else:
            await self.app(scope, receive, send)

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    @staticmethod
    async def _read_body(receive: Any) -> Optional[bytes]:
        """读取完整请求体（支持分片）。"""
        chunks: List[bytes] = []
        while True:
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(message.get("body", b""))
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return None
        return b"".join(chunks)

    @staticmethod
    def _replay_body(body: bytes, original_receive: Any) -> Any:
        """
        构造可重放请求体的 receive。

        第一次调用返回缓存的请求体，之后委托给原始 receive —— 而不是伪造
        http.disconnect。原因：Starlette 的 StreamingResponse 会并行监听
        disconnect（listen_for_disconnect），伪造的 disconnect 会被误判为
        "客户端断开"而取消整个流式响应（SSE 只出 hello 即被终止的真实事故）。
        """
        state = {"sent": False}

        async def receive() -> Dict[str, Any]:
            if not state["sent"]:
                state["sent"] = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await original_receive()

        return receive

    @staticmethod
    def _collect_strings(body: bytes, max_depth: int = 5, max_items: int = 200) -> List[str]:
        """递归收集 JSON body 中的所有字符串字段（供注入检测 / 长度约束）。

        覆盖 /chat 的 message、/mcp 的 params.arguments、/knowledge/add 的
        documents[].content、/eval/run 的用例文本等，避免只查顶层 message
        造成覆盖盲区。限制深度与条数，防止恶意嵌套放大收集成本。
        """
        try:
            data = json.loads(body.decode("utf-8", errors="ignore"))
        except Exception:
            return []
        texts: List[str] = []

        def walk(node: Any, depth: int) -> None:
            if len(texts) >= max_items or depth > max_depth:
                return
            if isinstance(node, str):
                texts.append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    walk(item, depth + 1)

        walk(data, 0)
        return texts

    async def _reject(self, send: Any, status: int, message: str) -> None:
        payload = json.dumps({"detail": message}, ensure_ascii=False).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json; charset=utf-8")],
        })
        await send({"type": "http.response.body", "body": payload})
        logger.warning(f"[EchoGuard] 拦截 {status}: {message}")

    async def _audit(self, path: str, subject: str, texts: List[str]) -> None:
        """脱敏审计日志：不落原始敏感内容，只记录哈希与脱敏摘要。"""
        from echoguide_guard.redaction import redact_text

        sample = " ".join(texts)[:120]
        digest = hashlib.sha256(sample.encode("utf-8")).hexdigest()[:16]
        summary = redact_text(sample)
        logger.info(
            f"[EchoGuard] 放行 path={path} subject={subject} sha256={digest} msg={summary!r}"
        )
