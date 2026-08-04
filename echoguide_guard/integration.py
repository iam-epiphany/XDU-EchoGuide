"""
EchoGuard 真实接入 —— FastAPI HTTP 中间件

把安全 Sidecar（echoguide_guard）的能力以中间件形式接入 EchoGuide 真实请求链
（/chat、/chat/stream、/eval/run、/knowledge/*、/skills/reload），修复
"Sidecar 与真实系统零引用"的架构缺口。

防护能力（按层）：
  1. 身份认证  —— ECHOGUIDE_GUARD_TOKEN 配置后，受保护端点要求 Bearer Token
  2. Prompt 注入检测 —— 复用 Sidecar 的注入标记正则（规则层第一道防线；
     RAG 间接注入由 prompt 中的安全边界 + 输出审计兜底）
  3. 敏感数据脱敏 —— 审计日志中的 API Key / 身份证 / 手机号等自动打码
  4. 限流 —— 用户维度滑动窗口（默认 30 次/分钟/用户）
  5. 输入约束 —— 消息长度上限（默认 2000 字），防止 token 爆炸

启用：环境变量 ECHOGUIDE_GUARD_ENABLED=1（默认关闭，便于本地开发）。
"""
import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

REDACTED = "[REDACTED]"

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
        self.enabled            = kwargs.get("enabled", os.getenv("ECHOGUIDE_GUARD_ENABLED", "0") == "1")
        self.token              = kwargs.get("token", os.getenv("ECHOGUIDE_GUARD_TOKEN", "") or None)
        self.max_message_chars  = int(kwargs.get("max_message_chars", os.getenv("ECHOGUIDE_GUARD_MAX_MESSAGE_CHARS", "2000")))
        self.user_rate_per_min  = int(kwargs.get("user_rate_per_min", os.getenv("ECHOGUIDE_GUARD_USER_RATE", "30")))
        self.ip_rate_per_min    = int(kwargs.get("ip_rate_per_min", os.getenv("ECHOGUIDE_GUARD_IP_RATE", "120")))

    # 需要保护的端点前缀
    PROTECTED_PREFIXES = ("/chat", "/eval/run", "/knowledge/", "/skills/reload")


class _RateLimiter:
    """用户/IP 维度滑动窗口限流。"""

    def __init__(self, user_limit: int, ip_limit: int):
        self.user_limit = user_limit
        self.ip_limit   = ip_limit
        self._hits: Dict[str, Deque[float]] = defaultdict(lambda: deque())

    def allow(self, key: str, limit: int, window_s: float = 60.0) -> bool:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= limit:
            return False
        q.append(now)
        return True


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
        method = scope.get("method", "GET")

        # 只保护敏感端点；静态/健康检查放行
        if method != "POST" or not path.startswith(self.settings.PROTECTED_PREFIXES):
            await self.app(scope, receive, send)
            return

        # 1. 身份认证（配置 token 后生效）
        if self.settings.token:
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"")
            expected = f"Bearer {self.settings.token}".encode()
            if auth != expected:
                await self._reject(send, 401, "未授权：缺少或错误的访问令牌")
                return

        # 2. 读取并缓存请求体（供注入检测 / 限流 / 输入约束）
        body = await self._read_body(receive)
        if body is None:
            await self.app(scope, receive, send)
            return

        user_id = self._extract_user_id(body)
        client = scope.get("client", ("", 0))[0] or "unknown"

        # 3. 限流（用户维度优先，IP 维度兜底）
        if not self._limiter.allow(f"user:{user_id}", self.settings.user_rate_per_min):
            await self._reject(send, 429, "请求过于频繁，请稍后再试")
            return
        if not self._limiter.allow(f"ip:{client}", self.settings.ip_rate_per_min):
            await self._reject(send, 429, "请求过于频繁，请稍后再试")
            return

        # 4. 注入检测 + 输入约束
        message = self._extract_message(body)
        if len(message) > self.settings.max_message_chars:
            await self._reject(send, 413, f"消息过长：上限 {self.settings.max_message_chars} 字")
            return
        if _INJECTION_RE.search(message or ""):
            await self._reject(send, 403, "检测到疑似注入内容，请求已拦截")
            return

        # 5. 放行：重放请求体给下游，并输出脱敏审计日志
        await self._audit(path, user_id, message)
        buffered = self._replay_body(body)
        await self.app(scope, buffered, send)

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
    def _replay_body(body: bytes) -> Any:
        """构造可重放请求体的 receive。"""
        async def receive() -> Dict[str, Any]:
            return {"type": "http.request", "body": body, "more_body": False}
        return receive

    @staticmethod
    def _extract_message(body: bytes) -> str:
        try:
            data = json.loads(body.decode("utf-8", errors="ignore"))
            return str(data.get("message", "")) if isinstance(data, dict) else ""
        except Exception:
            return ""

    @staticmethod
    def _extract_user_id(body: bytes) -> str:
        try:
            data = json.loads(body.decode("utf-8", errors="ignore"))
            uid = data.get("user_id", "anonymous") if isinstance(data, dict) else "anonymous"
            return str(uid)[:64]
        except Exception:
            return "anonymous"

    async def _reject(self, send: Any, status: int, message: str) -> None:
        payload = json.dumps({"detail": message}, ensure_ascii=False).encode()
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json; charset=utf-8")],
        })
        await send({"type": "http.response.body", "body": payload})
        logger.warning(f"[EchoGuard] 拦截 {status}: {message}")

    async def _audit(self, path: str, user_id: str, message: str) -> None:
        """脱敏审计日志：不落原始敏感内容，只记录哈希与脱敏摘要。"""
        from echoguide_guard.redaction import redact_text

        digest = hashlib.sha256((message or "").encode("utf-8")).hexdigest()[:16]
        summary = redact_text((message or "")[:120])
        logger.info(
            f"[EchoGuard] 放行 path={path} user={user_id} sha256={digest} msg={summary!r}"
        )
