"""SQLite users, password hashing and signed browser sessions."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

SESSION_COOKIE = "echoguide_session"
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
PBKDF2_ITERATIONS = 310_000


class UsernameExistsError(ValueError):
    """用户名已存在（唯一约束冲突），供 API 层精确映射 HTTP 409。"""


@dataclass(frozen=True)
class AuthUser:
    id: str
    username: str
    role: str
    pwd_ver: int = 0  # 密码版本：改密时递增，旧会话 token 立即失效

    def public(self) -> dict[str, str]:
        return {"id": self.id, "username": self.username, "role": self.role}


def _default_db_path() -> str:
    configured = os.getenv("ECHOGUIDE_DB_PATH")
    if configured:
        return configured
    return str(Path(__file__).resolve().parent.parent / "data" / "echoguide.db")


def _password_hash(password: str, *, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, digest_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(actual.hex(), digest_hex)
    except (TypeError, ValueError):
        return False


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _is_production() -> bool:
    return os.getenv("ECHOGUIDE_ENV", os.getenv("APP_ENV", "development")).lower() == "production"


def _session_secret() -> bytes:
    """会话签名密钥。生产环境缺密钥时 fail-closed，禁止回落到开发默认值。"""
    secret = os.getenv("JWT_SECRET_KEY") or os.getenv("SECRET_KEY")
    if not secret or secret == "echoguide-dev-only-secret":
        if _is_production():
            raise RuntimeError("生产环境必须配置 JWT_SECRET_KEY（或 SECRET_KEY）会话签名密钥")
        secret = secret or "echoguide-dev-only-secret"
        logger.warning("认证会话正在使用开发密钥；部署前请配置 JWT_SECRET_KEY")
    return secret.encode("utf-8")


def create_session_token(user: AuthUser, ttl_seconds: int = 7 * 24 * 3600) -> str:
    payload = {
        "sub": user.id,
        "username": user.username,
        "role": user.role,
        "pwd_ver": user.pwd_ver,
        "exp": int(time.time()) + ttl_seconds,
    }
    encoded = _b64encode(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    signature = _b64encode(hmac.new(_session_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def decode_session_token(token: str) -> Optional[AuthUser]:
    try:
        encoded, signature = token.split(".", 1)
        expected = _b64encode(hmac.new(_session_secret(), encoded.encode("ascii"), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(_b64decode(encoded))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return AuthUser(
            str(payload["sub"]),
            str(payload["username"]),
            str(payload["role"]),
            int(payload.get("pwd_ver", 0)),  # 旧 token 无该字段 → 0，兼容迁移前的会话
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
        # RuntimeError：生产环境缺密钥（_session_secret fail-closed）→ 视为未登录而非 500
        return None


class AuthStore:
    """Small synchronous SQLite store; calls are short and transaction-scoped."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """带显式提交与关闭的连接上下文。

        sqlite3 的 with 只提交/回滚事务、不关连接；这里两者都做，
        否则 INSERT/UPDATE 会在 close() 时被静默回滚。
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    pwd_ver INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 迁移：旧库补 pwd_ver 列（CREATE TABLE IF NOT EXISTS 不会给旧表加列）
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
            if "pwd_ver" not in columns:
                conn.execute("ALTER TABLE users ADD COLUMN pwd_ver INTEGER NOT NULL DEFAULT 0")
            # 管理员只从环境变量播种；未配置则不创建，杜绝硬编码默认口令
            admin_password = os.getenv("ECHOGUIDE_ADMIN_PASSWORD")
            existing = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
            if existing is None:
                if admin_password:
                    conn.execute(
                        "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                        ("admin", _password_hash(admin_password), "admin"),
                    )
                    logger.info("已从 ECHOGUIDE_ADMIN_PASSWORD 初始化管理员账号 admin")
                else:
                    logger.warning(
                        "未配置 ECHOGUIDE_ADMIN_PASSWORD，不创建默认管理员账号；"
                        "需要管理员时请配置该环境变量后重启（仅对新建库生效）"
                    )

    def create_user(self, username: str, password: str, role: str = "user") -> AuthUser:
        username = username.strip()
        if not USERNAME_RE.fullmatch(username):
            raise ValueError("用户名须为 3-32 位字母、数字、点、下划线或连字符")
        if len(password) < 6 or len(password) > 128:
            raise ValueError("密码长度须为 6-128 位")
        if role not in {"user", "admin"}:
            raise ValueError("无效角色")
        try:
            with self._connection() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                    (username, _password_hash(password), role),
                )
                return AuthUser(str(cursor.lastrowid), username, role)
        except sqlite3.IntegrityError as ex:
            raise UsernameExistsError("用户名已存在") from ex

    def authenticate(self, username: str, password: str) -> Optional[AuthUser]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, role, pwd_ver FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        if row is None or not _verify_password(password, row["password_hash"]):
            return None
        return AuthUser(str(row["id"]), row["username"], row["role"], int(row["pwd_ver"]))

    def get_user(self, user_id: str) -> Optional[AuthUser]:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT id, username, role, pwd_ver FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return AuthUser(str(row["id"]), row["username"], row["role"], int(row["pwd_ver"])) if row else None

    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        if len(new_password) < 6 or len(new_password) > 128:
            raise ValueError("新密码长度须为 6-128 位")
        with self._connection() as conn:
            row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
            if row is None or not _verify_password(current_password, row["password_hash"]):
                return False
            conn.execute(
                "UPDATE users SET password_hash = ?, pwd_ver = pwd_ver + 1 WHERE id = ?",
                (_password_hash(new_password), user_id),
            )
        return True


_store: Optional[AuthStore] = None
_store_lock = threading.Lock()


def get_auth_store() -> AuthStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = AuthStore()
    return _store


def _cookie_from_scope(scope: dict[str, Any], name: str) -> str:
    headers = dict(scope.get("headers", []))
    raw = headers.get(b"cookie", b"").decode("latin-1")
    for item in raw.split(";"):
        key, separator, value = item.strip().partition("=")
        if separator and key == name:
            return value
    return ""


def user_from_scope(scope: dict[str, Any]) -> Optional[AuthUser]:
    state = scope.setdefault("state", {})
    cached = state.get("auth_user")
    if isinstance(cached, AuthUser):
        return cached
    token = _cookie_from_scope(scope, SESSION_COOKIE)
    claimed = decode_session_token(token) if token else None
    if claimed is None:
        return None
    current = get_auth_store().get_user(claimed.id)
    if (
        current is None
        or current.username != claimed.username
        or current.role != claimed.role
        or current.pwd_ver != claimed.pwd_ver  # 改密后旧会话立即失效
    ):
        return None
    state["auth_user"] = current
    return current
