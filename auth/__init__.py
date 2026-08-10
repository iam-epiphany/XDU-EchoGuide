"""EchoGuide lightweight authentication package."""

from .service import (
    SESSION_COOKIE,
    AuthStore,
    AuthUser,
    create_session_token,
    decode_session_token,
    get_auth_store,
    user_from_scope,
)

__all__ = [
    "SESSION_COOKIE",
    "AuthStore",
    "AuthUser",
    "create_session_token",
    "decode_session_token",
    "get_auth_store",
    "user_from_scope",
]
