"""Lightweight authentication and authorization tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import auth.service as auth_service
from auth.service import AuthStore, create_session_token, decode_session_token


def _store(tmp_path) -> AuthStore:
    """管理员密码只从 ECHOGUIDE_ADMIN_PASSWORD 播种，测试统一显式设置。"""
    return AuthStore(str(tmp_path / "auth.db"))


def test_admin_seeded_from_env_password(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHOGUIDE_ADMIN_PASSWORD", "admin-secret-pass")
    store = _store(tmp_path)
    admin = store.authenticate("admin", "admin-secret-pass")
    assert admin is not None
    assert admin.role == "admin"

    with store._connect() as conn:
        encoded = conn.execute("SELECT password_hash FROM users WHERE username='admin'").fetchone()[0]
    assert encoded.startswith("pbkdf2_sha256$")
    assert "admin-secret-pass" not in encoded


def test_admin_not_seeded_without_env_password(tmp_path, monkeypatch):
    monkeypatch.delenv("ECHOGUIDE_ADMIN_PASSWORD", raising=False)
    store = _store(tmp_path)
    assert store.authenticate("admin", "anything") is None


def test_user_creation_authentication_and_password_change(tmp_path):
    store = _store(tmp_path)
    user = store.create_user("student01", "abcdef")
    assert user.role == "user"
    assert store.authenticate("student01", "wrong") is None
    assert store.authenticate("student01", "abcdef") == user
    assert store.change_password(user.id, "abcdef", "new-password") is True
    assert store.authenticate("student01", "abcdef") is None
    # 改密后 authenticate 返回的是 pwd_ver=1 的新版本用户（不是创建时的 pwd_ver=0）
    assert store.authenticate("student01", "new-password") == auth_service.AuthUser(
        user.id, user.username, user.role, pwd_ver=1,
    )


def test_signed_session_rejects_tampering(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ECHOGUIDE_ADMIN_PASSWORD", "admin-secret-pass")
    user = _store(tmp_path).authenticate("admin", "admin-secret-pass")
    token = create_session_token(user)
    assert decode_session_token(token) == user
    assert decode_session_token(token + "tampered") is None


def test_password_change_revokes_old_session(tmp_path, monkeypatch):
    """改密后旧 token 立即失效（pwd_ver 递增 + user_from_scope 比对）。"""
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    store = _store(tmp_path)
    # user_from_scope 走全局 store 单例：本测试显式指到临时库
    monkeypatch.setattr(auth_service, "_store", store)
    user = store.create_user("student01", "abcdef")
    token = create_session_token(user)
    assert decode_session_token(token) == user
    scope = {"headers": [(b"cookie", f"{auth_service.SESSION_COOKIE}={token}".encode("latin-1"))]}
    assert auth_service.user_from_scope(scope) == user
    assert store.change_password(user.id, "abcdef", "new-password") is True
    # token 内的 pwd_ver=0 已与库中 pwd_ver=1 不一致 → 会话吊销
    # 注意：user_from_scope 会把结果缓存进 scope["state"]，改密后须用新 scope 重新校验
    fresh_scope = {"headers": scope["headers"]}
    assert auth_service.user_from_scope(fresh_scope) is None


def test_old_token_without_pwd_ver_still_valid_when_unchanged(tmp_path, monkeypatch):
    """兼容：无 pwd_ver 字段的旧 token（pwd_ver=0）在密码未变时仍有效。"""
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    store = _store(tmp_path)
    user = store.create_user("student01", "abcdef")
    # 手工构造无 pwd_ver 字段的旧格式 token
    import base64
    import hashlib
    import hmac
    import json
    import time as time_mod

    payload = {"sub": user.id, "username": user.username, "role": user.role,
               "exp": int(time_mod.time()) + 3600}
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).decode("ascii").rstrip("=")
    signature = base64.urlsafe_b64encode(hmac.new(
        b"test-secret", encoded.encode("ascii"), hashlib.sha256
    ).digest()).decode("ascii").rstrip("=")
    legacy_token = f"{encoded}.{signature}"
    assert decode_session_token(legacy_token) == user


def test_production_secret_missing_fails_closed(tmp_path, monkeypatch):
    """生产环境缺少会话密钥必须拒绝签发（fail-closed），不能回落开发密钥。"""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("ECHOGUIDE_ADMIN_PASSWORD", "admin-secret-pass")
    user = _store(tmp_path).authenticate("admin", "admin-secret-pass")
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        create_session_token(user)
    # 解码侧同样视为未登录而非 500
    assert decode_session_token("anything.tampered") is None


def test_auth_routes_and_personal_endpoint_require_login(tmp_path, monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
    monkeypatch.setenv("ECHOGUIDE_ADMIN_PASSWORD", "admin-secret-pass")
    monkeypatch.setattr(auth_service, "_store", AuthStore(str(tmp_path / "auth.db")))

    import api.main as main

    client = TestClient(main.app)
    assert client.get("/auth/me").json()["authenticated"] is False
    assert client.get("/personal/schedule").status_code == 401

    login = client.post("/auth/login", json={"username": "admin", "password": "admin-secret-pass"})
    assert login.status_code == 200
    assert login.json()["user"]["role"] == "admin"
    assert client.get("/auth/me").json()["user"]["username"] == "admin"

    logout = client.post("/auth/logout")
    assert logout.status_code == 200
    assert client.get("/auth/me").json()["authenticated"] is False
