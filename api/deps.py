"""API 层认证依赖（登录用户 / 管理员 / 可观测权限）。"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, HTTPException, Request

from auth.service import AuthUser, user_from_scope


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


def cookie_secure() -> bool:
    # 本项目默认也支持本地 HTTP/Compose；正式 HTTPS 部署显式设为 1。
    return os.getenv("ECHOGUIDE_COOKIE_SECURE", "0") == "1"
