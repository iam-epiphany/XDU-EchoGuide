"""
query_ddl —— 考试与 DDL 倒计时查询工具。

返回未来 N 天内（含已过期未完成）的考试/截止任务及剩余天数，
供"我最近有什么考试""DDL 还有几天"类问题使用。
"""
from __future__ import annotations

from typing import Any, Dict

from personal.service import PersonalService


async def query_ddl_handler(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    查询考试与 DDL 安排（带倒计时）。

    params:
      horizon_days: 查询范围天数，默认 30（含今天到期与已过期未完成的）
    """
    service: PersonalService = context.get("personal_service")
    if service is None:
        return {"available": False, "message": "个人数据中心不可用，请稍后重试。"}

    try:
        horizon = min(max(int(params.get("horizon_days", 30)), 1), 180)
    except (TypeError, ValueError):
        horizon = 30

    user_id = (context.get("user_id") or "").strip() or "anonymous"
    items = await service.upcoming(user_id, horizon_days=horizon)
    return {
        "available": True,
        "horizon_days": horizon,
        "items": items,
        "total": len(items),
    }
