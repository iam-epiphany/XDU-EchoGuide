"""
query_schedule —— 个人课程表查询工具。

通过 context["user_id"] 读取该用户的课表（无则提示先导入），
按日期表达式（今天/明天/周X/YYYY-MM-DD）返回当天课程列表。
"""
from __future__ import annotations

from typing import Any, Dict

from personal.service import DateExprError, PersonalService


async def query_schedule_handler(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    查询个人课程表。

    params:
      date: 日期表达式，"今天"/"明天"/"后天"/"周X"/"星期X"/"YYYY-MM-DD"，默认今天
    """
    service: PersonalService = context.get("personal_service")
    user_id = (context.get("user_id") or "").strip() or "anonymous"
    when = str(params.get("date", "今天")).strip() or "今天"

    if user_id == "anonymous":
        return {"available": False, "auth_required": True, "message": "请先登录后使用个人课表。"}

    if service is None:
        return {"available": False, "message": "个人数据中心不可用，请稍后重试。"}

    try:
        result = await service.courses_on(user_id, when)
    except DateExprError as ex:
        return {"available": False, "message": str(ex), "date_expr": when}

    if not result["courses"]:
        result["message"] = (
            f"{result['weekday']}（第 {result['week_num']} 周）没有课程安排"
            if await service.store.count_schedule(user_id) > 0
            else "还没有导入课程表 —— 请先通过「我的课表」上传 .ics 文件或 JSON 课表。"
        )
    result["available"] = True
    return result
