"""
待办 / DDL / 考试管理工具 —— query_todo / add_todo / complete_todo。

对话场景：
  - "我有什么待办？"            → query_todo
  - "记一下周三前交实验报告"     → add_todo（kind=todo，due_at 解析由 LLM 给出）
  - "下周一考高数"              → add_todo（kind=exam）
  - "把XX标记为完成"            → complete_todo
"""
from __future__ import annotations

from typing import Any, Dict, List

from personal.service import PersonalService

KINDS = {"todo", "ddl", "exam"}


def _user_id(context: Any) -> str:
    return (context.get("user_id") or "").strip() or "anonymous"


async def query_todo_handler(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    查询待办清单。

    params:
      status: "open"（未完成，默认）/ "done" / "all"
      kinds:  过滤类型列表，如 ["ddl", "exam"]；不传返回全部类型
    """
    service: PersonalService = context.get("personal_service")
    if service is None:
        return {"available": False, "message": "个人数据中心不可用，请稍后重试。"}

    status = str(params.get("status", "open")).strip() or "open"
    raw_kinds = params.get("kinds") or []
    kinds = [k for k in raw_kinds if k in KINDS] or None

    todos = await service.list_todos(_user_id(context), status=status, kinds=kinds)
    return {"available": True, "status": status, "todos": todos, "total": len(todos)}


async def add_todo_handler(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    新增待办 / DDL / 考试安排。

    params:
      content: 事项内容（必填）
      kind:    "todo"（默认）/ "ddl"（截止任务）/ "exam"（考试）
      due_at:  截止/考试时间，"YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM"，可选
    """
    service: PersonalService = context.get("personal_service")
    if service is None:
        return {"available": False, "message": "个人数据中心不可用，请稍后重试。"}

    content = str(params.get("content", "")).strip()
    if not content:
        return {"available": False, "message": "缺少事项内容（content）。"}

    kind = str(params.get("kind", "todo")).strip() or "todo"
    if kind not in KINDS:
        kind = "todo"
    due_at = str(params.get("due_at", "")).strip() or None

    todo = await service.add_todo(_user_id(context), content, kind=kind, due_at=due_at)
    return {"available": True, "message": "已记录", "todo": todo}


async def complete_todo_handler(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    完成 / 恢复待办。

    params:
      id:   待办 id（必填）
      done:  true=标记完成（默认），false=恢复未完成
    """
    service: PersonalService = context.get("personal_service")
    if service is None:
        return {"available": False, "message": "个人数据中心不可用，请稍后重试。"}

    try:
        todo_id = int(params.get("id"))
    except (TypeError, ValueError):
        return {"available": False, "message": "缺少有效的待办 id。"}

    done = bool(params.get("done", True))
    todo = await service.complete_todo(_user_id(context), todo_id, done=done)
    if todo is None:
        return {"available": False, "message": f"待办 {todo_id} 不存在或不属于当前用户。"}
    return {"available": True, "message": "已标记完成" if done else "已恢复未完成", "todo": todo}
