"""
个人数据查询服务 —— 对话工具与 REST API 共用的业务逻辑层。

提供：
  - 日期表达式解析："今天/明天/周X/YYYY-MM-DD/X月X日" → date
  - 课程查询：某天的课程列表（含教学周过滤）、地点反查
  - 待办/DDL/考试：增查完成删、倒计时计算、未来 N 天安排汇总
  - 教学周列表的压缩（[1,2,3,5] → "1-3,5"）与展开

说明：周X 统一指本周的周X（"周五"在今天周三时指本周五，在今天周六时指本周五<已过>），
语义简单一致，避免跨周歧义。
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

from personal.store import PersonalStore
from personal.time_context import SEMESTER_START, SEMESTER_WEEKS, WEEKDAY_CN

_WEEKDAY_NUM = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}


class DateExprError(ValueError):
    """日期表达式无法解析。"""


# ── 教学周工具 ────────────────────────────────────────────────────────────────

def compress_weeks(weeks: List[int]) -> str:
    """教学周压缩：[1,2,3,5,6,7,9] → "1-3,5-7,9"；空列表 → ""（所有周）。"""
    if not weeks:
        return ""
    sorted_weeks = sorted(set(weeks))
    parts: List[str] = []
    start = prev = sorted_weeks[0]
    for w in sorted_weeks[1:] + [None]:
        if w is None or w != prev + 1:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = w if w is not None else prev
        else:
            prev = w
    return ",".join(parts)


def expand_weeks(expr: str) -> List[int]:
    """教学周展开："1-3,5" → [1,2,3,5]；"" / "all" → []（所有周）。"""
    expr = (expr or "").strip()
    if not expr or expr.lower() == "all":
        return []
    weeks: List[int] = []
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            weeks.extend(range(int(a), int(b) + 1))
        else:
            weeks.append(int(part))
    return sorted(set(weeks))


def week_in(w: int, weeks_expr: str) -> bool:
    """教学周 w 是否在 weeks 字段内；空（""）表示所有周。"""
    expanded = expand_weeks(weeks_expr)
    return True if not expanded else w in expanded


# ── 日期表达式解析 ────────────────────────────────────────────────────────────

def parse_date_expr(expr: Optional[str], today: Optional[date] = None) -> date:
    """
    解析日期表达式：
      "今天/今日"、"明天/明日"、"后天"、"昨天/昨日"、"前天"
      "周X/星期X/礼拜X"（本周该日）
      "YYYY-MM-DD"、"X月X日"（本年）、"这周/本周"（本周一）、"下周"（下周一）
    无法解析时抛 DateExprError。
    """
    today = today or date.today()
    expr = (expr or "今天").strip()
    if expr in ("今天", "今日"):
        return today
    if expr in ("明天", "明日"):
        return today + timedelta(days=1)
    if expr == "后天":
        return today + timedelta(days=2)
    if expr in ("昨天", "昨日"):
        return today - timedelta(days=1)
    if expr == "前天":
        return today - timedelta(days=2)
    if expr in ("这周", "本周"):
        return today - timedelta(days=today.weekday())
    if expr == "下周":
        return today - timedelta(days=today.weekday()) + timedelta(days=7)
    m = re.fullmatch(r"(?:周|星期|礼拜)([一二三四五六日天])", expr)
    if m:
        return today - timedelta(days=today.weekday() - _WEEKDAY_NUM[m.group(1)])
    try:
        return date.fromisoformat(expr)
    except ValueError:
        pass
    m = re.fullmatch(r"(\d{1,2})月(\d{1,2})日?", expr)
    if m:
        try:
            return date(today.year, int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass
    raise DateExprError(f"无法解析日期表达式: {expr!r}")


# ── 查询服务 ──────────────────────────────────────────────────────────────────

class PersonalService:
    """个人数据中心查询服务（对话工具与 REST API 共用）。"""

    def __init__(self, store: PersonalStore):
        self.store = store

    # ── 课程表 ────────────────────────────────────────────────────────────────

    async def import_courses(self, user_id: str, courses: List[Dict[str, Any]]) -> int:
        """导入课表（整表替换）：内部把 weeks 列表压缩为字符串存储。"""
        rows = []
        for c in courses:
            weeks = c.get("weeks", [])
            if isinstance(weeks, str):
                weeks_expr = weeks
            else:
                weeks_expr = compress_weeks([int(w) for w in weeks])
            rows.append({
                "course": c["course"],
                "day_of_week": int(c["day_of_week"]),
                "start_time": c["start_time"],
                "end_time": c["end_time"],
                "location": c.get("location", ""),
                "weeks": weeks_expr,
            })
        return await self.store.replace_schedule(user_id, rows)

    async def courses_on(
        self,
        user_id: str,
        when: Optional[str] = None,
        today: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        某天的课程。when 支持"今天/明天/周X/日期"。
        返回 {date, weekday, week_num, courses: [...]}；未导入课表时 courses 为空。
        """
        target = parse_date_expr(when, today)
        week = (target - SEMESTER_START).days // 7 + 1
        if week < 0:
            week = 0  # 开学前（假期）与 weekly_overview / time_context 保持一致，不返回负数周
        rows = await self.store.get_schedule(user_id)
        courses = []
        for r in rows:
            if r["day_of_week"] == target.weekday() and week_in(week, r["weeks"]):
                courses.append({
                    "course": r["course"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "location": r["location"],
                    "week_num": week,
                })
        courses.sort(key=lambda c: c["start_time"])
        return {
            "date": target.isoformat(),
            "weekday": WEEKDAY_CN[target.weekday()],
            "week_num": week,
            "courses": courses,
        }

    async def reverse_lookup(
        self,
        user_id: str,
        time_str: str,
        today: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """地点反查："下午 3 点在 B 栋 508 上的是什么课"。"""
        target = today or date.today()
        week = (target - SEMESTER_START).days // 7 + 1
        rows = await self.store.get_schedule(user_id)
        result = []
        for r in rows:
            if r["day_of_week"] != target.weekday() or not week_in(week, r["weeks"]):
                continue
            if r["start_time"] <= time_str <= r["end_time"]:
                result.append({
                    "course": r["course"],
                    "start_time": r["start_time"],
                    "end_time": r["end_time"],
                    "location": r["location"],
                    "week_num": week,
                })
        return result

    async def weekly_overview(self, user_id: str, today: Optional[date] = None) -> Dict[str, Any]:
        """本周课程总览（周一~周日），供前端课表面板展示。"""
        today = today or date.today()
        monday = today - timedelta(days=today.weekday())
        week = (monday - SEMESTER_START).days // 7 + 1
        in_semester = week >= 1
        if not in_semester:
            week = 0  # 开学前（假期）：周数显示为 0，由前端提示"假期/未开学"
        rows = await self.store.get_schedule(user_id)
        week_courses = [
            r for r in rows
            if week_in(week, r["weeks"])
        ]
        week_courses.sort(key=lambda r: (r["day_of_week"], r["start_time"]))
        return {
            "monday": monday.isoformat(),
            "week_num": week,
            "in_semester": in_semester,
            "courses": week_courses,
        }

    # ── 待办 / DDL / 考试 ─────────────────────────────────────────────────────

    async def list_todos(
        self,
        user_id: str,
        status: str = "open",
        kinds: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return await self.store.list_todos(user_id, status=status, kinds=kinds)

    async def add_todo(
        self,
        user_id: str,
        content: str,
        kind: str = "todo",
        due_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.store.add_todo(user_id, content, kind, due_at)

    async def complete_todo(self, user_id: str, todo_id: int, done: bool = True) -> Optional[Dict[str, Any]]:
        ok = await self.store.set_todo_done(user_id, todo_id, done)
        return await self.store.get_todo(user_id, todo_id) if ok else None

    async def delete_todo(self, user_id: str, todo_id: int) -> bool:
        return await self.store.delete_todo(user_id, todo_id)

    async def upcoming(
        self,
        user_id: str,
        horizon_days: int = 30,
        today: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        未来 horizon_days 内的 DDL/考试（含已过期未完成的），带倒计时。
        按截止时间排序；到期/过期项置顶。days_left 负数表示已过期。
        """
        today = today or date.today()
        deadline = today + timedelta(days=horizon_days)
        rows = await self.store.list_todos(
            user_id, status="open", kinds=["ddl", "exam"], due_before=deadline,
        )
        result = []
        for r in rows:
            due = r["due_at"]
            if not due:
                continue
            due_date = date.fromisoformat(due[:10])
            result.append({
                "id": r["id"],
                "content": r["content"],
                "kind": r["kind"],
                "due_at": due,
                "days_left": (due_date - today).days,
                "status": "已过期" if due_date < today else ("今天" if due_date == today else f"还剩{(due_date - today).days}天"),
            })
        result.sort(key=lambda x: (x["days_left"], x["due_at"]))
        return result

    async def overview(self, user_id: str, today: Optional[date] = None) -> Dict[str, Any]:
        """
        当日汇总（对话"我今天的安排"与前端共用）：
        课程 + 未完成待办 + 未来 7 天 DDL/考试倒计时。
        """
        today = today or date.today()
        schedule = await self.courses_on(user_id, "今天", today)
        todos = await self.list_todos(user_id, status="open", kinds=["todo"])
        ddls = await self.upcoming(user_id, horizon_days=7, today=today)
        return {
            "date": schedule["date"],
            "weekday": schedule["weekday"],
            "week_num": schedule["week_num"],
            "courses": schedule["courses"],
            "todos": todos,
            "upcoming": ddls,
            "has_schedule": await self.store.count_schedule(user_id) > 0,
        }

    # ── 提醒（对话内应答）─────────────────────────────────────────────────────

    async def reminders(self, user_id: str, today: Optional[date] = None) -> Dict[str, Any]:
        """
        对话式提醒汇总："我今天/最近有什么安排"。
        包含：今天剩余课程、明天课程、未来 7 天 DDL/考试（含今天到期与已过期）。
        """
        today = today or date.today()
        today_sched = await self.courses_on(user_id, "今天", today)
        tomorrow_sched = await self.courses_on(user_id, "明天", today)
        ddls = await self.upcoming(user_id, horizon_days=7, today=today)
        return {
            "today": today_sched,
            "tomorrow": tomorrow_sched,
            "upcoming": ddls,
            "has_schedule": await self.store.count_schedule(user_id) > 0,
        }
