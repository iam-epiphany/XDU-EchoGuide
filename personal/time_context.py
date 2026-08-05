"""
时间上下文 —— 让 Agent 知道「现在」。

在 BaseAgent._build_system_prompt 中统一注入 build_time_context() 的文本，
所有领域 Agent 都能感知：当前日期 / 星期几 / 现在是第几节课 / 当前教学周。
这是"今天有什么课""现在第几节"类问答的前提。

数据来源（公开信息，可用环境变量覆盖）：
  - 西电作息时间表（上午 8:30-12:00；下午秋冬春 14:00 / 夏季 14:30；晚上 19:00 起）
  - 2026-2027 学年校历：秋季学期 2026-09-07 开学，共 19 周
"""
from __future__ import annotations

import os
from datetime import date, datetime, time as dtime, timedelta
from typing import List, Optional, Tuple

# ── 学期配置（环境变量可覆盖）───────────────────────────────────────────────
# 秋季学期开学日（周一）。2026-2027 学年：2026-09-07 开学，19 周。
SEMESTER_START = date.fromisoformat(os.getenv("SEMESTER_START", "2026-09-07"))
SEMESTER_WEEKS = int(os.getenv("SEMESTER_WEEKS", "19"))
# 夏季作息启用月份（5-9 月下午 14:30 上课，其余月份 14:00）
SUMMER_MONTHS = {int(m) for m in os.getenv("SUMMER_SCHEDULE_MONTHS", "5,6,7,8,9").split(",")}

WEEKDAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# ── 作息时间表：[(节次名, 开始, 结束)] ────────────────────────────────────────
# 数据来源：西电教务处公开作息时间（上午 1-4 节、下午 5-8 节、晚上 9-10 节）。
_PERIODS_STANDARD: List[Tuple[str, str, str]] = [
    ("第1节", "08:30", "09:15"),
    ("第2节", "09:20", "10:05"),
    ("第3节", "10:25", "11:10"),
    ("第4节", "11:15", "12:00"),
    ("第5节", "14:00", "14:45"),
    ("第6节", "14:50", "15:35"),
    ("第7节", "15:55", "16:40"),
    ("第8节", "16:45", "17:30"),
    ("第9节", "19:00", "19:45"),
    ("第10节", "19:50", "20:35"),
]
# 夏季作息：下午 5-8 节整体后移 30 分钟，晚上 9-10 节后移 30 分钟
_PERIODS_SUMMER: List[Tuple[str, str, str]] = [
    *[(n, s, e) for n, s, e in _PERIODS_STANDARD[:4]],
    ("第5节", "14:30", "15:15"),
    ("第6节", "15:20", "16:05"),
    ("第7节", "16:25", "17:10"),
    ("第8节", "17:15", "18:00"),
    ("第9节", "19:30", "20:15"),
    ("第10节", "20:20", "21:05"),
]


def periods_for(now: Optional[datetime] = None) -> List[Tuple[str, str, str]]:
    """按日期返回适用作息表（夏季 5-9 月用夏季表，其余用标准表）。"""
    now = now or datetime.now()
    return _PERIODS_SUMMER if now.month in SUMMER_MONTHS else _PERIODS_STANDARD


def current_period(now: Optional[datetime] = None) -> Tuple[str, bool]:
    """
    返回 (当前所处节次描述, 是否在上课)。
    上课 → "第5节（14:00-14:45）"；课间/午休/晚上 → "非上课时间（14:45-15:55 课间）"。
    """
    now = now or datetime.now()
    t = now.time()
    periods = periods_for(now)

    def to_t(s: str) -> dtime:
        return dtime.fromisoformat(s)

    for name, start, end in periods:
        if to_t(start) <= t <= to_t(end):
            return f"{name}（{start}-{end}）", True
    # 课间描述：找最近的下一个节次开始时间
    for name, start, end in periods:
        if t < to_t(start):
            return f"非上课时间（下一节 {name} {start} 开始）", False
    return "非上课时间（今日课程已结束）", False


def week_num(now: Optional[datetime] = None) -> int:
    """当前教学周（开学前返回 0）。"""
    now = now or datetime.now()
    delta = (now.date() - SEMESTER_START).days
    if delta < 0:
        return 0
    return min(delta // 7 + 1, SEMESTER_WEEKS)


def build_time_context(now: Optional[datetime] = None) -> str:
    """
    生成注入 Agent 的时间上下文字块。所有 Agent 的 system prompt 统一拼接。
    """
    now = now or datetime.now()
    period_desc, in_class = current_period(now)
    wn = week_num(now)
    status = "上课中" if in_class else period_desc

    season = "夏季作息" if now.month in SUMMER_MONTHS else "秋冬春季作息"
    return (
        f"[当前时间]\n"
        f"- 日期：{now.strftime('%Y-%m-%d')}（{WEEKDAY_CN[now.weekday()]}）\n"
        f"- 时间：{now.strftime('%H:%M')}，当前状态：{status}\n"
        f"- 教学周：第 {wn} 周（学期开始日 {SEMESTER_START.isoformat()}，共 {SEMESTER_WEEKS} 周，{season}）\n"
        f"- 今天星期几以上面日期为准；回答「今天/明天/周几」类问题时基于此计算。"
    )
