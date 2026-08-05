"""
轻量 ICS 日历解析器 —— 面向教务系统导出的课程表。

教务系统（正方/强智等）通常支持导出 .ics 日历文件，结构为：
  BEGIN:VEVENT
  DTSTART:20260907T083000
  DTEND:20260907T100500
  SUMMARY:高等数学A
  LOCATION:南校区B栋101
  RRULE:FREQ=WEEKLY;BYDAY=MO,WE;UNTIL=20270117T235959Z
  END:VEVENT

本模块只实现课程表所需的最小子集（不依赖 icalendar 库）：
  - 时间值：本地时间 "YYYYMMDDTHHMMSS" / UTC "…Z" / ";TZID=…:…"
  - 重复规则：FREQ=WEEKLY + BYDAY（MO/TU/WE/TH/FR/SA/SU），UNTIL / COUNT / 无限
  - 无 RRULE 的事件按单周课程处理

重复课程展开为「每周几 + 起止时间 + 地点 + 教学周列表」：
  教学周 = 相对学期开始日（周一）的周序号，开学前的日期直接丢弃。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# 周缩写 → Python weekday（周一=0）
WEEKDAY_MAP: Dict[str, int] = {
    "MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6,
}


class ICSError(ValueError):
    """ICS 内容无法解析时抛出。"""


@dataclass
class Course:
    """一条展开后的课程记录（与 SQLite schedule 表一一对应）。"""
    course:      str
    day_of_week: int          # 0=周一 … 6=周日
    start_time:  str          # "08:30"
    end_time:    str          # "10:05"
    location:    str          # 上课地点，可能为空
    weeks:       List[int] = field(default_factory=list)  # 教学周列表，空=所有周

    def to_dict(self) -> Dict:
        return {
            "course": self.course,
            "day_of_week": self.day_of_week,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "location": self.location,
            "weeks": self.weeks,
        }


def _unfold(lines: List[str]) -> List[str]:
    """ICS 规范：以空格/制表符开头的行是上一行的续行，需拼接。"""
    unfolded: List[str] = []
    for line in lines:
        line = line.rstrip("\r")
        if not line:
            continue
        if line[0] in (" ", "\t") and unfolded:
            unfolded[-1] += line[1:]
        else:
            unfolded.append(line)
    return unfolded


def _parse_datetime_value(value: str, utc_to_local: bool = True) -> datetime:
    """
    解析 DTSTART/DTEND 值，返回本地（Asia/Shanghai）时间：
      "20260907T083000"                       本地时间
      "20260907T083000Z"                      UTC，+8 转本地
      "TZID=Asia/Shanghai:20260907T083000"    带时区参数
      "20260907"                               仅日期（按 00:00）

    utc_to_local=False：Z 后缀不转本地（用于 RRULE 的 UNTIL 边界比较，
    避免 "…T235959Z" 这种接近午夜的边界 +8h 后跨到次日）。
    """
    value = value.strip()
    # 去掉 "TZID=…:" 前缀（RFC5545 参数）
    if ":" in value and value.split(":", 1)[0].upper().startswith("TZID"):
        value = value.split(":", 1)[1]
    match = re.fullmatch(r"(\d{8})(?:T(\d{6}))?(Z)?", value)
    if not match:
        raise ICSError(f"无法解析时间值: {value!r}")
    d = datetime.strptime(match.group(1), "%Y%m%d")
    if match.group(2):
        d = d.replace(
            hour=int(match.group(2)[0:2]),
            minute=int(match.group(2)[2:4]),
            second=int(match.group(2)[4:6]),
        )
    if match.group(3) and utc_to_local:  # UTC → 本地（+8）
        d += timedelta(hours=8)
    return d


def _parse_rrule(value: str) -> Dict[str, str]:
    """解析 RRULE 参数（如 FREQ=WEEKLY;BYDAY=MO,WE;UNTIL=20270117T235959Z）。"""
    return dict(
        part.split("=", 1) for part in value.split(";") if "=" in part
    )


def _week_num(d: date, semester_start: date) -> int:
    """教学周序号（开学日为第 1 周的周一）。开学前的日期返回 0。"""
    delta = (d - semester_start).days
    if delta < 0:
        return 0
    return delta // 7 + 1


def _expand_recurrence(
    start_dt: datetime,
    days_of_week: List[int],
    until: Optional[date],
    count: Optional[int],
    semester_start: date,
    semester_weeks: int,
) -> List[int]:
    """
    把每周重复的课程展开为教学周列表。

    从 DTSTART 所在周起，对每个 BYDAY 逐周 +7 展开，
    直到 UNTIL / COUNT / 学期末三者的最小值；开学前与学期后的周丢弃。
    """
    start_day = start_dt.date()
    # DTSTART 所在周的周一（作为展开起点）
    week_monday = start_day - timedelta(days=start_day.weekday())
    weeks: List[int] = []
    # 教学周上限：学期末（以学期周数计）与 UNTIL 取早
    max_date = semester_start + timedelta(days=semester_weeks * 7 - 1)
    if until is not None:
        max_date = min(max_date, until)

    produced = 0
    for i in range(semester_weeks + 1):  # 最多展开到学期末
        d = week_monday + timedelta(days=i * 7)
        if d > max_date:
            break
        for dow in days_of_week:
            day = d + timedelta(days=dow)
            if day < start_day:
                continue  # 早于 DTSTART 的事件日期不产出
            if day > max_date:
                continue
            n = _week_num(day, semester_start)
            if n >= 1 and n not in weeks:
                weeks.append(n)
                produced += 1
                if count is not None and produced >= count:
                    return sorted(weeks)
    return sorted(weeks)


def parse_ics(text: str, semester_start: date, semester_weeks: int = 20) -> List[Course]:
    """
    解析 ICS 课程表文本，返回展开后的课程列表。

    semester_start：学期开始日（周一），用于计算教学周；
    semester_weeks：学期总周数，作为无 UNTIL/COUNT 时展开的上限。
    解析失败的事件会被跳过并记录在返回结构外 —— 调用方如需感知，
    可对返回结果长度与事件数对比。
    """
    events: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None

    for raw_line in _unfold(text.splitlines()):
        if ":" not in raw_line:
            continue
        name, _, value = raw_line.partition(":")
        name = name.upper().split(";")[0].strip()
        value = value.strip()
        if name == "BEGIN" and value.upper() == "VEVENT":
            current = {}
        elif name == "END" and value.upper() == "VEVENT":
            if current is not None:
                events.append(current)
            current = None
        elif current is not None:
            current[name] = value

    courses: List[Course] = []
    skipped = 0
    for ev in events:
        try:
            parsed = _parse_event(ev, semester_start, semester_weeks)
            courses.extend(parsed)
        except ICSError as ex:
            # 单条坏事件跳过（如缺 DTSTART 的占位事件），不拖垮整份课表导入
            skipped += 1
            if skipped <= 5:
                logger.warning(f"跳过无法解析的 ICS 事件（{ev.get('SUMMARY', '?')}）: {ex}")
    if skipped:
        logger.warning(f"ICS 解析完成，共跳过 {skipped} 条异常事件")
    return courses


def _parse_event(
    ev: Dict[str, str],
    semester_start: date,
    semester_weeks: int,
) -> List[Course]:
    """
    解析单个 VEVENT 为若干 Course。

    - 有 BYDAY 时按 BYDAY 展开（一门课周一周三都有 → 拆成两条 Course）；
      day_of_week 取 BYDAY 的星期，而不是 DTSTART 的星期
      （教务系统导出的 DTSTART 常是本周任一天的占位时间）。
    - 无 RRULE 的单次事件：只在 DTSTART 所在周开课。
    """
    if "DTSTART" not in ev:
        raise ICSError("缺少 DTSTART")
    start_dt = _parse_datetime_value(ev["DTSTART"])
    end_dt = _parse_datetime_value(ev.get("DTEND", ev["DTSTART"]))

    days_of_week: List[int] = [start_dt.weekday()]
    until: Optional[date] = None
    count: Optional[int] = None
    if "RRULE" in ev:
        rule = _parse_rrule(ev["RRULE"])
        if rule.get("FREQ", "WEEKLY").upper() != "WEEKLY":
            raise ICSError("仅支持 FREQ=WEEKLY 的重复课程")
        if rule.get("BYDAY"):
            try:
                days_of_week = [WEEKDAY_MAP[d.strip().upper()] for d in rule["BYDAY"].split(",")]
            except KeyError:
                raise ICSError(f"未知 BYDAY: {rule['BYDAY']}")
        if rule.get("UNTIL"):
            until = _parse_datetime_value(rule["UNTIL"], utc_to_local=False).date()
        if rule.get("COUNT"):
            count = int(rule["COUNT"])

    courses: List[Course] = []
    for dow in days_of_week:
        if "RRULE" in ev:
            # 每周重复：展开到 UNTIL / COUNT / 学期末
            weeks = _expand_recurrence(
                start_dt, [dow], until, count,
                semester_start, semester_weeks,
            )
        else:
            # 单次事件：只在 DTSTART 所在周开课
            n = _week_num(start_dt.date(), semester_start)
            weeks = [n] if n >= 1 else []
        # 该课程在所有周都不开课 → 丢弃（可能是上学期遗留事件）
        if not weeks:
            continue
        courses.append(Course(
            course=(ev.get("SUMMARY") or "未命名课程").strip(),
            day_of_week=dow,
            start_time=start_dt.strftime("%H:%M"),
            end_time=end_dt.strftime("%H:%M"),
            location=(ev.get("LOCATION") or "").strip(),
            weeks=weeks,
        ))
    return courses
