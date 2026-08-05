"""个人数据中心测试：ICS 解析、SQLite 存储、用户隔离、日期解析、课程/DDL 查询。

全部使用临时 SQLite 数据库（tmp_path），不触发 LLM、不污染真实数据。
"""
from __future__ import annotations

import asyncio
from datetime import date

import pytest

from personal.ics_parser import Course, parse_ics
from personal.service import (
    PersonalService,
    compress_weeks,
    expand_weeks,
    parse_date_expr,
    week_in,
)
from personal.store import PersonalStore

SEMESTER_START = date(2026, 9, 7)  # 2026-2027 秋季学期开学日（周一）

# 教务系统导出风格的 ICS 样例：
#  - 高等数学A：每周一 08:30-10:05，B-101，第 1-19 周（UNTIL 学期末）
#  - 大学英语：每周三 10:25-12:00，B-102，COUNT=8 次（第 2-9 周）
SAMPLE_ICS = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
DTSTART:20260907T083000
DTEND:20260907T100500
SUMMARY:高等数学A
LOCATION:南校区B栋101
RRULE:FREQ=WEEKLY;BYDAY=MO;UNTIL=20270117T235959Z
END:VEVENT
BEGIN:VEVENT
DTSTART:20260914T102500
DTEND:20260914T120000
SUMMARY:大学英语
LOCATION:B-102
RRULE:FREQ=WEEKLY;BYDAY=WE;COUNT=8
END:VEVENT
END:VCALENDAR
"""


def _service(tmp_path) -> PersonalService:
    store = PersonalStore(db_path=str(tmp_path / "test.db"))
    return PersonalService(store)


# ── ICS 解析 ──────────────────────────────────────────────────────────────────

def test_ics_parse_weekly_recurrence():
    courses = parse_ics(SAMPLE_ICS, SEMESTER_START, semester_weeks=19)
    assert len(courses) == 2

    math = next(c for c in courses if c.course == "高等数学A")
    assert math.day_of_week == 0            # 周一
    assert math.start_time == "08:30"
    assert math.end_time == "10:05"
    assert math.location == "南校区B栋101"
    assert math.weeks == list(range(1, 20))  # 每周一展开到学期末（第 19 周）


def test_ics_parse_count_limit():
    courses = parse_ics(SAMPLE_ICS, SEMESTER_START, semester_weeks=19)
    english = next(c for c in courses if c.course == "大学英语")
    # DTSTART 2026-09-14（第 2 周周一）所在周的周三起，共 8 次 → 第 2-9 周
    assert english.day_of_week == 2
    assert english.weeks == list(range(2, 10))


def test_ics_parse_single_event_without_rrule():
    ics = """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART:20260921T190000
DTEND:20260921T203500
SUMMARY:开学第一课（讲座）
LOCATION:大礼堂
END:VEVENT
END:VCALENDAR"""
    courses = parse_ics(ics, SEMESTER_START, semester_weeks=19)
    assert len(courses) == 1
    assert courses[0].weeks == [3]  # 2026-09-21 为第 3 周


def test_ics_parse_utc_and_tzid():
    ics = """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART;TZID=Asia/Shanghai:20260907T083000
DTEND;TZID=Asia/Shanghai:20260907T100500
SUMMARY:带时区参数
RRULE:FREQ=WEEKLY;BYDAY=TU
END:VEVENT
BEGIN:VEVENT
DTSTART:20260908T020000Z
DTEND:20260908T033000Z
SUMMARY:UTC时间
RRULE:FREQ=WEEKLY;BYDAY=WE
END:VEVENT
END:VCALENDAR"""
    courses = parse_ics(ics, SEMESTER_START, semester_weeks=19)
    by_name = {c.course: c for c in courses}
    assert by_name["带时区参数"].start_time == "08:30"
    # UTC 02:00 → 本地 10:00
    assert by_name["UTC时间"].start_time == "10:00"
    assert by_name["UTC时间"].day_of_week == 2


def test_ics_skips_bad_event_without_crashing():
    """回归：单条坏事件（缺 DTSTART）被跳过，不拖垮整份课表导入。"""
    ics = """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART:20260907T083000
DTEND:20260907T100500
SUMMARY:正常课程
RRULE:FREQ=WEEKLY;BYDAY=MO
END:VEVENT
BEGIN:VEVENT
SUMMARY:坏事件（没有 DTSTART）
END:VEVENT
END:VCALENDAR"""
    courses = parse_ics(ics, SEMESTER_START, semester_weeks=19)
    assert len(courses) == 1
    assert courses[0].course == "正常课程"


def test_ics_until_z_boundary_not_shifted_to_next_day():
    """回归：UNTIL=20270117T235959Z（UTC 周日深夜）不再 +8h 翻到次日，
    避免把 2027-01-18 周一的课多算一周。"""
    ics = """BEGIN:VCALENDAR
BEGIN:VEVENT
DTSTART:20260907T083000
DTEND:20260907T100500
SUMMARY:边界课程
RRULE:FREQ=WEEKLY;BYDAY=MO;UNTIL=20270117T235959Z
END:VEVENT
END:VCALENDAR"""
    courses = parse_ics(ics, SEMESTER_START, semester_weeks=20)
    assert len(courses) == 1
    # 2027-01-11 周一（第 19 周）≤ 2027-01-17 → 包含；2027-01-18 周一 → 排除
    assert courses[0].weeks[-1] == 19


def test_courses_on_vacation_returns_week_zero(tmp_path):
    """回归：开学前查询（暑假）返回 week_num=0，不出现负数周。"""
    service = _service(tmp_path)
    result = asyncio.run(service.courses_on("u1", "今天", today=date(2026, 8, 5)))
    assert result["week_num"] == 0
    assert result["courses"] == []


# ── 教学周工具 ────────────────────────────────────────────────────────────────

def test_weeks_compress_and_expand_roundtrip():
    weeks = [1, 2, 3, 5, 6, 7, 9]
    expr = compress_weeks(weeks)
    assert expr == "1-3,5-7,9"
    assert expand_weeks(expr) == weeks
    assert expand_weeks("") == []
    assert expand_weeks("all") == []


def test_week_in_empty_means_all():
    assert week_in(5, "") is True
    assert week_in(5, "3-8") is True
    assert week_in(2, "3-8") is False


# ── 日期表达式 ────────────────────────────────────────────────────────────────

def test_parse_date_expr():
    today = date(2026, 9, 9)  # 周三
    assert parse_date_expr("今天", today) == today
    assert parse_date_expr("明天", today) == date(2026, 9, 10)
    assert parse_date_expr("昨天", today) == date(2026, 9, 8)
    assert parse_date_expr("周三", today) == today
    assert parse_date_expr("周一", today) == date(2026, 9, 7)   # 本周一（已过）
    assert parse_date_expr("周五", today) == date(2026, 9, 11)  # 本周五（未到）
    assert parse_date_expr("2026-09-14", today) == date(2026, 9, 14)
    assert parse_date_expr("10月1日", today) == date(2026, 10, 1)
    assert parse_date_expr("这周", today) == date(2026, 9, 7)
    with pytest.raises(ValueError):
        parse_date_expr("下周三晚", today)


# ── 存储与用户隔离 ────────────────────────────────────────────────────────────

def test_store_schedule_crud(tmp_path):
    store = PersonalStore(db_path=str(tmp_path / "t.db"))
    courses = [
        {"course": "高数", "day_of_week": 0, "start_time": "08:30", "end_time": "10:05",
         "location": "B-101", "weeks": "1-16"},
    ]
    asyncio.run(store.replace_schedule("u1", courses))
    rows = asyncio.run(store.get_schedule("u1"))
    assert len(rows) == 1
    assert rows[0]["course"] == "高数"
    assert rows[0]["weeks"] == "1-16"

    asyncio.run(store.clear_schedule("u1"))
    assert asyncio.run(store.get_schedule("u1")) == []


def test_store_user_isolation(tmp_path):
    store = PersonalStore(db_path=str(tmp_path / "t.db"))
    asyncio.run(store.replace_schedule("u1", [
        {"course": "高数", "day_of_week": 0, "start_time": "08:30", "end_time": "10:05", "weeks": ""},
    ]))
    asyncio.run(store.replace_schedule("u2", [
        {"course": "英语", "day_of_week": 2, "start_time": "10:25", "end_time": "12:00", "weeks": ""},
    ]))
    u1 = asyncio.run(store.get_schedule("u1"))
    u2 = asyncio.run(store.get_schedule("u2"))
    assert [c["course"] for c in u1] == ["高数"]
    assert [c["course"] for c in u2] == ["英语"]


def test_todo_crud_and_done(tmp_path):
    store = PersonalStore(db_path=str(tmp_path / "t.db"))
    todo = asyncio.run(store.add_todo("u1", "交实验报告", kind="ddl", due_at="2026-09-14"))
    assert todo["content"] == "交实验报告"
    assert todo["done"] is False

    ok = asyncio.run(store.set_todo_done("u1", todo["id"], done=True))
    assert ok is True
    done_list = asyncio.run(store.list_todos("u1", status="done"))
    assert len(done_list) == 1 and done_list[0]["done"] is True

    # 用户隔离：u2 无法操作 u1 的待办
    assert asyncio.run(store.set_todo_done("u2", todo["id"])) is False
    assert asyncio.run(store.delete_todo("u2", todo["id"])) is False
    assert asyncio.run(store.delete_todo("u1", todo["id"])) is True


# ── 查询服务 ──────────────────────────────────────────────────────────────────

def test_courses_on_respects_weeks(tmp_path):
    service = _service(tmp_path)
    asyncio.run(service.import_courses("u1", [
        # 第 1-16 周的周一高数
        {"course": "高数", "day_of_week": 0, "start_time": "08:30", "end_time": "10:05",
         "location": "B-101", "weeks": list(range(1, 17))},
        # 第 3-16 周的周一体育（第 1、2 周不上）
        {"course": "体育", "day_of_week": 0, "start_time": "14:00", "end_time": "15:35",
         "location": "田径场", "weeks": list(range(3, 17))},
    ]))
    # 第 1 周（2026-09-07 周一）：只有高数
    w1 = asyncio.run(service.courses_on("u1", "2026-09-07"))
    assert [c["course"] for c in w1["courses"]] == ["高数"]
    assert w1["week_num"] == 1
    # 第 3 周（2026-09-21 周一）：高数 + 体育
    w3 = asyncio.run(service.courses_on("u1", "2026-09-21"))
    assert [c["course"] for c in w3["courses"]] == ["高数", "体育"]
    assert w3["week_num"] == 3


def test_courses_on_no_schedule_returns_empty(tmp_path):
    service = _service(tmp_path)
    result = asyncio.run(service.courses_on("u1", "今天", today=date(2026, 9, 7)))
    assert result["courses"] == []
    assert result["weekday"] == "周一"


def test_upcoming_days_left(tmp_path):
    service = _service(tmp_path)
    today = date(2026, 9, 7)
    asyncio.run(service.add_todo("u1", "高数期中考试", kind="exam", due_at="2026-09-21"))
    asyncio.run(service.add_todo("u1", "交实验报告", kind="ddl", due_at="2026-09-10"))
    asyncio.run(service.add_todo("u1", "买书", kind="todo", due_at="2026-09-08"))  # todo 不参与
    asyncio.run(service.add_todo("u1", "已过期的作业", kind="ddl", due_at="2026-09-05"))

    items = asyncio.run(service.upcoming("u1", horizon_days=30, today=today))
    assert [i["content"] for i in items] == ["已过期的作业", "交实验报告", "高数期中考试"]
    assert items[0]["days_left"] == -2 and items[0]["status"] == "已过期"
    assert items[1]["days_left"] == 3
    assert items[2]["status"] == "还剩14天"


def test_overview_aggregates(tmp_path):
    service = _service(tmp_path)
    today = date(2026, 9, 7)  # 周一，第 1 周
    asyncio.run(service.import_courses("u1", [
        {"course": "高数", "day_of_week": 0, "start_time": "08:30", "end_time": "10:05", "weeks": "1-16"},
    ]))
    asyncio.run(service.add_todo("u1", "取快递", kind="todo"))
    asyncio.run(service.add_todo("u1", "高数期中", kind="exam", due_at="2026-09-14"))

    ov = asyncio.run(service.overview("u1", today=today))
    assert ov["has_schedule"] is True
    assert ov["courses"][0]["course"] == "高数"
    assert ov["todos"][0]["content"] == "取快递"
    assert ov["upcoming"][0]["days_left"] == 7
    assert ov["weekday"] == "周一"
