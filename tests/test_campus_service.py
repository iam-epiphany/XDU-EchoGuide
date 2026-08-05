"""结构化公开信息测试：校车下一班计算、方向过滤、楼宇查询、数据缺失降级。

使用临时目录中的 JSON 数据文件（不依赖仓库 data/public 的真实数据）。
"""
from __future__ import annotations

import json
from datetime import date, datetime

from campus.store import CampusInfoStore


def _write(tmp_path, fname: str, data) -> None:
    (tmp_path / fname).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _store(tmp_path) -> CampusInfoStore:
    return CampusInfoStore(data_dir=str(tmp_path))


def _shuttle_data():
    return {
        "routes": [
            {"name": "南校区→北校区", "direction": "南→北", "pickup": "南校区东门",
             "duration_min": 60, "departures": ["07:00", "08:00", "09:00", "16:30"], "days": "weekdays"},
            {"name": "北校区→南校区", "direction": "北→南", "pickup": "北校区东门",
             "duration_min": 60, "departures": ["07:30", "08:30", "09:30", "17:00"], "days": "weekdays"},
            {"name": "南校区→北校区（双休）", "direction": "南→北", "pickup": "南校区东门",
             "duration_min": 60, "departures": ["07:30", "13:00", "18:00"], "days": "weekend"},
        ]
    }


def test_shuttle_weekday_weekend_filter(tmp_path):
    """工作日/周末班次按星期几过滤：周三只出工作日班次，周六只出双休班次。"""
    _write(tmp_path, "shuttle_schedule.json", _shuttle_data())
    store = _store(tmp_path)
    weekday = store.next_shuttle(now=datetime(2026, 9, 9, 8, 0))   # 周三
    weekend = store.next_shuttle(now=datetime(2026, 9, 12, 8, 0))  # 周六
    assert len(weekday["routes"]) == 2
    assert all("双休" not in r["route"] for r in weekday["routes"])
    assert len(weekend["routes"]) == 1
    assert "双休" in weekend["routes"][0]["route"]


def test_next_shuttle_today(tmp_path):
    _write(tmp_path, "shuttle_schedule.json", _shuttle_data())
    store = _store(tmp_path)
    now = datetime(2026, 9, 7, 8, 20)  # 8:20 → 下一班 9:00
    result = store.next_shuttle(now=now)
    assert result["available"] is True
    south_to_north = result["routes"][0]
    assert south_to_north["route"] == "南校区→北校区"
    assert south_to_north["next_departure"] == "09:00"
    assert south_to_north["minutes_left"] == 40
    assert south_to_north["is_today"] is True


def test_next_shuttle_after_last_returns_tomorrow(tmp_path):
    _write(tmp_path, "shuttle_schedule.json", _shuttle_data())
    store = _store(tmp_path)
    now = datetime(2026, 9, 7, 20, 0)  # 末班 16:30 已过 → 明天首班
    result = store.next_shuttle(now=now)
    route = result["routes"][0]
    assert route["next_departure"] == "07:00"
    assert route["is_today"] is False
    assert route["date"] == "2026-09-08"


def test_next_shuttle_direction_filter(tmp_path):
    _write(tmp_path, "shuttle_schedule.json", _shuttle_data())
    store = _store(tmp_path)
    now = datetime(2026, 9, 7, 8, 0)
    result = store.next_shuttle(direction="南→北", now=now)
    assert len(result["routes"]) == 1
    assert result["routes"][0]["direction"] == "南→北"


def test_next_shuttle_missing_data(tmp_path):
    store = _store(tmp_path)  # 目录中没有校车数据文件
    result = store.next_shuttle(now=datetime(2026, 9, 7, 8, 0))
    assert result["available"] is False
    assert "暂未录入" in result["message"]


def test_building_search_by_alias(tmp_path):
    _write(tmp_path, "buildings.json", [
        {"name": "信远楼", "aliases": ["信远"], "campus": "南校区",
         "location": "图书馆南侧", "description": "人文学院"},
    ])
    store = _store(tmp_path)
    found = store.find_building("信远")
    assert len(found) == 1 and found[0]["name"] == "信远楼"
    assert store.find_building("不存在的楼") == []


def test_venues_and_library(tmp_path):
    _write(tmp_path, "venues.json", [
        {"name": "南校区体育馆", "campus": "南校区", "open_hours": "8:00-22:00"},
    ])
    _write(tmp_path, "library.json", [
        {"name": "南校区图书馆", "campus": "南校区", "open_hours": "8:00-22:00"},
    ])
    store = _store(tmp_path)
    assert store.list_venues() == [{"name": "南校区体育馆", "campus": "南校区", "open_hours": "8:00-22:00"}]
    assert store.library_info()["libraries"][0]["open_hours"] == "8:00-22:00"


def test_library_missing_degrades_gracefully(tmp_path):
    store = _store(tmp_path)
    result = store.library_info()
    assert result["available"] is False
    assert "暂未录入" in result["message"]


def test_search_unknown_category(tmp_path):
    store = _store(tmp_path)
    result = store.search("hacker", "x")
    assert result["available"] is False


def test_reload_picks_up_new_data(tmp_path):
    store = _store(tmp_path)
    assert store.next_shuttle()["available"] is False
    _write(tmp_path, "shuttle_schedule.json", _shuttle_data())
    store.reload()
    assert store.next_shuttle()["available"] is True
