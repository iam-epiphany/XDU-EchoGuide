"""业务工具测试：天气（mock httpx）、课表/待办/DDL 工具、校园信息工具。

项目测试惯例：手写 Fake 打桩（无 mock 库），只测确定性行为。
"""
from __future__ import annotations

import json
import asyncio
from datetime import date, datetime, timedelta

import httpx

from campus.store import CampusInfoStore
from personal.service import PersonalService
from personal.store import PersonalStore
from tools import with_service
from tools.campus_tool import campus_info_handler
from tools.ddl_tool import query_ddl_handler
from tools.schedule_tool import query_schedule_handler
from tools.todo_tool import add_todo_handler, complete_todo_handler, query_todo_handler
from tools.weather import WMO_CODES, weather_handler


def _ctx(tmp_path, user_id="u1") -> dict:
    """构造带服务依赖的工具 context（与 api/main.py 的 with_service 注入等价）。"""
    service = PersonalService(PersonalStore(db_path=str(tmp_path / "t.db")))
    ctx = {"agent_type": "personal", "user_id": user_id}
    ctx.update({"personal_service": service, "campus_store": CampusInfoStore(data_dir=str(tmp_path))})
    return ctx


def _weather_data():
    return {
        "current": {"temperature_2m": 28.3, "weather_code": 2, "relative_humidity_2m": 60},
        "daily": {
            "time": ["2026-09-07", "2026-09-08"],
            "weather_code": [2, 61],
            "temperature_2m_max": [31.0, 26.0],
            "temperature_2m_min": [22.0, 19.0],
            "precipitation_probability_max": [10, 85],
            "wind_speed_10m_max": [12.0, 18.0],
        },
    }


class _FakeResponse:
    def __init__(self, data=None):
        self._data = data if data is not None else _weather_data()

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class _FakeClient:
    """手写 Fake：拦截 httpx.AsyncClient.get，按 URL 子串分派响应，记录全部调用。

    兼容无 Key 场景（单次 Open-Meteo 调用）：url/params 指向最后一次调用。
    """
    def __init__(self, responses=None, errors=None, timeout=None):
        self.url = None
        self.params = None
        self.calls = []          # [(url, params), ...] 按调用顺序记录
        self._responses = responses or {}  # {url子串: json_data}
        self._errors = errors or {}        # {url子串: 要抛出的异常}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def get(self, url, params, headers=None):
        self.url, self.params = url, params
        self.calls.append((url, dict(params)))
        if headers is not None:
            self.headers = dict(headers)
        for key, exc in self._errors.items():
            if key in url:
                raise exc
        for key, data in self._responses.items():
            if key in url:
                return _FakeResponse(data)
        return _FakeResponse()


# ── 天气 ──────────────────────────────────────────────────────────────────────

def test_weather_handler_calls_open_meteo(monkeypatch, tmp_path):
    fake = _FakeClient()
    monkeypatch.setattr("tools.weather.httpx.AsyncClient", lambda **kw: fake)

    result = asyncio.run(weather_handler({"place": "南校区", "days": 2}, {}))
    assert fake.url == "https://api.open-meteo.com/v1/forecast"
    assert fake.params["forecast_days"] == 2
    assert result["current"]["temperature"] == 28.3
    assert result["current"]["weather"] == WMO_CODES[2]          # 多云
    assert result["daily"][1]["weather"] == WMO_CODES[61]        # 小雨
    assert result["daily"][1]["precip_probability"] == 85
    assert "Open-Meteo" in result["source"]


def test_weather_handler_days_clamped(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr("tools.weather.httpx.AsyncClient", lambda **kw: fake)
    result = asyncio.run(weather_handler({"days": 99}, {}))  # 超上限 → 7
    assert fake.params["forecast_days"] == 7
    assert result["place"] == "南校区"  # 默认地点


def test_weather_timezone_must_not_be_pre_encoded(monkeypatch):
    """回归：timezone 交给 httpx 编码，预编码的 %2F 会被二次编码导致 Open-Meteo 400。"""
    fake = _FakeClient()
    monkeypatch.setattr("tools.weather.httpx.AsyncClient", lambda **kw: fake)
    asyncio.run(weather_handler({}, {}))
    assert fake.params["timezone"] == "Asia/Shanghai"


def test_weather_handler_bad_days_value_falls_back(monkeypatch):
    """回归：LLM 传 null/空串等异常 days 值时回退默认 3 天，不抛异常。"""
    fake = _FakeClient()
    monkeypatch.setattr("tools.weather.httpx.AsyncClient", lambda **kw: fake)
    result = asyncio.run(weather_handler({"days": None}, {}))
    assert fake.params["forecast_days"] == 3
    assert result["place"] == "南校区"


# ── 和风天气（主源，配置 QWEATHER_API_KEY 后启用）────────────────────────────

def _qweather_now():
    return {
        "code": "200",
        "now": {"temp": "28", "text": "多云", "humidity": "60", "windSpeed": "12"},
    }


def _qweather_daily():
    return {
        "code": "200",
        "daily": [
            {"fxDate": "2026-08-14", "textDay": "多云", "tempMax": "31", "tempMin": "22",
             "precip": 10, "windSpeedDay": "12"},
            {"fxDate": "2026-08-15", "textDay": "小雨", "tempMax": "26", "tempMin": "19",
             "precip": 85, "windSpeedDay": "18"},
        ],
    }


def test_weather_handler_uses_qweather_when_key_set(monkeypatch):
    """配置和风 Key 后：调用 devapi.qweather.com 的 now + 7d 两接口，不碰 Open-Meteo。"""
    monkeypatch.setattr("tools.weather.QWEATHER_KEY", "HE-test-key")
    monkeypatch.setattr("tools.weather.QWEATHER_BASE", "https://devapi.qweather.com")  # 固定域名，避免受 .env 影响
    fake = _FakeClient(responses={
        "/v7/weather/now": _qweather_now(),
        "/v7/weather/7d": _qweather_daily(),
    })
    monkeypatch.setattr("tools.weather.httpx.AsyncClient", lambda **kw: fake)

    result = asyncio.run(weather_handler({"place": "南校区", "days": 2}, {}))
    assert [u for u, _ in fake.calls] == [
        "https://devapi.qweather.com/v7/weather/now",
        "https://devapi.qweather.com/v7/weather/7d",
    ]
    # location 为 "经度,纬度"（和风坐标顺序与 Open-Meteo 相反，需反转）
    assert fake.calls[0][1]["location"] == "108.84,34.15"
    assert fake.calls[0][1]["key"] == "HE-test-key"
    # 和风返回中文天气描述，直接透传；与 Open-Meteo 路径保持同一输出结构
    assert result["current"]["temperature"] == "28"
    assert result["current"]["weather"] == "多云"
    assert result["daily"][1]["weather"] == "小雨"
    assert result["daily"][1]["precip_probability"] == 85
    assert "和风天气" in result["source"]


def test_weather_handler_uses_custom_qweather_host(monkeypatch):
    """回归：新账号专属 API Host（QWEATHER_API_HOST）优先于旧公共域名。"""
    monkeypatch.setattr("tools.weather.QWEATHER_KEY", "HE-test-key")
    monkeypatch.setattr("tools.weather.QWEATHER_BASE", "https://abc1234.def.qweatherapi.com")
    fake = _FakeClient(responses={
        "/v7/weather/now": _qweather_now(),
        "/v7/weather/7d": _qweather_daily(),
    })
    monkeypatch.setattr("tools.weather.httpx.AsyncClient", lambda **kw: fake)

    asyncio.run(weather_handler({}, {}))
    assert fake.calls[0][0] == "https://abc1234.def.qweatherapi.com/v7/weather/now"
    assert fake.calls[1][0] == "https://abc1234.def.qweatherapi.com/v7/weather/7d"


def test_weather_handler_falls_back_on_qweather_error_code(monkeypatch):
    """回归：和风返回非 200 业务码（如 402 免费额度用尽）→ 自动回退 Open-Meteo。"""
    monkeypatch.setattr("tools.weather.QWEATHER_KEY", "HE-test-key")
    monkeypatch.setattr("tools.weather.QWEATHER_BASE", "https://devapi.qweather.com")  # 固定域名，避免受 .env 影响
    fake = _FakeClient(responses={
        "/v7/weather/now": {"code": "402", "now": {}},  # HTTP 200，业务错误码
    })
    monkeypatch.setattr("tools.weather.httpx.AsyncClient", lambda **kw: fake)

    result = asyncio.run(weather_handler({}, {}))
    assert "Open-Meteo" in result["source"]
    assert fake.calls[0][0].startswith("https://devapi.qweather.com")
    assert fake.calls[-1][0] == "https://api.open-meteo.com/v1/forecast"


def test_weather_handler_falls_back_on_qweather_network_error(monkeypatch):
    """回归：和风网络异常 → 自动回退 Open-Meteo，工具不崩溃。"""
    monkeypatch.setattr("tools.weather.QWEATHER_KEY", "HE-test-key")
    monkeypatch.setattr("tools.weather.QWEATHER_BASE", "https://devapi.qweather.com")  # 固定域名，避免受 .env 影响
    fake = _FakeClient(errors={"/v7/weather/now": httpx.ConnectError("network down")})
    monkeypatch.setattr("tools.weather.httpx.AsyncClient", lambda **kw: fake)

    result = asyncio.run(weather_handler({}, {}))
    assert "Open-Meteo" in result["source"]


# ── 课表 / 待办 / DDL 工具 ───────────────────────────────────────────────────

def test_schedule_tool_without_schedule_guides_import(tmp_path):
    ctx = _ctx(tmp_path)
    result = asyncio.run(query_schedule_handler({"date": "今天"}, ctx))
    assert result["available"] is True
    assert result["courses"] == []
    assert "导入" in result["message"]


def test_schedule_tool_returns_courses(tmp_path):
    ctx = _ctx(tmp_path)
    service = ctx["personal_service"]
    asyncio.run(service.import_courses("u1", [
        {"course": "高数", "day_of_week": 0, "start_time": "08:30", "end_time": "10:05",
         "location": "B-101", "weeks": "1-16"},
    ]))
    # 2026-09-07 为开学第 1 周周一
    result = asyncio.run(query_schedule_handler({"date": "2026-09-07"}, ctx))
    assert result["courses"][0]["course"] == "高数"
    assert result["courses"][0]["location"] == "B-101"


def test_schedule_tool_user_isolation(tmp_path):
    ctx_a = _ctx(tmp_path, user_id="u1")
    ctx_b = _ctx(tmp_path, user_id="u2")
    asyncio.run(ctx_a["personal_service"].import_courses("u1", [
        {"course": "高数", "day_of_week": 0, "start_time": "08:30", "end_time": "10:05", "weeks": ""},
    ]))
    result_a = asyncio.run(query_schedule_handler({"date": "2026-09-07"}, ctx_a))
    result_b = asyncio.run(query_schedule_handler({"date": "2026-09-07"}, ctx_b))
    assert len(result_a["courses"]) == 1
    assert result_b["courses"] == []  # u2 看不到 u1 的课表


def test_todo_tool_full_flow(tmp_path):
    ctx = _ctx(tmp_path)
    added = asyncio.run(add_todo_handler(
        {"content": "交实验报告", "kind": "ddl", "due_at": "2026-09-14"}, ctx))
    assert added["available"] is True and added["todo"]["kind"] == "ddl"

    listed = asyncio.run(query_todo_handler({"status": "open"}, ctx))
    assert listed["total"] == 1

    done = asyncio.run(complete_todo_handler({"id": added["todo"]["id"]}, ctx))
    assert done["todo"]["done"] is True

    empty = asyncio.run(query_todo_handler({"status": "open"}, ctx))
    assert empty["total"] == 0


def test_add_todo_requires_content(tmp_path):
    ctx = _ctx(tmp_path)
    result = asyncio.run(add_todo_handler({"content": "  "}, ctx))
    assert result["available"] is False


def test_ddl_tool_returns_countdown(tmp_path):
    ctx = _ctx(tmp_path)
    service = ctx["personal_service"]
    today = date.today()
    asyncio.run(service.add_todo("u1", "高数期中", kind="exam", due_at=(today + timedelta(days=14)).isoformat()))
    result = asyncio.run(query_ddl_handler({"horizon_days": 30}, ctx))
    assert result["total"] == 1
    assert result["items"][0]["days_left"] == 14
    assert result["items"][0]["status"] == "还剩14天"


# ── 校园信息工具 ──────────────────────────────────────────────────────────────

def test_campus_tool_shuttle(tmp_path):
    (tmp_path / "shuttle_schedule.json").write_text(json.dumps({
        "routes": [{"name": "南校区→北校区", "direction": "南→北",
                    "departures": ["07:00", "08:00", "23:00"]}]
    }, ensure_ascii=False), encoding="utf-8")
    ctx = _ctx(tmp_path)
    result = asyncio.run(campus_info_handler({"category": "shuttle", "keyword": "南→北"}, ctx))
    assert result["available"] is True
    route = result["routes"][0]
    # 无论测试运行时刻如何，下一班都是今天或明天的某个班次
    assert route["next_departure"] in ("07:00", "08:00", "23:00")
    assert route["minutes_left"] >= 0


def test_campus_tool_unknown_category(tmp_path):
    ctx = _ctx(tmp_path)
    result = asyncio.run(campus_info_handler({"category": "x"}, ctx))
    assert result["available"] is False


# ── with_service 注入 ─────────────────────────────────────────────────────────

def test_with_service_injects_deps(tmp_path):
    async def handler(params, context):
        return {"has_service": context.get("personal_service") is not None}

    wrapped = with_service(handler, personal_service="svc")
    result = asyncio.run(wrapped({}, {"user_id": "u1"}))
    assert result == {"has_service": True}
