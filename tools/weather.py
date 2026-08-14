"""
get_weather —— 天气查询工具（和风天气免费版为主 + Open-Meteo 免费兜底）。

数据源（双源自动切换，对外接口完全一致）：
  1. 和风天气免费版（主源）：https://devapi.qweather.com，需在 dev.qweather.com
     注册免费订阅获取 API Key（以 HE 开头，约 1000 次/天，额度用完返回 402）。
     配置 .env 的 QWEATHER_API_KEY 后自动启用；国内访问快且稳定。
  2. Open-Meteo（兜底源）：https://api.open-meteo.com，免费无 Key，但为境外
     服务，国内网络可能不通。未配置和风 Key 或和风调用失败时回退至此。

坐标说明：内置西电南北校区近似坐标（可在此调整）：
  - 南校区（长安区）约 34.15, 108.84
  - 北校区（雁塔区太白南路）约 34.225, 108.915
返回结构化天气数据，由 Agent 转成自然语言（如"明天上午有雨，记得带伞"）。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
# 和风天气：2025-04 起每个开发者分配专属 API Host（控制台 → 设置 → API Host，形如
# abc1234xyz.def.qweatherapi.com），旧公共域名逐步停用，新 Key 必须配置自己的 Host。
# 未配置 QWEATHER_API_HOST 时回退 devapi.qweather.com（仅兼容老账号）。
QWEATHER_BASE = os.getenv("QWEATHER_API_HOST", "").strip() or "https://devapi.qweather.com"
if not QWEATHER_BASE.startswith(("http://", "https://")):
    QWEATHER_BASE = "https://" + QWEATHER_BASE  # 用户只填域名时自动补协议
QWEATHER_KEY = os.getenv("QWEATHER_API_KEY", "").strip()

# 地点 → (纬度, 经度)。如需精确坐标请调整。
PLACES: Dict[str, tuple] = {
    "南校区": (34.1500, 108.8400),
    "北校区": (34.2250, 108.9150),
    "西安":   (34.3416, 108.9398),
}

# WMO 天气代码 → 中文描述（Open-Meteo 返回 weather_code；和风直接返回中文 text，无需映射）
WMO_CODES: Dict[int, str] = {
    0: "晴", 1: "基本晴", 2: "多云", 3: "阴",
    45: "雾", 48: "冻雾",
    51: "毛毛雨", 53: "小雨", 55: "中雨",
    56: "冻毛毛雨", 57: "冻雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    77: "米雪",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "阵雪",
    95: "雷阵雨", 96: "雷雨伴冰雹", 99: "强雷雨伴冰雹",
}


def _describe(code: int) -> str:
    return WMO_CODES.get(code, f"天气代码{code}")


async def _fetch_open_meteo(place: str, days: int) -> Dict[str, Any]:
    """Open-Meteo 兜底源：返回与主源结构一致的天气数据（source 标注来源）。"""
    coord = PLACES.get(place) or PLACES.get("南校区")
    query_params = {
        "latitude": coord[0],
        "longitude": coord[1],
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,wind_speed_10m_max"
        ),
        "timezone": "Asia/Shanghai",  # 由 httpx 负责 URL 编码，不要预编码（否则二次编码导致 400）
        "forecast_days": days,
    }
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(OPEN_METEO_URL, params=query_params)
        resp.raise_for_status()
        data = resp.json()

    current = data.get("current", {})
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    return {
        "place": place,
        "requested_place": place,
        "current": {
            "temperature": current.get("temperature_2m"),
            "weather": _describe(current.get("weather_code", 0)),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
        },
        "daily": [
            {
                "date": d,
                "weather": _describe(daily["weather_code"][i]) if daily.get("weather_code") else "未知",
                "temp_max": daily["temperature_2m_max"][i] if daily.get("temperature_2m_max") else None,
                "temp_min": daily["temperature_2m_min"][i] if daily.get("temperature_2m_min") else None,
                "precip_probability": daily["precipitation_probability_max"][i] if daily.get("precipitation_probability_max") else None,
                "wind_speed_max": daily["wind_speed_10m_max"][i] if daily.get("wind_speed_10m_max") else None,
            }
            for i, d in enumerate(dates)
        ],
        "source": "Open-Meteo（免费数据源，仅供参考）",
    }


async def _fetch_qweather(place: str, days: int) -> Dict[str, Any]:
    """和风天气主源：实时 + N 天预报，返回与 Open-Meteo 路径相同的统一结构。

    location 参数用 "经度,纬度"（和风坐标顺序与 Open-Meteo 相反，需反转），
    直接以坐标定位，免去城市 ID 地理编码调用，节省免费额度。
    认证双保险：key 查询参数 + X-QW-Api-Key 请求头（新控制台推荐方式）。
    返回 code != 200 视为失败（如 402 额度用尽）。
    """
    lat, lon = PLACES.get(place) or PLACES.get("南校区")
    location = f"{lon},{lat}"
    query_params = {"location": location, "key": QWEATHER_KEY}  # 7d 接口固定返回 7 天，days 由上层切片
    headers = {"X-QW-Api-Key": QWEATHER_KEY}

    async with httpx.AsyncClient(timeout=10.0) as client:
        now_resp = await client.get(f"{QWEATHER_BASE}/v7/weather/now", params=query_params, headers=headers)
        now_resp.raise_for_status()
        now_data = now_resp.json()
        daily_resp = await client.get(f"{QWEATHER_BASE}/v7/weather/7d", params=query_params, headers=headers)
        daily_resp.raise_for_status()
        daily_data = daily_resp.json()

    if now_data.get("code") != "200" or daily_data.get("code") != "200":
        code = now_data.get("code") or daily_data.get("code")
        raise RuntimeError(f"和风天气返回错误码 {code}（可能为免费额度用尽）")

    now = now_data.get("now", {})
    daily = daily_data.get("daily", [])[:days]
    return {
        "place": place,
        "requested_place": place,
        "current": {
            "temperature": now.get("temp"),
            "weather": now.get("text"),
            "humidity": now.get("humidity"),
            "wind_speed": now.get("windSpeed"),
        },
        "daily": [
            {
                "date": d.get("fxDate"),
                "weather": d.get("textDay") or d.get("textNight"),
                "temp_max": d.get("tempMax"),
                "temp_min": d.get("tempMin"),
                "precip_probability": d.get("precip"),  # 降水概率百分比（免费版）
                "wind_speed_max": d.get("windSpeedDay"),
            }
            for d in daily
        ],
        "source": "和风天气（免费版，仅供参考）",
    }


async def weather_handler(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    查询天气（和风天气免费版为主，Open-Meteo 兜底）。

    params:
      place: "南校区" / "北校区" / "西安"，默认南校区
      days:  预报天数 1-7，默认 3
    """
    place = str(params.get("place", "南校区")).strip() or "南校区"
    try:
        days = min(max(int(params.get("days", 3)), 1), 7)
    except (TypeError, ValueError):
        days = 3  # LLM 传了 null/空串等异常值时回退默认 3 天

    # 配置了和风 Key → 主源优先；失败（网络/额度/错误码）回退 Open-Meteo
    if QWEATHER_KEY:
        try:
            return await _fetch_qweather(place, days)
        except Exception as exc:  # 网络异常、非 2xx、错误码等一律回退，不让工具崩溃
            logger.warning("和风天气查询失败，回退 Open-Meteo: %s", exc)
    return await _fetch_open_meteo(place, days)
