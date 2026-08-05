"""
get_weather —— 天气查询工具（Open-Meteo 免费 API，无需 API Key）。

坐标说明：内置西电南北校区近似坐标（可在此调整）：
  - 南校区（长安区）约 34.15, 108.84
  - 北校区（雁塔区太白南路）约 34.225, 108.915
返回结构化天气数据，由 Agent 转成自然语言（如"明天上午有雨，记得带伞"）。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# 地点 → (纬度, 经度)。如需精确坐标请调整。
PLACES: Dict[str, tuple] = {
    "南校区": (34.1500, 108.8400),
    "北校区": (34.2250, 108.9150),
    "西安":   (34.3416, 108.9398),
}

# WMO 天气代码 → 中文描述（Open-Meteo 返回 weather_code）
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


async def weather_handler(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    查询天气（Open-Meteo，免费无 Key）。

    params:
      place: "南校区" / "北校区" / "西安"，默认南校区
      days:  预报天数 1-7，默认 3
    """
    place = str(params.get("place", "南校区")).strip() or "南校区"
    try:
        days = min(max(int(params.get("days", 3)), 1), 7)
    except (TypeError, ValueError):
        days = 3  # LLM 传了 null/空串等异常值时回退默认 3 天

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
    result = {
        "place": place,
        "requested_place": params.get("place", "南校区"),
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
    return result
