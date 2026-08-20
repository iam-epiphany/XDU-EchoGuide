"""
query_campus_info —— 校园公开信息查询工具（结构化数据）。

覆盖校车时刻表 / 楼宇位置 / 运动场馆 / 图书馆开放时间，
数据来自 data/public/*.json（由维护者填充真实数据）。

注意：校车查询内部会与当前时间比对计算"下一班"，因此本工具
不设缓存（cache_ttl=0），保证时刻准确。
"""
from __future__ import annotations

from typing import Any, Dict

from campus.store import CampusInfoStore


async def campus_info_handler(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    查询校园公开信息。

    params:
      category: "auto"（汇总公开数据）/ "shuttle"（校车）/ "buildings"（楼宇）/
                "venues"（场馆）/ "library"（图书馆）
      keyword:  查询关键词：
                - shuttle：方向，如"南→北"（不传则返回两个方向下一班）
                - buildings：楼名/别名，如"信远楼"（不传返回全部）
                - venues：场馆名（不传返回全部）
                - library：忽略
    """
    store: CampusInfoStore = context.get("campus_store")
    if store is None:
        return {"available": False, "message": "校园信息数据源不可用，请稍后重试。"}

    category = str(params.get("category", "")).strip() or "auto"
    keyword = str(params.get("keyword", "")).strip() or None
    return store.search(category, keyword)
