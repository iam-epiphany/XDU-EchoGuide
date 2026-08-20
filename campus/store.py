"""
结构化公开信息 —— 校园公共数据加载与查询。

数据源：data/public/*.json（你提供真实数据后填充，模板见 data/public/README.md）：
  - shuttle_schedule.json  校车/班车时刻表（南↔北，按方向 + 发车时刻列表）
  - buildings.json         楼宇位置（楼名/别名 → 校区/位置/用途）
  - venues.json            运动场馆（名称/位置/开放时段/设施）
  - library.json           图书馆（各馆/自习区开放时间）

为什么不放进 RAG 知识库：校车"下一班几点"需要精确时间计算与当前时间比对，
RAG 只能返回文本片段，无法计算；结构化 JSON + 内存加载最直接，且方便更新。

数据缺失时（文件不存在或内容为空）查询返回空结果并附提示，不抛异常，
保证对话链路不受影响 —— 缺失项会在 README 的「待补充公开信息」中登记。
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
from datetime import date, datetime, time as dtime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 数据文件 → 顶层键
_FILES = ["shuttle_schedule.json", "buildings.json", "venues.json", "library.json"]


def _default_data_dir() -> str:
    root = pathlib.Path(__file__).parent.parent.resolve()
    return os.getenv("ECHOGUIDE_PUBLIC_DATA_DIR", str(root / "data" / "public"))


class CampusInfoStore:
    """
    公开信息加载与查询。

    启动时（或首次使用时）从 data/public/ 目录加载 JSON；
    数据缺失时相应类别返回空列表，不影响其他类别。
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or _default_data_dir()
        self._data: Dict[str, Any] = {}
        self.reload()

    # ── 加载 ──────────────────────────────────────────────────────────────────

    def reload(self) -> Dict[str, Any]:
        """重新加载全部数据文件，返回各文件的加载状态。"""
        self._data = {}
        status: Dict[str, str] = {}
        for fname in _FILES:
            path = pathlib.Path(self.data_dir) / fname
            if not path.exists():
                status[fname] = "missing"
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._data[fname] = json.load(f)
                status[fname] = "ok"
            except Exception as ex:
                logger.warning(f"公开信息文件加载失败: {path} — {ex}")
                status[fname] = f"error: {ex}"
        return status

    @property
    def load_status(self) -> Dict[str, str]:
        status: Dict[str, str] = {}
        for fname in _FILES:
            status[fname] = "ok" if fname in self._data else "missing"
        return status

    # ── 校车 ──────────────────────────────────────────────────────────────────

    def next_shuttle(
        self,
        direction: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        查询下一班校车。
        direction：方向关键词（南→北 / 北→南 / 南校区 / 北校区），不传则查两个方向。
        返回每方向的：下一班发车时间、剩余分钟、发车点、末班车信息。
        """
        now = now or datetime.now()
        data = self._data.get("shuttle_schedule.json", {})
        routes = data.get("routes", [])
        if not routes:
            return {"available": False, "message": "校车时刻表数据暂未录入，具体班次请以校车管理最新通知为准。"}

        results = []
        for route in routes:
            name = route.get("name", "")
            if direction and not self._match_direction(name, direction):
                continue
            if not self._runs_on_day(route, now):
                continue  # 工作日/周末班次按星期几过滤
            departures = [d for d in route.get("departures", []) if isinstance(d, str)]
            # 找今天 >= 当前时间的最近一班；没有则明天首班
            today = now.date()
            next_dep: Optional[str] = None
            next_date: Optional[date] = None
            for dep in departures:
                dep_t = dtime.fromisoformat(dep)
                if (today, dep_t) >= (now.date(), now.time()):
                    next_dep, next_date = dep, today
                    break
            if next_dep is None and departures:
                next_dep, next_date = departures[0], today + timedelta(days=1)

            if next_dep is None:
                continue
            dep_dt = datetime.combine(next_date, dtime.fromisoformat(next_dep))
            minutes = max(0, int((dep_dt - now).total_seconds() // 60))
            results.append({
                "route": name,
                "direction": route.get("direction", name),
                "pickup": route.get("pickup", ""),
                "duration_min": route.get("duration_min"),
                "next_departure": next_dep,
                "date": next_date.isoformat(),
                "minutes_left": minutes,
                "is_today": next_date == today,
                "last_departure": departures[-1] if departures else None,
                "note": route.get("note", ""),
            })

        if not results:
            return {"available": True, "message": "未找到该方向的班车信息，请确认方向（南→北 / 北→南）。"}
        return {"available": True, "routes": results}

    @staticmethod
    def _runs_on_day(route: Dict[str, Any], now: datetime) -> bool:
        """
        按星期过滤班次：route 的 days 字段
          "weekdays"（周一至周五）/ "weekend"（周六周日）/ "all"（默认，每天）。
        节假日/寒暑假班次无法按日期精确判断，由 note 说明。
        """
        days = str(route.get("days", "all")).strip().lower()
        if days == "weekdays":
            return now.weekday() < 5
        if days == "weekend":
            return now.weekday() >= 5
        return True

    @staticmethod
    def _match_direction(route_name: str, direction: str) -> bool:
        """
        方向匹配：
          - "南→北" / "北->南" 格式：按起终点在路线名中的先后顺序精确匹配
            （"南→北" 不会误配 "北校区→南校区" 路线）
          - 其他表述（如"南校区"）：包含匹配路线名
        """
        direction = direction.strip().replace("->", "→")
        if "→" in direction:
            start, _, end = direction.partition("→")
            start, end = start.strip(), end.strip()
            if start and end:
                i_start, i_end = route_name.find(start), route_name.find(end)
                return i_start != -1 and i_end != -1 and i_start < i_end
        return direction in route_name or route_name in direction

    # ── 楼宇 ──────────────────────────────────────────────────────────────────

    def find_building(self, keyword: str) -> List[Dict[str, Any]]:
        """按楼名/别名模糊查询楼宇信息。"""
        buildings = self._data.get("buildings.json", [])
        keyword = (keyword or "").strip()
        if not keyword:
            return buildings
        result = []
        for b in buildings:
            names = [b.get("name", "")] + [a for a in b.get("aliases", [])]
            if any(keyword in n for n in names if n):
                result.append(b)
        return result

    # ── 运动场馆 / 图书馆 ─────────────────────────────────────────────────────

    def list_venues(self, keyword: Optional[str] = None) -> List[Dict[str, Any]]:
        """运动场馆查询，keyword 匹配名称与别名。"""
        venues = self._data.get("venues.json", [])
        if not keyword:
            return venues
        result = []
        for v in venues:
            names = [v.get("name", "")] + [a for a in v.get("aliases", [])]
            if any(keyword in n for n in names if n):
                result.append(v)
        return result

    def library_info(self) -> Dict[str, Any]:
        """图书馆开放时间（多个馆/自习区）。"""
        libs = self._data.get("library.json", [])
        return {"libraries": libs} if libs else {"available": False, "message": "图书馆开放时间数据暂未录入，请以图书馆现场公告为准。"}

    def _freshness(self, filename: str) -> Dict[str, Any]:
        """公开数据缺少官方来源或时效字段时明确降级，避免被误当作实时信息。"""
        raw = self._data.get(filename)
        records = raw.get("routes", []) if isinstance(raw, dict) else (raw or [])
        if isinstance(raw, dict) and raw.get("source_url"):
            return {"source_url": raw.get("source_url"), "checked_at": raw.get("checked_at"), "valid_from": raw.get("valid_from"), "warning": None}
        sample = records[0] if records and isinstance(records[0], dict) else {}
        if sample.get("source_url") and sample.get("checked_at"):
            return {"source_url": sample.get("source_url"), "checked_at": sample.get("checked_at"), "valid_from": sample.get("valid_from"), "warning": None}
        return {"source_url": None, "checked_at": None, "valid_from": None, "warning": "该结构化数据未附可核验官方来源与时效字段，仅作演示参考，请以学校官方公告为准。"}

    def search(self, category: str, keyword: Optional[str] = None) -> Dict[str, Any]:
        """统一查询入口（MCP 工具 / REST API 共用）。"""
        category = (category or "").strip().lower()
        keyword = (keyword or "").strip() or None
        if category == "auto":
            # 无法由确定性数据自身判断用户意图时，统一返回各公开数据源的
            # 最新只读快照。调用方可以结合原始问题选择相关字段；这里不维护
            # 面向单个问法的关键词路由，新增公开数据类别也只需扩展本分支。
            shuttle = self.next_shuttle()
            shuttle["data_freshness"] = self._freshness("shuttle_schedule.json")
            library = self.library_info()
            library["data_freshness"] = self._freshness("library.json")
            return {
                "available": True,
                "query": keyword,
                "shuttle": shuttle,
                "library": library,
                "venues": self.list_venues(),
                "buildings": self.find_building(""),
                "message": "已读取校园公开结构化信息；若数据源未覆盖该事项，请以官方公告为准。",
            }
        if category in ("shuttle", "校车", "班车"):
            result = self.next_shuttle(keyword)
            result["data_freshness"] = self._freshness("shuttle_schedule.json")
            return result
        if category in ("buildings", "building", "楼宇", "教学楼", "楼"):
            return {"buildings": self.find_building(keyword or ""), "data_freshness": self._freshness("buildings.json")}
        if category in ("venues", "场馆", "运动场", "体育馆"):
            return {"venues": self.list_venues(keyword), "data_freshness": self._freshness("venues.json")}
        if category in ("library", "图书馆", "自习"):
            result = self.library_info()
            result["data_freshness"] = self._freshness("library.json")
            return result
        return {"available": False, "message": f"未知类别: {category}（支持 shuttle/buildings/venues/library）"}
