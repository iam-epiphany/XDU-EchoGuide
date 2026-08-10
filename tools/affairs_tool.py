"""AffairsAgent 的版本化办事流程查询工具。"""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Dict, List


def _data_path() -> Path:
    """与 campus/store、knowledge_base 共用 ECHOGUIDE_PUBLIC_DATA_DIR 配置。"""
    default_dir = Path(__file__).resolve().parents[1] / "data" / "public"
    public_dir = Path(os.getenv("ECHOGUIDE_PUBLIC_DATA_DIR", str(default_dir)))
    return public_dir / "affairs_processes.json"


@functools.lru_cache(maxsize=1)
def _load_processes() -> List[Dict[str, Any]]:
    """进程内缓存数据文件（更新后需重启生效；流程数据变更频率极低）。"""
    with _data_path().open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("办事流程数据格式错误")
    return data


async def query_affairs_process_handler(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    service = str(params.get("service") or "").strip().lower()
    if not service:
        raise ValueError("service 不能为空")
    processes = _load_processes()
    matches = []
    for item in processes:
        aliases = [str(alias).lower() for alias in item.get("aliases", [])]
        haystack = " ".join([
            str(item.get("id", "")),
            str(item.get("name", "")),
            *aliases,
        ]).lower()
        # aliases 已统一 lower，与 service/haystack 大小写一致
        if service in haystack or any(token in service for token in aliases):
            matches.append(item)
    if not matches:
        return {
            "found": False,
            "service": service,
            "available_services": [item.get("name") for item in processes],
            "message": "结构化流程库暂未覆盖该事项，可继续检索校园知识库。",
        }
    return {"found": True, "processes": matches[:3]}
