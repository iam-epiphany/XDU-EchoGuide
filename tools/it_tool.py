"""ITHelpAgent 的确定性故障诊断工具。"""
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
    return public_dir / "it_diagnostics.json"


@functools.lru_cache(maxsize=1)
def _load_rules() -> List[Dict[str, Any]]:
    """进程内缓存数据文件（更新后需重启生效）。"""
    with _data_path().open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("IT 诊断数据格式错误")
    return data


async def diagnose_it_issue_handler(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    system = str(params.get("system") or "").strip().lower()
    symptom = str(params.get("symptom") or "").strip().lower()
    error_code = str(params.get("error_code") or "").strip().lower()
    if not system and not symptom:
        raise ValueError("system 和 symptom 至少填写一项")

    ranked = []
    for rule in _load_rules():
        tokens = [str(token).lower() for token in rule.get("match", [])]
        score = sum(2 if token in system else 1 for token in tokens if token in f"{system} {symptom} {error_code}")
        if score:
            ranked.append((score, rule))
    ranked.sort(key=lambda item: item[0], reverse=True)
    if not ranked:
        return {
            "matched": False,
            "message": "未命中确定性诊断规则，建议继续使用 IT 知识库排查。",
            "contact": "西安电子科技大学信息网络技术中心",
        }
    best = ranked[0][1]
    return {
        "matched": True,
        "system": system,
        "symptom": symptom,
        "diagnosis": best,
    }
