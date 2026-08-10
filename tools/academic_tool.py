"""AcademicAgent 的确定性学业计算工具。"""
from __future__ import annotations

from typing import Any, Dict, List


async def calculate_weighted_score_handler(params: Dict[str, Any], context: Any) -> Dict[str, Any]:
    courses = params.get("courses")
    if not isinstance(courses, list) or not courses:
        raise ValueError("courses 必须是非空课程数组")
    if len(courses) > 50:
        raise ValueError("一次最多计算 50 门课程")

    contributions: List[Dict[str, Any]] = []
    total_credits = 0.0
    weighted_sum = 0.0
    for index, item in enumerate(courses, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {index} 门课程格式错误")
        name = str(item.get("name") or f"课程{index}").strip()[:80]
        try:
            credits = float(item["credits"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError) as ex:
            raise ValueError(f"{name} 的学分或成绩无效") from ex
        if credits <= 0 or credits > 30:
            raise ValueError(f"{name} 的学分必须在 0 到 30 之间")
        if score < 0 or score > 100:
            raise ValueError(f"{name} 的成绩必须在 0 到 100 之间")
        contribution = credits * score
        total_credits += credits
        weighted_sum += contribution
        contributions.append({
            "name": name,
            "credits": round(credits, 2),
            "score": round(score, 2),
            "weighted_contribution": round(contribution, 2),
        })

    return {
        "formula": "Σ(课程成绩×课程学分)/Σ课程学分",
        "total_credits": round(total_credits, 2),
        "weighted_score": round(weighted_sum / total_credits, 2),
        "contributions": contributions,
        "disclaimer": "这是加权学分成绩计算，不是学校官方 GPA 换算结果。",
    }
