"""
执行 Profile（Fast / Deep）—— 真实执行配置与选择决策。

职责：把「复杂度判定 + 置信度 + 关键词信号」映射为可执行的
模型/预算/检索深度配置。只做决策与描述，不执行（执行在 roles.py）。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ProfileName(Enum):
    FAST = "fast"
    DEEP = "deep"


@dataclass(frozen=True)
class ExecutionProfile:
    """真实执行配置：模型、思考模式、生成预算与检索深度。"""
    name: ProfileName
    model: str
    max_tokens: int
    thinking: bool
    rag_top_k: int
    use_rewrite: bool
    use_rerank: bool


def select_profile_name(
    complexity_mode: str,
    message: str,
    classifier_stage: str,
    confidence: float,
) -> ProfileName:
    """Profile 决策（纯函数）：复杂度/关键词/置信度 → Fast or Deep。

    - 非 single（parallel/dependent）→ Deep（协作链必须深度路径）；
    - 复杂语义关键词（政策/保研/比较等）→ Deep；
    - 确定性工具关键词（课表/校车/待办等）→ Fast；
    - 低置信度（< 0.80，与 intent_recognizer 的 embedding_threshold 联动）→ Deep。
    """
    if complexity_mode != "single":
        return ProfileName.DEEP
    deep_markers = ("转专业", "保研", "培养方案", "政策", "规定", "条件", "比较", "分析", "检索资料", "给出来源")
    if any(marker in message for marker in deep_markers):
        return ProfileName.DEEP
    deterministic_markers = (
        "加权", "平均成绩", "校园卡", "请假", "在读证明", "缓考",
        "校园网", "vpn", "统一身份认证", "教务系统", "课表", "待办",
        "校车", "天气", "图书馆", "体育馆",
    )
    if any(marker in message.lower() for marker in deterministic_markers):
        return ProfileName.FAST
    # 置信度低于 Embedding 命中线（0.80）视为低置信度 → DEEP，
    # 与 intent_recognizer 的 embedding_threshold 联动（改阈值时同步改这里）
    if classifier_stage == "llm" or (confidence and confidence < 0.80):
        return ProfileName.DEEP
    return ProfileName.FAST
