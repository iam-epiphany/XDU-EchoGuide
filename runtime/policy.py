"""
ExecutionPolicy —— Agent Runtime 的执行预算与策略。

大部分预算项默认值 = 收口前的魔法数字（3 Agent / 6 Task / 1024 合成 token）；
工具轮次从统一 2 升级为 Fast/Deep 分级上限（3/5），并新增无进展检测
（stagnant_round_limit）：护栏由"硬轮次"变为"分级上限 + 重复检测双保险"。
所有项可通过 ECHOGUIDE_RUNTIME_* 环境变量或代码注入覆盖。
frozen：运行期不可变，换策略即换实例。
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip())
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ExecutionPolicy:
    """执行预算与策略（默认值与原魔法数字一一对应）。"""

    max_agents: int = 3                  # 协作目标 Agent 上限（原 targets[:3]）
    max_tasks: int = 6                   # 协作任务 DAG 上限（原 6）
    max_tool_rounds_fast: int = 3        # Fast 路径工具轮次上限（保险丝；LLM 无 tool_use 即停）
    max_tool_rounds_deep: int = 5        # Deep 路径工具轮次上限（复杂任务留给 Deep）
    stagnant_round_limit: int = 2        # 无进展检测：连续 N 轮工具调用与上一轮完全重复 → 强制收尾
    max_model_calls: int = 0             # 单请求真实模型调用次数上限（0 = 仅计数不强限）
    max_tool_calls: int = 0              # 单请求工具调用总次数上限（0 = 仅计数不强限）
    max_retries: int = 1                 # 失败降级次数上限（Fast→Deep）
    synth_max_tokens: int = 1024         # 协作合成器输出预算（原硬编码 1024）
    guard_enabled: bool = True           # Runtime 层 Guard（CLI/内部调用同样受保护）
    guard_max_message_chars: int = 2000  # 单条消息长度上限（与 HTTP 层 GuardSettings 对齐）
    verifier_llm_enabled: bool = False   # 出口校验 LLM 判定（规则校验始终开启；仅 DEEP/执行路径）

    @classmethod
    def from_env(cls) -> "ExecutionPolicy":
        """从 ECHOGUIDE_RUNTIME_* 环境变量构建（缺省回落默认值）。"""
        return cls(
            max_agents=_int_env("ECHOGUIDE_RUNTIME_MAX_AGENTS", 3),
            max_tasks=_int_env("ECHOGUIDE_RUNTIME_MAX_TASKS", 6),
            max_tool_rounds_fast=_int_env("ECHOGUIDE_RUNTIME_MAX_TOOL_ROUNDS_FAST", 3),
            max_tool_rounds_deep=_int_env("ECHOGUIDE_RUNTIME_MAX_TOOL_ROUNDS_DEEP", 5),
            stagnant_round_limit=_int_env("ECHOGUIDE_RUNTIME_STAGNANT_ROUND_LIMIT", 2),
            max_model_calls=_int_env("ECHOGUIDE_RUNTIME_MAX_MODEL_CALLS", 0),
            max_tool_calls=_int_env("ECHOGUIDE_RUNTIME_MAX_TOOL_CALLS", 0),
            max_retries=_int_env("ECHOGUIDE_RUNTIME_MAX_RETRIES", 1),
            synth_max_tokens=_int_env("ECHOGUIDE_RUNTIME_SYNTH_MAX_TOKENS", 1024),
            guard_enabled=_bool_env("ECHOGUIDE_RUNTIME_GUARD_ENABLED", True),
            guard_max_message_chars=_int_env(
                "ECHOGUIDE_RUNTIME_GUARD_MAX_MESSAGE_CHARS", 2000
            ),
            verifier_llm_enabled=_bool_env("ECHOGUIDE_RUNTIME_VERIFIER_LLM", False),
        )
