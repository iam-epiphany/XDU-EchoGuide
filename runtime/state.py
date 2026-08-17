"""
RunState —— Agent Runtime 的单次运行状态载体。

一次请求（/chat、/chat/stream、CLI）对应一个 RunState：身份信息、路由结果、
执行计数器、错误记录与 middleware 自由扩展的 meta。执行摘要由 summary() 输出，
合并进 execution 元数据，是可观测性的统一出口（debug 面板可见）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RunState:
    """单次运行的执行状态（由 AgentRuntime.run 创建，贯穿整条请求链路）。"""

    request_id: str
    user_id: str = ""
    conv_id: str = ""
    trace_id: str = ""               # 与请求级 Trace 对齐（TraceMiddleware 回填）
    message: str = ""
    profile: str = ""                # fast / deep（路由后回填）
    complexity_mode: str = ""        # single / parallel / dependent（路由后回填）

    started_at: float = field(default_factory=time.monotonic)

    # 执行计数器（BudgetMiddleware 等中间件维护）
    step_count: int = 0              # 模型调用次数（before_model）
    tool_call_count: int = 0         # 工具调用次数（before_tool）
    tool_round_count: int = 0        # 工具调用轮次
    retry_count: int = 0             # 失败降级次数（Fast→Deep）
    input_tokens: int = 0
    output_tokens: int = 0

    # 错误记录（Guard 拒绝、预算超限、middleware 异常）
    errors: List[str] = field(default_factory=list)

    # middleware 自由扩展位（如 SkillMiddleware 的 skill_prompt_by_msg）
    meta: Dict[str, Any] = field(default_factory=dict)

    # 执行策略引用（BaseAgent 等执行层读取预算，缺省 None 时回落旧常量）
    policy: Optional[Any] = None

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def summary(self) -> Dict[str, Any]:
        """运行摘要（execution meta 的纯增量字段，不透出敏感信息）。"""
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "steps": self.step_count,
            "tool_calls": self.tool_call_count,
            "tool_rounds": self.tool_round_count,
            "retries": self.retry_count,
            "errors": list(self.errors),
        }
