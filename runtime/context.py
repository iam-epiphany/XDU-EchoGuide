"""
RunContext —— 单次运行的全量上下文：状态 + 策略 + 服务 + 事件出口。

业务执行体（编排器 core、Agent 工具循环）与中间件共享同一个 RunContext：
  - state：RunState（计数器 / 错误 / meta 扩展位）
  - policy：ExecutionPolicy（预算查询）
  - services：共享组件（skill_manager、req 等），由触发方注入
  - on_event：SSE 过程事件透传
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class RunContext:
    """一次 Agent 运行的上下文（中间件与执行体共享）。"""

    state: Any                                        # RunState
    policy: Any                                       # ExecutionPolicy
    services: Dict[str, Any] = field(default_factory=dict)
    on_event: Optional[Callable[[Dict[str, Any]], Any]] = None  # SSE 过程事件
