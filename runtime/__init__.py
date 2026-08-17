"""
EchoGuide Agent Runtime（Harness 收口层）。

模型之外的整套控制面统一收口到这里：
  - RunState：单次运行的执行状态（身份、计数器、错误、middleware 扩展位）
  - ExecutionPolicy：执行预算与策略（Agent/Task/工具轮次/合成 token/Guard）
  - RuntimeMiddleware / MiddlewareChain：生命周期钩子（run/model/tool/finish）
  - AgentRuntime：运行入口，编排器核心逻辑作为 core 在中间件链内执行

职责边界：业务 Agent 保持薄；可靠性（Guard、预算、Trace、Skill 解析）由
Runtime 保证。HTTP 层（Guard ASGI、语义缓存、记忆读写）不迁移到本层。
"""
from runtime.context import RunContext
from runtime.middleware import (
    BudgetExceeded, GuardRejection, MiddlewareChain, RuntimeMiddleware,
)
from runtime.policy import ExecutionPolicy
from runtime.runtime import AgentRuntime, default_middlewares
from runtime.state import RunState

__all__ = [
    "AgentRuntime",
    "BudgetExceeded",
    "ExecutionPolicy",
    "GuardRejection",
    "MiddlewareChain",
    "RunContext",
    "RunState",
    "RuntimeMiddleware",
    "default_middlewares",
]
