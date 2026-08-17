"""
具体中间件：Trace / Guard / Budget / Skill。

每个中间件只做一件事，注册进 AgentRuntime 后即对整个链路生效；
无状态设计（计数与缓存都落在 RunState 上），可安全复用同一实例。
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from runtime.context import RunContext
from runtime.middleware import BudgetExceeded, GuardRejection, RuntimeMiddleware

logger = logging.getLogger(__name__)


class TraceMiddleware(RuntimeMiddleware):
    """Trace 对齐：把请求级 trace_id 写入 RunState，保证 execution 可下钻到 Trace。"""

    name = "trace"

    async def before_run(self, ctx: RunContext) -> None:
        from core.tracing import begin_trace, current_trace

        trace = current_trace()
        if trace is None:
            trace = begin_trace("runtime")  # CLI / 评测等无请求级 trace 的路径
        ctx.state.trace_id = trace.trace_id


class GuardMiddleware(RuntimeMiddleware):
    """Runtime 层 Guard：消息长度 + Prompt 注入检测（CLI/内部调用同样受保护）。

    与 HTTP 层 EchoGuardMiddleware 的差异：HTTP 层失败关闭返回 503；Runtime 层
    检测不可用时放行（不阻断主链路），两者职责互补、互不替代。
    """

    name = "guard"

    def __init__(self, enabled: bool = True, max_message_chars: int = 2000):
        self._enabled = enabled
        self._max_message_chars = max_message_chars

    async def before_run(self, ctx: RunContext) -> None:
        if not self._enabled:
            return
        message = ctx.state.message or ""
        if len(message) > self._max_message_chars:
            raise GuardRejection(f"请求内容过长：上限 {self._max_message_chars} 字")
        try:
            from echoguide_guard.integration import find_injection_text

            hit = find_injection_text(message)
        except Exception as ex:
            logger.warning(f"runtime guard 检测不可用，放行: {ex}")
            return
        if hit:
            raise GuardRejection(f"检测到疑似指令注入（{hit}），请求已被拦截")


class BudgetMiddleware(RuntimeMiddleware):
    """执行预算计数：真实模型调用步数 / 工具调用次数；超限时抛出。

    计数发生在真实执行边界（ModelGateway 每次模型调用触发一次
    before_model），step_count 因此是模型调用次数而非 Agent handle 次数。
    """

    name = "budget"

    def __init__(self, max_tool_calls: int = 0, max_model_calls: int = 0):
        self._max_tool_calls = max_tool_calls  # 0 = 仅计数不强限
        self._max_model_calls = max_model_calls  # 0 = 仅计数不强限

    async def before_model(self, ctx: RunContext) -> None:
        # 先检查后计数：被预算拦截的调用不算"真实模型调用"（provider 未执行）
        if self._max_model_calls > 0 and ctx.state.step_count >= self._max_model_calls:
            raise BudgetExceeded(f"单请求模型调用超过预算上限 {self._max_model_calls} 次")
        ctx.state.step_count += 1

    async def before_tool(self, ctx: RunContext, tool_name: str, tool_input: Any) -> None:
        ctx.state.tool_call_count += 1
        if self._max_tool_calls > 0 and ctx.state.tool_call_count > self._max_tool_calls:
            raise BudgetExceeded(f"单请求工具调用超过预算上限 {self._max_tool_calls} 次")


class SkillMiddleware(RuntimeMiddleware):
    """Skill 解析缓存：before_model 按消息指纹解析一次并缓存到 state，
    注入点（BaseAgent._build_system_prompt）优先读缓存 —— 解析结果全链路一致。

    v4：Skills 不再按领域/角色过滤，模型看全部目录自主选择；命中提示依赖
    消息与历史，因此缓存键 = 消息 + 最近对话指纹（SkillManager.cache_key）。
    """

    name = "skill"

    async def before_model(self, ctx: RunContext) -> None:
        skill_manager = ctx.services.get("skill_manager")
        if skill_manager is None:
            return
        history = ctx.services.get("history")
        key = skill_manager.cache_key(ctx.state.message or "", history)
        prompts = ctx.state.meta.setdefault("skill_prompt_by_msg", {})
        if key in prompts:
            return
        try:
            prompt = skill_manager.prompt_for(ctx.state.message or "", None, history)
        except Exception as ex:
            logger.warning(f"skill 解析失败: {ex}")
            return
        prompts[key] = prompt or ""
