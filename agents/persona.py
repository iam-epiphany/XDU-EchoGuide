"""
领域人格与 Action 行为指引 —— 领域/动作维度的纯描述与纯策略。

职责划分（v5 收口）：
  - Domain（IntentDomain）：只提供业务语境与回答侧重点，不决定 Skill 或工具权限、
    不选择执行实体 —— 真正的 Agent 单位是 Task（TaskAgent Run，见 roles.py）；
  - Action（IntentAction）：决定怎么处理（执行策略 + 工具读写门禁）。
工具权限的另一半（Run 级写策略：非 REQUEST 一律 READ_ONLY）在
roles.py 的 write_policy_for。
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional

from core.domains import IntentAction, IntentDomain


# Action 层工具策略（公共工具层的读写门禁，职责划分：domain = 语境，action = 怎么处理）。
#   - QUERY：只开放只读/查询类工具，禁止状态修改类工具；
#   - REQUEST：允许按需开放写工具（任务级 allowed_write_tools 另行白名单）；
#   - GREETING / FEEDBACK：原则上不开放工具；
#   - COMPLAINT / OTHER：保守策略，只开放只读工具；
#   - None（预置请求/兼容路径）：不额外限制（Run 级写策略 write_policy_for
#     仍会在更外层兜底：非 REQUEST 一律 READ_ONLY）。
# 写工具集合由各工具自身声明（Tool.write=True，见 mcp/tool_manager.py 的 write_tools()），
# 不再手工维护黑名单 —— 新增写工具忘记声明时，只读动作下会被误开放的是它自己，
# 由门禁检查方从声明推导，漏声明直接不可写（fail-closed）。
def action_allows_tool(
    action: Optional[IntentAction],
    tool_name: str,
    write_tools: FrozenSet[str] = frozenset(),
) -> bool:
    """Action 层工具过滤（纯函数）：返回该动作下工具是否可暴露/执行。

    write_tools 由调用方从工具管理器推导（Tool.write 声明），缺省空集合
    （无写工具上下文时保守视为全只读）。
    """
    if action == IntentAction.REQUEST:
        return True  # 写工具：按需执行（任务级写能力白名单另行约束）
    if action in (IntentAction.GREETING, IntentAction.FEEDBACK):
        return False  # 原则上不开放工具
    if action is None:
        return True  # 动作未知：保持原有 allowlist 行为（兼容既有调用方）
    # QUERY / COMPLAINT / OTHER：保守策略，只开放只读工具
    return tool_name not in write_tools


# Action 行为指引（注入 system prompt）。职责划分：
# Domain 只提供语境，Action 决定执行策略。
ACTION_GUIDANCE: Dict[IntentAction, str] = {
    IntentAction.QUERY: "当前意图为查询：请准确查询并如实回答，不要执行任何修改状态的操作（如新增/删除/完成待办）。",
    IntentAction.REQUEST: "当前意图为请求办理：请积极调用工具解决问题，需要执行操作时按用户指令完成。",
    IntentAction.COMPLAINT: "当前意图为投诉/不满：请先识别具体问题点，再给出明确的解决路径或建议，语气克制。",
    IntentAction.GREETING: "当前意图为问候：请简洁友好回应即可，无需调用工具。",
    IntentAction.FEEDBACK: "当前意图为反馈：请简洁回应并感谢反馈，无需调用工具。",
    IntentAction.OTHER: "当前意图不明确：请保守处理，仅基于已有信息回答，不要执行任何修改状态的操作。",
}


# 领域人格（注入 system prompt 的 [领域人格] 段）。只包含业务语境、回答侧重点
# 与风格；具体 SOP、工具调用时机与权限均由 Skill / Runtime / Action 管理。
DOMAIN_PERSONA: Dict[IntentDomain, str] = {
    IntentDomain.ACADEMIC: (
        "当前问题属于学业支持语境，关注课程、考试、成绩与培养路径。"
        "回答应准确、结构清晰，并区分校级规则与学院规则。"
    ),
    IntentDomain.CAMPUS_LIFE: (
        "当前问题属于校园生活语境，关注地点、校区、开放时段和日常服务。"
        "回答应便于到达和执行，优先给出清晰的空间与时间信息。"
    ),
    IntentDomain.AFFAIRS: (
        "当前问题属于校务办事语境，关注流程、材料、部门与时效。"
        "回答应按办理顺序组织，并明确哪些信息需要向官方确认。"
    ),
    IntentDomain.IT_HELP: (
        "当前问题属于校园 IT 支持语境，关注现象、影响范围和可复现条件。"
        "回答应由低风险检查逐步收敛，并清楚区分自助处理与人工支持。"
    ),
    IntentDomain.PERSONAL: (
        "当前问题属于个人事务语境，关注用户自己的课程、待办与时间安排。"
        "回答应围绕时间顺序、优先级和用户当前数据展开。"
    ),
    IntentDomain.OTHER: (
        "当前问题不属于校园领域（如通用知识、编程、GitHub 等外部工具问题）："
        "以通用助手的方式直接回答，可用公共工具（含外部只读工具）辅助，保持简洁准确。"
    ),
}
