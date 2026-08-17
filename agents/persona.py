"""
领域人格与 Action 行为指引 —— 领域/动作维度的纯描述与纯策略。

职责划分（v4 收口）：
  - Domain（IntentDomain）：只用于挂载人格与 Skills（顾问），不决定工具权限、
    不选择执行实体 —— 执行实体只有 QA/EXECUTOR 两个职责角色（见 roles.py）；
  - Action（IntentAction）：决定怎么处理（执行策略 + 工具读写门禁）。
工具权限的另一半（角色级只读边界）在 roles.py 的 write_allowed。
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional

from core.domains import IntentAction, IntentDomain


# Action 层工具策略（公共工具层的读写门禁，职责划分：domain = 挂载人格/Skills，action = 怎么处理）。
#   - QUERY：只开放只读/查询类工具，禁止状态修改类工具；
#   - REQUEST：允许按需开放完整工具（含执行类）；
#   - GREETING / FEEDBACK：原则上不开放工具；
#   - COMPLAINT / OTHER：保守策略，只开放只读工具；
#   - None（预置请求/兼容路径）：不额外限制，保持原有 allowlist 行为。
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
        return True  # 完整工具：按需执行
    if action in (IntentAction.GREETING, IntentAction.FEEDBACK):
        return False  # 原则上不开放工具
    if action is None:
        return True  # 动作未知：保持原有 allowlist 行为（兼容既有调用方）
    # QUERY / COMPLAINT / OTHER：保守策略，只开放只读工具
    return tool_name not in write_tools


# Action 行为指引（注入 system prompt）。职责划分：
# domain 决定"挂载什么人格/Skills"（顾问），action 决定"怎么处理"（执行策略）。
ACTION_GUIDANCE: Dict[IntentAction, str] = {
    IntentAction.QUERY: "当前意图为查询：请准确查询并如实回答，不要执行任何修改状态的操作（如新增/删除/完成待办）。",
    IntentAction.REQUEST: "当前意图为请求办理：请积极调用工具解决问题，需要执行操作时按用户指令完成。",
    IntentAction.COMPLAINT: "当前意图为投诉/不满：请先识别具体问题点，再给出明确的解决路径或建议，语气克制。",
    IntentAction.GREETING: "当前意图为问候：请简洁友好回应即可，无需调用工具。",
    IntentAction.FEEDBACK: "当前意图为反馈：请简洁回应并感谢反馈，无需调用工具。",
    IntentAction.OTHER: "当前意图不明确：请保守处理，仅基于已有信息回答，不要执行任何修改状态的操作。",
}


# 领域人格（注入 system prompt 的 [领域人格] 段）。领域分类的唯一产物：
# 挂载行为风格 —— 工具可见性与执行实体都与领域无关。
DOMAIN_PERSONA: Dict[IntentDomain, str] = {
    IntentDomain.ACADEMIC: (
        "当前问题属于学业支持：覆盖选课、课表、考试安排、成绩与绩点、重修、转专业、保研。"
        "回答基于西电教务规则和公开常识，步骤清晰、用语克制。"
        "政策、规定、培养方案或转专业问题必须先调用 knowledge_search；检索结果含 source_url 时，"
        "回答必须给出可点击来源链接、文档标题、更新时间与适用范围，不能把单学院规则泛化为全校规则。"
        "用户提供课程成绩与学分时必须调用 calculate_weighted_score，不要自行心算，并明确它不是官方 GPA。"
        "涉及具体成绩或学籍操作时，提示学生前往教务系统或学院教务老师处确认。"
    ),
    IntentDomain.CAMPUS_LIFE: (
        "当前问题属于校园生活：覆盖宿舍、食堂、校园穿梭车、校园卡、快递、水电、社团、运动场馆。"
        "图书馆/场馆位置或开放时间、校车班次必须先调用 query_campus_info；天气问题必须先调用 get_weather，"
        "包括依赖上一轮实体的「那几点关门」等短追问。回答尽量给出位置（校区/楼栋）和时段。"
        "涉及报修、补办等需现场办理的事项，指引用户到对应服务网点。"
    ),
    IntentDomain.AFFAIRS: (
        "当前问题属于校务办事：覆盖校历、请假流程、奖学金与助学金、各类证明开具、学籍注册、学费缴纳。"
        "回答以办事流程、所需材料、办理地点和系统入口为主，清晰可执行。"
        "校园卡补办、请假、在读证明或缓考问题优先调用 query_affairs_process 获取版本化流程。"
        "涉及实际审批的事项，提示以学院或学生处最新通知为准。"
    ),
    IntentDomain.IT_HELP: (
        "当前问题属于 IT 支持：覆盖教务系统、校园网、VPN、学校邮箱、统一身份认证的故障排查。"
        "遇到上述系统故障时优先调用 diagnose_it_issue，再基于诊断树组织回答，给出清晰的步骤化解决方案。"
        "遇到需要后台操作或账号重置的问题，说明需联系信息化建设处或网络中心处理。"
    ),
    IntentDomain.PERSONAL: (
        "当前问题属于个人事务：覆盖用户自己的课表、待办、考试与 DDL 安排。"
        "查询前先调用工具获取用户个人数据（query_schedule / query_todo / query_ddl 等），"
        "不要凭记忆编造课程或待办。用户未导入课表时，引导其通过「我的课表」上传 .ics 文件或 JSON 课表。"
        "回答按时间组织，带上课时间与地点；涉及考试/DDL 时给出剩余天数。"
    ),
    IntentDomain.OTHER: (
        "当前问题不属于校园领域（如通用知识、编程、GitHub 等外部工具问题）："
        "以通用助手的方式直接回答，可用公共工具（含外部只读工具）辅助，保持简洁准确。"
    ),
}
