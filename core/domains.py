"""
EchoGuide 领域词表 —— 全系统唯一的领域定义与关键词来源。

设计动机（面试点）：
  早期版本中领域关键词在 intent_recognizer / orchestrator._collaboration_targets /
  api._should_use_knowledge 三处重复维护，必然漂移。
  本模块将「领域定义 + 关键词」收敛为单一事实来源，所有匹配逻辑统一引用。

同时修正两类匹配缺陷：
  1. 子串误命中：英文关键词必须整词匹配（\b 词边界），避免 "api" 命中 "capital"；
     中文关键词禁止单字（如旧版 "餐" 会命中 "餐补" 等无关场景），统一使用 ≥2 字词组。
  2. 单次命中评分：不再用 hits/len(keywords)（关键词多的领域永远低分），
     改为「首命中 0.55、每多命中 +0.2、上限 0.95」的边际衰减评分。
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Dict, List


class IntentDomain(Enum):
    """领域维度 —— 人格/Skills 挂载键（顾问，不做 Agent 路由）。

    执行实体只有 QA/EXECUTOR 两个职责角色（agents/roles.py 的 Role）；
    领域只决定挂载什么人格与 Skills，不决定工具可见性、不选执行角色。
    """
    ACADEMIC    = "academic"      # 学业支持
    CAMPUS_LIFE = "campus_life"   # 校园生活
    AFFAIRS     = "affairs"       # 校务咨询
    IT_HELP     = "it_help"       # IT 支持
    PERSONAL    = "personal"      # 个人助理（课表/待办/日程提醒）
    OTHER       = "other"


class IntentAction(Enum):
    """动作维度 —— 决定行为（查询/请求/问候/投诉/反馈等）。"""
    QUERY     = "query"       # 信息查询
    REQUEST   = "request"     # 请求操作
    GREETING  = "greeting"    # 问候
    COMPLAINT = "complaint"   # 投诉不满
    FEEDBACK  = "feedback"    # 正面反馈
    OTHER     = "other"


# ── 领域关键词（单一事实来源）──────────────────────────────────────────────
# 注意：中文关键词全部 ≥2 字；英文关键词匹配时按整词（词边界）处理。
DOMAIN_KEYWORDS: Dict[IntentDomain, List[str]] = {
    IntentDomain.ACADEMIC: [
        "选课", "考试", "成绩", "绩点", "学分", "重修", "保研", "转专业",
        "挂科", "补考", "培养方案", "先修课", "培养计划", "退改选", "期末考试",
    ],
    IntentDomain.CAMPUS_LIFE: [
        "食堂", "餐厅", "早餐", "午餐", "晚餐", "宿舍", "校车", "班车", "校园卡",
        "快递", "水电", "超市", "运动场", "体育馆", "社团", "充值",
        "门禁", "报修", "南校区", "北校区", "通勤", "图书馆", "自习",
    ],
    IntentDomain.AFFAIRS: [
        "校历", "请假", "奖学金", "助学金", "证明", "在读证明", "缴费", "学费",
        "注册", "学籍", "学生处", "教务处", "办事", "流程", "盖章", "假期",
        "校园卡补办", "补卡", "挂失", "补办", "缓考",
    ],
    IntentDomain.IT_HELP: [
        "教务系统", "校园网", "vpn", "邮箱", "统一身份认证", "登录不上", "报错",
        "密码重置", "验证码", "网络连不上", "无法访问", "账号", "激活", "配置",
        "证书", "重置密码",
    ],
    # 个人助理：课表类词归个人领域（"我的课表/今天有什么课"），
    # academic 只保留教务规则类词（选课流程/绩点算法）。
    # "考试" 两域共有：个人化表述（考试安排/倒计时）由个人化词组加权胜出，
    # 教务规则类（期末考试时间）仍由 academic 的多词命中胜出。
    IntentDomain.PERSONAL: [
        "课表", "课程表", "我的课表", "日程", "待办", "提醒", "作业", "考试",
        "考试安排", "ddl", "上课", "下课", "什么课", "第几节", "几点上课",
        "几点下课", "安排",
    ],
}


# 动作关键词（领域无关的通用模式，只用于 action 维度兜底）
# 重新定义（v4 收口）：REQUEST = 需要系统真正执行写操作或产生副作用；
# "帮我/我要/需要/办理" 等请求句式不再直接判 REQUEST——
#   "帮我查一下课表" → QUERY（咨询，不产生状态修改）
#   "帮我添加一个补办校园卡的待办" → REQUEST（写操作词"添加"）
# 写操作词（添加/删除/标记/记一下/提醒我…）权重高；查询词（怎么/什么/几点…）权重低。
ACTION_KEYWORDS: Dict[IntentAction, List[str]] = {
    IntentAction.COMPLAINT:  ["太差", "糟糕", "等了很久", "一直没人", "投诉", "不满"],
    IntentAction.QUERY:      ["?", "？", "怎么", "什么", "几点", "在哪", "什么时候", "如何", "哪里", "多少", "查一下", "查询", "看看", "帮我查", "怎么办"],
    IntentAction.REQUEST:    ["添加", "删除", "标记", "记一下", "记个", "提醒我", "创建", "新增", "设置", "设为", "帮我记", "完成"],
    IntentAction.GREETING:   ["你好", "您好", "嗨", "hello", "hi", "在吗", "早上好", "晚上好"],
}

# 动作关键词权重：写操作词 > 查询词（"帮我查" 是查询不是请求，避免请求句式吞掉查询）
ACTION_KEYWORD_WEIGHTS: Dict[str, float] = {
    "?": 0.2, "？": 0.2, "怎么": 0.3, "什么": 0.3, "几点": 0.35, "在哪": 0.35,
    "什么时候": 0.35, "如何": 0.3, "哪里": 0.35, "多少": 0.35, "怎么办": 0.35,
    "查一下": 0.55, "查询": 0.55, "看看": 0.5, "帮我查": 0.6,
    "添加": 0.7, "删除": 0.7, "标记": 0.7, "记一下": 0.7, "记个": 0.7,
    "提醒我": 0.7, "创建": 0.7, "新增": 0.7, "设置": 0.65, "设为": 0.65,
    "帮我记": 0.7, "完成": 0.6,
    "你好": 0.7, "您好": 0.7, "嗨": 0.7, "hello": 0.7, "hi": 0.7, "在吗": 0.65,
    "早上好": 0.7, "晚上好": 0.7,
    "太差": 0.6, "糟糕": 0.6, "等了很久": 0.6, "一直没人": 0.6, "投诉": 0.7, "不满": 0.6,
}

# 与 SkillManager 保持一致：内部注入逻辑只依赖这里的数据，不再各自维护关键词。
# 例如 skill 的 keywords 中单字 "餐" 已被替换为多字词组（见 skills/campus_life/SKILL.md）。


def domain_hit_score(message: str) -> tuple[IntentDomain | None, float]:
    """
    计算消息对每个领域的命中情况，返回 (最佳领域, 评分)。

    评分规则（修正旧版 hits/len(kws) 缺陷）：
      首个命中 0.55，每多命中一个关键词 +0.2，上限 0.95。
    """
    msg = (message or "").lower()

    # 高精度业务短语优先于通用词计数。它们既让级联分类器可以零 LLM
    # 直达，也解决“宿舍 + 校园网”“校园卡 + 补办”这类跨词表平局。
    if "待办" in msg and any(word in msg for word in ("课表", "课程", "空闲", "上课", "没课", "有空")):
        return IntentDomain.PERSONAL, 0.95
    if "校园卡" in msg and any(word in msg for word in ("丢", "挂失", "补办", "补卡")):
        return IntentDomain.AFFAIRS, 0.95
    if any(word in msg for word in ("校园网", "统一身份认证", "教务系统")) or keyword_hit("vpn", msg):
        return IntentDomain.IT_HELP, 0.95
    if any(word in msg for word in ("加权成绩", "加权学分", "平均成绩")):
        return IntentDomain.ACADEMIC, 0.95
    if any(word in msg for word in ("我的课表", "什么课", "待办", "课程表")):
        return IntentDomain.PERSONAL, 0.95
    if any(word in msg for word in ("天气", "校车", "班车", "图书馆")):
        return IntentDomain.CAMPUS_LIFE, 0.95

    best_domain: IntentDomain | None = None
    best_score = 0.0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(1 for kw in keywords if keyword_hit(kw, msg))
        if hits:
            score = min(0.95, 0.55 + 0.2 * (hits - 1))
            if score > best_score:
                best_domain, best_score = domain, score
    return best_domain, best_score


def action_hit_score(message: str) -> tuple[IntentAction | None, float]:
    """
    计算动作维度命中。领域词优先于通用疑问词，避免动作吞掉领域信息。

    动作关键词带权重（ACTION_KEYWORD_WEIGHTS）：
      显式意图词（帮我/我要/投诉）权重高；
      通用疑问词（怎么/什么/几点）权重低，避免平局时疑问词抢占请求语义。
    """
    msg = (message or "").lower()
    best_action: IntentAction | None = None
    best_score = 0.0
    for action, keywords in ACTION_KEYWORDS.items():
        score = sum(
            ACTION_KEYWORD_WEIGHTS.get(kw, 0.5) * (1 if keyword_hit(kw, msg) else 0)
            for kw in keywords
        )
        score = min(score, 0.95)
        if score > 0.0 and score > best_score:
            best_action, best_score = action, score
    return best_action, best_score


# ── 关键词匹配 ────────────────────────────────────────────────────────────────

_ASCII_RE_CACHE: Dict[str, re.Pattern] = {}


def keyword_hit(keyword: str, text: str) -> bool:
    """
    关键词命中检测（修正子串误命中）：

    - ASCII 关键词（如 api / vpn / it）必须整词出现。
      用 ASCII 字符类前后向断言（而非 \b —— \b 会把中文也当作词字符，
      导致 "vpn配置" 这类中英混合文本匹配失败）。这样 "capital" 不再命中 "api"。
    - 中文关键词（≥2 字）按子串匹配。
    - 中文单字关键词视为非法配置，静默不匹配（防止过拟合误命中）。
    """
    keyword = (keyword or "").strip().lower()
    if not keyword:
        return False
    if keyword.isascii():
        pattern = _ASCII_RE_CACHE.get(keyword)
        if pattern is None:
            # (?<![a-zA-Z0-9_]) 等价于"ASCII 词边界"：仅对 ASCII 字母数字下划线生效
            pattern = re.compile(
                rf"(?<![a-zA-Z0-9_]){re.escape(keyword)}(?![a-zA-Z0-9_])"
            )
            _ASCII_RE_CACHE[keyword] = pattern
        return bool(pattern.search(text))
    if len(keyword) < 2:
        return False  # 中文单字关键词过拟合，禁止使用
    return keyword in text
