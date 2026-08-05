"""用户画像信号检测测试。

画像提炼的成本控制核心：只有用户消息包含"偏好/背景声明"（画像信号）时
才调用 LLM 提炼，普通提问不触发。本测试验证信号检测的纯逻辑。
"""
from __future__ import annotations

from memory.conversation_memory import Message, MsgRole, _has_profile_signal


def _msgs(*texts) -> list:
    """构造消息列表，确保最后一条是用户消息（模拟"最近一条用户消息"）。"""
    msgs = [
        Message(role=MsgRole.USER if i % 2 == 0 else MsgRole.ASSISTANT, content=t)
        for i, t in enumerate(texts)
    ]
    if msgs and msgs[-1].role != MsgRole.USER:
        msgs[-1] = Message(role=MsgRole.USER, content=texts[-1])
    return msgs


def test_signal_detected_for_preference_statements():
    """含偏好/背景声明的消息应触发画像提炼。"""
    signals = [
        "我喜欢打羽毛球",
        "我是通信工程学院的学生",
        "我在南校区，住竹园公寓",
        "我大三了",
        "我经常去图书馆自习",
        "我打算考研",
        "我的专业是软件工程",
    ]
    for text in signals:
        assert _has_profile_signal(_msgs("你好", text)), f"应命中信号: {text}"


def test_no_signal_for_plain_questions():
    """普通提问不应触发画像提炼（省成本）。"""
    plain = [
        "南校区食堂几点关门？",
        "这学期选课什么时候开始？",
        "校车下一班几点？",
        "教务系统登录不上怎么办？",
        "帮我查一下今天的课",
    ]
    for text in plain:
        assert not _has_profile_signal(_msgs("你好", text)), f"不应命中信号: {text}"


def test_signal_checked_only_on_recent_user_messages():
    """只检查最近 2 条用户消息：很久之前的偏好声明不重复提炼。"""
    # 最后两条用户消息都是普通提问 → 不触发（即使更早消息含偏好）
    msgs = [
        Message(role=MsgRole.USER, content="我喜欢打羽毛球"),
        Message(role=MsgRole.ASSISTANT, content="好的"),
        Message(role=MsgRole.USER, content="食堂几点关门？"),
        Message(role=MsgRole.ASSISTANT, content="一般到晚上七点"),
        Message(role=MsgRole.USER, content="那校车呢？"),
        Message(role=MsgRole.ASSISTANT, content="下一班 13:10"),
    ]
    assert not _has_profile_signal(msgs)

    # 最后一条用户消息含信号 → 触发
    msgs[-1] = Message(role=MsgRole.USER, content="我最近在准备考研")
    assert _has_profile_signal(msgs)


def test_empty_or_assistant_only_messages():
    assert not _has_profile_signal([])
    assert not _has_profile_signal([Message(role=MsgRole.ASSISTANT, content="你好呀")])
