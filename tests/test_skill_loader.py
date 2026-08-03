"""Skill 加载器测试：front matter 解析、关键词匹配、prompt 注入、热加载。"""
from __future__ import annotations

import tempfile
from pathlib import Path

from core.skill_loader import SkillManager


def _write_skill(root: Path, name: str, body: str) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")


def _make_manager():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _write_skill(
        root,
        "academic",
        """---
name: 学业咨询规范
description: 选课等学业问题答复规范
keywords: 选课,课表,考试,成绩
agents: academic
enabled: true
---

# 学业咨询规范
选课分预选、正选、退改选阶段。
""",
    )
    _write_skill(
        root,
        "campus_life",
        """---
name: 校园生活向导规范
description: 食堂校车等生活问题答复规范
keywords: 食堂,宿舍,校车,校园卡
agents: campus_life
enabled: true
---

# 校园生活向导规范
食堂刷校园卡就餐。
""",
    )
    mgr = SkillManager(root_dir=str(root))
    mgr.load()
    return tmp, mgr


def test_loads_all_campus_skills():
    tmp, mgr = _make_manager()
    try:
        summary = mgr.summary()
        assert summary["count"] == 2
        assert summary["errors"] == []
        names = {s["name"] for s in summary["skills"]}
        assert names == {"学业咨询规范", "校园生活向导规范"}
    finally:
        tmp.cleanup()


def test_keyword_matching_routes_to_correct_agent():
    tmp, mgr = _make_manager()
    try:
        # 选课消息命中 academic skill
        assert mgr.prompt_for("这学期选课什么时候开始？", "academic")
        # 食堂消息命中 campus_life skill
        assert mgr.prompt_for("南校区食堂几点关门？", "campus_life")
        # 无关消息不注入
        assert not mgr.prompt_for("今天天气怎么样", "academic")
        # 非对应 Agent 不注入（关键词命中但 agents 不匹配）
        assert not mgr.prompt_for("选课时间？", "campus_life")
    finally:
        tmp.cleanup()


def test_prompt_injection_mentions_echoguide_and_rules():
    tmp, mgr = _make_manager()
    try:
        prompt = mgr.prompt_for("选课什么时候开始", "academic")
        assert "EchoGuide" in prompt
        assert "预选" in prompt
    finally:
        tmp.cleanup()


def test_reload_picks_up_new_skill():
    tmp, mgr = _make_manager()
    try:
        assert mgr.summary()["count"] == 2
        _write_skill(
            Path(tmp.name),
            "it_help",
            """---
name: 校园IT支持规范
description: 教务系统等IT问题答复规范
keywords: 教务系统,校园网,vpn
agents: it_help
enabled: true
---

# 校园IT支持规范
教务系统登录不上先清缓存。
""",
        )
        mgr.reload()
        summary = mgr.summary()
        assert summary["count"] == 3
        assert mgr.prompt_for("教务系统登录不上", "it_help")
    finally:
        tmp.cleanup()


def test_disabled_skill_is_skipped():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _write_skill(
        root,
        "disabled_skill",
        """---
name: 停用规范
description: 不应被加载
keywords: 停用
agents: academic
enabled: false
---

# 停用规范
""",
    )
    mgr = SkillManager(root_dir=str(root))
    mgr.load()
    try:
        assert mgr.summary()["count"] == 0
    finally:
        tmp.cleanup()
