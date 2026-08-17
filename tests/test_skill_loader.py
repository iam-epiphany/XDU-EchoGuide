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


def test_keyword_matching_hints_relevant_skill():
    """v5：目录常驻（含工具名）+ 关键词命中提示，正文经 use_skill_* 工具按需加载。"""
    tmp, mgr = _make_manager()
    try:
        # 选课消息：目录含工具名，且命中提示高亮学业规范并给出加载入口
        prompt = mgr.prompt_for("这学期选课什么时候开始？")
        assert "学业咨询规范" in prompt
        assert "use_skill_academic" in prompt
        assert "该请求可能涉及以下技能，可调用对应工具获取完整规范" in prompt
        # 食堂消息：命中提示高亮校园生活规范
        prompt = mgr.prompt_for("南校区食堂几点关门？")
        assert "校园生活向导规范" in prompt
        assert "use_skill_campus_life" in prompt
        # 无关消息：目录常驻但无命中提示，且完整正文不再注入
        prompt = mgr.prompt_for("今天天气怎么样")
        assert "use_skill_academic" in prompt
        assert "该请求可能涉及以下技能" not in prompt
        assert "选课分预选" not in prompt
        # agents 字段已废弃：关键词命中的消息不再因领域键被过滤
        assert mgr.prompt_for("选课时间？", "campus_life")
    finally:
        tmp.cleanup()


def test_tool_definitions_expose_each_skill():
    """每个启用的 Skill 生成一个只读工具声明（无参数，描述含触发词）。"""
    tmp, mgr = _make_manager()
    try:
        defs = mgr.tool_definitions()
        assert {d["name"] for d in defs} == {"use_skill_academic", "use_skill_campus_life"}
        for d in defs:
            assert d["input_schema"] == {
                "type": "object", "properties": {}, "additionalProperties": False,
            }
        academic = next(d for d in defs if d["name"] == "use_skill_academic")
        assert "触发词：选课、课表、考试、成绩" in academic["description"]
        assert "选课等学业问题答复规范" in academic["description"]
    finally:
        tmp.cleanup()


def test_skill_content_for_returns_full_body():
    """use_skill_* 工具加载完整正文；未知工具返回错误提示（工具名派生自目录名）。"""
    tmp, mgr = _make_manager()
    try:
        content = mgr.skill_content_for("use_skill_academic")
        assert "### 学业咨询规范" in content
        assert "选课分预选、正选、退改选阶段" in content
        assert "校园生活向导规范" not in content
        assert mgr.skill_content_for("use_skill_not_exists") == "技能 use_skill_not_exists 不存在或已停用"
    finally:
        tmp.cleanup()


def test_followup_question_inherits_skill_via_history():
    """追问感知回归：'那几点开门呢？' 自身无关键词，但结合历史仍高亮 campus skill。"""
    tmp, mgr = _make_manager()
    try:
        history = [
            {"role": "user", "content": "南校区食堂几点关门？"},
            {"role": "assistant", "content": "南校区食堂一般晚上七点关门。"},
        ]
        # 不带历史：无命中提示（追问继承失效时不高亮）
        assert "该请求可能涉及以下技能" not in mgr.prompt_for("那几点开门呢？")
        # 带历史：命中提示高亮校园生活规范（追问继承）
        prompt = mgr.prompt_for("那几点开门呢？", history=history)
        assert "该请求可能涉及以下技能" in prompt
        assert "校园生活向导规范" in prompt
    finally:
        tmp.cleanup()


def test_ascii_keyword_word_boundary_matching():
    """子串误命中回归：'api' 不命中 'capital'（命中提示维度）。"""
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _write_skill(
        root,
        "it_help",
        """---
name: 校园IT支持规范
description: IT 排障规范
keywords: api,校园网
agents: it_help
enabled: true
---

# 校园IT支持规范
调用 api 时注意鉴权。
""",
    )
    mgr = SkillManager(root_dir=str(root))
    mgr.load()
    try:
        # capital 不命中 api（整词边界），命中提示不出现
        assert "该请求可能涉及以下技能" not in mgr.prompt_for("capital 是什么意思")
        assert "该请求可能涉及以下技能" not in mgr.prompt_for("capital")
        # 调用 api 报错 → 命中提示出现
        assert "该请求可能涉及以下技能" in mgr.prompt_for("调用 api 报错")
    finally:
        tmp.cleanup()


def test_prompt_injection_mentions_echoguide_and_rules():
    tmp, mgr = _make_manager()
    try:
        prompt = mgr.prompt_for("选课什么时候开始", "academic")
        assert "EchoGuide" in prompt
        # v5：正文不再注入 system prompt（"预选"只出现在 SKILL.md 正文里）
        assert "预选" not in prompt
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
        assert mgr.tool_definitions() == []  # 停用 Skill 不出现在工具声明里
    finally:
        tmp.cleanup()
