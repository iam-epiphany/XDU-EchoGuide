"""Skill catalog、统一只读加载与资源路径保护测试。"""
from __future__ import annotations

import tempfile
from pathlib import Path

from core.skill_loader import SkillManager
from core.tracing import begin_trace, end_trace


def _write_skill(root: Path, name: str, body: str) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(body, encoding="utf-8")


def _make_manager():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    _write_skill(root, "course-planning", """---
name: 课程规划
description: 选课与培养方案的决策支持
keywords: 选课,培养方案
enabled: true
---
# Goal
先检索官方课程规则。
""")
    _write_skill(root, "campus-card-service", """---
name: 校园卡服务
description: 校园卡挂失与补办流程
keywords: 校园卡,挂失
enabled: true
---
# Goal
先挂失，再查询办理流程。
""")
    manager = SkillManager(root_dir=str(root))
    manager.load()
    return tmp, manager


def test_discovers_directory_skills_and_catalog_is_domain_independent():
    tmp, manager = _make_manager()
    try:
        assert {s["id"] for s in manager.summary()["skills"]} == {"course-planning", "campus-card-service"}
        prompt = manager.prompt_for("选课什么时候开始？", "campus_life")
        assert "course-planning" in prompt
        assert "校园卡服务" in prompt
        assert "load_skill" in prompt
        assert "先检索官方课程规则" not in prompt
    finally:
        tmp.cleanup()


def test_single_load_skill_tool_uses_allowlisted_ids():
    tmp, manager = _make_manager()
    try:
        definitions = manager.tool_definitions()
        assert len(definitions) == 1
        tool = definitions[0]
        assert tool["name"] == "load_skill"
        assert tool["input_schema"]["properties"]["skill_name"]["enum"] == ["campus-card-service", "course-planning"]
        assert "先检索官方课程规则" in manager.load_skill("course-planning")
        assert "不存在" in manager.load_skill("../course-planning")
    finally:
        tmp.cleanup()


def test_keyword_hint_followup_and_trace_observability():
    tmp, manager = _make_manager()
    begin_trace("skills")
    try:
        prompt = manager.prompt_for("那怎么挂失？", history=[{"role": "user", "content": "校园卡丢了"}])
        assert "校园卡服务" in prompt
        manager.load_skill("campus-card-service")
    finally:
        trace = end_trace()
        tmp.cleanup()
    assert trace is not None
    assert trace.tags["skills_prompted"] == "campus-card-service"
    assert trace.tags["skills_loaded"] == "campus-card-service"


def test_resource_loader_rejects_path_traversal():
    tmp, manager = _make_manager()
    try:
        root = Path(tmp.name) / "course-planning" / "references"
        root.mkdir()
        (root / "rules.md").write_text("official rules", encoding="utf-8")
        assert manager.load_skill_resource("course-planning", "rules.md") == "official rules"
        assert "路径不合法" in manager.load_skill_resource("course-planning", "../SKILL.md")
    finally:
        tmp.cleanup()
