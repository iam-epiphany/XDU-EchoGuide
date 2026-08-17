"""
EchoGuide（西电校园助手）Skill 加载器。

Skill 是一段可热加载的业务能力说明，用来补充 Agent 的 system prompt。
它适合放置企业话术、处理流程、合规边界、排障 SOP 等需要运营侧快速调整的规则。

匹配机制（修正旧版两类缺陷）：
  1. 子串误命中：关键词命中统一走 core.domains.keyword_hit ——
     ASCII 关键词整词匹配（\b 词边界，避免 "api" 命中 "capital"），
     中文关键词要求 ≥2 字（禁止单字过拟合）。
  2. 追问感知：当前消息未命中时，会继续匹配最近 2 轮用户消息，
     保证"南校区食堂几点关门？→ 那几点开门呢？"这类追问仍能注入对应 SOP。
"""
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from core.domains import keyword_hit

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """单个 Skill 的标准化表示，屏蔽 Markdown/JSON 等不同文件格式差异。"""
    name: str
    description: str
    content: str
    path: str
    keywords: List[str] = field(default_factory=list)
    agents: List[str] = field(default_factory=list)
    enabled: bool = True

    def matches(
        self,
        message: str,
        agent_type: Optional[str] = None,
        history: Optional[Iterable[Dict[str, str]]] = None,
    ) -> bool:
        """
        判断当前请求是否命中这个 Skill（v4：仅关键词驱动，领域键不再过滤）。

        agents 字段已废弃（Skill 不按领域分类，模型看全部目录自主选择）；
        参数保留仅为兼容旧调用方。keywords 为空 = 全局 Skill。
        history 非空：当前消息未命中时，回溯最近几轮用户消息（追问继承）。
        """
        if not self.enabled:
            return False

        if not self.keywords:
            return True

        if self._hit_any_keyword(message):
            return True

        # 追问继承：当前消息没有关键词时，看最近几轮用户消息
        if history:
            for m in history:
                if m.get("role") != "user":
                    continue
                if self._hit_any_keyword(str(m.get("content", ""))):
                    return True
        return False

    def _hit_any_keyword(self, text: str) -> bool:
        lowered = (text or "").lower()
        return any(keyword_hit(keyword, lowered) for keyword in self.keywords)

    def to_prompt_block(self, max_chars: int = 3200) -> str:
        """格式化为可直接拼入 system prompt 的文本块，并限制单个 Skill 长度。"""
        body = self.content.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "\n..."
        description = f"\n说明: {self.description}" if self.description else ""
        return f"### {self.name}{description}\n{body}"

    @property
    def tool_name(self) -> str:
        """Skill 对应的工具名（use_skill_<slug>），模型在工具循环内按需加载完整正文。

        SKILL.md 用父目录名做 slug，扁平文件用文件名（如 skills/refund.md → use_skill_refund）。
        """
        p = Path(self.path)
        slug = p.parent.name if p.name == "SKILL.md" else p.stem
        slug = re.sub(r"[^0-9a-zA-Z]+", "_", slug).strip("_").lower() or "skill"
        return f"use_skill_{slug}"

    def to_summary(self) -> Dict[str, Any]:
        """返回 API 可序列化摘要，避免把完整长文本默认暴露给健康检查。"""
        return {
            "name": self.name,
            "description": self.description,
            "path": self.path,
            "keywords": self.keywords,
            "agents": self.agents,
            "enabled": self.enabled,
            "content_chars": len(self.content),
        }


class SkillManager:
    """
    从目录中发现、解析并管理 Skills。

    支持两种常用结构：
      1. skills/refund/SKILL.md
      2. skills/refund.json / skills/refund.md / skills/refund.txt
    """

    SUPPORTED_SUFFIXES = {".md", ".txt", ".json"}

    def __init__(self, root_dir: str, max_prompt_chars: int = 5000):
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.max_prompt_chars = max_prompt_chars
        self._skills: List[Skill] = []
        self._errors: List[str] = []

    @property
    def skills(self) -> List[Skill]:
        return list(self._skills)

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    def load(self) -> List[Skill]:
        """重新扫描目录并加载 Skills；单个文件失败不会影响其他 Skill 生效。"""
        loaded: List[Skill] = []
        errors: List[str] = []

        if not self.root_dir.exists():
            logger.info(f"Skill 目录不存在，跳过加载: {self.root_dir}")
            self._skills = []
            self._errors = []
            return []

        for path in self._discover_files(self.root_dir):
            try:
                skill = self._load_file(path)
                if skill is not None and skill.enabled:
                    loaded.append(skill)
            except Exception as ex:
                msg = f"{path}: {ex}"
                errors.append(msg)
                logger.warning(f"Skill 加载失败: {msg}")

        self._skills = loaded
        self._errors = errors
        self._log_loaded_skills()
        return self.skills

    def reload(self) -> List[Skill]:
        """运行时热加载入口，供 API 调用。"""
        return self.load()

    def prompt_for(
        self,
        message: str,
        agent_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        构建 Skill 注入 prompt（v5：目录常驻 + 完整正文按需加载）。

        两段式（零 LLM 调用，只做关键词匹配与文本拼接）：
          1. 技能目录：全部 Skill 的名称 + 描述 + 触发词 + 对应工具名
             （模型建立"有哪些可用、如何加载"的预期）；
          2. 命中提示：当前消息/追问历史命中关键词的 Skill 高亮（免费引导，不强制）。
        完整 SKILL.md 正文不再注入 system prompt，由模型调用 use_skill_* 工具
        按需加载（Claude Code 式渐进披露）——正文加载见 tool_definitions/skill_content_for。
        """
        if not self._skills:
            return ""

        # 追问回溯范围：最近 2 轮用户消息
        follow_up_history = None
        if history:
            follow_up_history = [m for m in history if m.get("role") == "user"][-2:]

        # 1. 技能目录（含工具名，供模型按需调用加载完整规范）
        catalog_lines = ["可用技能目录（目录仅含摘要，调用对应工具可加载完整规范）："]
        for skill in self._skills:
            desc = f"（{skill.description}）" if skill.description else ""
            kws = "、".join(skill.keywords[:6]) if skill.keywords else "全局适用"
            catalog_lines.append(f"- {skill.name}{desc} 触发词: {kws}（工具: {skill.tool_name}）")
        catalog = "\n".join(catalog_lines)
        if len(catalog) > self.max_prompt_chars:
            catalog = catalog[:self.max_prompt_chars].rstrip() + "\n..."

        # 2. 命中提示（免费关键词引导）
        matched_names = []
        for skill in self._skills:
            if skill.matches(message, None, follow_up_history):
                matched_names.append(skill)
        hint = ""
        if matched_names:
            refs = "、".join(
                f"{skill.name}（{skill.tool_name}）" for skill in matched_names
            )
            hint = f"\n该请求可能涉及以下技能，可调用对应工具获取完整规范：{refs}"

        logger.info(
            "Skills 注入: 目录 %d 个, 命中提示 %s, message=%r",
            len(self._skills), matched_names and "、".join(s.name for s in matched_names) or "-",
            (message or "")[:80],
        )
        return (
            "以下是当前请求可用的 西电校园助手（EchoGuide）Skills。\n"
            "请优先遵循这些业务规则；如果与系统角色冲突，以系统角色和安全边界为准。\n"
            "需要完整规范时调用对应的 use_skill_* 工具加载。\n\n"
            f"{catalog}{hint}"
        )

    def tool_definitions(self) -> List[Dict[str, Any]]:
        """每个启用的 Skill 生成一个只读工具声明（无参数），供模型按需加载完整正文。"""
        defs = []
        for skill in self._skills:
            parts = [skill.description] if skill.description else []
            if skill.keywords:
                parts.append(f"触发词：{'、'.join(skill.keywords[:6])}")
            defs.append({
                "name": skill.tool_name,
                "description": "；".join(parts),
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
            })
        return defs

    def skill_for_tool(self, tool_name: str) -> Optional[Skill]:
        """按工具名反查 Skill（未命中返回 None），供工具执行层区分未知工具。"""
        for skill in self._skills:
            if skill.tool_name == tool_name:
                return skill
        return None

    def skill_content_for(self, tool_name: str) -> str:
        """按工具名返回 Skill 完整正文（模型工具循环内按需加载）；未知返回错误提示。"""
        skill = self.skill_for_tool(tool_name)
        if skill is None:
            return f"技能 {tool_name} 不存在或已停用"
        return skill.to_prompt_block(max_chars=12000)

    @staticmethod
    def cache_key(message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        """Skill 注入 prompt 的缓存键（消息 + 最近 2 轮用户消息指纹）。"""
        fp = ""
        if history:
            tail = "|".join(
                str(m.get("content", "")) for m in history[-2:]
            )
            import hashlib
            fp = hashlib.md5(tail.encode("utf-8")).hexdigest()[:8]
        return f"{str(message)[:200]}#{fp}"

    def summary(self) -> Dict[str, Any]:
        """返回 Skill 管理器状态，用于 /skills 接口和排障。"""
        return {
            "root_dir": str(self.root_dir),
            "count": len(self._skills),
            "skills": [skill.to_summary() for skill in self._skills],
            "errors": self.errors,
        }

    def _log_loaded_skills(self) -> None:
        """在控制台输出醒目的 Skill 加载结果，方便启动和热加载时确认生效状态。"""
        lines = [
            "",
            "================ EchoGuide Skills Loaded ================",
            f"目录: {self.root_dir}",
            f"数量: {len(self._skills)}",
        ]

        if self._skills:
            for index, skill in enumerate(self._skills, start=1):
                agents = ", ".join(skill.agents) if skill.agents else "all"
                keywords = ", ".join(skill.keywords[:8]) if skill.keywords else "all"
                if len(skill.keywords) > 8:
                    keywords += ", ..."
                lines.extend([
                    f"{index}. {skill.name}",
                    f"   agents: {agents}",
                    f"   keywords: {keywords}",
                    f"   path: {skill.path}",
                ])
        else:
            lines.append("未加载任何 Skill。")

        if self._errors:
            lines.append("解析错误:")
            lines.extend(f"  - {error}" for error in self._errors)

        lines.append("========================================================")
        logger.info("\n".join(lines))

    def _discover_files(self, root_dir: Path) -> Iterable[Path]:
        """发现可加载文件，优先读取目录规范文件 SKILL.md。"""
        skill_md_files = sorted(root_dir.rglob("SKILL.md"))
        yielded = {path.resolve() for path in skill_md_files}
        for path in skill_md_files:
            yield path

        for path in sorted(root_dir.rglob("*")):
            resolved = path.resolve()
            if resolved in yielded or not path.is_file():
                continue
            if path.name.startswith(".") or path.name.upper() == "README.MD":
                continue
            if path.suffix.lower() in self.SUPPORTED_SUFFIXES:
                yield path

    def _load_file(self, path: Path) -> Optional[Skill]:
        if path.suffix.lower() == ".json":
            return self._load_json(path)
        return self._load_text(path)

    def _load_json(self, path: Path) -> Optional[Skill]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("JSON Skill 必须是对象格式")

        content = str(raw.get("content") or raw.get("instructions") or "").strip()
        if not content:
            raise ValueError("缺少 content 或 instructions")

        return Skill(
            name=str(raw.get("name") or path.stem),
            description=str(raw.get("description") or ""),
            content=content,
            path=str(path),
            keywords=self._as_list(raw.get("keywords")),
            agents=[item.lower() for item in self._as_list(raw.get("agents"))],
            enabled=self._as_bool(raw.get("enabled"), default=True),
        )

    def _load_text(self, path: Path) -> Optional[Skill]:
        raw = path.read_text(encoding="utf-8")
        meta, body = self._split_front_matter(raw)
        body = body.strip()
        if not body:
            return None

        default_name = path.parent.name if path.name == "SKILL.md" else path.stem
        name = str(meta.get("name") or self._first_heading(body) or default_name)

        # 如果首行标题只是 Skill 名称，注入 prompt 时去掉它，减少重复噪音。
        body = self._strip_first_heading(body, name)

        return Skill(
            name=name,
            description=str(meta.get("description") or ""),
            content=body,
            path=str(path),
            keywords=self._as_list(meta.get("keywords")),
            agents=[item.lower() for item in self._as_list(meta.get("agents"))],
            enabled=self._as_bool(meta.get("enabled"), default=True),
        )

    def _split_front_matter(self, raw: str) -> tuple[Dict[str, Any], str]:
        """
        解析 Markdown 顶部的简单 front matter。

        这里刻意不用 PyYAML，避免为一个轻量配置格式新增运行时依赖。
        """
        text = raw.lstrip()
        if not text.startswith("---"):
            return {}, raw

        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            return {}, raw

        meta: Dict[str, Any] = {}
        end_idx: Optional[int] = None
        for idx, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                end_idx = idx
                break
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip("\"'")

        if end_idx is None:
            return {}, raw
        return meta, "\n".join(lines[end_idx + 1:])

    @staticmethod
    def _first_heading(body: str) -> Optional[str]:
        for line in body.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip() or None
        return None

    @staticmethod
    def _strip_first_heading(body: str, name: str) -> str:
        lines = body.splitlines()
        if not lines:
            return body
        first = lines[0].strip()
        if first.startswith("#") and first.lstrip("#").strip() == name:
            return "\n".join(lines[1:]).strip()
        return body

    @staticmethod
    def _as_list(value: Any) -> List[str]:
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [item.strip() for item in str(value).split(",") if item.strip()]

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in {"0", "false", "no", "off", "disabled"}
