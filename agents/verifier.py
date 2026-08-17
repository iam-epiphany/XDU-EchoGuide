"""
出口校验（Verifier / Grounding）：回答返回用户前的事实核查。

两层：
  1. 规则校验（免费、全量）：引用存在性（claim [n] 但无工具证据）、
     写操作落账（声称写入但未调用写工具）、实体一致性（回答中的日期/
     时间/电话/金额必须出现在工具证据或时间上下文中）；
  2. LLM 校验（可选，策略开关，仅 DEEP/执行路径）：一次廉价判定调用，
     判断回答是否被工具证据支撑；不通过追加免责声明。

设计原则：校验只标注不阻断主链路（honest-by-design）——flags 进
execution meta 与 Monitor 的 verification 计数，LLM 判定失败时给用户
追加免责声明而非吞掉回答。LLM 校验异常一律 fail-open（不阻断）。
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional

from core.domains import IntentAction

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[\d+\]")
_WRITE_VERB_RE = re.compile(r"已(?:添加|创建|记录|新增|完成|删除|更新|标记)")
_TIME_RE = re.compile(r"\d{1,2}\s*[:：]\s*\d{2}")
_PHONE_RE = re.compile(r"1[3-9]\d{9}")
_AMOUNT_RE = re.compile(r"\d+(?:\.\d+)?\s*元")
# 月日两种写法（8月17日 / 2026-08-17 / 2026年8月17日）
_MD_CN_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_MD_NUM_RE = re.compile(r"\d{4}\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})")


@dataclass
class VerificationResult:
    """一次出口校验的结果（只标注，不阻断）。"""

    flags: List[str] = field(default_factory=list)
    grounded: bool = True
    source: str = "rules"           # rules / rules+llm / skip
    disclaimer: str = ""            # LLM 判定未通过时的用户可见免责声明

    def summary(self) -> Dict[str, Any]:
        return {"flags": list(self.flags), "grounded": self.grounded, "source": self.source}


class ResponseVerifier:
    """规则 + 可选 LLM 两层出口校验。"""

    LLM_DISCLAIMER = "（部分内容未经工具证据完全支撑，请以官方渠道最新信息为准。）"

    def __init__(self, client: Optional[Any] = None, model: str = "", llm_enabled: bool = False,
                 gateway: Optional[Any] = None):
        self._client = client
        self._model = model
        self._llm_enabled = llm_enabled
        self._gateway = gateway  # 统一模型调用入口（编排器注入；None 时直接调用）

    # ── 规则校验（纯函数，免费）──────────────────────────────────────────────

    @staticmethod
    def _rule_flags(
        content: str,
        tools_used: List[str],
        tool_evidence: List[Dict[str, Any]],
        write_tools: FrozenSet[str],
    ) -> List[str]:
        flags: List[str] = []
        if _CITATION_RE.search(content or "") and not tool_evidence:
            flags.append("citation_without_evidence")
        if _WRITE_VERB_RE.search(content or "") and not (set(tools_used or []) & set(write_tools)):
            flags.append("write_claim_without_tool")
        if ResponseVerifier._unverified_facts(content or "", tool_evidence or []):
            flags.append("unverified_facts")
        return flags

    @staticmethod
    def _unverified_facts(content: str, tool_evidence: List[Dict[str, Any]]) -> List[str]:
        """回答中的硬事实（日期/时间/电话/金额）必须出现在证据或时间上下文中。

        日期两种写法统一规范为「M月D日」比较；时间/电话/金额按字符串包含比较。
        时间上下文（当前日期/周次）视为可信事实池的一部分，避免误伤
        "今天 8月17日"这类来自系统注入的信息。上限 3 条，只做标注。
        """
        from personal.time_context import build_time_context

        trusted = build_time_context()
        for item in tool_evidence or []:
            for key in ("title", "content", "source_url", "updated_at"):
                value = str(item.get(key) or "")
                if value:
                    trusted += "\n" + value

        def month_day_set(text: str) -> set:
            out = set()
            for m in _MD_CN_RE.finditer(text):
                out.add(f"{int(m.group(1))}月{int(m.group(2))}日")
            for m in _MD_NUM_RE.finditer(text):
                out.add(f"{int(m.group(1))}月{int(m.group(2))}日")
            return out

        unmatched: List[str] = sorted(month_day_set(content) - month_day_set(trusted))
        for regex in (_TIME_RE, _PHONE_RE, _AMOUNT_RE):
            for hit in regex.findall(content):
                if hit.strip() not in trusted:
                    unmatched.append(hit.strip())
                if len(unmatched) >= 3:
                    return unmatched[:3]
        return unmatched[:3]

    # ── LLM 校验（可选，fail-open）───────────────────────────────────────────

    async def _llm_grounded(
        self, req: Any, content: str, tool_evidence: List[Dict[str, Any]],
    ) -> Optional[bool]:
        """判定回答是否被工具证据支撑：True/False，None = 校验不可用（放行）。"""
        if self._client is None or not self._model:
            return None
        evidence = "\n".join(
            f"- {str(item.get('title', ''))}: {str(item.get('content', ''))[:400]}"
            for item in (tool_evidence or [])
        )[:3000]
        system = (
            "你是 EchoGuide 的出口校验器。判断助手回答中的事实性陈述是否被工具证据支撑："
            "回答若包含证据中没有的信息（具体日期、金额、电话、政策条款），判定不通过；"
            "通用建议、流程指引或基于证据的合理推论可以通过。只输出 JSON："
            '{"grounded": true/false, "reason": "一句话原因"}'
        )
        user = (
            f"用户请求: {req.message}\n\n工具证据:\n{evidence or '（无）'}\n\n"
            f"助手回答:\n{content[:2000]}"
        )
        try:
            if self._gateway is not None:
                result = await self._gateway.call(
                    client=self._client,
                    model=self._model,
                    messages=[{"role": "user", "content": user}],
                    state=getattr(req, "state", None),
                    span_name="verifier_llm",
                    max_tokens=256,
                    system=system,
                )
                resp = result.response
            else:
                resp = await self._client.messages.create(
                    model=self._model,
                    max_tokens=256,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            text = "".join(
                b.text for b in resp.content if getattr(b, "type", "") == "text"
            ).strip().strip("`")
            if text.startswith("json"):
                text = text[4:]
            data = json.loads(text)
            return bool(data.get("grounded", True))
        except Exception as ex:
            logger.warning(f"LLM 出口校验不可用，按放行处理: {ex}")
            return None

    # ── 主入口 ────────────────────────────────────────────────────────────────

    async def verify(
        self,
        req: Any,
        content: str,
        tools_used: List[str],
        tool_evidence: List[Dict[str, Any]],
        profile: str,
        write_tools: FrozenSet[str],
    ) -> VerificationResult:
        """执行出口校验：规则全量 + 可选 LLM（DEEP/执行路径）。"""
        if not (content or "").strip():
            return VerificationResult(source="skip")

        flags = self._rule_flags(content, tools_used, tool_evidence, write_tools)
        source = "rules"

        use_llm = (
            self._llm_enabled
            and (profile == "deep" or req.action == IntentAction.REQUEST)
        )
        if use_llm:
            source = "rules+llm"
            grounded = await self._llm_grounded(req, content, tool_evidence)
            if grounded is False:
                flags.append("llm_ungrounded")
                return VerificationResult(
                    flags=flags, grounded=False, source=source,
                    disclaimer=self.LLM_DISCLAIMER,
                )
        return VerificationResult(flags=flags, grounded=not flags, source=source)
