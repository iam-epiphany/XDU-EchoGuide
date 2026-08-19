"""
出口校验（Verifier）：Post-response Consistency / Grounding Check。

定位：回答返回用户前的**出口一致性 / 依据检查**，不是"能消灭幻觉的事实
核查系统"——它只标注风险、不阻断主链路（honest-by-design），也不承诺
"绝对正确"。

重点检查（规则层，免费全量）：
  - 回答声明了引用（[n]），但没有真实 Tool Evidence；
  - 回答声称已经执行写操作（"已添加/已完成"），但实际没有调用对应写工具；
  - needs_knowledge=true，但执行链没有发生知识检索（expected_retrieval_missing）。
  - 有 Tool Evidence 时，回答是否明显超出证据 → 由可选 LLM 判定承接。

LLM 判定（可选，策略开关，仅 DEEP/执行路径）：一次廉价判定调用，判断回答
是否被工具证据支撑；不通过追加免责声明，异常一律 fail-open（不阻断）。

证据来源统一：Tool Evidence 由执行链从工具结果泛化采集（含标题/来源 URL
的条目列表即视为证据），Verifier 不特殊认识任何具体工具名。
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
        needs_knowledge: bool = False,
    ) -> List[str]:
        flags: List[str] = []
        if _CITATION_RE.search(content or "") and not tool_evidence:
            flags.append("citation_without_evidence")
        if _WRITE_VERB_RE.search(content or "") and not (set(tools_used or []) & set(write_tools)):
            flags.append("write_claim_without_tool")
        # 检索需求闭环：意图判定需要知识检索，但最终执行链没有出现任何
        # retrieval evidence → 标记异常（只标注不阻断，提示 RAG 链路未落地）
        if needs_knowledge and not tool_evidence:
            flags.append("expected_retrieval_missing")
        return flags

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
        needs_knowledge: bool = False,
    ) -> VerificationResult:
        """执行出口校验：规则全量 + 可选 LLM（DEEP/执行路径）。"""
        if not (content or "").strip():
            return VerificationResult(source="skip")

        flags = self._rule_flags(content, tools_used, tool_evidence, write_tools, needs_knowledge)
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
