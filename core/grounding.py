"""
答案 Grounding 层：sentence-level citation 标注与证据支持度判定。

设计动机（可信 Agent / Grounded RAG 定位）：
  - 旧行为：模型自觉输出 [n]（经常不输出或越界），末尾追加链接列表。
    Citation 0.18 / Faithfulness 0.33 的直接原因就是"引用依赖模型自觉"。
  - 本模块把引用从"模型自觉"改为"执行层后置保证"（与工具调用后置引用
    同一哲学）：回答生成后，按句拆分 → 逐句与真实使用的证据做确定性匹配
    （字符 2-gram Dice + bge 余弦）→ 支持的句子追加 [i] 标注；末尾来源
    区按 [i] 编号列出证据。索引永远落在证据范围内（引用的正确性不再依赖
    模型），无证据支持的句子不加引用（暴露给 Verifier 做 unsupported 标注）。

匹配阈值（bge 同构嵌入标定）：
  - min_dice=0.16：直接引用/高重叠句（"校园卡需要身份证"这类陈述句通常
    ≥0.3）；低于 0.16 视为无词面依据。
  - min_cos=0.52：同义改写句（"需要带什么材料"→"携带本人身份证"语义等价
    但词面不重叠）；bge 对 paraphrase 的相似度实测在 0.55-0.75 区间。
  任一指标达标即视为"该句有证据支持"。

不引入任何 benchmark query 特化规则：匹配完全由句子与证据的文本相似度驱动。
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

MIN_DICE = 0.16
MIN_COS = 0.52


def split_sentences(text: str) -> List[str]:
    """中文/英文混合句子拆分：按句末标点与换行切分，标点保留在句尾。

    引用标注按句粒度进行：一个事实性 claim 对应一个句子。
    """
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    return [p.strip() for p in parts if p.strip()]


def _bigrams(s: str) -> set:
    s = re.sub(r"[\s，。！？、,.!?：:；;\"'“”‘’（）()\[\]【】*#\-]", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def dice_coef(a: str, b: str) -> float:
    """字符 2-gram Dice 系数（确定性词面重叠，无外部依赖）。"""
    ga, gb = _bigrams(a), _bigrams(b)
    if not ga or not gb:
        return 0.0
    return 2.0 * len(ga & gb) / (len(ga) + len(gb))


async def cosine_sim(a: str, b: str) -> float:
    """bge 同构嵌入余弦（语义改写匹配），embedder 不可用返回 0。"""
    try:
        from mcp.embeddings import get_embedder

        embedder = get_embedder()
        if embedder is None:
            return 0.0
        vecs = await asyncio.to_thread(embedder.embed_documents, [a, b[:500]])
        x, y = vecs[0], vecs[1]
        dot = sum(float(i) * float(j) for i, j in zip(x, y))
        nx = sum(float(i) * float(i) for i in x) ** 0.5
        ny = sum(float(i) * float(i) for i in y) ** 0.5
        return float(dot / (nx * ny)) if nx and ny else 0.0
    except Exception:
        return 0.0


def _strip_markdown_links(text: str) -> str:
    """剔除 [text](url) 形式的 Markdown 链接（其 [n] 是链接标签不是引用）。"""
    return re.sub(r"\[[^\]]*\]\([^)]*\)", "", text or "")


def existing_citations(answer: str) -> List[int]:
    """解析回答中已有的 [n] 引用索引（模型自觉输出的，供合并/校验）。"""
    stripped = _strip_markdown_links(answer or "")
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", stripped)})


def strip_citation_markers(answer: str) -> str:
    """移除正文中独立的 [n] 引用标记（保留 Markdown 链接）。

    引用由 Grounding 层统一重标，避免"模型标错 + 执行层再标"造成
    重复/越界引用。
    """
    text = answer or ""
    # 先保护 Markdown 链接（占位符替换），再删独立 [n]
    links: List[str] = []
    def _save(m: "re.Match") -> str:
        links.append(m.group(0))
        return f"\x00{len(links) - 1}\x00"
    text = re.sub(r"\[[^\]]*\]\([^)]*\)", _save, text)
    text = re.sub(r"\[\d+\]", "", text)
    for i, link in enumerate(links):
        text = text.replace(f"\x00{i}\x00", link)
    return text


async def match_evidence(sentence: str, evidences: List[Dict[str, Any]]) -> Tuple[int, float, float]:
    """句子 → 最佳证据匹配：返回 (evidence_idx, dice, cos)。

    evidence_idx = -1 表示无任何证据可匹配（证据为空）。
    """
    if not evidences:
        return -1, 0.0, 0.0
    best_idx, best_dice = -1, 0.0
    for i, ev in enumerate(evidences):
        dice = dice_coef(sentence, str(ev.get("content") or ""))
        if dice > best_dice:
            best_dice, best_idx = dice, i
    cos = 0.0
    if best_idx >= 0:
        cos = await cosine_sim(sentence, str(evidences[best_idx].get("content") or ""))
    return best_idx, best_dice, cos


def supported(dice: float, cos: float) -> bool:
    """任一指标达标即视为有证据支持（词面依据 或 语义等价）。"""
    return dice >= MIN_DICE or cos >= MIN_COS


async def annotate_citations(
    answer: str,
    evidences: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    句子级引用标注（确定性后置保证）：

      1. 剥离模型自觉输出的 [n]（防重复/越界）
      2. 按句拆分，逐句匹配最佳证据
      3. 有支持的句子追加 [i]（i = 证据在 evidences 中的 1-based 序号）
      4. 无支持的句子不加引用（保留原句，交给 Verifier 标注 unsupported）

    返回 {text, sentences, citation_indices, unsupported_sentences}
    citation_indices 恒在 [1, len(evidences)] 内 —— 引用正确性由执行层保证。
    """
    if not evidences:
        return {"text": answer or "", "sentences": [], "citation_indices": [], "unsupported_sentences": []}

    body = strip_citation_markers(answer or "")
    sentences = split_sentences(body)
    cited_sents: List[Dict[str, Any]] = []
    unsupported: List[str] = []
    indices: set = set()

    for sent in sentences:
        idx, dice, cos = await match_evidence(sent, evidences)
        if idx >= 0 and supported(dice, cos):
            cited_sents.append({
                "sentence": sent,
                "evidence_idx": idx,
                "dice": round(dice, 4),
                "cos": round(cos, 4),
            })
            indices.add(idx + 1)
        else:
            unsupported.append(sent)

    # 拼接：支持句加 [i]，unsupported 句保持原样
    out_parts: List[str] = []
    for sent in sentences:
        mark = next(
            (f"[{c['evidence_idx'] + 1}]" for c in cited_sents if c["sentence"] == sent),
            "",
        )
        out_parts.append(f"{sent}{mark}")
    annotated = "\n".join(out_parts)

    return {
        "text": annotated,
        "sentences": cited_sents,
        "citation_indices": sorted(indices),
        "unsupported_sentences": unsupported,
    }


def build_source_section(evidences: List[Dict[str, Any]], cited_indices: List[int]) -> str:
    """末尾来源区：按 [i] 编号列出被引用的证据（title + url）。"""
    lines = []
    for i, ev in enumerate(evidences):
        if (i + 1) not in cited_indices:
            continue
        title = str(ev.get("title") or "公开来源").strip()
        url = str(ev.get("source_url") or "").strip()
        updated = str(ev.get("updated_at") or "").strip()
        if url:
            line = f"[{i + 1}] [{title}]({url})"
        else:
            line = f"[{i + 1}] {title}"
        if updated:
            line = f"{line}（更新：{updated}）"
        lines.append(line)
    if not lines:
        return ""
    return "\n### 可核验来源\n" + "\n".join(lines)


async def grounding_trace(
    answer: str,
    evidences: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """完整 Grounding trace（可观测性/评测用，不修改回答）：
    每句的证据索引、相似度与支持度，供错误分析与 faithfulness 校验。"""
    sentences = split_sentences(answer or "")
    per_sentence = []
    for sent in sentences:
        idx, dice, cos = await match_evidence(sent, evidences)
        per_sentence.append({
            "sentence": sent,
            "evidence_idx": idx,
            "evidence_title": str(evidences[idx].get("title", "")) if idx >= 0 else "",
            "dice": round(dice, 4),
            "cos": round(cos, 4),
            "supported": supported(dice, cos),
        })
    supported_n = sum(1 for s in per_sentence if s["supported"])
    return {
        "sentence_count": len(per_sentence),
        "supported_ratio": round(supported_n / len(per_sentence), 4) if per_sentence else 0.0,
        "sentences": per_sentence,
    }
