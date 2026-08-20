"""端到端回答 Grounding Trace 分析：定位 Faithfulness / Citation 低的根因。

方法（计划 §5.1，先 Trace 后改代码）：
  1. 跑对话用例的 Agent 生成（与 run_dialog_eval 同一 Orchestrator，跳过 Judge
     评分以省成本），记录每轮：question / answer / tool_evidence（真实使用的证据）。
  2. 把回答按句拆分，逐句与每个证据 chunk 做确定性匹配
     （字符 2-gram Dice 系数 + bge 向量余弦），得到句子级证据支持度。
  3. 解析回答中已有的 [n] 引用（模型自标引用）。
  4. 输出分类统计：
     情况 A：答案正确但无引用 → citation pipeline 问题
     情况 B：答案含证据不支持的句子 → generation grounding 问题
     情况 C：证据已检索但未被利用 → context format 问题

用法：
  python evaluation/error_analysis/dialog_trace_analysis.py [--smoke] [--out ...]

输出：data/eval/error_analysis/dialog_trace.json（每轮完整 trace + 汇总）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import re
import statistics
import sys

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv()

from evaluation.cases import load_dialog_cases

# 句子级支持度阈值：低于该 Dice 系数视为"该句没有直接证据支持"
_SUPPORT_DICE = 0.18
# 证据匹配也看向量相似度（证据内容被截断 800 字，字符重叠可能失真）
_SUPPORT_COS = 0.55


def split_sentences(text: str) -> list[str]:
    """中文/英文混合句子拆分：按句末标点与换行切分，标点保留在句尾。"""
    if not text:
        return []
    parts = re.split(r"(?<=[。！？!?；;\n])", text)
    return [p.strip() for p in parts if p.strip()]


def _chars(text: str) -> list[str]:
    return [ch for ch in text if not ch.isspace()]


def dice_coef(a: str, b: str) -> float:
    """字符 2-gram Dice 系数：句子与证据的词汇重叠度（确定性，无外部依赖）。"""
    def bigrams(s: str) -> set:
        s = re.sub(r"[\s，。！？、,.!?：:；;\"'“”‘’（）()\[\]【】]", "", s)
        return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}
    ga, gb = bigrams(a), bigrams(b)
    if not ga or not gb:
        return 0.0
    return 2.0 * len(ga & gb) / (len(ga) + len(gb))


async def _embed_sim(sentence: str, chunk: str) -> float:
    """句子与证据的向量相似度（bge 同构嵌入，失败返回 0）。"""
    try:
        from mcp.embeddings import get_embedder

        embedder = get_embedder()
        if embedder is None:
            return 0.0
        import asyncio as _a

        vecs = await _a.to_thread(embedder.embed_documents, [sentence, chunk[:500]])
        a, b = vecs[0], vecs[1]
        dot = sum(float(x) * float(y) for x, y in zip(a, b))
        na = sum(float(x) * float(x) for x in a) ** 0.5
        nb = sum(float(x) * float(x) for x in b) ** 0.5
        return float(dot / (na * nb)) if na and nb else 0.0
    except Exception:
        return 0.0


def extract_citations(answer: str) -> list[int]:
    """解析回答中的 [n] 引用（剔除 Markdown 链接标签）。"""
    stripped = re.sub(r"\[[^\]]*\]\([^)]*\)", "", answer or "")
    return sorted({int(m) for m in re.findall(r"\[(\d+)\]", stripped)})


async def analyze_sentence(sentence: str, evidences: list[dict]) -> dict:
    """单句证据匹配：返回 best evidence、Dice、cos、supported。"""
    best = {"dice": 0.0, "cos": 0.0, "idx": -1, "title": "", "supported": False}
    for i, ev in enumerate(evidences):
        chunk = str(ev.get("content") or "")
        dice = dice_coef(sentence, chunk)
        if dice > best["dice"]:
            best["dice"] = dice
            best["idx"] = i
            best["title"] = str(ev.get("title") or "")
    if best["idx"] >= 0:
        best["cos"] = await _embed_sim(sentence, str(evidences[best["idx"]].get("content") or ""))
    best["supported"] = best["dice"] >= _SUPPORT_DICE or best["cos"] >= _SUPPORT_COS
    return best


async def run_trace(cases, smoke: bool) -> list[dict]:
    from agents.agent_orchestrator import Request as OrcReq

    from api import state

    state._build_runtime()
    orchestrator = state._orchestrator

    turns_trace = []
    for i, case in enumerate(cases):
        questions = case.get("turns") or [case.get("question")]
        questions = [str(q) for q in questions if str(q).strip()]
        conv_id = f"trace_{i}"
        user_id = "trace_user"
        history: list[dict] = []
        for turn_idx, question in enumerate(questions):
            orch_req = OrcReq(
                message=question,
                user_id=user_id,
                conv_id=conv_id,
                context="",
                history=history[-6:] if history else None,
            )
            orch_result = await orchestrator.run(orch_req)
            answer = orch_result.response
            evidences = list(getattr(orch_result, "tool_evidence", []) or [])
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

            sentences = [s for s in split_sentences(answer)]
            analyzed = []
            for s in sentences:
                match = await analyze_sentence(s, evidences)
                analyzed.append({
                    "sentence": s,
                    "evidence_idx": match["idx"],
                    "evidence_title": match["title"],
                    "dice": round(match["dice"], 4),
                    "cos": round(match["cos"], 4),
                    "supported": match["supported"],
                })
            citations = extract_citations(answer)
            n_sources = len(evidences)
            turns_trace.append({
                "case_id": case.get("id", f"case-{i}"),
                "turn": turn_idx,
                "question": question,
                "answer": answer,
                "n_sources": n_sources,
                "source_titles": [str(e.get("title", "")) for e in evidences],
                "citations_in_answer": citations,
                "citation_valid": all(1 <= c <= n_sources for c in citations),
                "has_citation": len(citations) > 0,
                "sentences": analyzed,
                "supported_sentence_ratio": round(
                    sum(1 for a in analyzed if a["supported"]) / len(analyzed), 4
                ) if analyzed else 0.0,
            })
            if smoke:
                break
        if smoke:
            break
    return turns_trace


def summarize(turns_trace: list[dict]) -> dict:
    n = len(turns_trace)
    has_ev = [t for t in turns_trace if t["n_sources"] > 0]
    no_ev = [t for t in turns_trace if t["n_sources"] == 0]
    cited = [t for t in turns_trace if t["has_citation"]]
    valid_cited = [t for t in cited if t["citation_valid"]]
    unsupported_sents = [
        (t, a) for t in turns_trace for a in t["sentences"] if not a["supported"]
    ]
    return {
        "total_turns": n,
        "turns_with_evidence": len(has_ev),
        "turns_without_evidence": len(no_ev),
        "turns_with_any_citation": len(cited),
        "citation_rate": round(len(cited) / n, 4) if n else 0.0,
        "turns_with_valid_citation": len(valid_cited),
        "citation_correctness": round(len(valid_cited) / len(cited), 4) if cited else None,
        "avg_supported_sentence_ratio": round(
            statistics.mean(t["supported_sentence_ratio"] for t in has_ev), 4
        ) if has_ev else None,
        "unsupported_sentence_count": len(unsupported_sents),
        "unsupported_examples": [
            {"case_id": t["case_id"], "turn": t["turn"], "sentence": a["sentence"],
             "dice": a["dice"], "cos": a["cos"], "evidence_title": a["evidence_title"],
             "question": t["question"]}
            for t, a in unsupported_sents[:15]
        ],
        "no_evidence_turns": [
            {"case_id": t["case_id"], "turn": t["turn"], "question": t["question"]}
            for t in no_ev
        ],
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="回答 Grounding Trace 分析")
    parser.add_argument("--smoke", action="store_true", help="只跑前 3 轮（联调）")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cases = load_dialog_cases()
    if args.smoke:
        cases = cases[:2]
    trace = await run_trace(cases, smoke=args.smoke)
    summary = summarize(trace)
    report = {"summary": summary, "turns": trace}

    out = args.out or ROOT / "data" / "eval" / "error_analysis" / "dialog_trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
