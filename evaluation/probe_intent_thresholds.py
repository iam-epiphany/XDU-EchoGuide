"""意图识别 Embedding 阈值对比探针：0.80 vs 0.85（margin 均 0.10）。

背景：
  默认阈值从旧模型标定值 0.74/0.08 上调到 0.85/0.10（宁紧勿松：LLM 托底只保护
  漏判、不保护误判）。0.85 是否偏紧、0.80 是否更合适，用真实 bge 模型对一组
  典型问题打分，按两种阈值模拟级联路由对比，用数据决定。

用法：
  python evaluation/probe_intent_thresholds.py
  # 需要本地 bge 模型缓存（ECHOGUIDE_MODEL_CACHE_DIR）或可联网下载

输出：
  - 每个问题：Pattern 命中、Embedding top1/top2 分数与 margin、
    两种阈值下的路由结果（pattern / embedding / llm）；
  - 汇总：0.80 命中但 0.85 落入 LLM 的问题（"牺牲品"清单，看是否值得为它们
    降低阈值）；两阈值下 Embedding 命中的误判数（看 0.80 多放进来多少错路由）。

与生产链路同款嵌入方式：意图模板匹配为**同构嵌入**（用户消息与模板都走
embed_documents、都不带 bge-zh 指令前缀——指令前缀只用于 RAG 检索的 query 侧，
用在模板匹配会把同义文本相似度压到 ~0.79 导致级联空转）；Pattern / 追问形态 /
Embedding 判据与 core/intent_recognizer.py 完全一致（pattern 有信号时不跳过，
pattern=OTHER 且追问形态 → 直接 LLM）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 支持直接 python evaluation/probe_intent_thresholds.py 运行

from core.domains import IntentDomain, domain_hit_score  # noqa: E402
from core.intent_recognizer import IntentRecognizer, _DOMAIN_TEMPLATES, _cosine  # noqa: E402

PATTERN_THRESHOLD = 0.90
# 对比的两个候选配置（阈值, margin）；共享 margin 0.10，只对比阈值影响
CONFIGS = {"0.80": (0.80, 0.10), "0.85": (0.85, 0.10)}
IS_FOLLOWUP_SHAPED = IntentRecognizer._is_followup_shaped

# (问题, 期望领域或 None=应走 LLM, 备注)
_CASES: List[Tuple[str, Optional[str], str]] = [
    # ── 各领域完整问句（模板原文与改写）─────────────────────────────
    ("这学期选课什么时候开始？", "academic", "模板原文"),
    ("绩点怎么算的？", "academic", "模板原文"),
    ("重修怎么报名？", "academic", "模板原文"),
    ("挂科了要不要重修？", "academic", "改写"),
    ("毕业要修多少学分？", "academic", "改写"),
    ("南校区食堂几点关门？", "campus_life", "模板原文"),
    ("图书馆几点开门？", "campus_life", "模板原文"),
    ("校车最后一班几点？", "campus_life", "模板原文"),
    ("宿舍怎么报修？", "campus_life", "改写"),
    ("校园卡在哪充值？", "campus_life", "模板原文"),
    ("奖学金什么时候评？", "affairs", "模板原文"),
    ("请假流程怎么走？", "affairs", "模板原文"),
    ("在读证明在哪开？", "affairs", "模板原文"),
    ("学费怎么交？", "affairs", "改写"),
    ("教务系统登录不上", "it_help", "模板原文"),
    ("VPN怎么配置？", "it_help", "模板原文"),
    ("校园网连不上", "it_help", "改写"),
    ("今天有什么课？", "personal", "模板原文"),
    ("帮我记个待办", "personal", "模板原文"),
    ("明天第几节在哪上课？", "personal", "改写"),
    # ── 追问（形态检测已跳过 Embedding，期望 LLM）──────────────────
    ("那几点开门呢？", None, "追问-省略"),
    ("那几点关门呢？", None, "追问-省略"),
    ("最早一班呢？", None, "追问-省略"),
    ("几点？", None, "追问-省略"),
    ("什么时候？", None, "追问-省略"),
    ("那选课呢？", None, "追问-带主题词"),
    ("那保研呢？", None, "追问-带主题词"),
    # ── 高危/模糊（无领域关键词，应走 LLM 而不是 Embedding 猜）──────
    ("那什么时候放假？", None, "新话题短句（无关键词）"),
    ("什么时候放假", None, "无关键词完整句"),
    ("怎么办理", None, "无领域"),
    ("今天天气怎么样？", "campus_life", "天气词在词表"),
    ("谢谢", None, "收尾语"),
    ("好的", None, "收尾语"),
    # ── 跨领域易混（看 Embedding 分得清吗）─────────────────────────
    ("考试安排", "personal", "academic vs personal"),
    ("补办校园卡", "affairs", "campus_life vs affairs"),
    ("我要补办校园卡，要什么材料？", "affairs", "补办-材料"),
]


def route_for(pat: Tuple[Optional[IntentDomain], float], emb_score: float,
              emb_margin: float, emb_domain: Optional[IntentDomain],
              threshold: float, margin: float) -> str:
    """复刻 intent_recognizer 的级联判据（追问跳过另由调用方标注）。

    含双确认与分歧仲裁：
      - Pattern 高置信 + Embedding 方向一致 → pattern（免费直返）；
      - Pattern 高置信 + Embedding 方向分歧/未命中 → llm（仲裁）；
      - Embedding 达标且无 pattern 弱信号/方向一致 → embedding；
      - 其余 → llm。
    """
    pat_hit = pat[0] not in (None, IntentDomain.OTHER)
    if pat_hit and pat[1] >= PATTERN_THRESHOLD:
        if emb_domain == pat[0]:
            return "pattern"
        return "llm⇄"  # 关键词与 Embedding 分歧 → LLM 仲裁
    if (emb_domain not in (None, IntentDomain.OTHER)
            and emb_score >= threshold and emb_margin >= margin):
        if pat_hit and emb_domain != pat[0]:
            return "llm⇄"  # pattern 弱信号与 Embedding 分歧 → LLM 仲裁
        return "embedding"
    return "llm"


def main() -> int:
    from mcp.embeddings import LocalEmbedder

    embedder = LocalEmbedder()
    if not embedder.available:
        print(f"[失败] bge Embedding 模型不可用: {embedder._model.error}")
        print("请先在可联网/已有模型缓存的环境运行（模型会被缓存到本地）。")
        return 1

    tpl_texts = [t for d in _DOMAIN_TEMPLATES for t in _DOMAIN_TEMPLATES[d]]
    tpl_vecs = embedder.embed_documents(tpl_texts)
    tpl_index: Dict[IntentDomain, List[Tuple[str, list]]] = {}
    i = 0
    for domain in _DOMAIN_TEMPLATES:
        n = len(_DOMAIN_TEMPLATES[domain])
        tpl_index[domain] = list(zip(_DOMAIN_TEMPLATES[domain], tpl_vecs[i:i + n]))
        i += n

    rows = []  # (msg, expect, note, routes, emb_domain, pat_display)
    print(f"{'问题':<22}{'期望':<9}{'pattern':<9}{'emb top1':<15}{'margin':<8}"
          f"{'0.80':<9}{'0.85':<9}  备注")
    print("-" * 100)
    for msg, expect, note in _CASES:
        pat = domain_hit_score(msg)
        pat_display = f"{pat[0].value}@{pat[1]:.2f}" if pat[0] else "-"

        qv = embedder.embed_documents([msg])[0]   # 同构嵌入（与生产一致）
        scored = sorted(
            ((max(_cosine(qv, v) for _, v in vecs), d) for d, vecs in tpl_index.items()),
            key=lambda x: x[0], reverse=True,
        )
        emb_score, emb_domain = scored[0]
        emb_margin = max(0.0, scored[0][0] - scored[1][0])
        # 真实级联：追问形态为最高优先级（强信号无条件；弱信号仅当 pattern 无信号）
        followup_skip = IS_FOLLOWUP_SHAPED(
            msg, has_pattern_signal=pat[0] not in (None, IntentDomain.OTHER))

        routes = {}
        for name, (thr, mg) in CONFIGS.items():
            routes[name] = "llm✂" if followup_skip else route_for(pat, emb_score, emb_margin, emb_domain, thr, mg)
            if routes[name] == "embedding" and not followup_skip:
                # Embedding 命中即定路由：期望领域不匹配 → 误判
                if expect is not None and emb_domain.value != expect:
                    routes[name] += "✗"
                elif expect is None:
                    routes[name] += "✗"
        rows.append((msg, expect, note, routes, emb_domain, pat_display))

        diff = " ←差异" if routes["0.80"].rstrip("✗") != routes["0.85"].rstrip("✗") else ""
        print(f"{msg:<22}{expect or 'llm':<9}{pat_display:<9}"
              f"{emb_domain.value}@{scored[0][0]:.3f}{'':<3}{emb_margin:.3f}{'':<5}"
              f"{routes['0.80']:<9}{routes['0.85']:<9}{note}{' ✂' if followup_skip else ''}{diff}")

    print("-" * 100)
    n_followup = sum(1 for r in rows if r[3]["0.80"].startswith("llm✂"))
    print(f"追问形态（跳过 Embedding 走 LLM）: {n_followup} 个")
    for name in CONFIGS:
        judged = [r for r in rows if not r[3][name].startswith("llm✂")]
        wrong = sum(1 for r in judged if r[3][name].endswith("✗"))
        hits = sum(1 for r in judged if r[3][name].startswith("embedding"))
        print(f"  {name}: Embedding 命中 {hits} 个，其中误判 {wrong} 个")

    # "牺牲品"：非追问句在 0.80 命中、0.85 落 LLM
    sacrificed = [r[0] for r in rows
                  if not r[3]["0.85"].startswith("llm✂")
                  and r[3]["0.80"].startswith("embedding")
                  and r[3]["0.85"].startswith("llm")]
    print(f"\n0.80 命中但 0.85 落入 LLM 的问题（\"牺牲品\"，看是否值得为它们降阈值）:")
    for m in sacrificed:
        print(f"  - {m}")
    print("\n判断口径：牺牲品若多为高价值高频问句 → 选 0.80；若多为模糊/边缘句 →")
    print("0.85 更稳（多付的 LLM 调用可控，且追问/模糊句已由形态检测分流）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
