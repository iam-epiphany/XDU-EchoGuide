"""意图识别 Embedding 阈值重标定探针（MiniLM → bge 模型切换后分数分布不同）。

背景：
  意图识别的 Embedding 级联策略用「用户消息 vs 领域模板」的余弦相似度决策：
    - embedding_threshold（默认 0.80）：≥ 阈值才考虑 Embedding 命中；
    - embedding_margin（默认 0.10）：与第二候选领域的间隔 ≥ 该值才可信。
  有 LLM 兜底时阈值宁紧勿松（高阈值只是多付 LLM 调用，低阈值会静默误路由）；
  默认值按真实 bge 分布标定（probe_intent_thresholds.py 与 calibrate 脚本同源）。

用法：
  python evaluation/calibrate_intent_thresholds.py

输出：
  - 每个领域的正例（同领域模板）最低分 / 负例（异领域模板）最高分；
  - 建议阈值：正例最低分与负例最高分之间的分离区间中点；
  - 建议 margin：全部正例 margin 的最小值（保守取第一四分位）。
  将建议值写入 .env 的 ECHOGUIDE_INTENT_EMBEDDING_THRESHOLD / _MARGIN 即可。

说明：
  - 与生产链路同款嵌入方式：意图模板匹配为**同构嵌入**（用户消息与模板都走
    embed_documents、都不带 bge-zh 指令前缀——指令前缀只用于 RAG 检索的
    query 侧，用在模板匹配会把同义文本相似度压到 ~0.79 导致级联空转）；
  - 需要本地模型缓存或网络（mcp.embeddings）；模型不可用时给出明确提示。
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # 支持直接 python evaluation/calibrate_intent_thresholds.py 运行

from core.domains import IntentDomain  # noqa: E402
from core.intent_recognizer import _DOMAIN_TEMPLATES, _cosine  # noqa: E402

# 代表性用户问法（各领域 5-7 条，覆盖模板的改写表述，不含模板原文）
_PROBE_QUERIES: Dict[IntentDomain, List[str]] = {
    IntentDomain.ACADEMIC: [
        "什么时候开始选课？", "这学期选课从哪天开始", "绩点怎么计算",
        "挂科了要不要重修", "保研需要什么条件", "退改选能改几门课",
    ],
    IntentDomain.CAMPUS_LIFE: [
        "南校食堂晚上几点关门", "校车最晚一班是几点", "宿舍灯坏了找谁修",
        "校园卡丢了去哪补办", "图书馆几点开门", "快递站周末营业吗",
    ],
    IntentDomain.AFFAIRS: [
        "奖学金什么时候评选", "我想请两天假怎么办", "在读证明去哪里开",
        "学费怎么交", "补办校园卡要带什么材料", "助学金申请流程",
    ],
    IntentDomain.IT_HELP: [
        "教务系统进不去怎么办", "校园网连不上怎么回事", "VPN 怎么配置",
        "学校邮箱登录不了", "统一身份认证密码忘了",
    ],
    IntentDomain.PERSONAL: [
        "我今天有没有课", "看看我的课表", "明天下午在哪上课",
        "这周哪天没课", "帮我记一个待办", "我最近的考试安排",
    ],
}


def main() -> int:
    from mcp.embeddings import LocalEmbedder

    embedder = LocalEmbedder()
    if not embedder.available:
        print(f"[失败] bge Embedding 模型不可用: {embedder._model.error}")
        print("请先在可联网/已有模型缓存的环境运行（模型会被缓存到本地）。")
        return 1

    # 领域模板 → 文档侧向量（与 _load_template_embeddings 一致）
    tpl_texts = [t for d in _DOMAIN_TEMPLATES for t in _DOMAIN_TEMPLATES[d]]
    tpl_vecs = embedder.embed_documents(tpl_texts)
    tpl_index: Dict[IntentDomain, List[tuple]] = {}
    i = 0
    for domain in _DOMAIN_TEMPLATES:
        n = len(_DOMAIN_TEMPLATES[domain])
        tpl_index[domain] = list(zip(_DOMAIN_TEMPLATES[domain], tpl_vecs[i:i + n]))
        i += n

    positives, negatives, margins = [], [], []
    print(f"{'领域':<12}{'正例最低':>10}{'负例最高':>10}{'margin 最低':>12}")
    print("-" * 46)
    for domain in _DOMAIN_TEMPLATES:
        pos_scores, neg_scores, dom_margins = [], [], []
        for q in _PROBE_QUERIES.get(domain, []):
            qv = embedder.embed_documents([q])[0]   # 同构嵌入（与生产一致）
            pos = max(_cosine(qv, v) for _, v in tpl_index[domain])
            neg = max(
                _cosine(qv, v) for d2, vecs in tpl_index.items()
                if d2 != domain for _, v in vecs
            )
            pos_scores.append(pos)
            neg_scores.append(neg)
            dom_margins.append(pos - neg)
            positives.append(pos)
            negatives.append(neg)
            margins.append(pos - neg)
        print(f"{domain.value:<12}{min(pos_scores):>10.4f}{max(neg_scores):>10.4f}"
              f"{min(dom_margins):>12.4f}")

    print("-" * 46)
    min_pos, max_neg = min(positives), max(negatives)
    print(f"全部正例最低分: {min_pos:.4f}   全部负例最高分: {max_neg:.4f}")
    print(f"分离区间: [{max_neg:.4f}, {min_pos:.4f}]"
          f"{'（重叠！阈值无法完全分离）' if max_neg >= min_pos else ''}")
    threshold = round((min_pos + max_neg) / 2, 4)
    margins_sorted = sorted(margins)
    margin_rec = round(margins_sorted[len(margins_sorted) // 4], 4)  # 第一四分位（保守）
    print(f"\n建议配置（写入 .env）:")
    print(f"  ECHOGUIDE_INTENT_EMBEDDING_THRESHOLD={threshold}")
    print(f"  ECHOGUIDE_INTENT_EMBEDDING_MARGIN={margin_rec}")
    print(f"\n参考：正例 margin 均值 {statistics.mean(margins):.4f}，"
          f"中位数 {statistics.median(margins):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
