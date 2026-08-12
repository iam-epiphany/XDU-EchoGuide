"""切块策略对比：递归分隔符（旧） vs Markdown 结构感知（新）。

为什么做这个对比：
  旧切块（_split_recursive 递归分隔符）把 anydoc 输出的 Markdown 当纯文本切，
  标题行/表格行可能被拆散、块内没有标题上下文；新切块（_chunk_text 结构感知）
  注入标题链、标题边界成块、表格/代码块整体保留。本脚本在同一份文档集上分别
  跑两种切法，量化结构质量差异（不改 embedding、不依赖模型，纯离线可跑）。

指标说明：
  - heading_in_chunk：含标题行的块占比。旧切法标题行只是普通文本，随缘入块；
    新切法标题链注入块首，比例应为 100%（有标题文档）
  - broken_table_groups：被拆散到多个块的表格行组数。旧切法按句/逗号切，
    表格行大概率散落；新切法表格原子成块，应为 0
  - over_budget_chunks：块长超过 chunk_size 的块数（含链头预算）。旧切法
    500 字上限；新切法链头计入预算，同样不应超
  - avg_chunk_len / max_chunk_len：块长分布参考

用法：
  python evaluation/compare_chunking.py [--dir 学习文档] [--chunk-size 500] [--overlap 60]

说明：
  - 需要可解析的文档（.pdf/.docx/...），解析走 mcp.document_parser（anydoc）；
  - 纯离线：不启动服务、不下载模型。
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mcp.document_parser import parse_document  # noqa: E402
from mcp.knowledge_base import KnowledgeBase  # noqa: E402

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")


def split_old(kb: KnowledgeBase, text: str, chunk_size: int, overlap: int) -> list[str]:
    """旧切法：递归分隔符（等价于结构感知改造前的 _chunk_text）。"""
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]
    return [c for c, _, _ in kb._split_recursive(text, kb._CHUNK_SEPARATORS, chunk_size, overlap)]


def split_new(kb: KnowledgeBase, text: str, chunk_size: int, overlap: int) -> list[str]:
    return kb._chunk_text(text, chunk_size, overlap)


def group_table_rows(chunks: list[str]) -> int:
    """统计被拆散到多个块的表格行组数（连续 | 行视作一组，跨块即算拆散）。"""
    broken = 0
    in_table = False
    for c in chunks:
        rows = [l for l in c.split("\n") if l.lstrip().startswith("|")]
        if not rows:
            in_table = False
            continue
        if in_table:
            broken += 1  # 上一块的表格延续到本块 → 被拆散
        in_table = True
    return broken


def analyze(kb: KnowledgeBase, text: str, chunk_size: int, overlap: int) -> dict:
    old, new = split_old(kb, text, chunk_size, overlap), split_new(kb, text, chunk_size, overlap)

    def heading_blocks(chunks: list[str]) -> int:
        """旧口径：块内任何行以 # 开头（标题行作为正文被切进块）。"""
        return sum(1 for c in chunks if any(_HEADING_RE.match(l) for l in c.split("\n")))

    def chain_blocks(chunks: list[str]) -> int:
        """新口径：块首行是标题链（去井号后以 > 连接，单标题块首即标题）。"""
        return sum(1 for c in chunks
                   if re.match(r"^[^#|\s].* > ", c.split("\n")[0]) or _HEADING_RE.match(c.split("\n")[0]))

    def stats(chunks: list[str]) -> dict:
        return {
            "n_chunks": len(chunks),
            "avg_len": round(statistics.mean(len(c) for c in chunks), 1) if chunks else 0,
            "max_len": max((len(c) for c in chunks), default=0),
            "over_budget": sum(1 for c in chunks if len(c) > chunk_size),
            "heading_chunks": heading_blocks(chunks),
            "chain_chunks": chain_blocks(chunks),
            "broken_tables": group_table_rows(chunks),
        }

    return {"old": stats(old), "new": stats(new)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", default=str(ROOT / "学习文档"), help="待切分文档目录")
    ap.add_argument("--chunk-size", type=int, default=500)
    ap.add_argument("--overlap", type=int, default=60)
    args = ap.parse_args()

    kb = KnowledgeBase.__new__(KnowledgeBase)
    docs = sorted(Path(args.dir).glob("*.*"))
    if not docs:
        print(f"目录无文件: {args.dir}")
        sys.exit(1)

    print(f"{'文档':<30} {'旧块数':>6} {'新块数':>6} {'旧标题行入块':>9} {'新链头入块':>8} "
          f"{'旧拆表格':>7} {'新拆表格':>7} {'旧超预算':>7} {'新超预算':>7}")
    print("-" * 108)
    totals = {"old": {}, "new": {}}
    for path in docs:
        try:
            doc = parse_document(path.name, path.read_bytes())[0]
        except Exception as ex:
            print(f"{path.name:<30} 解析失败: {ex}")
            continue
        r = analyze(kb, doc["content"], args.chunk_size, args.overlap)
        for side in ("old", "new"):
            for k, v in r[side].items():
                totals[side].setdefault(k, []).append(v)
        print(f"{path.name:<30} {r['old']['n_chunks']:>6} {r['new']['n_chunks']:>6} "
              f"{r['old']['heading_chunks']:>9} {r['new']['chain_chunks']:>8} "
              f"{r['old']['broken_tables']:>7} {r['new']['broken_tables']:>7} "
              f"{r['old']['over_budget']:>7} {r['new']['over_budget']:>7}")
    print("-" * 108)

    def total(side: str, key: str):
        return sum(totals[side].get(key, []))

    print("汇总：")
    print(f"  块数合计:           旧 {total('old', 'n_chunks'):>5}  |  新 {total('new', 'n_chunks'):>5}")
    print(f"  标题行入块:         旧 {total('old', 'heading_chunks'):>5}  |  新 {total('new', 'chain_chunks'):>5}")
    print(f"  拆散表格组数:       旧 {total('old', 'broken_tables'):>5}  |  新 {total('new', 'broken_tables'):>5}")
    print(f"  超预算块数:         旧 {total('old', 'over_budget'):>5}  |  新 {total('new', 'over_budget'):>5}")
    print("    注：超预算来自 overlap 尾块（旧）与 overlap 尾块+原子表格块（新），均为设计行为")
    avg_old = statistics.mean(totals["old"]["avg_len"]) if totals["old"].get("avg_len") else 0
    avg_new = statistics.mean(totals["new"]["avg_len"]) if totals["new"].get("avg_len") else 0
    print(f"  平均块长:           旧 {avg_old:>6.1f}  |  新 {avg_new:>6.1f} 字")


if __name__ == "__main__":
    main()
