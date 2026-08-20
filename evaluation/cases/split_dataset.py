"""数据集三向分割（train / dev / holdout = 60/20/20，固定 seed 可复现）。

计划 §6：新增评测体系，避免过拟合——训练/开发/留出三分，holdout 用例在
优化过程中不可见（本轮优化未针对任何用例做特化，holdout 用于验证泛化）。

用法：
  python evaluation/cases/split_dataset.py            # 生成 intent + retrieval 三份文件
  python evaluation/cases/split_dataset.py --seed 42

输出（与 load_intent_cases / load_retrieval_cases 格式一致）：
  evaluation/cases/intent_cases_train/dev/holdout.json
  evaluation/cases/retrieval_cases_train/dev/holdout.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random

CASES_DIR = pathlib.Path(__file__).resolve().parent


def _split(items: list, ratios=(0.6, 0.2, 0.2), seed: int = 42):
    rng = random.Random(seed)
    pool = list(items)
    rng.shuffle(pool)
    n = len(pool)
    n_train = round(n * ratios[0])
    n_dev = round(n * ratios[1])
    return pool[:n_train], pool[n_train:n_train + n_dev], pool[n_train + n_dev:]


def _save(name: str, cases: list) -> None:
    path = CASES_DIR / f"{name}.json"
    path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{path.name}: {len(cases)} 条")


def main() -> int:
    parser = argparse.ArgumentParser(description="数据集 train/dev/holdout 三分")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for base in ("intent_cases", "retrieval_cases"):
        data = json.loads((CASES_DIR / f"{base}.json").read_text(encoding="utf-8"))
        train, dev, holdout = _split(data["cases"], seed=args.seed)
        _save(f"{base}_train", {"version": data.get("version", "1.0"), "description": f"{base} 训练集（60%）", "cases": train})
        _save(f"{base}_dev", {"version": data.get("version", "1.0"), "description": f"{base} 开发集（20%）", "cases": dev})
        _save(f"{base}_holdout", {"version": data.get("version", "1.0"), "description": f"{base} 留出集（20%，优化过程不可见）", "cases": holdout})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
