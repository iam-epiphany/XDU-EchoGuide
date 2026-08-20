"""评测标注集加载器：统一从 evaluation/cases/*.json 读取评测用例。

标注集文件：
  - intent_cases.json    意图识别标注集（expected 为 domain 或 action 值）
  - retrieval_cases.json RAG 检索标注集（query → 相关文档标题）
  - dialog_cases.json    端到端对话评测集（含 golden_answer 要点）

加载函数与 evaluation/evaluator.py 的数据结构对齐：
  IntentTestCase / RetrievalTestCase / EndToEndEvaluator 的 dialog case 字典。
"""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List

from evaluation.evaluator import IntentTestCase, RetrievalTestCase

CASES_DIR = pathlib.Path(__file__).resolve().parent


def load_intent_cases(path: str | pathlib.Path | None = None) -> List[IntentTestCase]:
    """加载意图标注集。context.history 多轮用例原样透传。"""
    p = pathlib.Path(path) if path else CASES_DIR / "intent_cases.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    cases = []
    for item in data["cases"]:
        cases.append(IntentTestCase(
            message=str(item["message"]),
            expected_intent=str(item["expected"]),
            context=item.get("context") or None,
        ))
    return cases


def load_retrieval_cases(path: str | pathlib.Path | None = None) -> List[RetrievalTestCase]:
    """加载检索标注集：query → relevant_titles（知识库实际文档标题）。"""
    p = pathlib.Path(path) if path else CASES_DIR / "retrieval_cases.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return [
        RetrievalTestCase(
            query=str(item["query"]),
            relevant_titles=[str(t) for t in item["relevant_titles"]],
        )
        for item in data["cases"]
    ]


def load_dialog_cases(path: str | pathlib.Path | None = None) -> List[Dict[str, Any]]:
    """加载端到端对话评测集（含 golden_answer 要点）。"""
    p = pathlib.Path(path) if path else CASES_DIR / "dialog_cases.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return [dict(item) for item in data["cases"]]
