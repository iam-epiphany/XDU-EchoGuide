"""
亮点：端到端 Agent 评测框架

核心问题：如何评测端到端 Agent？

评测维度：
  1. 意图识别准确率 —— 预测意图 vs 标注意图，计算 Accuracy / F1
  2. 响应质量评分 —— 用 LLM 作为评判者（LLM-as-Judge），
     从相关性、准确性、完整性、有用性四个维度打分
  3. 端到端对话评测 —— 模拟完整多轮对话，评估整体体验
  4. 回归测试 —— 与历史基线对比，防止性能退化

LLM-as-Judge 是评测 Agent 质量的关键技术：
  人工标注成本高、主观性强；用 LLM 评判可以规模化、可重复。
"""
import asyncio
import json
import logging
import pathlib
import re
import subprocess
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.intent_recognizer import IntentAction, IntentDomain, IntentRecognizer

logger = logging.getLogger(__name__)

# 不达标样本日志的截断长度：评测用例为内部数据集，问题保留 200 字符便于复盘，
# Agent 回答可能很长，截断 800 字符控制单条日志体积。
_LOG_QUESTION_MAX = 200
_LOG_RESPONSE_MAX = 800


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class IntentTestCase:
    message:          str
    expected_intent:  str
    context:          Optional[Dict[str, Any]] = None


@dataclass
class QualityScores:
    """LLM-as-Judge 评分结果。"""
    relevance:    float   # 相关性：回答是否针对问题
    accuracy:     float   # 准确性：信息是否正确
    completeness: float   # 完整性：是否完整解决问题
    helpfulness:  float   # 有用性：用户是否能据此行动
    judge_failed: bool = False
    error: Optional[str] = None

    @property
    def overall(self) -> float:
        return statistics.mean([self.relevance, self.accuracy, self.completeness, self.helpfulness])


@dataclass
class EvalResult:
    test_id:    str
    passed:     bool
    scores:     Dict[str, float]
    detail:     str = ""
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """评测报告。"""
    timestamp:        str
    total:            int
    passed:           int
    pass_rate:        float
    avg_scores:       Dict[str, float]
    regressions:      List[str]          # 相比基线退化的指标
    recommendations:  List[str]
    results:          List[EvalResult]
    retrieval:        Optional[Dict[str, Any]] = None  # RAG 检索硬指标（HitRate@K/Recall@K/MRR）
    provenance:       Dict[str, Any] = field(default_factory=dict)
    judge:             Dict[str, Any] = field(default_factory=dict)


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

class LLMJudge:
    """
    用 LLM 评判 Agent 响应质量。

    为什么用 LLM 而不是人工？
    - 可规模化：数千条测试用例自动评测
    - 可重复：相同输入得到稳定评分
    - 多维度：同时评估相关性、准确性等多个维度

    注意：LLM Judge 本身也有偏差，建议定期用人工标注校准。
    """

    JUDGE_PROMPT = """你是一个西电校园助手答复质量评估专家。请对以下校园助手响应进行评分。

用户问题: {question}
Agent 响应: {response}
{context_section}

请从以下四个维度评分（0.0-1.0），返回 JSON：
- relevance: 响应是否直接针对用户问题（0=完全无关，1=完全相关）
- accuracy: 信息是否准确无误（0=明显错误，1=完全正确）
- completeness: 是否完整解决了用户需求（0=完全没解决，1=完全解决）
- helpfulness: 用户能否据此采取行动（0=毫无帮助，1=非常有帮助）

只返回 JSON，例如: {{"relevance": 0.9, "accuracy": 0.8, "completeness": 0.7, "helpfulness": 0.85}}"""

    def __init__(self, client: AsyncAnthropic, model: str):
        self._client = client
        self._model  = model

    async def judge(
        self,
        question: str,
        response: str,
        context: Optional[str] = None,
    ) -> QualityScores:
        ctx_section = f"背景信息: {context}" if context else ""
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            response=response,
            context_section=ctx_section,
        )
        prompt = self._clean_text(prompt)
        # 最多重试 2 次：LLM 偶尔返回纯文本/格式漂移，重试能显著降低误判
        for attempt in range(2):
            try:
                resp = await self._client.messages.create(
                    model=self._model, max_tokens=256, temperature=0.0,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text
                data = self._parse_scores(raw)
                if data is None:
                    raise ValueError("Judge 输出缺少 JSON")
                return QualityScores(**data)
            except Exception as ex:
                logger.warning(
                    f"LLM Judge 第 {attempt + 1} 次失败: {ex} "
                    f"(question={str(question)[:_LOG_QUESTION_MAX]!r})"
                )
                if attempt == 0:
                    # 重试时提示必须输出严格 JSON，减少格式漂移
                    prompt = (
                        prompt
                        + "\n\n注意：上次输出无法解析。请只输出一个 JSON 对象，"
                        "不要包含任何其他文字、注释或 Markdown 代码块。"
                    )
        return QualityScores(
            0.5, 0.5, 0.5, 0.5,
            judge_failed=True,
            error="Judge 连续 2 次输出无法解析",
        )

    @staticmethod
    def _parse_scores(raw: str) -> Optional[Dict[str, float]]:
        """
        从 Judge 输出中提取分数 JSON。

        兼容三种形态：
          - 纯 JSON 对象：{"relevance": 0.9, ...}
          - Markdown 代码块包裹：```json {...} ```
          - JSON 前后有少量说明文字
        """
        text = (raw or "").strip()
        # 去掉 ```json ... ``` 代码块
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e <= s:
            return None
        try:
            data = json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return {
            "relevance": max(0.0, min(1.0, float(data.get("relevance", 0.5)))),
            "accuracy": max(0.0, min(1.0, float(data.get("accuracy", 0.5)))),
            "completeness": max(0.0, min(1.0, float(data.get("completeness", 0.5)))),
            "helpfulness": max(0.0, min(1.0, float(data.get("helpfulness", 0.5)))),
        }

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    # ── 生成端扩展指标：Faithfulness / Answer Correctness ────────────────────

    FAITHFULNESS_PROMPT = """你是 RAG 生成质量评估专家。判断回答是否**忠实于**给定的知识来源（不发生幻觉）。

用户问题: {question}
知识来源: {context}
Agent 回答: {response}

评分规则（0.0-1.0）：
- 回答中的所有事实性陈述都能在知识来源中找到依据 → 1.0
- 部分内容在来源中找不到依据（编造/幻觉）→ 按无依据内容占比扣分
- 回答大量编造来源中不存在的信息 → 接近 0.0
- 知识来源为空/不相关，无法判断 → 给 0.5

只返回 JSON，例如: {{"faithfulness": 0.9}}"""

    ANSWER_CORRECTNESS_PROMPT = """你是 RAG 生成质量评估专家。对比 Agent 回答与标准答案，评估其**正确性**。

用户问题: {question}
标准答案: {golden}
Agent 回答: {response}

评分规则（0.0-1.0）：
- 与标准答案信息一致且完整 → 1.0
- 信息正确但不够完整 → 0.6-0.9
- 部分错误 → 0.3-0.5
- 完全错误/答非所问 → 0.0

只返回 JSON，例如: {{"correctness": 0.85}}"""

    async def judge_faithfulness(self, question: str, response: str, context: str) -> tuple[float, bool]:
        """回答忠实性：回答是否被检索上下文支持（无幻觉）。失败兜底 0.5 并标记。"""
        prompt = self.FAITHFULNESS_PROMPT.format(
            question=question, context=context[:3000], response=response
        )
        return await self._judge_scalar(prompt, "faithfulness", question=question)

    async def judge_answer_correctness(self, question: str, response: str, golden: str) -> tuple[float, bool]:
        """答案正确性：与标准答案的一致性（需要用例提供 golden_answer）。"""
        prompt = self.ANSWER_CORRECTNESS_PROMPT.format(
            question=question, golden=golden[:2000], response=response
        )
        return await self._judge_scalar(prompt, "correctness", question=question)

    async def _judge_scalar(
        self, prompt: str, key: str, question: str = "",
    ) -> tuple[float, bool]:
        """单指标 Judge：输出 {"key": 0.0-1.0}。返回 (分数, 是否失败)。

        失败兜底 0.5 并显式标记，与 judge() 的 judge_failed 语义一致，
        避免失败样本静默混入平均分。
        """
        prompt = self._clean_text(prompt)
        for attempt in range(2):
            try:
                resp = await self._client.messages.create(
                    model=self._model, max_tokens=128, temperature=0.0,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text
                data = self._parse_raw_json(raw)
                if data is None or key not in data:
                    raise ValueError(f"Judge 输出缺少 {key}")
                return max(0.0, min(1.0, float(data[key]))), False
            except Exception as ex:
                logger.warning(
                    f"Judge({key}) 第 {attempt + 1} 次失败: {ex} "
                    f"(question={str(question)[:_LOG_QUESTION_MAX]!r})"
                )
                if attempt == 0:
                    prompt = prompt + "\n\n注意：上次输出无法解析。请只输出一个 JSON 对象。"
        return 0.5, True

    @staticmethod
    def _parse_raw_json(raw: str) -> Optional[Dict[str, Any]]:
        """
        从 Judge 输出中提取 JSON 对象（兼容代码块包裹/前后说明文字）。
        不做键名归一化，保留原始键。
        """
        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
        s, e = text.find("{"), text.rfind("}")
        if s == -1 or e <= s:
            return None
        try:
            data = json.loads(text[s:e + 1])
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None


# ── 意图识别评测 ──────────────────────────────────────────────────────────────

class IntentEvaluator:
    """
    评测意图识别的准确率和 F1（领域 × 动作双维度）。

    expected_intent 取值约定：
      - 领域值（academic/campus_life/affairs/it_help/other）→ 比较预测的 domain
      - 动作值（query/request/greeting/complaint/feedback）→ 比较预测的 action
    用例可通过 context.history 提供多轮对话，评测追问继承能力。
    """

    DOMAIN_VALUES = {d.value for d in IntentDomain}
    ACTION_VALUES = {a.value for a in IntentAction}

    def __init__(self, recognizer: IntentRecognizer):
        self._recognizer = recognizer

    async def evaluate(self, cases: List[IntentTestCase]) -> Dict[str, Any]:
        predictions, ground_truth = [], []
        case_details: List[Dict[str, Any]] = []

        for case in cases:
            expected = case.expected_intent
            history = (case.context or {}).get("history") if case.context else None
            result = await self._recognizer.recognize(case.message, history=history)

            if expected in self.DOMAIN_VALUES:
                predicted = result.domain.value
            elif expected in self.ACTION_VALUES:
                predicted = result.action.value
            else:
                predicted = result.intent.value

            predictions.append(predicted)
            ground_truth.append(expected)
            case_details.append({
                "message": case.message,
                "expected": expected,
                "predicted": predicted,
                "domain": result.domain.value,
                "action": result.action.value,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            })

        # 纯 Python 计算指标
        correct = sum(p == g for p, g in zip(predictions, ground_truth))
        accuracy = correct / len(predictions) if predictions else 0.0

        # 每类 F1
        labels = sorted(set(ground_truth + predictions))
        per_class: Dict[str, Dict[str, float]] = {}
        for label in labels:
            tp = sum(p == label and g == label for p, g in zip(predictions, ground_truth))
            fp = sum(p == label and g != label for p, g in zip(predictions, ground_truth))
            fn = sum(p != label and g == label for p, g in zip(predictions, ground_truth))
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_class[label] = {"precision": prec, "recall": rec, "f1": f1}

        macro_f1 = statistics.mean(v["f1"] for v in per_class.values()) if per_class else 0.0

        return {
            "accuracy":   round(accuracy, 4),
            "macro_f1":   round(macro_f1, 4),
            "per_class":  per_class,
            "total":      len(cases),
            "correct":    correct,
            "cases":      case_details,
        }


# ── RAG 检索硬指标（无 LLM，确定性）──────────────────────────────────────────

@dataclass
class RetrievalTestCase:
    """检索评测用例：query + 知识库中相关文档的标题。"""
    query:           str
    relevant_titles: List[str]


def compute_retrieval_metrics(
    results: List[List[Dict[str, Any]]],
    relevant: List[List[str]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    纯函数：计算 RAG 检索硬指标（无需 LLM，可离线单测）。

    results:  每个用例的检索结果（每项含 title 字段）
    relevant: 每个用例的相关文档标题
    指标：
      - HitRate@K: 至少一个相关文档出现在 Top-K 的用例占比
      - Recall@K:  相关文档被召回到 Top-K 的比例（逐用例平均）
      - MRR:       第一个相关文档排名的倒数（逐用例平均）
    """
    assert_len = len(results)
    if assert_len != len(relevant):
        raise ValueError(f"results({assert_len}) 与 relevant({len(relevant)}) 用例数必须一致")
    hits, recalls, mrrs = [], [], []
    cases: List[Dict[str, Any]] = []

    for res, rel in zip(results, relevant):
        top_titles = [str(item.get("title", "")) for item in res[:top_k]]
        rel_set = set(rel)
        hit = any(t in rel_set for t in top_titles)
        recalled = sum(1 for t in top_titles if t in rel_set)
        recall = recalled / len(rel_set) if rel_set else 0.0
        rank = next((i + 1 for i, t in enumerate(top_titles) if t in rel_set), None)
        mrr = 1.0 / rank if rank else 0.0
        hits.append(hit)
        recalls.append(recall)
        mrrs.append(mrr)
        cases.append({
            "relevant": sorted(rel_set),
            "top_titles": top_titles,
            "hit": hit,
            "recall": round(recall, 4),
            "mrr": round(mrr, 4),
        })

    return {
        "hit_rate@K": round(sum(hits) / len(hits), 4) if hits else 0.0,
        "recall@K":   round(statistics.mean(recalls), 4) if recalls else 0.0,
        "mrr":        round(statistics.mean(mrrs), 4) if mrrs else 0.0,
        "top_k":      top_k,
        "total":      len(results),
        "cases":      cases,
    }


def citation_correctness(answer: str, sources: List[Any]) -> Dict[str, Any]:
    """
    引用正确性（确定性）：解析回答中的 [n] 引用，校验是否都在来源范围内。

    sources: 检索到的来源列表（每项 dict 或 str）
    返回: {total, valid, invalid, score, has_citation}
    """
    # 先剔除 Markdown 链接 [text](url) —— 其 [n] 是链接标签而非引用序号，
    # 避免把链接标签数字误判为引用编号。
    stripped = re.sub(r"\[[^\]]*\]\([^)]*\)", "", answer or "")
    indices = sorted({int(m) for m in re.findall(r"\[(\d+)\]", stripped)})
    n_sources = len(sources)
    valid = [i for i in indices if 1 <= i <= n_sources]
    return {
        "total": len(indices),
        "valid": len(valid),
        "invalid": [i for i in indices if not (1 <= i <= n_sources)],
        "score": round(len(valid) / len(indices), 4) if indices else None,
        "has_citation": len(indices) > 0,
    }


class RetrievalEvaluator:
    """RAG 检索硬指标评测：真实调用知识库，输出 HitRate@K / Recall@K / MRR。"""

    def __init__(self, knowledge_base):
        self._kb = knowledge_base

    async def run(self, cases: List[RetrievalTestCase], top_k: int = 5) -> Dict[str, Any]:
        results, relevant = [], []
        for case in cases:
            items = await asyncio.to_thread(self._kb.search, case.query, top_k=top_k)
            results.append(items)
            relevant.append(case.relevant_titles)
        metrics = compute_retrieval_metrics(results, relevant, top_k=top_k)
        for case, detail in zip(cases, metrics["cases"]):
            detail["query"] = case.query
        return metrics


# ── 端到端评测器 ──────────────────────────────────────────────────────────────

class EndToEndEvaluator:
    """
    端到端 Agent 评测。

    评测流程：
      1. 运行意图识别评测（准确率/F1）
      2. 运行对话质量评测（LLM-as-Judge）
      3. 与历史基线对比（回归检测）
      4. 生成可操作的优化建议
    """

    # 质量及格线
    PASS_THRESHOLD = 0.75

    def __init__(
        self,
        orchestrator,
        recognizer: IntentRecognizer,
        api_key:  str,
        base_url: Optional[str] = None,
        model:    str = "claude-3-5-sonnet-20241022",
        judge_api_key:  Optional[str] = None,
        judge_base_url: Optional[str] = None,
        judge_model:    Optional[str] = None,
        baseline_path: Optional[str] = None,
        knowledge_base: Optional[Any] = None,
    ):
        """
        双模型 LLM-as-Judge：

        生成模型（api_key/base_url/model）与评判模型（judge_*）可分离，
        消除"自己给自己打分"的自评偏差。judge_* 缺省时退化为同模型（向后兼容）。
        knowledge_base: 传入时启用 RAG 检索硬指标（HitRate@K/Recall@K/MRR）
        与生成端 Citation Correctness / Faithfulness 评测。
        """
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = AsyncAnthropic(**kwargs)

        judge_kwargs: Dict[str, Any] = {"api_key": judge_api_key or api_key}
        if judge_base_url:
            judge_kwargs["base_url"] = judge_base_url
        judge_client = AsyncAnthropic(**judge_kwargs)

        self._orchestrator     = orchestrator
        self._judge            = LLMJudge(judge_client, judge_model or model)
        self._judge_model      = judge_model or model
        # 独立 Judge：只有显式配置了不同的 API Key / 端点 / 模型才算独立
        self._judge_independent = (
            bool(judge_api_key and judge_api_key != api_key)
            or bool(judge_base_url and judge_base_url != base_url)
            or bool(judge_model and judge_model != model)
        )
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._retrieval_evaluator = RetrievalEvaluator(knowledge_base) if knowledge_base else None
        self._history:         List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._baseline: Optional[EvalReport] = self._load_baseline()

    async def run(
        self,
        intent_cases:    Optional[List[IntentTestCase]] = None,
        dialog_cases:    Optional[List[Dict[str, Any]]] = None,
        routing_cases:   Optional[List[Dict[str, Any]]] = None,
        retrieval_cases: Optional[List[RetrievalTestCase]] = None,
        promote_baseline: bool = False,
        dataset: str = "built_in_cases_v1",
    ) -> EvalReport:
        """
        运行完整评测。

        intent_cases: 意图识别测试用例（含追问继承用例）
        dialog_cases:
          - 单轮: [{"question": "..."}]
          - 多轮: [{"turns": ["第一轮", "第二轮", ...]}]
          - 可选字段: golden_answer（Answer Correctness 用）
        routing_cases: 意图领域评测用例 [{"turns": [...], "expected_agent": "campus_life"}]（expected_agent 兼容命名，比较实际识别领域）
        retrieval_cases: RAG 检索硬指标用例 [{"query": ..., "relevant_titles": [...]}]
          需要 knowledge_base（检索端 HitRate@K/Recall@K/MRR）。
        dataset: 用例来源标识（内置/自定义），写入 provenance。
        """
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {
            "relevance": [], "accuracy": [], "completeness": [], "helpfulness": [],
            "faithfulness": [], "answer_correctness": [], "citation_correctness": [],
        }

        # 1. 意图识别评测
        intent_metrics: Dict[str, Any] = {}
        if intent_cases:
            intent_metrics = await self._intent_evaluator.evaluate(intent_cases)
            passed = intent_metrics["accuracy"] >= self.PASS_THRESHOLD
            results.append(EvalResult(
                test_id="intent_recognition",
                passed=passed,
                scores={"accuracy": intent_metrics["accuracy"], "macro_f1": intent_metrics["macro_f1"]},
                detail=f"准确率 {intent_metrics['accuracy']:.1%}，Macro-F1 {intent_metrics['macro_f1']:.3f}",
                metadata={
                    "total": intent_metrics.get("total", 0),
                    "correct": intent_metrics.get("correct", 0),
                    "cases": intent_metrics.get("cases", []),
                },
            ))

        # 2. 对话质量评测（调用 orchestrator 产出回复，再用独立 Judge 模型评分）
        if dialog_cases:
            for i, case in enumerate(dialog_cases):
                case_results = await self._evaluate_dialog_case(case, i)
                results.extend(case_results)
                for r in case_results:
                    for k in all_scores:
                        if k in r.scores:
                            all_scores[k].append(r.scores[k])

        # 3. 意图领域评测（追问继承 / 请求句式是否正确识别领域）
        if routing_cases:
            routing_results = await self._evaluate_routing_cases(routing_cases)
            results.extend(routing_results)
            all_scores["routing_accuracy"] = [  # 兼容键名：实为领域识别准确率
                r.scores.get("accuracy", 0.0) for r in routing_results
            ]

        # 4. RAG 检索硬指标（HitRate@K / Recall@K / MRR，真实调用知识库，无 LLM）
        retrieval_metrics: Optional[Dict[str, Any]] = None
        if retrieval_cases and self._retrieval_evaluator is not None:
            retrieval_metrics = await self._retrieval_evaluator.run(retrieval_cases)
            results.append(EvalResult(
                test_id="rag_retrieval",
                passed=retrieval_metrics["hit_rate@K"] >= 0.5,
                scores={
                    "hit_rate@K": retrieval_metrics["hit_rate@K"],
                    "recall@K": retrieval_metrics["recall@K"],
                    "mrr": retrieval_metrics["mrr"],
                },
                detail=(
                    f"HitRate@{retrieval_metrics['top_k']} {retrieval_metrics['hit_rate@K']:.1%}，"
                    f"Recall@{retrieval_metrics['top_k']} {retrieval_metrics['recall@K']:.3f}，"
                    f"MRR {retrieval_metrics['mrr']:.3f}"
                ),
                metadata={"cases": retrieval_metrics["cases"]},
            ))
            all_scores["hit_rate@K"] = [retrieval_metrics["hit_rate@K"]]
            all_scores["recall@K"] = [retrieval_metrics["recall@K"]]
            all_scores["mrr"] = [retrieval_metrics["mrr"]]

        # 5. 汇总
        avg_scores = {
            k: round(statistics.mean(v), 4) for k, v in all_scores.items() if v
        }
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]

        passed_count = sum(1 for r in results if r.passed)
        pass_rate    = passed_count / len(results) if results else 0.0

        # 6. 回归检测
        regressions = self._detect_regressions(avg_scores)

        # 7. 优化建议
        recommendations = self._recommendations(avg_scores, intent_metrics)

        report = EvalReport(
            timestamp=datetime.now().isoformat(),
            total=len(results),
            passed=passed_count,
            pass_rate=round(pass_rate, 4),
            avg_scores=avg_scores,
            regressions=regressions,
            recommendations=recommendations,
            results=results,
            retrieval=retrieval_metrics,
            provenance={
                "dataset": dataset,
                "dataset_counts": {"intent": len(intent_cases or []), "routing": len(routing_cases or []), "retrieval": len(retrieval_cases or []), "dialog": len(dialog_cases or [])},
                "code_commit": self._git_commit(),
                "generator_model": self._model,
                "judge_model": self._judge_model,
                "threshold": self.PASS_THRESHOLD,
            },
            judge={
                "judge_independent": self._judge_independent,
                "note": "独立 Judge 需单独配置并人工核验；失败样本不应解释为模型质量分数。",
            },
        )
        self._history.append(report)
        if promote_baseline:
            self._save_baseline(report)
        return report

    @staticmethod
    def _git_commit() -> str:
        try:
            return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2).stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int) -> List[EvalResult]:
        """评测单轮或多轮对话用例。"""
        from agents.agent_orchestrator import Request as OrcReq

        questions = self._dialog_turns(case)
        if not questions:
            return []

        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        user_id = str(case.get("user_id") or "eval_user")
        golden_answer = str(case.get("golden_answer") or "").strip()  # Answer Correctness 用
        history: List[Dict[str, str]] = []
        results: List[EvalResult] = []

        for turn_idx, question in enumerate(questions):
            context = self._history_context(history)
            orch_req = OrcReq(
                message=question,
                user_id=user_id,
                conv_id=conv_id,
                context=context,
                history=history[-6:] if history else None,
            )
            orch_result = await self._orchestrator.run(orch_req)
            actual_answer = orch_result.response

            scores = await self._judge.judge(question, actual_answer, context=context or None)
            passed = scores.overall >= self.PASS_THRESHOLD
            score_dict = {
                "relevance": scores.relevance,
                "accuracy": scores.accuracy,
                "completeness": scores.completeness,
                "helpfulness": scores.helpfulness,
                "overall": scores.overall,
            }
            metadata: Dict[str, Any] = {
                "question": question,
                "response": actual_answer,
                "agent_type": orch_result.agent_type.value,
                "intent": orch_result.intent.value if orch_result.intent else None,
                "turn": turn_idx,
                "conv_id": conv_id,
                "judge_failed": scores.judge_failed,
                "judge_error": scores.error,
            }

            # 生成端 RAG 硬指标：Citation Correctness（确定性）+ Faithfulness（Judge）
            if self._retrieval_evaluator is not None:
                # 只用本次 Agent 真正调用 knowledge_search 获得的证据；禁止二次
                # 检索替代，从而避免把未被模型使用的材料当作“引用依据”。
                sources = list(getattr(orch_result, "tool_evidence", []) or [])
                source_titles = [str(s.get("title", "")) for s in sources]
                citation = citation_correctness(actual_answer, source_titles)
                if citation["score"] is not None:
                    score_dict["citation_correctness"] = citation["score"]
                metadata["citation"] = citation
                metadata["sources"] = source_titles

                if sources:
                    sources_text = "\n".join(
                        f"[{i + 1}] {s.get('title', '')}: {s.get('content', '')[:200]}"
                        for i, s in enumerate(sources)
                    )
                    faithfulness, faithfulness_failed = await self._judge.judge_faithfulness(
                        question, actual_answer, sources_text,
                    )
                    score_dict["faithfulness"] = faithfulness
                    metadata["faithfulness"] = faithfulness
                    metadata["faithfulness_failed"] = faithfulness_failed

            # Answer Correctness：用例提供 golden_answer 时才评测
            if golden_answer:
                correctness, correctness_failed = await self._judge.judge_answer_correctness(
                    question, actual_answer, golden_answer,
                )
                score_dict["answer_correctness"] = correctness
                metadata["answer_correctness"] = correctness
                metadata["answer_correctness_failed"] = correctness_failed

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": actual_answer})

            test_id = f"dialog_{case_idx}" if len(questions) == 1 else f"dialog_{case_idx}_turn_{turn_idx}"

            # 不达标样本留痕：Judge 自身失败单独记录（评判故障 ≠ 质量差），
            # 质量不达标则连同问题/回答/低分指标一并落日志，供事后复盘。
            # Judge 失败时分数是 0.5 兜底，必须排除在"低分指标"外避免误读。
            failed_flag_of = {
                "faithfulness": "faithfulness_failed",
                "answer_correctness": "answer_correctness_failed",
            }
            if scores.judge_failed:
                logger.warning(
                    f"[Eval] Judge 失败 test_id={test_id} conv_id={conv_id} "
                    f"question={str(question)[:_LOG_QUESTION_MAX]!r} error={scores.error}"
                )
            else:
                failed_flags = [
                    k for k in ("faithfulness_failed", "answer_correctness_failed")
                    if metadata.get(k)
                ]
                low = {
                    k: round(v, 3)
                    for k, v in score_dict.items()
                    if k != "overall" and isinstance(v, float) and v < self.PASS_THRESHOLD
                    and not metadata.get(failed_flag_of.get(k, ""), False)
                }
                if not passed or low or failed_flags:
                    logger.warning(
                        f"[Eval] 质量不达标 test_id={test_id} conv_id={conv_id} user_id={user_id} "
                        f"question={str(question)[:_LOG_QUESTION_MAX]!r} "
                        f"overall={scores.overall:.3f} 低分指标={low} "
                        f"judge_failed_flags={failed_flags or None} "
                        f"agent_type={orch_result.agent_type.value} "
                        f"intent={(orch_result.intent.value if orch_result.intent else None)} "
                        f"judge_model={self._judge_model} "
                        f"response={str(actual_answer or '')[:_LOG_RESPONSE_MAX]!r}"
                    )

            results.append(EvalResult(
                test_id=test_id,
                passed=passed,
                scores=score_dict,
                detail=f"Q: {question[:30]}... → 综合评分 {scores.overall:.3f}",
                metadata=metadata,
            ))

        return results

    async def _evaluate_routing_cases(self, cases: List[Dict[str, Any]]) -> List[EvalResult]:
        """
        意图领域评测：多轮对话跑完编排器，比较实际识别领域与期望领域。

        v3 语义：执行实体是职责角色（qa/executor），领域只作为挂载键；
        本评测验证"领域分类是否正确"（expected_agent 字段保留兼容命名，
        取值仍为领域值 academic/campus_life/...）。

        重点覆盖两类历史缺陷：
          1. 请求句式（"我要请假怎么走流程"）必须识别为对应领域
          2. 追问（"那几点开门呢？"）必须继承上一轮领域，不落回默认
        """
        from agents.agent_orchestrator import Request as OrcReq

        results: List[EvalResult] = []
        for idx, case in enumerate(cases):
            turns = self._dialog_turns(case)
            expected_agent = str(case.get("expected_agent", ""))
            if not turns or not expected_agent:
                continue

            conv_id = f"eval_routing_{idx}"
            user_id = str(case.get("user_id") or "eval_user")
            history: List[Dict[str, str]] = []
            passed = True
            details = []

            for turn_idx, question in enumerate(turns):
                orch_req = OrcReq(
                    message=question,
                    user_id=user_id,
                    conv_id=conv_id,
                    context=self._history_context(history),
                    history=history[-6:] if history else None,
                )
                orch_result = await self._orchestrator.run(orch_req)
                actual_domain = orch_result.domain.value if orch_result.domain else ""
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": orch_result.response})

                turn_ok = actual_domain == expected_agent
                passed = passed and turn_ok
                details.append({
                    "turn": turn_idx,
                    "question": question,
                    "expected_agent": expected_agent,
                    "actual_domain": actual_domain,
                    "ok": turn_ok,
                })

            results.append(EvalResult(
                test_id=f"routing_{idx}",
                passed=passed,
                scores={"accuracy": 1.0 if passed else 0.0},
                detail=f"期望领域 {expected_agent} → {'全部命中' if passed else '存在偏离'}: " +
                       "; ".join(
                           f"turn{d['turn']}: {d['actual_domain']}{'(✓)' if d['ok'] else '(✗)'}"
                           for d in details
                       ),
                metadata={"case": details},
            ))
        return results

    @staticmethod
    def _dialog_turns(case: Dict[str, Any]) -> List[str]:
        turns = case.get("turns")
        if isinstance(turns, list):
            return [str(t) for t in turns if str(t).strip()]
        question = case.get("question")
        return [str(question)] if question else []

    @staticmethod
    def _history_context(history: List[Dict[str, str]]) -> str:
        if not history:
            return ""
        lines = [f"{m['role']}: {m['content']}" for m in history[-8:]]
        return "[评测多轮历史]\n" + "\n".join(lines)

    def _detect_regressions(self, current: Dict[str, float]) -> List[str]:
        """与上一次评测对比，找出退化超过 5% 的指标。"""
        prev_report = self._history[-1] if self._history else self._baseline
        if prev_report is None:
            return []
        prev = prev_report.avg_scores
        regressions = []
        for metric, value in current.items():
            if metric in prev and prev[metric] > 0:
                delta = (value - prev[metric]) / prev[metric]
                if delta < -0.05:
                    regressions.append(
                        f"{metric}: {prev[metric]:.3f} → {value:.3f} (退化 {abs(delta):.1%})"
                    )
        return regressions

    def _recommendations(
        self,
        scores: Dict[str, float],
        intent_metrics: Dict[str, Any],
    ) -> List[str]:
        recs = []
        if scores.get("intent_accuracy", 1.0) < 0.90:
            recs.append("意图识别准确率 < 90%：增加 Few-shot 示例，或对低 F1 的意图类别补充训练数据")
        if scores.get("relevance", 1.0) < 0.75:
            recs.append("相关性偏低：检查 Agent system_prompt，确保 Agent 聚焦于用户问题")
        if scores.get("completeness", 1.0) < 0.75:
            recs.append("完整性偏低：Agent 可能过早结束回答，考虑在 prompt 中要求提供完整解决方案")
        if scores.get("helpfulness", 1.0) < 0.75:
            recs.append("有用性偏低：回答可能过于抽象，考虑要求 Agent 提供具体操作步骤")
        if scores.get("hit_rate@K", 1.0) < 0.8:
            recs.append("检索 HitRate@K < 80%：检查知识库覆盖度与分块质量，必要时补充文档或调低相关性阈值")
        if scores.get("recall@K", 1.0) < 0.6:
            recs.append("检索 Recall@K 偏低：确认查询改写链路生效（Agent 调用 knowledge_search 应走 search_with_rewrite）")
        if scores.get("faithfulness", 1.0) < 0.8:
            recs.append("回答忠实性偏低：提示 Agent 严格基于检索结果作答，禁止编造来源中不存在的信息")
        if not recs:
            recs.append("所有指标均达标，继续保持")
        return recs

    @property
    def history(self) -> List[EvalReport]:
        return self._history

    def _load_baseline(self) -> Optional[EvalReport]:
        if not self._baseline_path or not self._baseline_path.exists():
            return None
        try:
            data = json.loads(self._baseline_path.read_text(encoding="utf-8"))
            return self._report_from_dict(data)
        except Exception as ex:
            logger.warning(f"读取评测基线失败: {ex}")
            return None

    def _save_baseline(self, report: EvalReport) -> None:
        if not self._baseline_path:
            return
        try:
            self._baseline_path.parent.mkdir(parents=True, exist_ok=True)
            self._baseline_path.write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._baseline = report
        except Exception as ex:
            logger.warning(f"保存评测基线失败: {ex}")

    @staticmethod
    def _report_from_dict(data: Dict[str, Any]) -> EvalReport:
        return EvalReport(
            timestamp=data.get("timestamp", ""),
            total=int(data.get("total", 0)),
            passed=int(data.get("passed", 0)),
            pass_rate=float(data.get("pass_rate", 0.0)),
            avg_scores=dict(data.get("avg_scores", {})),
            regressions=list(data.get("regressions", [])),
            recommendations=list(data.get("recommendations", [])),
            results=[
                EvalResult(
                    test_id=r.get("test_id", ""),
                    passed=bool(r.get("passed", False)),
                    scores=dict(r.get("scores", {})),
                    detail=r.get("detail", ""),
                    metadata=dict(r.get("metadata", {})),
                )
                for r in data.get("results", [])
            ],
            retrieval=data.get("retrieval"),
            provenance=dict(data.get("provenance", {})),
            judge=dict(data.get("judge", {})),
        )


# ── 内置测试用例（开箱即用）──────────────────────────────────────────────────

DEFAULT_INTENT_CASES: List[IntentTestCase] = [
    # 领域维度（路由依据）—— 覆盖"请求句式"不再丢领域
    IntentTestCase("这学期选课什么时候开始？",   "academic"),
    IntentTestCase("帮我查一下我的课表",          "personal"),
    IntentTestCase("今天有什么课？",              "personal"),
    IntentTestCase("我最近的考试安排？",          "personal"),
    IntentTestCase("南校区食堂几点关门？",        "campus_life"),
    # 校园卡补办走事务/材料指引 → affairs（与 domains 0.95 规则、demo_cases 一致）
    IntentTestCase("校园卡丢了怎么补办？",        "affairs"),
    IntentTestCase("帮我查一下校园卡余额",        "campus_life"),
    IntentTestCase("奖学金什么时候评定？",        "affairs"),
    IntentTestCase("我要请假怎么走流程",          "affairs"),
    IntentTestCase("教务系统登录不上怎么办？",    "it_help"),
    IntentTestCase("校园网连不上",                "it_help"),
    # 动作维度（行为依据）
    IntentTestCase("你好",                        "greeting"),
    IntentTestCase("这个助手很实用！",            "feedback"),
    IntentTestCase("宿舍热水一直不来！",          "complaint"),
    # 追问继承（对话感知）：短句无领域关键词，应从历史继承领域
    IntentTestCase(
        "那几点开门呢？",
        "campus_life",
        context={"history": [
            {"role": "user", "content": "南校区食堂几点关门？"},
            {"role": "assistant", "content": "南校区食堂一般晚上七点关门。"},
        ]},
    ),
    IntentTestCase(
        "怎么重置？",
        "it_help",
        context={"history": [
            {"role": "user", "content": "教务系统密码忘了怎么办？"},
            {"role": "assistant", "content": "可以通过统一身份认证自助重置密码。"},
        ]},
    ),
]

DEFAULT_DIALOG_CASES: List[Dict[str, Any]] = [
    {"question": "这学期选课什么时候开始？我想提前准备一下"},
    {"question": "教务系统一直登录不上，报错说密码错误"},
    {"question": "南校区食堂晚上几点关门？"},
    {"question": "我要办在读证明，需要带什么材料？"},
    {"turns": ["你好，我想问下校车时刻", "南校区到北校区的", "末班车是几点？"]},
    {"turns": ["南校区食堂几点关门？", "那几点开门呢？"]},
]

# 路由评测用例：验证请求句式与追问继承的路由正确性
DEFAULT_ROUTING_CASES: List[Dict[str, Any]] = [
    {"turns": ["我要请假怎么走流程"], "expected_agent": "affairs"},
    # 校园卡补办 → affairs（与 intent 用例、demo_cases 口径一致）
    {"turns": ["校园卡丢了怎么补办"], "expected_agent": "affairs"},
    {"turns": ["帮我查一下校园卡余额"], "expected_agent": "campus_life"},
    {"turns": ["南校区食堂几点关门？", "那几点开门呢？"], "expected_agent": "campus_life"},
    {"turns": ["教务系统登录不上怎么办？", "怎么重置密码？"], "expected_agent": "it_help"},
]

# RAG 检索硬指标用例：query → 知识库默认文档中应被召回的相关标题
DEFAULT_RETRIEVAL_CASES: List[RetrievalTestCase] = [
    RetrievalTestCase("这学期选课什么时候开始？", ["选课指南"]),
    RetrievalTestCase("选课分几个阶段？", ["选课指南"]),
    RetrievalTestCase("校园穿梭车怎么预约？", ["校园穿梭车（校车）"]),
    RetrievalTestCase("南校区食堂几点关门？", ["食堂与餐饮"]),
    RetrievalTestCase("宿舍几点关门？", ["宿舍管理"]),
    RetrievalTestCase("图书馆开放时间？", ["图书馆"]),
    RetrievalTestCase("校历有什么重要时间节点？", ["校历与重要时间节点"]),
]
