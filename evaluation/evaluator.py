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
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.intent_recognizer import IntentAction, IntentDomain, IntentRecognizer

logger = logging.getLogger(__name__)


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
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text
                data = self._parse_scores(raw)
                if data is None:
                    raise ValueError("Judge 输出缺少 JSON")
                return QualityScores(**data)
            except Exception as ex:
                logger.warning(f"LLM Judge 第 {attempt + 1} 次失败: {ex}")
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
            "relevance": float(data.get("relevance", 0.5)),
            "accuracy": float(data.get("accuracy", 0.5)),
            "completeness": float(data.get("completeness", 0.5)),
            "helpfulness": float(data.get("helpfulness", 0.5)),
        }

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")


# ── 意图识别评测 ──────────────────────────────────────────────────────────────

class IntentEvaluator:
    """
    评测意图识别的准确率和 F1（领域 × 动作双维度）。

    expected_intent 取值约定：
      - 领域值（academic/campus_life/affairs/it_help/other）→ 比较预测的 domain
      - 动作值（query/request/greeting/complaint/feedback/escalation）→ 比较预测的 action
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
    ):
        """
        双模型 LLM-as-Judge：

        生成模型（api_key/base_url/model）与评判模型（judge_*）可分离，
        消除"自己给自己打分"的自评偏差。judge_* 缺省时退化为同模型（向后兼容）。
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
        self._intent_evaluator = IntentEvaluator(recognizer)
        self._history:         List[EvalReport] = []
        self._baseline_path = pathlib.Path(baseline_path) if baseline_path else None
        self._baseline: Optional[EvalReport] = self._load_baseline()

    async def run(
        self,
        intent_cases:    Optional[List[IntentTestCase]] = None,
        dialog_cases:    Optional[List[Dict[str, Any]]] = None,
        routing_cases:   Optional[List[Dict[str, Any]]] = None,
    ) -> EvalReport:
        """
        运行完整评测。

        intent_cases: 意图识别测试用例（含追问继承用例）
        dialog_cases:
          - 单轮: [{"question": "..."}]
          - 多轮: [{"turns": ["第一轮", "第二轮", ...]}]
        routing_cases: 路由评测用例 [{"turns": [...], "expected_agent": "campus_life"}]
        """
        results: List[EvalResult] = []
        all_scores: Dict[str, List[float]] = {
            "relevance": [], "accuracy": [], "completeness": [], "helpfulness": []
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

        # 3. 路由评测（追问继承 / 请求句式是否路由到正确 Agent）
        if routing_cases:
            routing_results = await self._evaluate_routing_cases(routing_cases)
            results.extend(routing_results)
            all_scores["routing_accuracy"] = [
                r.scores.get("accuracy", 0.0) for r in routing_results
            ]

        # 4. 汇总
        avg_scores = {
            k: round(statistics.mean(v), 4) for k, v in all_scores.items() if v
        }
        if intent_metrics:
            avg_scores["intent_accuracy"] = intent_metrics["accuracy"]

        passed_count = sum(1 for r in results if r.passed)
        pass_rate    = passed_count / len(results) if results else 0.0

        # 5. 回归检测
        regressions = self._detect_regressions(avg_scores)

        # 6. 优化建议
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
        )
        self._history.append(report)
        self._save_baseline(report)
        return report

    async def _evaluate_dialog_case(self, case: Dict[str, Any], case_idx: int) -> List[EvalResult]:
        """评测单轮或多轮对话用例。"""
        from agents.agent_orchestrator import Request as OrcReq

        questions = self._dialog_turns(case)
        if not questions:
            return []

        conv_id = str(case.get("conv_id") or f"eval_{case_idx}")
        user_id = str(case.get("user_id") or "eval_user")
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

            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": actual_answer})

            test_id = f"dialog_{case_idx}" if len(questions) == 1 else f"dialog_{case_idx}_turn_{turn_idx}"
            results.append(EvalResult(
                test_id=test_id,
                passed=passed,
                scores={
                    "relevance": scores.relevance,
                    "accuracy": scores.accuracy,
                    "completeness": scores.completeness,
                    "helpfulness": scores.helpfulness,
                    "overall": scores.overall,
                },
                detail=f"Q: {question[:30]}... → 综合评分 {scores.overall:.3f}",
                metadata={
                    "question": question,
                    "response": actual_answer,
                    "agent_type": orch_result.agent_type.value,
                    "intent": orch_result.intent.value if orch_result.intent else None,
                    "turn": turn_idx,
                    "conv_id": conv_id,
                    "judge_failed": scores.judge_failed,
                    "judge_error": scores.error,
                },
            ))

        return results

    async def _evaluate_routing_cases(self, cases: List[Dict[str, Any]]) -> List[EvalResult]:
        """
        路由评测：多轮对话跑完编排器，比较实际 Agent 与期望 Agent。

        重点覆盖两类历史缺陷：
          1. 请求句式（"我要请假怎么走流程"）必须路由到领域 Agent
          2. 追问（"那几点开门呢？"）必须继承上一轮领域，不落到默认 Agent
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
                actual_agent = orch_result.agent_type.value
                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": orch_result.response})

                turn_ok = actual_agent == expected_agent
                passed = passed and turn_ok
                details.append({
                    "turn": turn_idx,
                    "question": question,
                    "expected_agent": expected_agent,
                    "actual_agent": actual_agent,
                    "domain": orch_result.domain.value if orch_result.domain else None,
                    "ok": turn_ok,
                })

            results.append(EvalResult(
                test_id=f"routing_{idx}",
                passed=passed,
                scores={"accuracy": 1.0 if passed else 0.0},
                detail=f"期望 {expected_agent} → {'全部命中' if passed else '存在偏离'}: " +
                       "; ".join(
                           f"turn{d['turn']}: {d['actual_agent']}{'(✓)' if d['ok'] else '(✗)'}"
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
        )


# ── 内置测试用例（开箱即用）──────────────────────────────────────────────────

DEFAULT_INTENT_CASES: List[IntentTestCase] = [
    # 领域维度（路由依据）—— 覆盖"请求句式"不再丢领域
    IntentTestCase("这学期选课什么时候开始？",   "academic"),
    IntentTestCase("帮我查一下我的课表",          "personal"),
    IntentTestCase("今天有什么课？",              "personal"),
    IntentTestCase("我最近的考试安排？",          "personal"),
    IntentTestCase("南校区食堂几点关门？",        "campus_life"),
    IntentTestCase("校园卡丢了怎么补办？",        "campus_life"),
    IntentTestCase("帮我查一下校园卡余额",        "campus_life"),
    IntentTestCase("奖学金什么时候评定？",        "affairs"),
    IntentTestCase("我要请假怎么走流程",          "affairs"),
    IntentTestCase("教务系统登录不上怎么办？",    "it_help"),
    IntentTestCase("校园网连不上",                "it_help"),
    # 动作维度（行为依据）
    IntentTestCase("我要找辅导员",                "escalation"),
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
    {"turns": ["校园卡丢了怎么补办"], "expected_agent": "campus_life"},
    {"turns": ["帮我查一下校园卡余额"], "expected_agent": "campus_life"},
    {"turns": ["南校区食堂几点关门？", "那几点开门呢？"], "expected_agent": "campus_life"},
    {"turns": ["教务系统登录不上怎么办？", "怎么重置密码？"], "expected_agent": "it_help"},
]
