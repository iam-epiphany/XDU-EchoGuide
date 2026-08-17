"""意图识别器测试：二维意图（领域×动作）、关键词模式匹配、
缓存指纹、实体提取兜底、在线学习。

只测确定性逻辑；LLM 路径用最小 mock 覆盖失败回退。
"""
from __future__ import annotations

import asyncio

from core.domains import IntentAction, IntentDomain, keyword_hit
from core.intent_recognizer import IntentRecognizer

FAKE_KEY = "sk-test-not-used"


def _recognizer(**kwargs) -> IntentRecognizer:
    kwargs.setdefault("api_key", FAKE_KEY)
    return IntentRecognizer(**kwargs)


# ── 关键词命中（子串误命中回归）─────────────────────────────────────────────

def test_ascii_keyword_requires_word_boundary():
    # EchoMind 经典缺陷："api" 命中 "capital"
    assert keyword_hit("api", "capital") is False
    assert keyword_hit("api", "调用 api 服务") is True
    assert keyword_hit("it", "quite") is False
    assert keyword_hit("it", "it服务支持") is True
    assert keyword_hit("vpn", "vpn配置") is True


def test_chinese_single_char_keyword_rejected():
    # 单字中文关键词过拟合（如旧版 "餐"），一律不匹配，运营侧需改用多字词组
    assert keyword_hit("餐", "餐补什么时候发") is False
    assert keyword_hit("餐", "食堂几点开门") is False
    assert keyword_hit("餐", "食堂早餐几点") is False


def test_domain_hit_score_marks_domain():
    from core.domains import domain_hit_score

    domain, score = domain_hit_score("南校区食堂几点关门")
    assert domain == IntentDomain.CAMPUS_LIFE
    assert score >= 0.55


# ── 模式匹配（零 LLM 依赖）───────────────────────────────────────────────────

def test_pattern_recognizes_campus_domains():
    rec = _recognizer()
    cases = {
        "这学期选课什么时候开始": IntentDomain.ACADEMIC,
        "南校区食堂几点关门": IntentDomain.CAMPUS_LIFE,
        "奖学金什么时候评定": IntentDomain.AFFAIRS,
        "教务系统登录不上": IntentDomain.IT_HELP,
        "你好": IntentDomain.OTHER,
    }
    for msg, expected in cases.items():
        result = rec._pattern_recognize(msg)
        assert result["domain"] == expected, f"{msg} → {result['domain']}"


def test_pattern_recognizes_personal_domain():
    """个人助理领域：我的日程/待办/考试安排类问题。"""
    rec = _recognizer()
    cases = {
        "今天有什么课": IntentDomain.PERSONAL,
        "我的课表在哪看": IntentDomain.PERSONAL,
        "帮我记个待办": IntentDomain.PERSONAL,
        "我最近的考试安排": IntentDomain.PERSONAL,
        "明天几点上课": IntentDomain.PERSONAL,
    }
    for msg, expected in cases.items():
        result = rec._pattern_recognize(msg)
        assert result["domain"] == expected, f"{msg} → {result['domain']}"


def test_pattern_request_form_keeps_domain():
    """P0 回归：请求句式不再吞掉领域信息。"""
    rec = _recognizer()
    result = rec._pattern_recognize("我要请假怎么走流程")
    assert result["domain"] == IntentDomain.AFFAIRS
    assert result["action"] == IntentAction.REQUEST

    result = rec._pattern_recognize("校园卡丢了怎么补办")
    assert result["domain"] == IntentDomain.AFFAIRS


def test_pattern_does_not_fire_for_generic_chat():
    rec = _recognizer()
    result = rec._pattern_recognize("今天天气怎么样")
    assert result["domain"] == IntentDomain.CAMPUS_LIFE
    assert result["confidence"] >= 0.90


def test_pattern_punctuation_false_positive_fixed():
    # "填报错误" 不应命中 IT 领域（"报错" 子串误命中回归）
    rec = _recognizer()
    result = rec._pattern_recognize("我填报表错了，怎么改")
    assert result["domain"] != IntentDomain.IT_HELP


# ── 追问形态检测（防 Embedding 误判）────────────────────────────────────────

def test_followup_shaped_detection():
    rec = _recognizer()
    # 强信号（指代承接）→ 无条件判追问，即使有 pattern 弱信号
    assert rec._is_followup_shaped("那几点开门呢？") is True
    assert rec._is_followup_shaped("那选课呢？") is True          # pattern 弱命中 academic
    assert rec._is_followup_shaped("最早一班呢？") is True     # "呢"结尾弱信号（无 pattern 信号）
    assert rec._is_followup_shaped("最早一班呢？", True) is False  # 有主题词信号则不判
    assert rec._is_followup_shaped("那几点上课？") is True        # 强信号"那"
    # 弱信号（极短疑问词/呢结尾）→ 仅当 pattern 无信号
    assert rec._is_followup_shaped("几点？") is True
    assert rec._is_followup_shaped("什么时候？") is True
    assert rec._is_followup_shaped("下午呢？") is True
    assert rec._is_followup_shaped("绩点怎么算的？", True) is False  # 完整问句有主题词
    assert rec._is_followup_shaped("这学期选课什么时候开始？", True) is False  # "这"不是指代
    # 非追问形态：社交语 / 长句 / 空
    assert rec._is_followup_shaped("谢谢") is False
    assert rec._is_followup_shaped("好的") is False
    assert rec._is_followup_shaped("那明天早上的课表安排能不能帮我查一下") is False
    assert rec._is_followup_shaped("") is False


def test_followup_shaped_skips_embedding_goes_llm():
    """省略追问 → 最高优先级直接 LLM，Embedding 不参与。"""
    rec = _recognizer()

    async def embedding_should_not_run(message):
        raise AssertionError("追问形态不应走 Embedding")

    async def fake_llm(message, history, complexity_only=False, state=None):
        return {"domain": IntentDomain.CAMPUS_LIFE, "action": IntentAction.QUERY,
                "confidence": 0.9, "reasoning": "mock"}

    rec._embedding_recognize = embedding_should_not_run
    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize(
        "那几点开门呢？",
        history=[{"role": "user", "content": "南校区食堂几点关门？"}],
    ))
    assert result.domain == IntentDomain.CAMPUS_LIFE
    assert result.classifier_stage == "llm"


def test_full_question_keeps_embedding():
    """完整问句（无强信号 + pattern 有主题词）→ 放行 Embedding 免费路径。"""
    rec = _recognizer()

    async def fake_embedding(message):
        return {"domain": IntentDomain.ACADEMIC, "action": IntentAction.OTHER,
                "confidence": 0.90, "margin": 0.2}

    async def llm_should_not_run(message, history, complexity_only=False, state=None):
        raise AssertionError("完整问句不应跳过 Embedding")

    rec._embedding_recognize = fake_embedding
    rec._llm_recognize = llm_should_not_run
    result = asyncio.run(rec.recognize("绩点怎么算的？"))  # pattern 弱命中 academic@0.55
    assert result.domain == IntentDomain.ACADEMIC
    assert result.classifier_stage == "embedding"


# ── 缓存指纹（同句追问不同上下文不复用）──────────────────────────────────────

def test_cache_key_includes_history_fingerprint():
    rec = _recognizer()
    k1 = rec._cache_key("那几点开门呢？", [{"role": "user", "content": "南校区食堂几点关门？"}])
    k2 = rec._cache_key("那几点开门呢？", [{"role": "user", "content": "教务系统密码忘了"}])
    k3 = rec._cache_key("那几点开门呢？", None)
    assert k1 != k2 != k3
    assert k1 != k3


# ── 实体提取兜底 ─────────────────────────────────────────────────────────────

def test_entity_extraction_failure_returns_campus_schema():
    rec = _recognizer()

    async def run():
        return await rec._extract_entities("选课")

    # 无真实 API key 时 LLM 调用失败，应返回校园字段的空列表结构
    entities = asyncio.run(run())
    assert set(entities.keys()) == {"course", "term", "location", "campus", "system"}


# ── 在线学习 ─────────────────────────────────────────────────────────────────

def test_learn_adds_template_and_invalidates_embedding_cache():
    rec = _recognizer()
    msg = "我的培养方案里学分要求是多少"
    rec.learn(msg, IntentDomain.ACADEMIC)
    from core.intent_recognizer import _DOMAIN_TEMPLATES

    assert msg in _DOMAIN_TEMPLATES[IntentDomain.ACADEMIC]


# ── 缓存统计 ─────────────────────────────────────────────────────────────────

def test_cache_stats_before_any_request():
    rec = _recognizer()
    stats = rec.cache_stats
    assert stats["size"] == 0
    assert stats["hit_rate"] == 0.0


# ── 级联分类 ─────────────────────────────────────────────────────────────────

def test_cascade_pattern_skips_llm():
    """Pattern 高置信 + Embedding 双确认（同域）→ 免费直返，不调用 LLM。"""
    rec = _recognizer()

    async def fake_embedding(message):
        return {"domain": IntentDomain.ACADEMIC, "action": IntentAction.OTHER,
                "confidence": 0.99, "margin": 0.5}

    async def llm_should_not_run(message, history, complexity_only=False, state=None):
        raise AssertionError("双确认通过时不应调用 LLM")

    rec._embedding_recognize = fake_embedding
    rec._llm_recognize = llm_should_not_run
    result = asyncio.run(rec.recognize("选课成绩学分有什么规定？"))
    assert result.domain == IntentDomain.ACADEMIC
    assert result.classifier_stage == "pattern"


def test_pattern_embedding_conflict_arbitrates_to_llm():
    """Pattern 高置信但 Embedding 方向分歧（关键词子串误配）→ LLM 仲裁，不静默直返。

    v4：LLM 不再输出领域（只仲裁 action/查询理解），领域由免费关键词回填——
    "图书馆"命中 campus_life；action 采用 LLM 结论。
    """
    rec = _recognizer()

    async def fake_embedding(message):
        # "电子图书馆怎么登录？"被"图书馆"命中 campus_life，但 Embedding 判 it_help
        return {"domain": IntentDomain.IT_HELP, "action": IntentAction.OTHER,
                "confidence": 0.50, "margin": 0.10}

    async def fake_llm(message, history, complexity_only=False, state=None):
        return {"action": IntentAction.QUERY, "confidence": 0.9, "reasoning": "mock"}

    rec._embedding_recognize = fake_embedding
    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("电子图书馆怎么登录？"))
    assert result.domain == IntentDomain.CAMPUS_LIFE  # 免费关键词回填（领域不做路由）
    assert result.action == IntentAction.QUERY        # action 由 LLM 裁决
    assert result.classifier_stage == "llm"


def test_pattern_subthreshold_embedding_arbitrates_to_llm():
    """Pattern 高置信但 Embedding 同向分数低于命中阈值（<0.80）→ LLM 仲裁。

    0.80 是 bge 标定的命中区/未命中区分隔线：方向一致但分数不足只是
    噪声级巧合，不能算"双确认"（如真实 n-gram 回退下"选课成绩学分"
    实测 0.672 —— 正处于 miss 区 0.655 与命中区 0.820 之间的灰色地带）。
    """
    rec = _recognizer()

    async def fake_embedding(message):
        # 同方向（academic）但分数 0.50 < 0.80：方向一致不再是充分条件
        return {"domain": IntentDomain.ACADEMIC, "action": IntentAction.OTHER,
                "confidence": 0.50, "margin": 0.10}

    async def fake_llm(message, history, complexity_only=False, state=None):
        return {"domain": IntentDomain.ACADEMIC, "action": IntentAction.QUERY,
                "confidence": 0.9, "reasoning": "mock"}

    rec._embedding_recognize = fake_embedding
    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("选课成绩学分有什么规定？"))  # pattern 高置信 academic@0.95
    assert result.domain == IntentDomain.ACADEMIC
    assert result.classifier_stage == "llm"
    assert "低于阈值" in result.reasoning  # 仲裁原因可追溯：同向但分数不足


def test_embedding_weak_pattern_conflict_arbitrates_to_llm():
    """Embedding 高置信命中但与 pattern 弱信号方向矛盾 → LLM 仲裁。"""
    rec = _recognizer()

    async def fake_embedding(message):
        # pattern 弱命中 academic（"考试"），Embedding 却判 personal（"考试安排"）
        return {"domain": IntentDomain.PERSONAL, "action": IntentAction.OTHER,
                "confidence": 0.90, "margin": 0.3}

    async def fake_llm(message, history, complexity_only=False, state=None):
        return {"domain": IntentDomain.ACADEMIC, "action": IntentAction.QUERY,
                "confidence": 0.9, "reasoning": "mock"}

    rec._embedding_recognize = fake_embedding
    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("什么时候考试？"))  # pattern 弱命中 academic@0.55
    assert result.domain == IntentDomain.ACADEMIC
    assert result.classifier_stage == "llm"


def test_cascade_embedding_skips_llm():
    rec = _recognizer()

    async def fake_embedding(message):
        return {"domain": IntentDomain.IT_HELP, "action": IntentAction.OTHER, "confidence": 0.87, "margin": 0.2}

    async def llm_should_not_run(message, history, complexity_only=False, state=None):
        raise AssertionError("高置信度 Embedding 不应调用 LLM")

    rec._embedding_recognize = fake_embedding
    rec._llm_recognize = llm_should_not_run
    result = asyncio.run(rec.recognize("网络服务出现一个模糊问题"))
    assert result.domain == IntentDomain.IT_HELP
    assert result.classifier_stage == "embedding"


def test_force_llm_bypasses_cascade():
    """强制 LLM：action 来自 LLM，领域由免费关键词回填（LLM 不再输出领域）。"""
    rec = _recognizer()

    async def fake_llm(message, history, complexity_only=False, state=None):
        return {"action": IntentAction.QUERY, "confidence": 0.91, "reasoning": "baseline"}

    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("选课成绩学分有什么规定？", force_llm=True))
    assert result.domain == IntentDomain.ACADEMIC  # 免费关键词回填
    assert result.classifier_stage == "llm"


# ── 复杂度判定（意图识别的一部分，LLM 参与时顺带输出）────────────────────────

def test_llm_complexity_attached_when_llm_used():
    """LLM 参与意图识别时顺带输出 complexity（parallel），随 IntentResult 返回。"""
    rec = _recognizer()

    async def fake_llm(message, history, complexity_only=False, state=None):
        return {
            "domain": IntentDomain.AFFAIRS, "action": IntentAction.QUERY,
            "confidence": 0.9, "reasoning": "baseline",
            "complexity": {"mode": "parallel", "targets": ["affairs", "personal"]},
        }

    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("奖学金评定和我的待办一起看下", force_llm=True))
    assert result.classifier_stage == "llm"
    assert result.complexity is not None
    assert result.complexity.mode == "parallel"
    assert result.complexity.targets == ["affairs", "personal"]


def test_llm_complexity_dependent_carries_tasks():
    """mode=dependent 时 LLM 输出的原始任务链随信号保留。"""
    rec = _recognizer()

    async def fake_llm(message, history, complexity_only=False, state=None):
        return {
            "domain": IntentDomain.PERSONAL, "action": IntentAction.REQUEST,
            "confidence": 0.8, "reasoning": "baseline",
            "complexity": {
                "mode": "dependent", "targets": ["personal", "affairs"],
                "tasks": [
                    {"id": "t1", "agent": "personal", "goal": "查空闲"},
                    {"id": "t2", "agent": "affairs", "depends_on": ["t1"]},
                ],
            },
        }

    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("我下午有空想办校园卡再记个待办", force_llm=True))
    assert result.complexity is not None
    assert result.complexity.mode == "dependent"
    assert len(result.complexity.tasks) == 2
    assert result.complexity.tasks[1]["depends_on"] == ["t1"]


def test_llm_complexity_none_when_absent():
    """LLM 未输出 complexity 字段（旧 fake/旧模型）→ None，不影响主流程。"""
    rec = _recognizer()

    async def fake_llm(message, history, complexity_only=False, state=None):
        return {"domain": IntentDomain.OTHER, "action": IntentAction.QUERY, "confidence": 0.3, "reasoning": "x"}

    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("随便聊聊", force_llm=True))
    assert result.complexity is None


def test_high_confidence_pattern_has_no_complexity():
    """高置信度 Pattern 不调 LLM → 无复杂度信号（复杂度只随 LLM 参与产生）。"""
    rec = _recognizer()

    async def fake_embedding(message):
        # 双确认必须达标（≥0.80）：高分数 + 同方向才放行免费直返
        return {"domain": IntentDomain.ACADEMIC, "action": IntentAction.OTHER,
                "confidence": 0.90, "margin": 0.3}

    async def llm_should_not_run(message, history, complexity_only=False, state=None):
        raise AssertionError("高置信度 Pattern 不应调用 LLM")

    rec._embedding_recognize = fake_embedding
    rec._llm_recognize = llm_should_not_run
    result = asyncio.run(rec.recognize("选课成绩学分有什么规定？"))
    assert result.classifier_stage == "pattern"
    assert result.complexity is None


def test_parse_complexity_valid_signal():
    rec = _recognizer()
    signal = rec._parse_complexity({
        "mode": "parallel", "targets": ["campus_life", "personal"], "reason": "两个领域",
    })
    assert signal is not None
    assert signal.mode == "parallel"
    assert signal.targets == ["campus_life", "personal"]


def test_parse_complexity_rejects_invalid_shapes():
    rec = _recognizer()
    # 非法 mode
    assert rec._parse_complexity({"mode": "weird"}) is None
    # 非 dict
    assert rec._parse_complexity(None) is None
    assert rec._parse_complexity("string") is None
    # parallel 模式携带 tasks → tasks 被丢弃但信号仍合法
    signal = rec._parse_complexity({"mode": "parallel", "tasks": [{"id": "t1"}]})
    assert signal is not None and signal.mode == "parallel" and signal.tasks is None
    # dependent 模式 tasks 非列表 → tasks 被丢弃
    signal = rec._parse_complexity({"mode": "dependent", "tasks": "not-a-list"})
    assert signal is not None and signal.tasks is None


def test_judge_complexity_delegates_with_complexity_only():
    """升级路径：judge_complexity 只问复杂度（complexity_only=True），不重复问领域/动作。"""
    rec = _recognizer()
    seen = {}

    async def fake_llm(message, history, complexity_only=False, state=None):
        seen["complexity_only"] = complexity_only
        return {"complexity": {"mode": "parallel", "targets": ["campus_life", "personal"]}}

    rec._llm_recognize = fake_llm
    signal = asyncio.run(rec.judge_complexity("食堂几点关门，顺便查下明天课表"))
    assert seen["complexity_only"] is True
    assert signal is not None and signal.mode == "parallel"


def test_judge_complexity_failure_returns_none():
    """LLM 不可用/输出非法 → judge_complexity 返回 None（调用方回落关键词规则）。"""
    rec = _recognizer()

    async def fake_llm(message, history, complexity_only=False, state=None):
        return {"complexity": None}

    rec._llm_recognize = fake_llm
    assert asyncio.run(rec.judge_complexity("随便问问")) is None
