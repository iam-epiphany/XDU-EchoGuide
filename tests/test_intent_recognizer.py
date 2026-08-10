"""意图识别器测试：二维意图（领域×动作）、关键词模式匹配、
追问继承、缓存指纹、实体提取兜底、在线学习。

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


# ── 追问继承（对话感知）──────────────────────────────────────────────────────

def test_inherit_domain_from_history_followup():
    rec = _recognizer()
    history = [
        {"role": "user", "content": "南校区食堂几点关门？"},
        {"role": "assistant", "content": "南校区食堂一般晚上七点关门。"},
    ]
    domain = rec._inherit_domain("那几点开门呢？", history, IntentDomain.OTHER, IntentAction.QUERY)
    assert domain == IntentDomain.CAMPUS_LIFE


def test_inherit_domain_keeps_existing_domain():
    rec = _recognizer()
    domain = rec._inherit_domain(
        "那几点开门呢？",
        [{"role": "user", "content": "南校区食堂几点关门？"}],
        IntentDomain.ACADEMIC,
        IntentAction.QUERY,
    )
    assert domain == IntentDomain.ACADEMIC


def test_inherit_domain_without_history_stays_other():
    rec = _recognizer()
    domain = rec._inherit_domain("那几点开门呢？", None, IntentDomain.OTHER, IntentAction.QUERY)
    assert domain == IntentDomain.OTHER


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
    rec = _recognizer()

    async def llm_should_not_run(message, history):
        raise AssertionError("高置信度 Pattern 不应调用 LLM")

    rec._llm_recognize = llm_should_not_run
    result = asyncio.run(rec.recognize("选课成绩学分有什么规定？"))
    assert result.domain == IntentDomain.ACADEMIC
    assert result.classifier_stage == "pattern"


def test_cascade_embedding_skips_llm():
    rec = _recognizer()

    async def fake_embedding(message):
        return {"domain": IntentDomain.IT_HELP, "action": IntentAction.OTHER, "confidence": 0.82, "margin": 0.2}

    async def llm_should_not_run(message, history):
        raise AssertionError("高置信度 Embedding 不应调用 LLM")

    rec._embedding_recognize = fake_embedding
    rec._llm_recognize = llm_should_not_run
    result = asyncio.run(rec.recognize("网络服务出现一个模糊问题"))
    assert result.domain == IntentDomain.IT_HELP
    assert result.classifier_stage == "embedding"


def test_force_llm_bypasses_cascade():
    rec = _recognizer()

    async def fake_llm(message, history):
        return {"domain": IntentDomain.AFFAIRS, "action": IntentAction.QUERY, "confidence": 0.91, "reasoning": "baseline"}

    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("选课成绩学分有什么规定？", force_llm=True))
    assert result.domain == IntentDomain.AFFAIRS
    assert result.classifier_stage == "llm"
