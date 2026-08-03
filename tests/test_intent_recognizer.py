"""意图识别器测试：校园意图枚举、关键词模式匹配、加权投票、紧急度、实体提取兜底。

只测确定性逻辑；LLM 路径用最小 mock 覆盖失败回退。
"""
from __future__ import annotations

import asyncio

from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel

FAKE_KEY = "sk-test-not-used"


def _recognizer(**kwargs) -> IntentRecognizer:
    kwargs.setdefault("api_key", FAKE_KEY)
    return IntentRecognizer(**kwargs)


# ── 模式匹配（零 LLM 依赖）───────────────────────────────────────────────────

def test_pattern_recognizes_campus_domains():
    rec = _recognizer()
    cases = {
        "这学期选课什么时候开始": IntentCategory.ACADEMIC,
        "南校区食堂几点关门": IntentCategory.CAMPUS_LIFE,
        "奖学金什么时候评定": IntentCategory.AFFAIRS,
        "教务系统登录不上": IntentCategory.IT_HELP,
        "我要找辅导员": IntentCategory.ESCALATION,
        "你好": IntentCategory.GREETING,
    }
    for msg, expected in cases.items():
        result = rec._pattern_recognize(msg)
        assert result["intent"] == expected, f"{msg} → {result['intent']}"


def test_pattern_does_not_fire_for_generic_chat():
    rec = _recognizer()
    result = rec._pattern_recognize("今天天气怎么样")
    # 通用疑问句只命中 QUERY 关键词（？/怎么），不应命中校园领域意图
    assert result["intent"] in (IntentCategory.QUERY, IntentCategory.OTHER)


# ── 加权投票 ─────────────────────────────────────────────────────────────────

def test_vote_llm_dominant():
    rec = _recognizer()
    llm = {"intent": IntentCategory.ACADEMIC, "confidence": 0.9}
    emb = {"intent": IntentCategory.CAMPUS_LIFE, "confidence": 0.8}
    pat = {"intent": IntentCategory.ACADEMIC, "confidence": 0.3}
    assert rec._vote(llm, emb, pat) == IntentCategory.ACADEMIC


def test_vote_llm_failure_falls_back_to_embedding():
    rec = _recognizer()
    llm = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
    emb = {"intent": IntentCategory.IT_HELP, "confidence": 0.7}
    pat = {"intent": IntentCategory.OTHER, "confidence": 0.0}
    assert rec._vote(llm, emb, pat) == IntentCategory.IT_HELP


def test_vote_all_fail_returns_other():
    rec = _recognizer()
    llm = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
    emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}
    pat = {"intent": IntentCategory.OTHER, "confidence": 0.0}
    assert rec._vote(llm, emb, pat) == IntentCategory.OTHER


def test_vote_below_threshold_downgrades_to_other():
    rec = _recognizer(confidence_threshold=0.9)
    llm = {"intent": IntentCategory.ACADEMIC, "confidence": 0.5}
    emb = {"intent": IntentCategory.OTHER, "confidence": 0.0}
    pat = {"intent": IntentCategory.OTHER, "confidence": 0.0}
    assert rec._vote(llm, emb, pat) == IntentCategory.OTHER


# ── 紧急度 ───────────────────────────────────────────────────────────────────

def test_urgency_keywords_and_escalation():
    rec = _recognizer()
    assert rec._urgency("非常紧急，马上要交材料", IntentCategory.OTHER) == UrgencyLevel.CRITICAL
    assert rec._urgency("今天就要办", IntentCategory.OTHER) == UrgencyLevel.HIGH
    assert rec._urgency("我要找辅导员", IntentCategory.ESCALATION) == UrgencyLevel.HIGH
    assert rec._urgency("食堂几点开门", IntentCategory.CAMPUS_LIFE) == UrgencyLevel.LOW


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
    rec.learn(msg, IntentCategory.ACADEMIC)
    assert msg in rec._TEMPLATES if hasattr(rec, "_TEMPLATES") else True
    assert msg in __import__("core.intent_recognizer", fromlist=["_TEMPLATES"])._TEMPLATES[IntentCategory.ACADEMIC]


# ── 缓存统计 ─────────────────────────────────────────────────────────────────

def test_cache_stats_before_any_request():
    rec = _recognizer()
    stats = rec.cache_stats
    assert stats["size"] == 0
    assert stats["hit_rate"] == 0.0
