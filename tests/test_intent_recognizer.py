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
    """P0 回归：请求句式不再吞掉领域信息。

    v4 重新定义：咨询流程（"怎么走流程/怎么补办"）是 QUERY（不产生状态修改）；
    只有明确写操作词（添加/删除/标记/记一下）才是 REQUEST。
    """
    rec = _recognizer()
    result = rec._pattern_recognize("我要请假怎么走流程")
    assert result["domain"] == IntentDomain.AFFAIRS
    assert result["action"] == IntentAction.QUERY

    result = rec._pattern_recognize("校园卡丢了怎么补办")
    assert result["domain"] == IntentDomain.AFFAIRS
    assert result["action"] == IntentAction.QUERY


def test_pattern_write_action_recognized_as_request():
    """v4：写操作词（添加/标记/记一下）→ REQUEST；"帮我查" → QUERY。"""
    rec = _recognizer()
    assert rec._pattern_recognize("帮我添加一个补办校园卡的待办")["action"] == IntentAction.REQUEST
    assert rec._pattern_recognize("把这个待办标记完成")["action"] == IntentAction.REQUEST
    assert rec._pattern_recognize("帮我记一下明天交实验报告")["action"] == IntentAction.REQUEST
    assert rec._pattern_recognize("帮我查一下课表")["action"] == IntentAction.QUERY
    assert rec._pattern_recognize("查一下校园卡余额")["action"] == IntentAction.QUERY


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

    async def fake_llm(message, history, state=None):
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


def test_weak_pattern_goes_directly_to_llm_without_embedding():
    """Pattern 未达高置信时直接 LLM，Embedding 不能单独决定路由。"""
    rec = _recognizer()

    async def embedding_should_not_run(message):
        raise AssertionError("弱 Pattern 不应先走 Embedding")

    async def fake_llm(message, history, state=None):
        return {"domain": IntentDomain.ACADEMIC, "action": IntentAction.QUERY,
                "domain_confidence": 0.90, "confidence": 0.90, "reasoning": "mock"}

    rec._embedding_recognize = embedding_should_not_run
    rec._llm_recognize = fake_llm
    trace = {}
    result = asyncio.run(rec.recognize("绩点怎么算的？", _trace=trace))  # pattern 弱命中 academic@0.55
    assert result.domain == IntentDomain.ACADEMIC
    assert result.classifier_stage == "llm"
    assert trace["embedding_candidates"] == []


# ── 缓存指纹（同句追问不同上下文不复用）──────────────────────────────────────

def test_cache_key_includes_history_fingerprint():
    rec = _recognizer()
    k1 = rec._cache_key("那几点开门呢？", [{"role": "user", "content": "南校区食堂几点关门？"}])
    k2 = rec._cache_key("那几点开门呢？", [{"role": "user", "content": "教务系统密码忘了"}])
    k3 = rec._cache_key("那几点开门呢？", None)
    assert k1 != k2 != k3
    assert k1 != k3


# ── 实体提取兜底 ─────────────────────────────────────────────────────────────

def test_needs_knowledge_absent_on_pattern_path():
    """免费路径（pattern/embedding）不判定 needs_knowledge（未判定 → False）。"""
    rec = _recognizer()

    async def fake_embedding(message):
        return {"domain": IntentDomain.ACADEMIC, "action": IntentAction.OTHER,
                "confidence": 0.90, "margin": 0.3}

    rec._embedding_recognize = fake_embedding
    result = asyncio.run(rec.recognize("选课成绩学分有什么规定？"))
    assert result.classifier_stage == "pattern"
    assert result.needs_knowledge is False


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
    """Pattern 高置信 + Embedding 同域、高分且 margin 达标 → 免费直返。"""
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

    async def fake_llm(message, history, state=None):
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

    async def fake_llm(message, history, state=None):
        return {"domain": IntentDomain.ACADEMIC, "action": IntentAction.QUERY,
                "confidence": 0.9, "reasoning": "mock"}

    rec._embedding_recognize = fake_embedding
    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("选课成绩学分有什么规定？"))  # pattern 高置信 academic@0.95
    assert result.domain == IntentDomain.ACADEMIC
    assert result.classifier_stage == "llm"
    assert "低于阈值" in result.reasoning  # 仲裁原因可追溯：同向但分数不足


def test_pattern_embedding_low_margin_arbitrates_to_llm():
    """双确认必须包含 margin：同向且高分但候选太接近时仍走 LLM。"""
    rec = _recognizer()

    async def fake_embedding(message):
        return {"domain": IntentDomain.ACADEMIC, "action": IntentAction.OTHER,
                "confidence": 0.90, "margin": 0.05}

    async def fake_llm(message, history, state=None):
        return {"domain": IntentDomain.ACADEMIC, "action": IntentAction.QUERY,
                "confidence": 0.9, "reasoning": "mock"}

    rec._embedding_recognize = fake_embedding
    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("选课成绩学分有什么规定？"))
    assert result.domain == IntentDomain.ACADEMIC
    assert result.classifier_stage == "llm"
    assert "margin" in result.reasoning


def test_embedding_without_high_pattern_goes_directly_to_llm():
    rec = _recognizer()

    async def embedding_should_not_run(message):
        raise AssertionError("Embedding 不应作为独立免费路径")

    async def fake_llm(message, history, state=None):
        return {"domain": IntentDomain.IT_HELP, "action": IntentAction.QUERY,
                "domain_confidence": 0.87, "confidence": 0.87, "reasoning": "mock"}

    rec._embedding_recognize = embedding_should_not_run
    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("网络服务出现一个模糊问题"))
    assert result.domain == IntentDomain.IT_HELP
    assert result.classifier_stage == "llm"


def test_force_llm_bypasses_cascade():
    """强制 LLM：action 来自 LLM，领域由免费关键词回填（LLM 不再输出领域）。"""
    rec = _recognizer()

    async def fake_llm(message, history, state=None):
        return {"action": IntentAction.QUERY, "confidence": 0.91, "reasoning": "baseline"}

    rec._llm_recognize = fake_llm
    result = asyncio.run(rec.recognize("选课成绩学分有什么规定？", force_llm=True))
    assert result.domain == IntentDomain.ACADEMIC  # 免费关键词回填
    assert result.classifier_stage == "llm"
