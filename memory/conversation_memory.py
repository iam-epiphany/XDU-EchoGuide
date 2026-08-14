"""
亮点：多轮对话记忆管理（四层记忆金字塔，对应 TencentDB-Agent-Memory 思路）

四层记忆架构，模拟人类记忆机制：
  0. 原文层 L0（SQLite）—— 原始对话全量落库，永不丢失，是证据链的锚点
  1. 工作记忆（Redis）—— 当前会话的最近 N 条消息，毫秒级读写
  2. 情景记忆（ChromaDB）—— 跨会话历史检索；压缩时生成"场景块"（layer=scenario）
     优先注入，普通片段次之（L2 场景层）
  3. 原子事实（SQLite）—— 从对话提炼的结构化事实，每条带来源会话与轮次，
     可沿 turn_id 下钻回 L0 原文（L1 事实层，白盒可溯源）；
     只存画像未覆盖的细粒度事实（决定/状态/计划/细节），上下文阶段按需召回
  4. 用户画像（ChromaDB + SQLite）—— 长期偏好聚合，版本历史可回滚（L3 画像层）

关键设计：
  - 上下文构建时四层记忆融合：L3 画像常驻注入（紧凑聚合），L1 事实按需召回
    （与当前提问相关才注入，不与 L3 重复注入）
  - 工作记忆超过阈值时自动压缩（LLM 场景化摘要），防止 context 爆炸
  - 提炼画像时一次 LLM 调用同时产出"画像 + 原子事实"（零额外成本），
    且只有检测到画像信号才调用 LLM（llm_call_count 统计调用率）；
    被画像覆盖的事实（偏好/实体条目是其子串）不落 L1（L1/L3 分工去重）
  - Embedding 由本地 bge 中文模型生成（mcp.embeddings，ONNX，不依赖外部 API）；
    模型不可用时回退 ChromaDB 内置 all-MiniLM-L6-v2（collection 名随向量空间切换）
"""
import hashlib
import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import chromadb
import redis.asyncio as redis
from anthropic import AsyncAnthropic

from mcp.embeddings import get_embedder
from memory.layered_store import LayeredStore

logger = logging.getLogger(__name__)


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


# ── 画像信号检测 ──────────────────────────────────────────────────────────────
# 用户消息含"偏好/背景声明"时才值得用 LLM 提炼画像；普通提问（"食堂几点关门"）
# 不触发，避免每轮对话都调用 LLM 提炼造成成本浪费与画像污染。
# 模式用 {0,N} 宽容中间词（如"我最近在准备考研"），避免只匹配紧邻表述。
_PROFILE_SIGNAL_PATTERNS = (
    r"我.{0,6}(喜欢|不喜欢|讨厌|最爱|爱)",
    r"我是.{0,12}(学院|专业|年级|校区)",
    r"我在.{0,8}(校区|宿舍|公寓|学院|部门)",
    r"我(大|研)[一二三四五]",
    r"我.{0,6}(经常|平时|每天|周末|习惯|打算|准备|计划|希望)",
    r"我的.{0,8}(专业|学院|宿舍|校区|爱好|课表)",
    r"我(想|要|决定|报名|参加).{0,6}(学|考|去|选)",
)
_PROFILE_SIGNAL_RE = re.compile("|".join(_PROFILE_SIGNAL_PATTERNS))


def _has_profile_signal(messages: List["Message"]) -> bool:
    """最近 2 条用户消息是否包含画像信号（偏好/背景声明）。"""
    user_texts = [m.content for m in messages if m.role == MsgRole.USER][-2:]
    return any(_PROFILE_SIGNAL_RE.search(t or "") for t in user_texts)


# ── L1/L3 分工去重 + L1 按需召回（纯规则，零 LLM 成本）────────────────────────
# 分层原则（对应记忆金字塔）：L3 画像 = 聚合画像（偏好/实体，紧凑常驻注入）；
# L1 事实 = 画像之外的细粒度可溯源事实（决定/状态/计划/细节，带证据链）。
# 同一信息不双写：事实被画像条目覆盖（条目是事实的规范化子串，或事实全文
# 已在画像中）则不落 L1；上下文阶段 L1 按需召回（与当前提问相关才注入），
# 不与 L3 重复注入。


def _norm_mem_text(text: str) -> str:
    """记忆文本规范化：去空白与标点、小写（覆盖判定与 bigram 共用的清洗）。"""
    if not text:
        return ""
    return re.sub(r"[\s\W_]+", "", str(text)).lower()


def _profile_entries(profile: Dict[str, Any]) -> List[str]:
    """展平画像为条目列表（preferences + entities 各字段）。"""
    entries: List[str] = []
    for pref in profile.get("preferences") or []:
        if isinstance(pref, str) and pref:
            entries.append(pref)
    for vals in (profile.get("entities") or {}).values():
        if isinstance(vals, list):
            entries.extend(str(v) for v in vals if isinstance(v, str) and v)
        elif isinstance(vals, str) and vals:
            entries.append(vals)
    return entries


def _fact_subsumed_by_profile(fact: str, profile: Dict[str, Any]) -> bool:
    """
    事实是否已被画像覆盖（L1/L3 分工去重的判定核心）：
      1. 事实全文是画像条目文本（preferences + entities）的规范化子串 —— 逐字重复；
      2. 画像某条目（偏好/实体）是事实的规范化子串 —— 事实只是条目的扩充陈述。
    任一命中即视为"画像已覆盖"：该事实不落 L1（写入侧）、不注入上下文（读取侧）。
    注意只对偏好/实体条目判定：画像 dict 可能携带 facts 等附加键，不能整份参与。
    """
    nf = _norm_mem_text(fact)
    if not nf:
        return True
    entries_text = _norm_mem_text("\n".join(_profile_entries(profile)))
    if nf in entries_text:
        return True
    return any(
        ne and ne in nf
        for ne in (_norm_mem_text(e) for e in _profile_entries(profile))
    )


# L1 按需召回停用 bigram：高频无区分度（"用户"是事实统一主语），命中不计相关性。
# 时间词（今天/明天）也停用 —— 事实与查询里的时间表述太常见，不承载主题相关性。
_FACT_STOP_BIGRAMS = frozenset({
    "用户", "我们", "你们", "他们", "这个", "那个", "什么", "怎么", "可以",
    "需要", "就是", "已经", "现在", "没有", "不是", "如果", "因为", "所以",
    "还有", "自己", "今天", "明天", "上次", "平时", "周末",
})


def _fact_bigrams(text: str) -> set:
    """字符 bigram 集合（中文 1 字产出 1 个 bigram）。"""
    norm = _norm_mem_text(text)
    return {norm[i:i + 2] for i in range(len(norm) - 1)}


def _fact_relevant_to_query(query: str, fact: str) -> bool:
    """
    L1 按需召回：事实与查询共享 ≥1 个非停用 bigram 即相关。

    中文场景下字符 bigram 对"考研/准备""补办/校园卡"这类术语命中可靠；
    近义改写可能漏召（可接受：L2 情景记忆 + L3 画像仍兜底）。
    查询无有效 bigram（过短/纯停用词）时视为不相关，由调用方对空查询回退全量。
    """
    q_bigrams = _fact_bigrams(query) - _FACT_STOP_BIGRAMS
    if not q_bigrams:
        return False
    return bool(_fact_bigrams(fact) & q_bigrams)


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的完整上下文（四层记忆融合）。"""
    recent_messages:  List[Message]   # 工作记忆：最近对话
    relevant_history: List[str]       # 情景记忆：语义相关历史片段（L2 场景块优先）
    user_profile:     Dict[str, Any]  # 用户画像：长期偏好、常用实体（L3）
    summary:          str             # 当前会话摘要（压缩后）
    facts:            List[str]       # 原子事实：按需召回的 L1 事实（与当前提问相关）
    memory_trace:     Dict[str, Any]  # 各层命中统计（透出 API/前端，白盒演示用）

    @staticmethod
    def _clean(text: str) -> str:
        """移除 Unicode 代理字符，防止编码错误。"""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """将记忆上下文格式化为 LLM 可用的文本（场景 → 事实 → 历史 → 画像 → 最近）。"""
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.facts:
            parts.append("[用户事实]\n" + "\n".join(f"- {self._clean(f)}" for f in self.facts[:8]))
        if self.relevant_history:
            parts.append("[相关历史]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        if self.user_profile:
            parts.append(f"[用户画像]\n{json.dumps(self.user_profile, ensure_ascii=True)}")
        if self.recent_messages:
            parts.append("[最近对话]")
            for m in self.recent_messages:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """
    四层记忆管理器（对应记忆金字塔）：
      L0 原文层 —— LayeredStore.raw_messages（SQLite，永不丢失）
      L1 事实层 —— LayeredStore.facts（结构化原子事实，带证据链）
      L2 场景层 —— ChromaDB episodic（layer=scenario 场景块优先检索）
      L3 画像层 —— ChromaDB profile + LayeredStore.profile_history（版本可回滚）
    工作记忆存 Redis（TTL 24h）。
    """

    WORKING_MAX   = 20    # 工作记忆最大条数，超过则触发压缩
    COMPRESS_AT   = 15    # 达到此条数时压缩，保留摘要 + 最近 5 条
    HISTORY_TOP_K = 5     # 情景记忆检索返回条数（场景块优先取 2 条）

    def __init__(
        self,
        redis_url:    str = "redis://localhost:6379/0",
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = "claude-3-5-sonnet-20241022",
        layered_store: Optional[LayeredStore] = None,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model

        # L0/L1/L3 数据底座（SQLite，可注入以便测试替换）
        self._layered = layered_store or LayeredStore()
        # 记忆模块自身的 LLM 调用计数（画像提炼 + 摘要压缩），供评测统计调用率
        self.llm_call_count = 0

        # 增量提炼并发锁（每 user:conv 一把，防连续对话时后台任务重叠提炼同一区间）
        self._extract_locks: Dict[str, "asyncio.Lock"] = {}

        self._redis = redis.from_url(redis_url, decode_responses=True)
        # ChromaDB：优先连接独立服务（docker compose 模式），连不上则降级为本地嵌入式
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            chroma = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            chroma.heartbeat()  # 测试连接
            logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
            chroma = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # Embedding：本地 bge 模型优先；不可用时回退 MiniLM 向量空间。
        # collection 名随向量空间切换，两种模型的向量绝不混存。
        embedding_function = get_embedder()
        if embedding_function is None:
            logger.warning("本地 Embedding 模型不可用，记忆回退 MiniLM 向量空间")
        episodic_name = "episodic_v3" if embedding_function is not None else "episodic_v2"
        profile_name = "user_profile_v2" if embedding_function is not None else "user_profile"

        # 情景记忆：存储历史对话片段
        self._episodic = chroma.get_or_create_collection(
            episodic_name,
            metadata={"hnsw:space": "cosine", "description": "情景记忆（cosine，bge）"},
            embedding_function=embedding_function,
        )
        # 用户画像：存储提炼出的偏好和实体（只按 user_id 精确读取，不做向量查询）
        self._profile = chroma.get_or_create_collection(
            profile_name, metadata={"hnsw:space": "cosine", "description": "用户画像（bge）"},
            embedding_function=embedding_function,
        )

        # 跨模型迁移：旧向量空间的原始文本重写入当前空间（仅当当前为空时）
        self._migrate_collections(chroma)

    def _migrate_collections(self, chroma: Any) -> None:
        """把旧向量空间的记忆/画像重写入当前 collection（跨模型需重嵌入）。

        仅当当前 collection 为空时进行（幂等），失败不阻断服务启动。
        """
        for current, previous in (
            (self._episodic, "episodic_v2"),
            (self._profile, "user_profile"),
        ):
            if current.name == previous or current.count() != 0:
                continue  # 本就是回退空间 / 已有数据
            try:
                old = chroma.get_collection(previous)
                records = old.get(include=["documents", "metadatas"])
                ids = records.get("ids") or []
                docs = records.get("documents") or []
                metas = records.get("metadatas") or []
                if ids and docs:
                    current.upsert(ids=ids, documents=docs, metadatas=metas)
                    logger.info("记忆已从 %s 重新索引 %d 条到 %s",
                                previous, len(ids), current.name)
            except Exception as ex:
                logger.debug("记忆迁移 %s 跳过（可忽略）: %s", previous, ex)

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时自动压缩。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=self._safe_text(content), metadata=clean_metadata)
        key = self._wm_key(user_id, conv_id)

        # 追加到 Redis 列表（左推，最新在前）
        await self._redis.lpush(key, json.dumps({
            "role":      msg.role.value,
            "content":   msg.content,
            "ts":        msg.timestamp.isoformat(),
            "metadata":  msg.metadata,
        }))
        await self._redis.expire(key, 86400)  # 24h TTL

        # L0 原文落库（四层金字塔的证据链锚点，永不丢失）。
        # 写入失败只告警不阻断主链路（记忆是增强，不是依赖）。
        try:
            await self._layered.append_raw(
                user_id, conv_id, msg.role.value, msg.content, clean_metadata
            )
        except Exception as ex:
            logger.warning(f"L0 原文写入失败 user={user_id}: {ex}")

        # 超过压缩阈值时触发压缩
        if await self._redis.llen(key) >= self.COMPRESS_AT:
            await self._compress(user_id, conv_id)

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        """
        从当前对话中提炼用户偏好与原子事实，更新用户画像（L3）与事实层（L1）。

        设计要点（对应记忆金字塔治理）：
        1. 成本控制 —— 只有最近用户消息包含"画像信号"（偏好/背景声明）才调用 LLM
           提炼，普通提问（"食堂几点关门"）不重复提炼；
        2. 一次调用双产出 —— LLM 一次返回 {preferences, entities, facts}：
           画像仍按用户聚合单条（L3），事实写入 facts 表（L1，带来源会话与
           轮次，可沿证据链下钻回 L0 原文）；零额外 LLM 成本；
        3. 版本治理 —— 每次提炼把画像快照写入 profile_history（L3 版本历史，
           可回滚），reason 记录触发信号；
        4. L1/L3 分工去重 —— 被画像覆盖的事实（偏好/实体条目是其规范化子串，
           或事实全文已在画像中）不落 L1，只保留画像之外的细粒度可溯源事实；
           与既有 active 事实按文本去重（LLM 合并后的重复提炼不落库）；
        5. 增量提炼（对齐 TencentDB-Agent-Memory）—— extract_marks 水位记录
           上次提炼的最大 turn，信号命中只提炼上次之后的新消息（L0 原文区间），
           老消息不重复喂 LLM；首次提炼水位为 0 取全量（预热）；无增量跳过；
           同会话后台任务按 user:conv 串行化并在锁内重读水位（防并发重复）；
           提炼成功才推进水位（失败不推进，下次幂等重试）。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = await self._get_working_memory(user_id, conv_id)
        if not messages:
            return
        if not _has_profile_signal(messages):
            logger.debug(f"无画像信号，跳过提炼: {user_id}")
            return

        lock = self._extract_locks.setdefault(f"{user_id}:{conv_id}", asyncio.Lock())
        async with lock:
            await self._extract_incremental(user_id, conv_id, messages)

    async def _extract_incremental(
        self, user_id: str, conv_id: str, messages: List["Message"]
    ) -> None:
        """锁内执行：取增量区间 → 一次 LLM 双产出 → 落库 → 推进水位。"""
        last_turn = await self._layered.get_extract_mark(user_id, conv_id)
        max_turn = await self._layered.get_last_turn(user_id, conv_id)
        incremental = await self._layered.get_raw_range(user_id, conv_id, last_turn + 1)
        if not incremental:
            if max_turn > 0:
                # L0 有记录但水位后无新消息 → 上次已提炼到顶，跳过（防并发/重复提炼）
                logger.debug(f"无增量消息，跳过提炼: {user_id}")
                return
            # L0 完全无记录（写失败告警后的异常兜底）→ 回退工作记忆全量提炼
            incremental = messages

        if isinstance(incremental[0], dict):  # L0 原始行（get_raw_range 返回字典）
            text = self._safe_text("\n".join(
                f"{r.get('role', '')}: {r.get('content', '')}" for r in incremental
            ))
        else:  # 回退路径：工作记忆 Message 对象
            text = self._safe_text("\n".join(
                f"{m.role.value}: {m.content}" for m in incremental
            ))
        existing = await self._get_profile(user_id)
        existing_text = self._safe_text(json.dumps(existing, ensure_ascii=False)) if existing else "（无既有画像）"
        prompt = f"""从以下西电校园用户对话中提炼用户偏好、关键实体和原子事实，返回 JSON。
对话:
{text}

既有画像（合并时保留仍有效的信息，去除过时条目）:
{existing_text}

返回格式: {{"preferences": ["..."], "entities": {{"院系专业": [], "年级": [], "校区": [], "诉求类型": []}}, "facts": [{{"fact": "原子事实一句话", "category": "preference|entity|decision|status"}}]}}
要求：preferences 去重合并，最多 20 条；entities 每个字段最多 10 条；
facts 只提炼「画像未覆盖」的细粒度可溯源事实（做出的决定、当前状态、具体计划、
细节），身份背景与偏好归 preferences/entities，严禁与两者内容重复，最多 10 条。"""
        prompt = self._safe_text(prompt)

        try:
            self.llm_call_count += 1
            resp = await self._client.messages.create(
                model=self._model, max_tokens=768, temperature=0.0,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text
            s, e = raw.find("{"), raw.rfind("}") + 1
            profile_data = json.loads(raw[s:e])

            # 结构兜底 + 上限截断（LLM 输出不可信）
            prefs = profile_data.get("preferences") if isinstance(profile_data.get("preferences"), list) else []
            profile_data["preferences"] = prefs[:20]
            entities = profile_data.get("entities") if isinstance(profile_data.get("entities"), dict) else {}
            profile_data["entities"] = {
                k: (v[:10] if isinstance(v, list) else v) for k, v in entities.items()
            }

            # L1 原子事实落库（带证据链：来源会话 + 当前轮次锚点）
            facts_raw = profile_data.get("facts") if isinstance(profile_data.get("facts"), list) else []
            source_turn = await self._layered.get_last_turn(user_id, conv_id)
            facts = [
                {"fact": str(f.get("fact") or "").strip(),
                 "category": str(f.get("category") or "preference"),
                 "source_conv": conv_id, "source_turn": source_turn}
                for f in facts_raw[:10] if isinstance(f, dict) and str(f.get("fact") or "").strip()
            ]
            # L1/L3 分工去重：画像已覆盖（偏好/实体条目是事实子串等）的事实不落 L1，
            # 避免同一信息既在画像又在事实层（LLM 输出不可信，代码侧硬约束兜底）
            facts = [
                f for f in facts
                if not _fact_subsumed_by_profile(f["fact"], profile_data)
            ]
            added = await self._layered.add_facts(user_id, facts)
            if added:
                logger.info(f"原子事实新增 {added} 条: {user_id}")

            # L3 画像：用户级单条（聚合）+ 版本历史（可回滚）
            doc_id = f"{user_id}_profile"   # 用户级单条（聚合）
            doc_text = self._safe_text(json.dumps(profile_data, ensure_ascii=False))

            # 直接传 documents，让 ChromaDB 内置模型生成 embedding（不依赖外部 API）
            await asyncio.to_thread(self._profile.upsert,
                ids=[doc_id],
                documents=[doc_text],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat()}],
            )
            await self._layered.save_profile_version(
                user_id, doc_text, reason=f"signal: {conv_id}"
            )
            # 提炼成功才推进水位（失败不推进，下次幂等重试同一区间）
            await self._layered.set_extract_mark(user_id, conv_id, max_turn)
            logger.info(f"用户画像已更新: {user_id}（{len(prefs)} 条偏好，{added} 条新事实）")
        except Exception as ex:
            logger.warning(f"更新用户画像失败: {ex}")

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """
        构建完整的记忆上下文（四层融合）。

        query 用于从情景记忆中检索语义相关的历史片段，同时作为 L1 事实
        按需召回的关联查询：无关事实不注入（L3 画像常驻注入，不与 L3 重复）；
        query 为空时保守回退全量，不破坏无查询场景。
        """
        # 1. 工作记忆（当前会话最近消息）
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        recent = await self._get_working_memory(user_id, conv_id)

        # 2. 情景记忆（跨会话语义检索，L2 场景块优先）
        search_query = query or (recent[-1].content if recent else "")
        history, layer_counts = await self._search_episodic(
            user_id, search_query
        )

        # 3. 用户画像（L3，聚合画像 —— 常驻注入；先读：L1 过滤依赖画像）
        profile = await self._get_profile(user_id)

        # 4. 原子事实（L1，细粒度可溯源事实 —— 按需召回）
        #    两层过滤：① 画像已覆盖的事实不注入（兼容存量数据，与 L3 不重复）；
        #    ② 与当前提问相关才注入（query 为空时保守回退全量）。
        facts = await self._list_facts(user_id)
        injected_facts = [
            f["fact"] for f in facts
            if not _fact_subsumed_by_profile(f["fact"], profile)
            and (not search_query or _fact_relevant_to_query(search_query, f["fact"]))
        ][:8]

        # 5. 会话摘要（如果已压缩过）
        summary = await self._redis.get(self._summary_key(user_id, conv_id)) or ""

        # 6. 各层命中统计（白盒 trace：透出给 API/前端演示）
        memory_trace = {
            "layers": {
                "raw": await self._layered.count_raw(user_id),
                "facts": len(injected_facts),
                "facts_total": len(facts),
                "scenario": layer_counts["scenario"],
                "segments": layer_counts["segment"],
                "profile_versions": await self._layered.count_profile_versions(user_id),
            }
        }

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
            facts=injected_facts,
            memory_trace=memory_trace,
        )

    async def _list_facts(self, user_id: str) -> List[Dict[str, Any]]:
        """读取用户当前有效原子事实（L1；读取失败降级为空，不阻断主链路）。"""
        try:
            return await self._layered.list_facts(user_id)
        except Exception as ex:
            logger.warning(f"读取原子事实失败: {ex}")
            return []

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆压缩（L2 场景层生成）：
          1. 用 LLM 对旧消息生成**场景化摘要**（任务/结论/关键实体），
             即 L2 场景块 —— 用于跨会话快速恢复"当时在做什么、结论是什么"
          2. 摘要存 Redis（覆盖旧摘要，会话内快速参考）
          3. 场景块存入情景记忆（ChromaDB，layer="scenario"），检索时优先注入；
             对话原文由 L0 层兜底（不再截断存 metadata）
          4. 工作记忆只保留最近 5 条
        """
        messages = await self._get_working_memory(user_id, conv_id)
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-5]   # 保留最近 5 条
        keep        = messages[-5:]

        # LLM 场景化摘要
        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in to_compress))
        prompt = self._safe_text(
            f"总结以下对话。用 3-5 句话输出场景化摘要，必须包含："
            f"1) 涉及的任务/问题；2) 结论或给出的答案；3) 关键实体（人名/地点/数字/日期）。\n"
            f"对话:\n{text}"
        )
        try:
            self.llm_call_count += 1
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                thinking={"type": "disabled"},
                messages=[{"role": "user", "content": prompt}],
            )
            summary = self._safe_text(resp.content[0].text).strip()
        except Exception:
            summary = f"对话包含 {len(to_compress)} 条消息（摘要生成失败）"

        # 存摘要到 Redis
        skey = self._summary_key(user_id, conv_id)
        old_summary = await self._redis.get(skey) or ""
        new_summary = self._safe_text(f"{old_summary}\n{summary}").strip()
        await self._redis.setex(skey, 86400, new_summary)

        # 场景块存入情景记忆（L2，原文由 L0 兜底）
        await self._store_episodic(user_id, conv_id, text, summary, layer="scenario")

        # 重置工作记忆为最近 5 条
        key = self._wm_key(user_id, conv_id)
        await self._redis.delete(key)
        for m in reversed(keep):
            await self._redis.lpush(key, json.dumps({
                "role": m.role.value, "content": m.content,
                "ts": m.timestamp.isoformat(), "metadata": m.metadata,
            }))
        await self._redis.expire(key, 86400)
        logger.info(f"工作记忆压缩完成: {user_id}/{conv_id}，场景块 {len(summary)} 字")

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        key  = self._wm_key(user_id, conv_id)
        raws = await self._redis.lrange(key, 0, self.WORKING_MAX - 1)
        msgs = []
        for raw in reversed(raws):  # Redis lpush 最新在前，reversed 还原时序
            # 单条损坏（半写入/旧格式/手工改键）不应拖垮整个对话链路：
            # 跳过坏条目并告警，与 ChromaDB 各处的降级语义保持一致。
            try:
                d = json.loads(raw)
                msgs.append(Message(
                    role=MsgRole(d["role"]),
                    content=d["content"],
                    timestamp=datetime.fromisoformat(d["ts"]),
                    metadata=d.get("metadata", {}),
                ))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as ex:
                logger.warning(
                    "工作记忆条目损坏已跳过 user=%s conv=%s raw=%r: %s",
                    user_id, conv_id, raw[:120], ex,
                )
        return msgs

    async def _search_episodic(
        self, user_id: str, query: str
    ) -> tuple[List[str], Dict[str, int]]:
        """
        语义检索情景记忆（分层注入）：
          1. 先查 L2 场景块（layer=scenario，跨会话"快速恢复工作场景"）
          2. 再取普通片段（segment，旧数据无 layer 字段按普通处理）
        返回 (合并文档, 各层命中计数)。
        """
        counts = {"scenario": 0, "segment": 0}
        query_text = self._safe_text(query).strip()
        if not query_text:
            return [], counts

        # L2 场景块优先（n=2）
        scenario = await self._query_episodic(
            user_id, query_text, n=2, layer="scenario"
        )
        counts["scenario"] = len(scenario)

        # 普通片段（排除场景块，Python 侧过滤兼容旧数据无 layer 字段）
        segments = await self._query_episodic(
            user_id, query_text, n=self.HISTORY_TOP_K
        )
        segments = [d for d in segments if d not in scenario]
        counts["segment"] = len(segments)

        return scenario + segments, counts

    async def _query_episodic(
        self, user_id: str, query_text: str, n: int, layer: Optional[str] = None
    ) -> List[str]:
        """单次 ChromaDB 语义查询；layer 指定时按 metadata 精确过滤。"""
        try:
            where: Dict[str, Any] = {"user_id": self._safe_text(user_id)}
            if layer:
                where["layer"] = layer
            results = await asyncio.to_thread(self._episodic.query,
                query_texts=[query_text],
                n_results=n,
                where=where,
                include=["documents", "metadatas"],
            )
            docs = results["documents"][0] if results["documents"] else []
            metas = results["metadatas"][0] if results.get("metadatas") else []
            kept = []
            for d, mt in zip(docs, metas):
                if not isinstance(d, str) or not d.strip():
                    continue
                # 普通片段查询时排除场景块（旧数据无 layer 字段视为普通）
                if layer is None and (mt or {}).get("layer") == "scenario":
                    continue
                kept.append(self._safe_text(d))
            return kept
        except Exception as ex:
            logger.warning(f"情景记忆检索失败: {ex}")
            return []

    async def _store_episodic(
        self,
        user_id: str,
        conv_id: str,
        text: str,
        summary: str,
        layer: str = "segment",
    ) -> None:
        """
        将压缩后的对话片段存入情景记忆（ChromaDB）。
        layer 标记记忆层级：scenario（L2 场景块，检索优先）或 segment（普通片段）。
        原文全文不再截断入 metadata —— L0 层已全量落库（证据链锚点）。
        """
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = hashlib.md5(f"{user_id}{conv_id}{time.time()}".encode()).hexdigest()
            # 直接传 documents，ChromaDB 内置模型自动生成 embedding
            await asyncio.to_thread(self._episodic.add,
                ids=[doc_id],
                documents=[summary],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat(), "layer": layer}],
            )
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """获取用户画像（取最新一条）。"""
        try:
            results = await asyncio.to_thread(self._profile.get, where={"user_id": user_id}, limit=1)
            if results["documents"]:
                return json.loads(results["documents"][0])
        except Exception:
            pass
        return {}

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{user_id}:{conv_id}"

    @property
    def layered_store(self) -> LayeredStore:
        """分层记忆存储（L0/L1/L3 数据底座），供编排器注入共享实例。"""
        return self._layered

    async def close(self) -> None:
        """关闭异步 Redis 连接，供 FastAPI lifespan 调用。"""
        await self._redis.aclose()

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{user_id}:{conv_id}"

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成 ChromaDB 可接受的普通 UTF-8 字符串。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 Redis/ChromaDB 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value
