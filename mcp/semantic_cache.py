"""
语义缓存（Semantic Caching）—— GPTCache 思路的轻量实现，双层隔离 + 上下文依赖性判定。

原理：把 (查询 → 回答) 存入 ChromaDB 向量库；新查询先做语义检索，
如果与历史查询的 embedding 相似度 ≥ 阈值（默认 0.85），直接复用历史回答。

双层设计（解决用户隔离问题）：
  - Global 缓存（semantic_cache_global）：只存**上下文无关**的公共答案，
    任何用户可复用（语义匹配，容忍近义改写）。
  - User 缓存（semantic_cache_user）：按 user_id 隔离的个性化答案，
    同一用户不同问法可语义复用，跨用户互不可见。

决策模型（v3 起）：
  早期版本把"完整动态上下文指纹 context_fp（md5 画像+摘要+历史+最近对话）"
  作为语义缓存的硬命中条件：只要有记忆上下文就强制进 User 层，且 where
  同时硬过滤 user_id + context_fp —— 指纹稍有变化（画像更新、历史增长）
  即 miss，命中率被严重牺牲。v3 改为先判断请求的**上下文依赖性**：
    - 公共事实查询（global）→ Global 层语义匹配；
    - 依赖用户画像/身份但可复用（user）→ User 层语义匹配（仅 user_id 分区）；
    - 强上下文依赖（追问/省略句/指代/个人数据/状态改变）→ 直接 bypass
      语义缓存（计 bypass，不算 miss）。
  完整动态指纹只适合 Exact Cache / Prompt Cache 这类"完全相同计算"的
  安全复用，不适合作为 Semantic Cache 的硬命中条件，故本模块不再计算指纹。

读取策略（防绕过个性化推理）：
  - global → 只查 Global 缓存（公共答案）；
  - user + 有效身份 → 只查 User 缓存；miss 后**不回退 Global**，
    防止公共答案绕过个性化 Agent 推理；
  - skip（强上下文依赖）→ 直接跳过缓存（计 bypass，不算 miss）；
  - user 但匿名 → 无法按身份隔离，跳过缓存。

注意：缓存读取必须发生在 Memory Context 获取**之后**，否则无法判断
请求是否依赖历史上下文。

与 TTL 精确缓存（MCPToolManager._cache）的区别：
  精确缓存要求参数完全相同；语义缓存容忍近义改写（"选课什么时候开始？" ≈
  "选课几时开始？"）。
"""
import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, Optional

import chromadb

from mcp.embeddings import get_embedder

logger = logging.getLogger(__name__)

# 语义缓存各 tier 的 collection 名（v4/v5 起使用 bge 中文向量空间；
# 本地模型不可用时回退 v2/v3 的 MiniLM 空间，保证相似度分数语义一致）。
COLLECTION_GLOBAL = "semantic_cache_global_v4"      # bge 空间（当前）
COLLECTION_USER = "semantic_cache_user_v5"          # bge 空间（当前）
PREVIOUS_COLLECTION_GLOBAL = "semantic_cache_global_v2"  # MiniLM 空间（回退）
PREVIOUS_COLLECTION_USER = "semantic_cache_user_v3"     # MiniLM 空间（回退）
_COSINE_METADATA = {"hnsw:space": "cosine", "description": "EchoGuide semantic cache (cosine, bge)"}

# 上下文依赖性三态
DEP_GLOBAL = "global"  # 公共事实查询：答案不依赖上下文，可进 Global 层
DEP_USER = "user"      # 依赖用户画像/身份但可复用，可进 User 层
DEP_SKIP = "skip"      # 强上下文依赖：追问/省略句/指代/个人数据/状态改变，直接 bypass

# 第一人称指代（"我们"单独剔除，避免"我们学校图书馆几点关门？"这类公共问题误判）
_PRIVATE_PRONOUNS = ("我的", "俺", "咱", "我")
# 指代词（"其"剔除：易误伤"其实/其他/其它"等完整查询）
_DEICTIC_TOKENS = ("那", "这", "它")
# 事实查询疑问词
_QUESTION_WORDS = (
    "几点", "怎么", "什么", "哪里", "哪儿", "如何", "在哪",
    "多少", "是否", "有没有", "何时", "为什么", "怎样",
)


def classify_context_dependence(
    message: str,
    ctx_text: str = "",
    domain: Optional[str] = None,
    action: Optional[str] = None,
    classifier_stage: Optional[str] = None,
) -> str:
    """
    判断请求的上下文依赖性（纯函数，读写两侧共用同一规则）。

    规则优先级（命中即返回，越靠前越保守）：
      1. 编排信号（写入侧有编排结果时传入；读取侧为 None）：
           personal/other 领域、追问继承 stage="history"、请求/投诉/反馈
           动作 → skip（个人数据/状态改变不适合缓存）；
      2. 追问/指代/省略信号（优先于第一人称）：
           含指代词（那/这/它）且 ≤ 12 字 → skip（答案依赖上文话题）；
           以"呢"结尾且 ≤ 8 字 → skip；
           极短问句（≤ 5 字且问号结尾）→ skip（必然是省略句/依赖上下文）；
           "吗"不单独作为追问信号 —— "图书馆几点关门吗？"是完整疑问句，
           按事实查询处理；
      3. 第一人称指代（我/我的/俺/咱，排除"我们"）→ user（依赖用户画像）；
      4. 事实查询句式（含疑问词）→ global（公共答案，语义匹配）；
      5. 无记忆上下文 → global；
      6. 兜底（有上下文但无信号）→ user（保守：防公共答案绕过个性化推理）。

    保守性说明：Global 层是余弦模糊匹配（≥ 阈值即命中），把追问/指代句
    误判为 global 可能**错误命中**语义相似的公共答案（如"那几点开门？"
    可能命中"食堂几点开门"的缓存），因此追问/省略/指代信号必须优先判 skip。
    """
    msg = (message or "").strip()
    if not msg:
        return DEP_SKIP  # 空消息无法判定，直接 bypass

    # 1. 编排信号（仅写入侧有编排结果时可用）
    if domain in ("personal", "other"):
        return DEP_SKIP
    if classifier_stage == "history":
        return DEP_SKIP
    if action in ("request", "complaint", "feedback"):
        return DEP_SKIP

    # 2. 追问/指代/省略信号（优先于第一人称）
    if len(msg) <= 12 and any(tok in msg for tok in _DEICTIC_TOKENS):
        return DEP_SKIP
    if len(msg) <= 8 and msg.rstrip("?？!！.。 ").endswith("呢"):
        return DEP_SKIP
    if len(msg) <= 5 and msg.endswith(("？", "?")):
        return DEP_SKIP

    # 3. 第一人称指代（排除"我们"，避免"我们学校…"误伤）
    if "我们" not in msg and "我校" not in msg and any(tok in msg for tok in _PRIVATE_PRONOUNS):
        return DEP_USER

    # 4. 事实查询句式 → 公共答案
    if any(w in msg for w in _QUESTION_WORDS):
        return DEP_GLOBAL

    # 5. 无记忆上下文 → 公共答案
    if not (ctx_text or "").strip():
        return DEP_GLOBAL

    # 6. 兜底：有上下文但无法判定 → 保守走 User 层
    return DEP_USER


def cache_tier(dependence: str, user_id: Optional[str] = None) -> Optional[str]:
    """
    三层判定结果 → 写入层映射（纯函数，只做映射，不重复判断 domain/action）：

      - "global" → "global"（公共答案，任何用户可复用）；
      - "user" + 有效身份 → "user"（按 user_id 分区）；
      - 其余（skip / 匿名 user）→ None（不写入）。

    返回 "global" / "user" / None。
    """
    if dependence == DEP_GLOBAL:
        return "global"
    if dependence == DEP_USER and user_id and user_id != "anonymous":
        return "user"
    return None


def cache_read_tier(dependence: str, user_id: Optional[str] = None) -> Optional[str]:
    """
    三层判定结果 → 读取层映射（纯函数，只做映射）：

      - "global" → "global"（公共答案）；
      - "user" + 有效身份 → "user"（只查 User 层，miss 不回退 Global，
        防止公共答案绕过个性化 Agent 推理）；
      - 其余（skip 强上下文依赖 / 匿名 user）→ None（跳过缓存）。

    返回 "user" / "global" / None。
    """
    if dependence == DEP_GLOBAL:
        return "global"
    if dependence == DEP_USER and user_id and user_id != "anonymous":
        return "user"
    return None


def _entry_id(query: str, user_id: Optional[str] = None) -> str:
    """
    缓存条目 ID（防跨用户覆盖）：

      - Global：md5(query)；
      - User：md5(user_id + query) —— 同用户同问题 upsert 覆盖只留最新答案；
        不同 user_id 互不覆盖。不再包含上下文指纹（User 层按语义匹配）。
    """
    if user_id:
        return hashlib.md5(
            f"{user_id}\x00{query}".encode("utf-8")
        ).hexdigest()
    return hashlib.md5(query.strip().encode("utf-8")).hexdigest()


class SemanticCache:
    """基于 ChromaDB 的双层语义缓存（Global + User 隔离 + 上下文依赖性判定）。"""

    DEFAULT_THRESHOLD = 0.85   # 相似度阈值：>= 命中即复用（0.9 命中率过低，实际形同虚设）
    DEFAULT_TTL_S = 86400      # 缓存条目有效期 24h

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        threshold: float = DEFAULT_THRESHOLD,
        ttl_s: float = DEFAULT_TTL_S,
        enabled: bool = True,
    ):
        self.threshold = threshold
        self.ttl_s = ttl_s
        self.enabled = enabled
        self._hits = 0
        self._misses = 0
        self._bypass = 0

        if not enabled:
            self._global = None
            self._user = None
            return

        try:
            client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            client.heartbeat()
        except Exception:
            logger.info("语义缓存使用本地嵌入式模式")
            client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # Embedding：本地 bge 模型优先；不可用时回退 MiniLM 空间（collection 名随
        # 向量空间切换，两种模型的向量绝不混存）。
        embedding_function = get_embedder()
        if embedding_function is None:
            logger.warning("本地 Embedding 模型不可用，语义缓存回退 MiniLM 向量空间")
        global_name = COLLECTION_GLOBAL if embedding_function is not None else PREVIOUS_COLLECTION_GLOBAL
        user_name = COLLECTION_USER if embedding_function is not None else PREVIOUS_COLLECTION_USER

        # 缓存不迁移：旧缓存的相似度阈值基于旧向量空间，冷启动可避免误命中。
        self._global = client.get_or_create_collection(
            global_name, metadata=_COSINE_METADATA, embedding_function=embedding_function)
        self._user = client.get_or_create_collection(
            user_name, metadata=_COSINE_METADATA, embedding_function=embedding_function)

    # ── 读写 ──────────────────────────────────────────────────────────────────

    def get(self, query: str, user_id: Optional[str] = None, dependence: str = DEP_GLOBAL) -> Optional[Dict[str, Any]]:
        """
        语义检索缓存：相似度 ≥ 阈值且未过期 → 返回缓存条目，否则 None。

        读取层由 cache_read_tier 决定（纯映射）：
          - "global" → 只查 Global 缓存（公共答案，语义匹配）；
          - "user" + 身份有效 → 只查 User 缓存（where 仅过滤 user_id），
            **miss 不回退 Global**；
          - "skip"（强上下文依赖）或匿名 user → 跳过缓存（计 bypass，不算 miss）。
        """
        if not self.enabled or self._global is None or not (query or "").strip():
            self._misses += 1
            return None

        tier = cache_read_tier(dependence, user_id)
        if tier == "user":
            return self._query_collection(
                self._user,
                query,
                where={"user_id": str(user_id)},
            )
        if tier == "global":
            return self._query_collection(self._global, query)

        # skip（强上下文依赖）或匿名 user：从未发起语义查询，计 bypass 而非 miss
        self._bypass += 1
        return None

    def _query_collection(
        self,
        collection: Any,
        query: str,
        where: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        try:
            results = collection.query(
                query_texts=[query.strip()],
                n_results=1,
                where=where,
            )
            if not results["documents"] or not results["documents"][0]:
                self._misses += 1
                return None

            score = round(max(0.0, min(1.0, 1.0 - float(results["distances"][0][0]))), 4)
            if score < self.threshold:
                self._misses += 1
                return None

            meta = results["metadatas"][0][0]
            ts = float(meta.get("ts", 0))
            if time.time() - ts > self.ttl_s:
                self._misses += 1
                return None

            self._hits += 1
            logger.info(f"语义缓存命中: {query[:30]!r} 相似度 {score} tier={meta.get('tier', 'global')}")
            return {
                "response": meta.get("response", ""),
                "domain": meta.get("domain", "other"),
                "agent_type": meta.get("agent_type", ""),
                "score": score,
                "tier": meta.get("tier", "global"),
                "knowledge_used": bool(meta.get("knowledge_used", False)),
            }
        except Exception as ex:
            logger.warning(f"语义缓存查询失败: {ex}")
            self._misses += 1
            return None

    def put(
        self,
        query: str,
        response: str,
        domain: str = "other",
        agent_type: str = "",
        user_id: Optional[str] = None,
        dependence: str = DEP_GLOBAL,
        knowledge_used: bool = False,
    ) -> None:
        """
        写入缓存条目。

        - dependence="global" → Global 缓存（doc_id = md5(query)）；
        - dependence="user" + 有效身份 → User 缓存
          （doc_id = md5(user_id+query)，同用户同问题只留最新）；
        - dependence="skip" / 匿名 user → 静默跳过（强上下文依赖或
          身份不可用的答案不入任何缓存，防污染 Global 公共缓存）。
        """
        if not self.enabled or self._global is None or not (query or "").strip():
            return
        if not (response or "").strip() or len(response) < 20:
            return  # 过短/空回复不缓存
        try:
            tier = cache_tier(dependence, user_id)
            if tier == "user":
                # User 缓存：doc_id 含 user_id，防跨用户覆盖
                self._user.upsert(
                    ids=[_entry_id(query, user_id=user_id)],
                    documents=[query.strip()],
                    metadatas=[{
                        "response": response,
                        "domain": str(domain),
                        "agent_type": str(agent_type),
                        "user_id": str(user_id),
                        "tier": "user",
                        "ts": str(time.time()),
                        "knowledge_used": bool(knowledge_used),
                    }],
                )
            elif tier == "global":
                # Global 缓存：只收上下文无关的公共答案
                self._global.upsert(
                    ids=[_entry_id(query)],
                    documents=[query.strip()],
                    metadatas=[{
                        "response": response,
                        "domain": str(domain),
                        "agent_type": str(agent_type),
                        "tier": "global",
                        "ts": str(time.time()),
                        "knowledge_used": bool(knowledge_used),
                    }],
                )
            # else: skip / 匿名 user → 静默跳过（强上下文依赖或身份不可用）
        except Exception as ex:
            logger.warning(f"语义缓存写入失败: {ex}")

    async def aget(self, query: str, user_id: Optional[str] = None, dependence: str = DEP_GLOBAL) -> Optional[Dict[str, Any]]:
        """在线程池执行同步 Chroma 查询，避免阻塞 FastAPI 事件循环。"""
        return await asyncio.to_thread(self.get, query, user_id, dependence)

    async def aput(self, query: str, response: str, **kwargs: Any) -> None:
        """在线程池执行同步 Chroma 写入，避免阻塞 FastAPI 事件循环。"""
        await asyncio.to_thread(self.put, query, response, **kwargs)

    # ── 统计 ──────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "tiers": "global(公共语义匹配) + user(user_id 分区)",
            "hits": self._hits,
            "misses": self._misses,
            "bypass": self._bypass,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }
