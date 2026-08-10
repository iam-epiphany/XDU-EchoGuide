"""
语义缓存（Semantic Caching）—— GPTCache 思路的轻量实现，双层隔离 + 上下文指纹。

原理：把 (查询 → 回答) 存入 ChromaDB 向量库；新查询先做语义检索，
如果与历史查询的 embedding 相似度 ≥ 阈值（默认 0.85），直接复用历史回答。

双层设计（解决用户隔离问题）：
  - Global 缓存（semantic_cache_global）：只存**不依赖用户上下文**的答案
    （请求时无画像、无历史，即 context_fp 为空）。任何用户可复用。
  - User 缓存（semantic_cache_user）：按 (user_id, context_fp) 双重隔离。
    - user_id：不同用户的个性化答案互不覆盖（doc_id 含 user_id）、互不命中；
    - context_fp：同一用户在不同对话上下文中的回答互不污染
      ——"那几点开门？" 在食堂话题下缓存的答案不会命中图书馆话题的同款追问。
  - personal 领域（课表/待办/DDL 等个人数据）仍由调用方整体禁止入缓存。

读取策略（防绕过个性化推理）：
  - 有上下文（context_fp 非空）→ 只查 User 缓存；miss 后**不回退 Global**，
    保证带用户上下文的请求一定走正常 Agent 推理；
  - 无上下文 → 只查 Global 缓存（公共答案）；
  - 有上下文但身份匿名 → 无法按身份隔离，直接跳过缓存。

注意：缓存读取必须发生在 Memory Context 获取**之后**，否则无法判断
请求是否依赖历史上下文、也无法计算 context_fp。

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

logger = logging.getLogger(__name__)

# 语义缓存各 tier 的 collection 名
COLLECTION_GLOBAL = "semantic_cache_global_v2"
COLLECTION_USER = "semantic_cache_user_v2"
_COSINE_METADATA = {"hnsw:space": "cosine", "description": "EchoGuide semantic cache (cosine)"}


def cache_tier(domain: str, has_user_context: bool, user_id: Optional[str] = None) -> Optional[str]:
    """
    决定一次回答写入哪一层缓存（纯函数，可离线单测）。

    规则：
      - personal / other 领域：不缓存（None）—— 个人数据实时变化，不能复用；
      - 无用户上下文（无画像/历史）：Global 缓存（任何用户可复用）；
      - 有用户上下文且 user_id 有效：User 缓存（按 user_id + 上下文指纹隔离）；
      - 有用户上下文但 user_id 匿名：不缓存 —— 匿名会话无稳定身份，
        写 User 缓存会与其他匿名会话串扰。

    返回 "global" / "user" / None。
    """
    if domain in ("personal", "other"):
        return None
    if not has_user_context:
        return "global"
    if user_id and user_id != "anonymous":
        return "user"
    return None


def context_fingerprint(context_text: str) -> str:
    """
    上下文指纹：对记忆上下文文本（画像/摘要/相关历史/最近对话）取短哈希。

    空上下文返回空串 —— 表示该请求**不依赖**用户上下文，可进 Global 缓存。
    同一用户在不同对话上下文（历史不同）会得到不同指纹，
    从而避免"那几点开门？"这类追问错误命中其他话题的旧缓存。
    """
    text = (context_text or "").strip()
    if not text:
        return ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def cache_read_tier(context_fp: str, user_id: Optional[str] = None) -> Optional[str]:
    """
    决定读取哪一层缓存（纯函数）：

      - context_fp 非空（请求依赖用户上下文）：
          * user_id 有效 → "user"（只查 User 缓存，绝不回退 Global，
            防止公共答案绕过个性化 Agent 推理）；
          * 匿名 → None（无法按身份隔离，跳过缓存直接走正常推理）。
      - context_fp 为空（请求不依赖上下文）→ "global"（公共答案）。

    返回 "user" / "global" / None。
    """
    if context_fp:
        return "user" if (user_id and user_id != "anonymous") else None
    return "global"


def _entry_id(query: str, user_id: Optional[str] = None, context_fp: str = "") -> str:
    """
    缓存条目 ID（防跨用户/跨上下文覆盖）：

      - Global：md5(query)（共享集合，行为不变，与 user_id 无关）；
      - User：md5(user_id + context_fp + query) ——
        "同 query + 不同 user_id" 或 "同 user_id + 不同上下文" 都会生成不同 ID，
        upsert 互不覆盖，真正并存。
    """
    if user_id and context_fp:
        return hashlib.md5(
            f"{user_id}\x00{context_fp}\x00{query}".encode("utf-8")
        ).hexdigest()
    return hashlib.md5(query.strip().encode("utf-8")).hexdigest()


class SemanticCache:
    """基于 ChromaDB 的双层语义缓存（Global + User 隔离 + 上下文指纹）。"""

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

        # 缓存不迁移：旧缓存的相似度阈值基于 L2，冷启动可避免误命中。
        self._global = client.get_or_create_collection(COLLECTION_GLOBAL, metadata=_COSINE_METADATA)
        self._user = client.get_or_create_collection(COLLECTION_USER, metadata=_COSINE_METADATA)

    # ── 读写 ──────────────────────────────────────────────────────────────────

    def get(self, query: str, user_id: Optional[str] = None, context_fp: str = "") -> Optional[Dict[str, Any]]:
        """
        语义检索缓存：相似度 ≥ 阈值且未过期 → 返回缓存条目，否则 None。

        读取层由 cache_read_tier 决定（纯函数）：
          - 有上下文（context_fp 非空）+ 身份有效 → 只查 User 缓存
            （where 同时过滤 user_id 与 context_fp），**miss 不回退 Global**；
          - 无上下文 → 只查 Global 缓存；
          - 有上下文但匿名 → 跳过缓存（无法隔离）。
        """
        if not self.enabled or self._global is None or not (query or "").strip():
            self._misses += 1
            return None

        tier = cache_read_tier(context_fp, user_id)
        if tier == "user":
            return self._query_collection(
                self._user,
                query,
                where={"$and": [
                    {"user_id": str(user_id)},
                    {"context_fp": context_fp},
                ]},
            )
        if tier == "global":
            return self._query_collection(self._global, query)

        self._misses += 1  # 上下文相关但身份匿名：无法隔离，跳过
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
        context_fp: str = "",
        knowledge_used: bool = False,
    ) -> None:
        """
        写入缓存条目。

        - 带有效 user_id 且 context_fp 非空 → User 缓存
          （doc_id 含 user_id+context_fp+query，防跨用户/跨上下文覆盖）；
        - context_fp 为空（上下文无关）→ Global 缓存（doc_id = md5(query)，行为不变）；
        - 上下文相关但身份不可用（匿名/空 user_id）→ 跳过，防止个性化回答
          污染 Global 公共缓存。
        """
        if not self.enabled or self._global is None or not (query or "").strip():
            return
        if not (response or "").strip() or len(response) < 20:
            return  # 过短/空回复不缓存
        try:
            doc_id = _entry_id(query, user_id=user_id, context_fp=context_fp)
            if user_id and user_id != "anonymous" and context_fp and self._user is not None:
                # User 缓存：doc_id 含 user_id + context_fp，防跨用户/跨上下文覆盖
                self._user.upsert(
                    ids=[doc_id],
                    documents=[query.strip()],
                    metadatas=[{
                        "response": response,
                        "domain": str(domain),
                        "agent_type": str(agent_type),
                        "user_id": str(user_id),
                        "context_fp": context_fp,
                        "tier": "user",
                        "ts": str(time.time()),
                        "knowledge_used": bool(knowledge_used),
                    }],
                )
            elif not context_fp:
                # Global 缓存：只收不依赖用户上下文的答案
                self._global.upsert(
                    ids=[doc_id],
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
            else:
                # 上下文相关但身份不可用（匿名/空 user_id）：不入任何缓存，
                # 防止上下文相关的个性化回答污染 Global 公共缓存
                logger.warning(f"上下文相关但身份不可用，跳过缓存写入: {query[:30]!r}")
        except Exception as ex:
            logger.warning(f"语义缓存写入失败: {ex}")

    async def aget(self, query: str, user_id: Optional[str] = None, context_fp: str = "") -> Optional[Dict[str, Any]]:
        """在线程池执行同步 Chroma 查询，避免阻塞 FastAPI 事件循环。"""
        return await asyncio.to_thread(self.get, query, user_id, context_fp)

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
            "tiers": "global + user(user_id + 上下文指纹隔离)",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }
