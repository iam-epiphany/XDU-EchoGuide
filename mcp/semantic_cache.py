"""
语义缓存（Semantic Caching）—— GPTCache 思路的轻量实现，双层隔离。

原理：把 (查询 → 回答) 存入 ChromaDB 向量库；新查询先做语义检索，
如果与历史查询的 embedding 相似度 ≥ 阈值（默认 0.85），直接复用历史回答。

双层设计（解决用户隔离问题）：
  - Global 缓存（semantic_cache_global）：只存**不依赖用户上下文**的答案
    （请求时无画像、无历史，即 mem_ctx.to_prompt_text() 为空）。任何用户可复用。
  - User 缓存（semantic_cache_user）：按 user_id 维度隔离，存**个性化答案**
    （回答受用户画像/历史影响）。查询带 where={"user_id": ...} 过滤，
    用户 A 的缓存永远不会命中用户 B。
  - personal 领域（课表/待办/DDL 等个人数据）仍由调用方整体禁止入缓存。

收益：
  - 重复/近似问题不再触发 LLM 调用 → 成本趋近于 0
  - 首字延迟从秒级降到毫秒级
  - 回答一致性：同一问题永远得到同一份答复
  - 用户隔离：个性化答案只在本用户内复用，杜绝跨用户串扰

与 TTL 精确缓存（MCPToolManager._cache）的区别：
  精确缓存要求参数完全相同；语义缓存容忍近义改写（"选课什么时候开始？" ≈
  "选课几时开始？"）。
"""
import hashlib
import logging
import time
from typing import Any, Dict, Optional

import chromadb

logger = logging.getLogger(__name__)

# 语义缓存各 tier 的 collection 名
COLLECTION_GLOBAL = "semantic_cache_global"
COLLECTION_USER = "semantic_cache_user"


def cache_tier(domain: str, has_user_context: bool, user_id: Optional[str] = None) -> Optional[str]:
    """
    决定一次回答写入哪一层缓存（纯函数，可离线单测）。

    规则：
      - personal / other 领域：不缓存（None）—— 个人数据实时变化，不能复用；
      - 无用户上下文（无画像/历史）：Global 缓存（任何用户可复用）；
      - 有用户上下文且 user_id 有效：User 缓存（按 user_id 隔离）；
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


class SemanticCache:
    """基于 ChromaDB 的双层语义缓存（Global + User 隔离）。"""

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

        self._global = client.get_or_create_collection(COLLECTION_GLOBAL)
        self._user = client.get_or_create_collection(COLLECTION_USER)

    # ── 读写 ──────────────────────────────────────────────────────────────────

    def get(self, query: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        语义检索缓存：相似度 ≥ 阈值且未过期 → 返回缓存条目，否则 None。

        查找顺序：User 缓存（仅当 user_id 有效）→ Global 缓存。
        User 缓存按 user_id 隔离；Global 缓存的答案不依赖用户上下文，可安全复用。
        """
        if not self.enabled or self._global is None or not (query or "").strip():
            self._misses += 1
            return None

        # 1. User 缓存（个性化答案，按 user_id 隔离）
        if user_id and user_id != "anonymous" and self._user is not None:
            hit = self._query_collection(self._user, query, where={"user_id": user_id})
            if hit is not None:
                return hit

        # 2. Global 缓存（上下文无关答案，任何用户可复用）
        return self._query_collection(self._global, query)

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

            score = round(1.0 - results["distances"][0][0], 4)
            if score < self.threshold:
                self._misses += 1
                return None

            meta = results["metadatas"][0][0]
            ts = float(meta.get("ts", 0))
            if time.time() - ts > self.ttl_s:
                self._misses += 1
                return None

            self._hits += 1
            logger.info(f"语义缓存命中: {query[:30]!r} 相似度 {score}")
            return {
                "response": meta.get("response", ""),
                "domain": meta.get("domain", "other"),
                "agent_type": meta.get("agent_type", ""),
                "score": score,
                "tier": meta.get("tier", "global"),
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
    ) -> None:
        """写入缓存条目。user_id 非空写 User 缓存（隔离），否则写 Global 缓存。"""
        if not self.enabled or self._global is None or not (query or "").strip():
            return
        if not (response or "").strip() or len(response) < 20:
            return  # 过短/空回复不缓存
        try:
            doc_id = hashlib.md5(query.strip().encode("utf-8")).hexdigest()
            if user_id and user_id != "anonymous" and self._user is not None:
                self._user.upsert(
                    ids=[doc_id],
                    documents=[query.strip()],
                    metadatas=[{
                        "response": response,
                        "domain": str(domain),
                        "agent_type": str(agent_type),
                        "user_id": str(user_id),
                        "tier": "user",
                        "ts": str(time.time()),
                    }],
                )
            else:
                self._global.upsert(
                    ids=[doc_id],
                    documents=[query.strip()],
                    metadatas=[{
                        "response": response,
                        "domain": str(domain),
                        "agent_type": str(agent_type),
                        "tier": "global",
                        "ts": str(time.time()),
                    }],
                )
        except Exception as ex:
            logger.warning(f"语义缓存写入失败: {ex}")

    # ── 统计 ──────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "tiers": "global + user(按 user_id 隔离)",
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }
