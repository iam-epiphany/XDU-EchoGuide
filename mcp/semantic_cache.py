"""
语义缓存（Semantic Caching）—— GPTCache 思路的轻量实现。

原理：把 (查询 → 回答) 存入 ChromaDB 向量库；新查询先做语义检索，
如果与历史查询的 embedding 相似度 ≥ 阈值（默认 0.90），直接复用历史回答。

收益：
  - 重复/近似问题不再触发 LLM 调用 → 成本趋近于 0
  - 首字延迟从秒级降到毫秒级
  - 回答一致性：同一问题永远得到同一份答复

与 TTL 精确缓存（MCPToolManager._cache）的区别：
  精确缓存要求参数完全相同；语义缓存容忍近义改写（"选课什么时候开始？" ≈
  "选课几时开始？"）。
"""
import logging
import time
from typing import Any, Dict, Optional

import chromadb

logger = logging.getLogger(__name__)


class SemanticCache:
    """基于 ChromaDB 的语义缓存。"""

    COLLECTION_NAME = "semantic_cache"
    DEFAULT_THRESHOLD = 0.90   # 相似度阈值：>= 命中即复用
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
            self._collection = None
            return

        try:
            self._collection = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            ).get_or_create_collection(self.COLLECTION_NAME)
        except Exception:
            logger.info("语义缓存使用本地嵌入式模式")
            self._collection = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            ).get_or_create_collection(self.COLLECTION_NAME)

    # ── 读写 ──────────────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        语义检索缓存：相似度 ≥ 阈值且未过期 → 返回缓存条目，否则 None。
        """
        if not self.enabled or self._collection is None or not (query or "").strip():
            self._misses += 1
            return None

        try:
            results = self._collection.query(
                query_texts=[query.strip()],
                n_results=1,
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
            }
        except Exception as ex:
            logger.warning(f"语义缓存查询失败: {ex}")
            self._misses += 1
            return None

    def put(self, query: str, response: str, domain: str = "other", agent_type: str = "") -> None:
        """写入缓存条目。"""
        if not self.enabled or self._collection is None or not (query or "").strip():
            return
        if not (response or "").strip() or len(response) < 20:
            return  # 过短/空回复不缓存
        try:
            import hashlib

            doc_id = hashlib.md5(query.strip().encode("utf-8")).hexdigest()
            self._collection.upsert(
                ids=[doc_id],
                documents=[query.strip()],
                metadatas=[{
                    "response": response,
                    "domain": str(domain),
                    "agent_type": str(agent_type),
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
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }
