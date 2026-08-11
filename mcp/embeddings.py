"""
本地向量模型服务（ONNX 轻量推理）：Embedding + Rerank 统一入口。

背景：
  旧链路 Embedding 用 ChromaDB 内置 all-MiniLM-L6-v2（英文模型，中文语义弱），
  Rerank 用 LLM API 打分（秒级延迟 + token 成本）。
  本模块改为进程内 onnxruntime 推理中文优化模型，无 torch 依赖：
    - Embedding: onnx-community/bge-small-zh-v1.5-ONNX（bge-small-zh-v1.5，512 维，~95MB）
    - Rerank:    Xenova/bge-reranker-base（cross-encoder，fp16 ~570MB）
  模型文件经 HF（尊重 HF_ENDPOINT 镜像）下载到 ECHOGUIDE_MODEL_CACHE_DIR
  （Docker 镜像构建时预下载 / 运行期首次调用懒加载）。

设计要点：
  1. 懒加载 + 单例：首个调用才下载/加载 ONNX session；加载失败返回 None，
     由调用方降级（知识库回退 chroma 默认模型、意图识别回退 n-gram、重排回退 LLM）。
     失败带 5 分钟冷却，之后自动重试（网络瞬时故障可自愈）。
  2. 池化自实现：ONNX 导出仅含 BERT 主体（不含 pooling），
     mean pooling（attention_mask 加权）→ L2 归一化
     （注意 chromadb 的 normalize_embeddings 只做类型规整，不是归一化）。
  3. bge-zh 指令前缀：query 侧加「为这个句子生成表示以用于检索相关文章：」。
     chromadb 0.5.x 的 collection 调用无法区分 query/document（统一 __call__），
     而 BAAI 明确不建议给 passage 加指令 → chromadb 路径默认不加前缀
     （ECHOGUIDE_EMBED_PREFIX_MODE=both 可切换）；直连调用方（意图识别模板匹配）
     通过 embed_query() / embed_documents() 显式区分两侧。
  4. 线程安全：onnxruntime session 可并发 run；懒加载用锁保护。
"""
import logging
import os
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── 模型与配置 ────────────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_MODEL = "onnx-community/bge-small-zh-v1.5-ONNX"
DEFAULT_RERANK_MODEL = "Xenova/bge-reranker-base"
# bge-zh v1.5 官方 query 指令（只用于 query 侧，文档/passage 侧不加）
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

# 模型文件按优先级尝试（不同仓库提供的量化格式不同，取第一个存在的）
_EMBEDDING_FILE_PRIORITY = (
    "onnx/model.onnx", "onnx/model_fp16.onnx", "onnx/model_quantized.onnx",
)
_RERANK_FILE_PRIORITY = (
    "onnx/model_fp16.onnx", "onnx/model_int8.onnx",
    "onnx/model_uint8.onnx", "onnx/model.onnx",
)
# tokenizer / 配置配套文件（tokenizer.json 必需，其余缺失不阻断）
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json",
                    "special_tokens_map.json", "config.json")

_MAX_SEQ_LEN = 512          # bge 系列最大序列长度
_RETRY_COOLDOWN_S = 300.0   # 加载失败后的冷却时间（避免每次调用都重试网络下载）
_DOWNLOAD_TIMEOUT_S = 30.0  # 单文件下载超时（快速失败，不拖垮降级链路）


def _env(key: str, default: str) -> str:
    return os.getenv(key, default).strip()


def _hf_base() -> str:
    """HF 端点（尊重 HF_ENDPOINT 镜像，如 https://hf-mirror.com）。"""
    return os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def model_cache_dir() -> Path:
    """模型缓存目录（ECHOGUIDE_MODEL_CACHE_DIR，默认 ~/.cache/echoguide_models）。"""
    path = Path(_env("ECHOGUIDE_MODEL_CACHE_DIR",
                     str(Path.home() / ".cache" / "echoguide_models"))).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download_file(repo_id: str, filename: str, dest: Path) -> bool:
    """下载 HF 文件到本地。urllib 实现（超时快速失败，无第三方重试拖慢降级）。

    返回是否可用；本地已有文件时直接复用（幂等）。下载中途失败清理 .part 残留。
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True
    url = f"{_hf_base()}/{repo_id}/resolve/main/{filename}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "EchoGuide/1.0"})
        with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT_S) as resp, \
                open(tmp, "wb") as out:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
        if tmp.stat().st_size > 0:
            tmp.rename(dest)
            return True
        tmp.unlink(missing_ok=True)
        return False
    except Exception as ex:
        tmp.unlink(missing_ok=True)
        logger.debug("模型文件下载失败 %s/%s: %s", repo_id, filename, ex)
        return False


def _ensure_model(repo_id: str, model_priority: List[str], cache_dir: Path) -> Path:
    """确保模型文件 + tokenizer 配套文件在本地缓存，返回 ONNX 模型文件路径。

    - 模型文件：按优先级取第一个下载成功的（量化格式可配置）；
    - 全部不可用或 tokenizer.json 缺失 → 抛 RuntimeError（调用方降级）。
    """
    repo_dir = cache_dir / repo_id.replace("/", "--")
    for fname in _TOKENIZER_FILES:
        _download_file(repo_id, fname, repo_dir / fname)
    for fname in model_priority:
        dest = repo_dir / fname
        if _download_file(repo_id, fname, dest):
            return dest
    raise RuntimeError(
        f"模型 {repo_id} 的 ONNX 文件均不可用（检查网络或 ECHOGUIDE_MODEL_CACHE_DIR）")


class _LoadedModel:
    """onnxruntime session + tokenizer 的懒加载容器（线程安全，带失败冷却）。"""

    def __init__(self, repo_id: str, model_priority: List[str], cache_dir: Optional[Path] = None):
        self._repo_id = repo_id
        self._model_priority = model_priority
        self._cache_dir = cache_dir or model_cache_dir()
        self._lock = threading.Lock()
        self._session: Any = None
        self._tokenizer: Any = None
        self._error: Optional[str] = None
        self._last_attempt = 0.0

    def ensure_loaded(self) -> Any:
        """返回 onnxruntime session；不可用返回 None（冷却期内不重试）。"""
        with self._lock:
            if self._session is not None:
                return self._session
            now = time.monotonic()
            if self._error and now - self._last_attempt < _RETRY_COOLDOWN_S:
                return None
            self._last_attempt = now
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                model_path = _ensure_model(self._repo_id, self._model_priority, self._cache_dir)
                tok_path = model_path.parent.parent / "tokenizer.json"
                if not tok_path.exists():
                    found = list(model_path.parent.parent.rglob("tokenizer.json"))
                    tok_path = found[0] if found else None
                if tok_path is None:
                    raise RuntimeError(f"模型 {self._repo_id} 缺少 tokenizer.json")

                tokenizer = Tokenizer.from_file(str(tok_path))
                tokenizer.enable_truncation(max_length=_MAX_SEQ_LEN)
                tokenizer.enable_padding(
                    pad_id=tokenizer.token_to_id("[PAD]") or 0, pad_token="[PAD]",
                )
                session = ort.InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"],
                )
                self._session = session
                self._tokenizer = tokenizer
                self._error = None
                logger.info("本地模型已加载: %s（%s）", self._repo_id, model_path.name)
                return session
            except Exception as ex:
                self._session = None
                self._error = str(ex)
                logger.warning("本地模型 %s 加载失败（调用方将降级，%.0f 秒后重试）: %s",
                               self._repo_id, _RETRY_COOLDOWN_S, ex)
                return None

    def tokenize(self, texts: Any) -> Dict[str, np.ndarray]:
        """tokenize → 按 session 实际输入名构造 feeds（兼容无 token_type_ids 的模型）。

        texts 为 List[str]（单文本）或 List[Tuple[str, str]]（句子对，reranker 用，
        tokenizer 自动拼为 [CLS] a [SEP] b [SEP]）。
        """
        if self._session is None:
            raise RuntimeError(f"模型不可用: {self._error}")
        encodings = self._tokenizer.encode_batch(texts)
        feeds: Dict[str, np.ndarray] = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        if any(inp.name == "token_type_ids" for inp in self._session.get_inputs()):
            feeds["token_type_ids"] = np.array(
                [e.type_ids for e in encodings], dtype=np.int64)
        return feeds

    def run(self, texts: Any) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """session.run 包装：返回 (输出, feeds)，feeds 供调用方复用（避免二次 tokenize）。

        输出为第一个张量（last_hidden_state / logits）。
        """
        session = self.ensure_loaded()
        if session is None:
            raise RuntimeError(f"模型不可用: {self._error}")
        feeds = self.tokenize(texts)
        return np.asarray(session.run(None, feeds)[0], dtype=np.float32), feeds

    @property
    def error(self) -> Optional[str]:
        return self._error


class LocalEmbedder:
    """bge-zh 系列 ONNX Embedding（mean pooling + L2 归一化）。

    与 chromadb EmbeddingFunction 协议兼容（提供 __call__）：
      - chromadb collection 路径：__call__，前缀模式由 ECHOGUIDE_EMBED_PREFIX_MODE
        决定（none=两侧都不加指令，both=两侧都加；默认 none —— 0.5.x 无法区分
        query/document，BAAI 不建议给 passage 加指令）；
      - 直连路径：embed_query() 带指令前缀 / embed_documents() 不带。
    """

    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[Path] = None):
        self._model_name = model_name or _env("ECHOGUIDE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._prefix_mode = _env("ECHOGUIDE_EMBED_PREFIX_MODE", "none").strip().lower()
        self._model = _LoadedModel(self._model_name, list(_EMBEDDING_FILE_PRIORITY), cache_dir)

    @property
    def available(self) -> bool:
        return self._model.ensure_loaded() is not None

    def embed_texts(self, texts: List[str], *, is_query: bool = False) -> List[List[float]]:
        """核心嵌入。is_query=True 时加 bge-zh 指令前缀（仅直连调用方应传 True）。

        mean pooling（attention_mask 加权）→ L2 归一化；向量空间与 chromadb
        collection 一致（cosine 相似度）。
        """
        if not texts:
            return []
        prepared = [QUERY_INSTRUCTION + t if is_query else t for t in texts]
        hidden, feeds = self._model.run(prepared)                     # [batch, seq, hidden]
        mask_f = feeds["attention_mask"].astype(np.float32)[:, :, None]
        pooled = (hidden * mask_f).sum(axis=1) / mask_f.sum(axis=1).clip(min=1e-9)
        norm = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
        return [v.tolist() for v in (pooled / norm).astype(np.float32)]

    def embed_query(self, texts: List[str]) -> List[List[float]]:
        """query 侧（带指令前缀）。"""
        return self.embed_texts(texts, is_query=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """文档侧（不带指令）。"""
        return self.embed_texts(texts, is_query=False)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """chromadb EmbeddingFunction 协议入口（0.5.x 无法区分 query/document）。"""
        return self.embed_texts(input, is_query=(self._prefix_mode == "both"))


class LocalReranker:
    """bge-reranker 系列 cross-encoder 重排（score = sigmoid(logit) ∈ [0,1]）。"""

    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[Path] = None):
        self._model_name = model_name or _env("ECHOGUIDE_RERANK_MODEL", DEFAULT_RERANK_MODEL)
        self._model = _LoadedModel(self._model_name, list(_RERANK_FILE_PRIORITY), cache_dir)

    @property
    def available(self) -> bool:
        return self._model.ensure_loaded() is not None

    def score(self, query: str, texts: List[str]) -> List[float]:
        """query 与每个候选的相关性分数（sigmoid 归一化到 [0,1]）。"""
        if not texts:
            return []
        logits, _ = self._model.run([(query, t) for t in texts])
        logits = logits.reshape(-1)
        return [float(1.0 / (1.0 + np.exp(-x))) for x in logits]

    def rerank(self, query: str, items: List[Any], top_k: int) -> List[Any]:
        """按相关度降序重排 items（item 为任意对象，仅排序不修改），取 top_k。"""
        if not items or top_k <= 0:
            return items[:top_k]
        scores = self.score(query, [str(item) for item in items])
        order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
        return [items[i] for i in order[:top_k]]


# ── 全局单例（懒加载，失败冷却后自动重试）─────────────────────────────────────

_embedder_holder: Dict[str, Any] = {"instance": None, "failed_at": 0.0}
_reranker_holder: Dict[str, Any] = {"instance": None, "failed_at": 0.0}
_singleton_lock = threading.Lock()


def _get_singleton(holder: Dict[str, Any], factory: Any) -> Any:
    with _singleton_lock:
        if holder["instance"] is not None:
            return holder["instance"]
        if holder["failed_at"] and time.monotonic() - holder["failed_at"] < _RETRY_COOLDOWN_S:
            return None
        try:
            instance = factory()
            if instance.available:  # 触发加载；失败走 except 分支
                holder["instance"] = instance
                return instance
            raise RuntimeError("模型不可用")
        except Exception as ex:
            holder["failed_at"] = time.monotonic()
            logger.warning("模型初始化失败（%.0f 秒后自动重试）: %s", _RETRY_COOLDOWN_S, ex)
            return None


def get_embedder() -> Optional[LocalEmbedder]:
    """全局共享 Embedder；不可用时返回 None（调用方降级，冷却后自动重试）。"""
    return _get_singleton(_embedder_holder, LocalEmbedder)


def get_reranker() -> Optional[LocalReranker]:
    """全局共享 Reranker；不可用时返回 None（调用方降级，冷却后自动重试）。"""
    return _get_singleton(_reranker_holder, LocalReranker)


def reset_singletons() -> None:
    """清空全局单例（测试用）。"""
    with _singleton_lock:
        _embedder_holder.update(instance=None, failed_at=0.0)
        _reranker_holder.update(instance=None, failed_at=0.0)


def preload_models() -> None:
    """预下载 Embedding + Rerank 模型到缓存目录（Dockerfile 构建阶段调用）。

    任一模型不可用即抛错，让镜像构建失败（而不是带一个残废镜像）。
    """
    embedder = LocalEmbedder()
    if not embedder.available:
        raise RuntimeError(f"Embedding 模型预下载失败: {embedder._model.error}")
    reranker = LocalReranker()
    if not reranker.available:
        raise RuntimeError(f"Rerank 模型预下载失败: {reranker._model.error}")
    logger.info("模型预下载完成: %s + %s", embedder._model_name, reranker._model_name)
