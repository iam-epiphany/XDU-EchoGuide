"""
亮点：MCP 工具调用框架

核心问题：工具调用出错（检索不全、召回不好）怎么优化？

本模块的答案：
  1. 查询改写（Query Rewriting）—— 用 LLM 把用户原始问题扩写成多个角度的子查询，
     再合并去重，解决"召回不全"问题。
  2. 结果重排（Reranking）—— 对召回结果用 LLM 打分，按相关性重新排序，
     解决"召回不好/排序差"问题。
  3. 熔断器（Circuit Breaker）—— 连续失败超阈值时自动断开，防止雪崩。
  4. 结果缓存（TTL Cache）—— 相同参数直接返回缓存，减少重复调用。
  5. 降级策略（Fallback）—— 工具不可用时返回有意义的降级结果。

检索优化链路（查询改写 → 并行召回 → 去重 → LLM 重排）是 /search 演示接口
与 Agentic RAG 主链路（Agent 调用 knowledge_search）共用的同一套链路：
注册工具时设置 use_rewrite=True，call() 即自动走完整优化链路。
"""
import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, List, Optional

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED    = "closed"     # 正常
    OPEN      = "open"       # 熔断，拒绝请求
    HALF_OPEN = "half_open"  # 探测恢复


@dataclass
class ToolResult:
    success:        bool
    data:           Any
    tool_name:      str
    error:          Optional[str] = None
    cached:         bool = False
    latency_ms:     float = 0.0
    reranked:       bool = False   # 是否经过重排


@dataclass
class ToolStats:
    """工具运行时统计，供 Monitor 读取。"""
    total:              int = 0
    success:            int = 0
    failed:             int = 0
    total_latency_ms:   float = 0.0
    consecutive_fails:  int = 0

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total if self.total else 0.0


# ── 熔断器 ────────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    三态熔断器：CLOSED → OPEN → HALF_OPEN → CLOSED

    连续失败 failure_threshold 次后打开；
    打开 recovery_s 秒后进入 HALF_OPEN 探测；
    探测成功则关闭，失败则重新打开。
    """

    def __init__(self, failure_threshold: int = 5, recovery_s: float = 60.0):
        self.threshold   = failure_threshold
        self.recovery_s  = recovery_s
        self.state       = CircuitState.CLOSED
        self.fail_count  = 0
        self.opened_at:  Optional[float] = None

    def allow(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.opened_at >= self.recovery_s:  # type: ignore
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        return True  # HALF_OPEN：放行一次探测

    def record_success(self) -> None:
        self.fail_count = 0
        self.state = CircuitState.CLOSED

    def record_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= self.threshold:
            self.state     = CircuitState.OPEN
            self.opened_at = time.monotonic()
            logger.warning(f"熔断器打开（连续失败 {self.fail_count} 次）")


# ── 工具定义 ──────────────────────────────────────────────────────────────────

class ToolEffect(Enum):
    """工具副作用声明（fail-closed）：未声明 = 不暴露、不可写。

    - READ: 只读，无副作用（检索/查询）；
    - WRITE: 修改系统内部状态（如待办数据）；
    - EXTERNAL_SIDE_EFFECT: 对外部系统产生副作用（如发送通知、调用外部写接口）。
    """
    READ                 = "read"
    WRITE                = "write"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"


@dataclass
class Tool:
    name:        str
    description: str
    handler:     Callable                    # async (params, context) -> Any
    schema:      Dict[str, Any]              # JSON Schema
    cache_ttl:   float = 0.0                 # 0 = 不缓存
    timeout_s:   float = 30.0
    supports_rerank: bool = False            # 是否支持结果重排
    use_rewrite: bool = False                # 调用时是否自动走「查询改写→并行召回→去重→重排」链路
    fallback:    Optional[Callable] = None    # sync/async (params, context, error) -> Any
    agent_exposed: bool = True               # 是否暴露给 Agent 的 function calling
    # 副作用声明（fail-closed）：None = 未声明 → 不暴露给 Agent、不可写。
    # 新增工具必须显式声明 effect；忘记声明时只影响它自己（不可见不可写），
    # 不会被只读动作误当只读工具开放。
    effect:      Optional[ToolEffect] = None

    # 运行时状态（不参与构造）
    stats:   ToolStats    = field(default_factory=ToolStats, init=False)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker, init=False)

    @property
    def is_agent_visible(self) -> bool:
        """Agent 可见性 = 显式暴露 AND 显式声明副作用（fail-closed）。

        未声明 effect 的工具即使 agent_exposed=True 也不进入 Agent 工具面，
        避免"新增副作用工具忘记声明却被当只读工具开放"。
        """
        return self.agent_exposed and self.effect is not None

    @property
    def is_write(self) -> bool:
        """是否写工具：WRITE / EXTERNAL_SIDE_EFFECT 均视为有副作用（写集合成员）。"""
        return self.effect in (ToolEffect.WRITE, ToolEffect.EXTERNAL_SIDE_EFFECT)


# ── MCP 工具管理器 ────────────────────────────────────────────────────────────

class MCPToolManager:
    """
    MCP 工具调用框架。

    核心优化链路（针对检索类工具）：
      用户查询 → 查询改写（多角度子查询）→ 并行召回 → 结果重排 → 返回 Top-K
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        rerank_backend: Optional[str] = None,
        gateway: Optional[Any] = None,  # 统一模型调用入口（改写/重排兜底 LLM 调用）
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)
        self._model  = model
        self._gateway = gateway
        # 重排后端（构造参数优先，否则读环境变量）：
        #   local（默认）= 本地 bge-reranker 毫秒级打分，不可用自动降级 LLM；
        #   llm = LLM 打分（旧行为，秒级延迟 + token 成本）；
        #   off = 关闭重排，按原顺序截断 Top-K。
        self._rerank_backend = (
            rerank_backend or os.getenv("ECHOGUIDE_RERANK_BACKEND", "local")
        ).strip().lower()
        self._tools: Dict[str, Tool] = {}
        self._cache: Dict[str, tuple] = {}   # key → (result, expire_at)

    # ── 注册 / 注销 ───────────────────────────────────────────────────────────

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def write_tools(self) -> FrozenSet[str]:
        """读写门禁的写工具集合：由各工具声明的 effect 推导。

        新增有副作用的工具只需显式声明 effect=WRITE / EXTERNAL_SIDE_EFFECT，
        无需维护任何黑名单（旧版 WRITE_TOOLS 手工登记，新增写工具容易漏登记）；
        未声明 effect 的工具天然不在写集合（fail-closed，不会被只读动作误开放）。
        """
        return frozenset(name for name, t in self._tools.items() if t.is_write)

    # ── 核心调用 ──────────────────────────────────────────────────────────────

    async def call(
        self,
        name: str,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        *,
        use_cache: bool = True,
        rerank_top_k: int = 0,          # >0 时对结果重排，取 Top-K
        use_rewrite: Optional[bool] = None,  # None=按工具配置，False=绕过，True=强制走改写链路
    ) -> ToolResult:
        """
        调用工具，完整执行链：
          缓存检查 → 熔断检查 → 参数校验 → 执行（含超时）→ 缓存写入 → 可选重排

        use_rewrite=True 的工具（如 knowledge_search）自动走
        「查询改写 → 并行召回 → 去重 → LLM 重排」完整检索优化链路
        （与 /search 演示接口共用），Agentic RAG 主链路因此获得同样优化。
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(success=False, data=None, tool_name=name, error=f"工具不存在: {name}")

        # 熔断检查：先于改写链路，避免熔断期所有改写调用绕过熔断继续打后端，
        # 也避免熔断状态被 rewrite 链路掩盖（子查询各自 fallback 导致假成功）。
        if not tool.breaker.allow():
            error = f"工具熔断中: {name}，请稍后重试"
            return await self._fallback_result(tool, params, context, error)

        # 查询改写检索链路：注册时 use_rewrite=True 的工具，任何调用方（Agent 工具循环、
        # /search 等）都自动走完整优化链路；子查询的参数（min_score/domain 等）原样透传，
        # 保证领域过滤、相关性阈值等语义在改写链路中不丢失。
        if tool.use_rewrite and use_rewrite is not False:
            return await self.search_with_rewrite(
                name,
                params.get("query", ""),
                top_k=params.get("top_k", 5),
                context=context,
                extra_params={
                    k: v for k, v in params.items() if k not in ("query", "top_k")
                },
            )

        # 缓存命中
        if use_cache and tool.cache_ttl > 0:
            cached = self._get_cache(name, params)
            if cached is not None:
                tool.stats.total += 1
                tool.stats.success += 1
                return ToolResult(success=True, data=cached, tool_name=name, cached=True)

        t0 = time.monotonic()
        tool.stats.total += 1
        try:
            # 参数校验（根据 JSON Schema 的 required 和 properties.type）
            self._validate_params(tool, params)

            data = await asyncio.wait_for(tool.handler(params, context), timeout=tool.timeout_s)
            latency = (time.monotonic() - t0) * 1000

            tool.stats.success += 1
            tool.stats.consecutive_fails = 0
            tool.stats.total_latency_ms += latency
            tool.breaker.record_success()

            # 写缓存
            if tool.cache_ttl > 0:
                self._set_cache(name, params, data, tool.cache_ttl)

            # 重排（针对返回列表的检索工具）
            reranked = False
            if rerank_top_k > 0 and tool.supports_rerank and isinstance(data, list):
                query = params.get("query", "")
                data, reranked = await self._rerank(query, data, rerank_top_k), True

            return ToolResult(success=True, data=data, tool_name=name,
                              latency_ms=latency, reranked=reranked)

        except asyncio.TimeoutError:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具超时: {name} ({tool.timeout_s}s)")
            return await self._fallback_result(tool, params, context, "执行超时")

        except Exception as ex:
            tool.stats.failed += 1
            tool.stats.consecutive_fails += 1
            tool.breaker.record_failure()
            logger.error(f"工具异常: {name} — {ex}")
            return await self._fallback_result(tool, params, context, str(ex))

    async def _fallback_result(
        self,
        tool: Tool,
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]],
        error: str,
    ) -> ToolResult:
        """工具不可用时返回降级结果，而不是把空错误直接暴露给调用方。"""
        if tool.fallback is None:
            return ToolResult(success=False, data=None, tool_name=tool.name, error=error)
        try:
            data = tool.fallback(params, context, error)
            if asyncio.iscoroutine(data):
                data = await data
            return ToolResult(
                success=True,
                data=data,
                tool_name=tool.name,
                error=error,
            )
        except Exception as ex:
            logger.error(f"工具降级失败: {tool.name} — {ex}")
            return ToolResult(success=False, data=None, tool_name=tool.name, error=f"{error}; fallback失败: {ex}")

    # ── 查询改写（解决召回不全）────────────────────────────────────────────────

    async def rewrite_query(self, query: str, n: int = 3) -> List[str]:
        """
        用 LLM 将原始查询改写为 n 个不同角度的子查询。

        目的：单一查询往往只能召回某一角度的文档，
        多角度子查询并行检索后合并，显著提升召回率。

        示例：
          原始: "选课流程"
          改写: ["如何选课", "选课分几个阶段", "选课有什么要求"]
        """
        prompt = f"""将以下用户查询改写为 {n} 个不同角度的搜索子查询，用于检索知识库。
要求：每个子查询角度不同，覆盖原始问题的不同方面。
原始查询: "{query}"
返回 JSON 数组，例如: ["子查询1", "子查询2", "子查询3"]"""
        prompt = self._clean_text(prompt)
        try:
            async def _one():
                if self._gateway is not None:
                    result = await self._gateway.call(
                        client=self._client,
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        span_name="rewrite_query",
                        max_tokens=256, temperature=0.3,
                        thinking={"type": "disabled"},
                    )
                    return result.response
                return await self._client.messages.create(
                    model=self._model, max_tokens=256, temperature=0.3,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}],
                )
            resp = await asyncio.wait_for(_one(), timeout=15.0)
            raw = resp.content[0].text
            s, e = raw.find("["), raw.rfind("]") + 1
            queries = json.loads(raw[s:e])
            # 原始查询也保留，去重
            return list(dict.fromkeys([query] + queries))
        except Exception as ex:
            logger.warning(f"查询改写失败，使用原始查询: {ex}")
            return [query]

    async def search_with_rewrite(
        self,
        tool_name: str,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        完整的检索优化链路：查询改写 → 并行召回 → 去重 → 重排 → Top-K

        这是解决"检索不全、召回不好"的完整方案，同时服务于：
          - /search 演示接口（直接调用本方法）
          - Agentic RAG 主链路（use_rewrite=True 的工具经 call() 自动进入本链路）

        extra_params: 透传给每个子查询的额外参数（如 min_score/domain），
        保证领域过滤、相关性阈值等语义在改写链路中不丢失。
        """
        # 1. 查询改写：生成多角度子查询
        sub_queries = await self.rewrite_query(query, n=3)
        logger.info(f"查询改写: {query!r} → {sub_queries}")

        # 2. 并行召回：所有子查询同时检索（use_rewrite=False 防递归：子查询本身不再改写）
        recall_k = max(top_k, 5)
        tasks = [
            self.call(
                tool_name,
                {"query": q, "top_k": recall_k, **(extra_params or {})},
                context,
                use_cache=True,
                use_rewrite=False,
            )
            for q in sub_queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 3. 合并去重（按内容哈希去重）；剔除降级占位条目 —— fallback 数据不是
        # 真实检索结果，混入合并池会被 LLM 重排成"证据"误导 Agent 引用。
        seen, merged = set(), []
        for r in results:
            if isinstance(r, ToolResult) and r.success and isinstance(r.data, list):
                for item in r.data:
                    if isinstance(item, dict) and item.get("fallback"):
                        continue
                    key = hashlib.md5(str(item).encode()).hexdigest()
                    if key not in seen:
                        seen.add(key)
                        merged.append(item)

        if not merged:
            errors = "; ".join(
                r.error for r in results
                if isinstance(r, ToolResult) and r.error
            )
            detail = "所有子查询均无结果"
            if errors:
                detail = f"{detail}（{errors}）"
            return ToolResult(success=False, data=[], tool_name=tool_name, error=detail)

        # 4. 重排：用 LLM 对合并结果按相关性打分，取 Top-K
        reranked = await self._rerank(query, merged, top_k)
        return ToolResult(success=True, data=reranked, tool_name=tool_name, reranked=True)

    # ── 结果重排（解决召回不好）──────────────────────────────────────────────

    async def _rerank(self, query: str, items: List[Any], top_k: int) -> List[Any]:
        """
        结果重排：本地 cross-encoder 优先，LLM 兜底。

        解决问题：向量检索的相似度分数不等于"对用户有用"，重排后
        Top-K 质量显著提升。后端由 ECHOGUIDE_RERANK_BACKEND 控制：
          - local（默认）：bge-reranker-base 本地打分（毫秒级，零 token 成本）；
            模型不可用自动降级 LLM；
          - llm：LLM 打分（旧行为，理解语义最强但秒级延迟）；
          - off：不重排，按原顺序截断 Top-K。
        """
        if len(items) <= top_k:
            return items
        if self._rerank_backend == "off":
            return items[:top_k]
        if self._rerank_backend == "local":
            reranked = await self._rerank_local(query, items, top_k)
            if reranked is not None:
                return reranked
            logger.info("本地重排不可用，降级 LLM 重排")
        return await self._rerank_llm(query, items, top_k)

    async def _rerank_local(self, query: str, items: List[Any], top_k: int) -> Optional[List[Any]]:
        """本地 bge-reranker 重排（线程池执行，避免阻塞事件循环）。

        模型不可用返回 None，由 _rerank 决定降级路径。
        """
        try:
            from mcp.embeddings import get_reranker

            reranker = get_reranker()
            if reranker is None:
                return None
            return await asyncio.to_thread(reranker.rerank, query, items, top_k)
        except Exception as ex:
            logger.warning(f"本地重排失败: {ex}")
            return None

    async def _rerank_llm(self, query: str, items: List[Any], top_k: int) -> List[Any]:
        """
        LLM 对召回结果重新打分排序（本地模型不可用时的兜底）。

        解决问题：向量检索的相似度分数不等于"对用户有用"，
        LLM 能理解语义相关性，重排后 Top-K 质量显著提升。
        """
        if len(items) <= top_k:
            return items

        # 将结果序列化为文本供 LLM 评分
        items_text = "\n".join(f"{i}. {json.dumps(item, ensure_ascii=False)[:200]}"
                               for i, item in enumerate(items))
        prompt = f"""根据用户查询，对以下检索结果按相关性打分（0-10），返回 JSON 数组。
用户查询: "{query}"
检索结果:
{items_text}

返回格式（按相关性降序排列的索引列表）: [最相关的索引, ..., 最不相关的索引]
只返回 JSON 数组，不要其他文字。"""
        prompt = self._clean_text(prompt)

        try:
            async def _one():
                if self._gateway is not None:
                    result = await self._gateway.call(
                        client=self._client,
                        model=self._model,
                        messages=[{"role": "user", "content": prompt}],
                        span_name="rerank_llm",
                        max_tokens=256, temperature=0.0,
                        thinking={"type": "disabled"},
                    )
                    return result.response
                return await self._client.messages.create(
                    model=self._model, max_tokens=256, temperature=0.0,
                    thinking={"type": "disabled"},
                    messages=[{"role": "user", "content": prompt}],
                )
            resp = await asyncio.wait_for(_one(), timeout=15.0)
            raw = resp.content[0].text
            s, e = raw.find("["), raw.rfind("]") + 1
            order: List[int] = json.loads(raw[s:e])
            reranked = [items[i] for i in order if 0 <= i < len(items)]
            return reranked[:top_k]
        except Exception as ex:
            logger.warning(f"重排失败，返回原始顺序: {ex}")
            return items[:top_k]

    # ── 缓存 ──────────────────────────────────────────────────────────────────

    def _cache_key(self, name: str, params: Dict) -> Optional[str]:
        """缓存键。params 含不可序列化值时返回 None（跳过缓存），不抛异常。

        此前 json.dumps 失败会直接打穿 call()；而写缓存失败又会被 except 吞成
        "工具失败"并计入熔断 —— 序列化问题不应污染工具状态。
        """
        try:
            serialized = json.dumps(params, sort_keys=True)
        except (TypeError, ValueError):
            logger.warning("缓存键序列化失败，跳过缓存: %s", name)
            return None
        return f"{name}:{hashlib.md5(serialized.encode()).hexdigest()}"

    def _get_cache(self, name: str, params: Dict) -> Optional[Any]:
        key = self._cache_key(name, params)
        if key is None:
            return None
        if key in self._cache:
            data, expire_at = self._cache[key]
            if time.monotonic() < expire_at:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, name: str, params: Dict, data: Any, ttl: float) -> None:
        key = self._cache_key(name, params)
        if key is None:
            return
        if len(self._cache) >= 5000:
            # 清掉最旧的 1/4
            for k in list(self._cache)[:1250]:
                del self._cache[k]
        self._cache[key] = (data, time.monotonic() + ttl)

    # ── 参数校验（宽容模式）─────────────────────────────────────────────────
    #
    # LLM 生成的工具参数偶尔会带错误类型（如把 3 传成 "3"、把 true 传成 "false" 字符串）。
    # 这里做宽容类型转换而不是直接拒绝：能转就转，转不了才报错。
    # 这样工具的语义（JSON Schema）仍然生效，同时避免一次手滑的参数类型把整轮对话打断。

    _TYPE_MAP = {"string": str, "number": (int, float), "integer": int, "boolean": bool, "array": list, "object": dict}

    @staticmethod
    def _coerce(value: Any, expected: str, key: str) -> Any:
        """把 value 转成 JSON Schema 期望的类型；无法转换时抛 ValueError。"""
        if expected == "string":
            if isinstance(value, str):
                return value
            return str(value)  # 数字/布尔 → 字符串（如 id 5 → "5"）
        if expected == "integer":
            if isinstance(value, bool):
                raise ValueError(f"参数 {key} 类型错误: 期望 integer")
            if isinstance(value, int):
                return value
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                raise ValueError(f"参数 {key} 无法转换为 integer: {value!r}")
        if expected == "number":
            if isinstance(value, bool):
                raise ValueError(f"参数 {key} 类型错误: 期望 number")
            if isinstance(value, (int, float)):
                return value
            try:
                return float(str(value).strip())
            except (TypeError, ValueError):
                raise ValueError(f"参数 {key} 无法转换为 number: {value!r}")
        if expected == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "1", "yes", "是", "完成", "done"):
                    return True
                if lowered in ("false", "0", "no", "否", "未完成"):
                    return False
            raise ValueError(f"参数 {key} 无法转换为 boolean: {value!r}")
        if expected == "array":
            if isinstance(value, list):
                return value
            if isinstance(value, str) and value.strip():
                return [value]  # 单值 → 单元素数组（如 kinds: "ddl" → ["ddl"]）
            raise ValueError(f"参数 {key} 类型错误: 期望 array")
        if expected == "object":
            if isinstance(value, dict):
                return value
            raise ValueError(f"参数 {key} 类型错误: 期望 object")
        return value

    def _validate_params(self, tool: Tool, params: Dict[str, Any]) -> None:
        """根据工具的 JSON Schema 校验参数，并把可转换的参数就地规整为正确类型。"""
        schema = tool.schema
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        for field in required:
            if field not in params:
                raise ValueError(f"工具 {tool.name} 缺少必需参数: {field}")

        for key, value in params.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type and expected_type in self._TYPE_MAP:
                    params[key] = self._coerce(value, expected_type, key)

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 LLM 请求编码失败。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    # ── 统计 ──────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            name: {
                "total": t.stats.total,
                "success_rate": round(t.stats.success_rate, 3),
                "avg_latency_ms": round(t.stats.avg_latency_ms, 1),
                "consecutive_fails": t.stats.consecutive_fails,
                "circuit_state": t.breaker.state.value,
            }
            for name, t in self._tools.items()
        }
