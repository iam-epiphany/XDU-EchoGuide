"""
亮点：利用 Monitor 监控在线表现（Profile / Model / Tool 维度）

核心问题：如何利用 Monitor 监控系统的在线表现？

本模块的答案：
  1. 实时采集 —— 每隔 N 秒从 Orchestrator 和 ToolManager 拉取最新统计
  2. 异常检测 —— Z-score 统计方法，自动发现指标突变
  3. 有限反馈 —— Fast Profile 不健康时标记 Orchestrator 临时升级 Deep
     （不引入 RL / Bandit / 在线学习，不做实例级智能路由）
  4. 优化建议 —— 基于规则生成可操作的优化建议（不是空话）
  5. 告警 —— 超阈值时打日志 + 可选 Webhook

监控维度（真实业务/执行维度，不是复制的 Agent 对象）：
  - Profile 成功率 / 平均与 P50/P95 延迟（每 Profile 单执行实例）
  - Model / Provider 维度由 Profile 统计承接（错误率、延迟）
  - Tool 成功率 / 延迟 / 连续失败 / 熔断状态
  - Task 状态与 DAG blocked/failed 数量、Runtime 模型/工具调用计数、
    Verifier flags（Orchestrator.observability_counts）
"""
import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

import httpx
from prometheus_client import Gauge, start_http_server

logger = logging.getLogger(__name__)


# ── 数据结构 ──────────────────────────────────────────────────────────────────

class Severity(Enum):
    INFO     = "info"
    WARNING  = "warning"
    ERROR    = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    severity:    Severity
    metric:      str
    message:     str
    value:       float
    threshold:   float
    ts:          str = field(default_factory=lambda: datetime.now().isoformat())
    resolved:    bool = False


@dataclass
class Suggestion:
    """可操作的优化建议。"""
    title:       str
    detail:      str
    action:      str    # 具体操作步骤
    priority:    int    # 1-10


# ── 异常检测 ──────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    基于滑动窗口 Z-score 的异常检测。

    Z-score = |当前值 - 均值| / 标准差
    超过 sensitivity 倍标准差则判定为异常。
    """

    def __init__(self, window: int = 60, sensitivity: float = 2.5):
        self._window      = window
        self._sensitivity = sensitivity
        self._history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window))

    def record(self, metric: str, value: float) -> Optional[Dict[str, Any]]:
        """记录一个数据点，如果异常则返回异常信息，否则返回 None。"""
        buf = self._history[metric]
        buf.append(value)

        if len(buf) < self._window // 2:
            return None  # 数据不足，不检测

        mean  = statistics.mean(buf)
        stdev = statistics.stdev(buf) if len(buf) > 1 else 0.0
        if stdev == 0:
            return None

        z = abs(value - mean) / stdev
        if z > self._sensitivity:
            return {
                "metric":   metric,
                "value":    value,
                "mean":     mean,
                "z_score":  round(z, 2),
                "severity": "high" if z > self._sensitivity * 1.5 else "medium",
            }
        return None


# ── 性能监控器 ────────────────────────────────────────────────────────────────

class PerformanceMonitor:
    """
    Profile / Tool 在线表现监控。

    与 Orchestrator 的联动（有限反馈，不伪装成智能路由）：
      Monitor 采集 → Fast Profile 成功率显著偏低 →
      set_fast_health(False) → Orchestrator 临时把本应走 Fast 的请求升级 Deep
      → 恢复后自动回落。

    不做实例级路由惩罚：每 Profile 单执行实例（同构实例无路由意义），
    未来接入真正异构 Model/Provider 池后再增加实例级 routing score。
    """

    # 告警阈值（agent_* 指标现按 Profile 采集，label 为 fast / deep）
    THRESHOLDS = {
        "agent_success_rate":  (0.90, Severity.ERROR,   "less_than"),
        "tool_success_rate":   (0.95, Severity.WARNING,  "less_than"),
        "agent_avg_ms":        (3000, Severity.WARNING,  "greater_than"),
        "tool_avg_ms":         (5000, Severity.ERROR,    "greater_than"),
    }

    def __init__(
        self,
        orchestrator,
        tool_manager,
        interval_s:       float = 10.0,
        webhook_url:      Optional[str] = None,
        prometheus_port:  Optional[int] = None,   # None = 不启动
    ):
        self._orchestrator = orchestrator
        self._tool_manager = tool_manager
        self._interval     = interval_s
        self._webhook      = webhook_url
        self._detector     = AnomalyDetector()

        self._alerts:      List[Alert]      = []
        self._suggestions: List[Suggestion] = []
        self._active       = False
        self._task:        Optional[asyncio.Task] = None

        # Prometheus 指标（可选）
        self._prom: Dict[str, Any] = {}
        if prometheus_port:
            self._setup_prometheus(prometheus_port)

    def _setup_prometheus(self, port: int) -> None:
        self._prom = {
            "agent_success_rate": Gauge("agent_success_rate", "Profile 成功率", ["agent"]),
            "agent_avg_latency_ms": Gauge("agent_avg_latency_ms", "Profile 平均延迟", ["agent"]),
            "agent_p95_latency_ms": Gauge("agent_p95_latency_ms", "Profile P95 延迟", ["agent"]),
            "agent_requests_total": Gauge("agent_requests_total", "Profile 请求总数", ["agent"]),
            "agent_in_flight": Gauge("agent_in_flight", "Profile 当前并发", ["agent"]),
            "tool_success_rate":  Gauge("tool_success_rate", "工具成功率", ["tool"]),
            "tool_avg_latency_ms": Gauge("tool_avg_latency_ms", "工具平均延迟", ["tool"]),
            "tool_requests_total": Gauge("tool_requests_total", "工具请求总数", ["tool"]),
            "task_status": Gauge("task_status", "Task 状态计数", ["status"]),
        }
        start_http_server(port)
        logger.info(f"Prometheus 已启动: :{port}")

    # ── 生命周期 ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._active:
            return
        self._active = True
        self._task   = asyncio.create_task(self._loop())
        logger.info(f"Monitor 已启动，采集间隔 {self._interval}s")

    async def stop(self) -> None:
        self._active = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    # ── 采集循环 ──────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        while self._active:
            try:
                await self._collect()
            except Exception as ex:
                logger.error(f"Monitor 采集异常: {ex}")
            await asyncio.sleep(self._interval)

    async def _collect(self) -> None:
        """
        采集 Profile / Tool 的实时统计与聚合计数，检测异常，生成建议。

        关键：这里读取的 stats 就是 Orchestrator/ToolManager 在处理请求时
        实时更新的数据，Monitor 不需要额外埋点。
        """
        agent_stats = self._orchestrator.get_stats()
        tool_stats  = self._tool_manager.get_stats()

        # ── Profile 指标（每 Profile 单执行实例，key = fast / deep）──────────
        for profile_key, s in agent_stats.items():
            sr  = s["success_rate"]
            ms  = s["avg_ms"]

            # 异常检测
            for metric, value in [("agent_success_rate", sr), ("agent_avg_ms", ms)]:
                anomaly = self._detector.record(f"{metric}:{profile_key}", value)
                if anomaly:
                    logger.warning(f"异常检测 [{profile_key}] {metric}={value:.3f} z={anomaly['z_score']}")

            # 阈值告警
            self._check_threshold("agent_success_rate", sr, profile_key)
            self._check_threshold("agent_avg_ms", ms, profile_key)

            # Prometheus
            if "agent_success_rate" in self._prom:
                self._prom["agent_success_rate"].labels(agent=profile_key).set(sr)
                self._prom["agent_avg_latency_ms"].labels(agent=profile_key).set(ms)
                self._prom["agent_p95_latency_ms"].labels(agent=profile_key).set(s.get("p95_ms", 0))
                self._prom["agent_requests_total"].labels(agent=profile_key).set(s["total"])
                self._prom["agent_in_flight"].labels(agent=profile_key).set(s.get("in_flight", 0))

        # ── Profile 级有限反馈（Fast/Deep 选择）─────────────────────────────
        # Fast 路径成功率显著偏低且样本足够 → 标记 Fast 不健康，Orchestrator
        # 临时把本应走 Fast 的请求升级 Deep；恢复后自动回落（不引入 RL/Bandit）。
        fast_health = self._fast_health(agent_stats)
        health_updater = getattr(self._orchestrator, "set_fast_health", None)
        if health_updater is not None:
            health_updater(fast_health)

        # ── 工具指标 ──────────────────────────────────────────────────────────
        for tool_name, s in tool_stats.items():
            sr = s["success_rate"]
            ms = s["avg_latency_ms"]
            cf = s["consecutive_fails"]

            self._check_threshold("tool_success_rate", sr, tool_name)
            self._check_threshold("tool_avg_ms", ms, tool_name)

            if "tool_success_rate" in self._prom:
                self._prom["tool_success_rate"].labels(tool=tool_name).set(sr)
                self._prom["tool_avg_latency_ms"].labels(tool=tool_name).set(ms)
                self._prom["tool_requests_total"].labels(tool=tool_name).set(s.get("total_calls", 0))

            # 连续失败 → 生成具体建议
            if cf >= 3:
                self._add_suggestion(Suggestion(
                    title=f"工具 {tool_name} 连续失败 {cf} 次",
                    detail=f"成功率 {sr:.1%}，平均延迟 {ms:.0f}ms，熔断状态: {s['circuit_state']}",
                    action="1. 检查工具依赖服务是否正常\n2. 查看错误日志\n3. 考虑增加超时时间或降级策略",
                    priority=9,
                ))

        # ── 真实业务/执行维度计数（Task 状态 / Runtime 调用 / Verifier flags）─
        obs = getattr(self._orchestrator, "observability_counts", lambda: {})()
        task_status = obs.get("task_status") or {}
        runtime_counts = obs.get("runtime") or {}
        verifier_flags = obs.get("verification") or {}

        for status, count in task_status.items():
            if "task_status" in self._prom:
                self._prom["task_status"].labels(status=status).set(count)
        blocked = task_status.get("blocked", 0)
        failed = task_status.get("failed", 0)
        if blocked >= 3:
            self._add_suggestion(Suggestion(
                title=f"DAG 被阻塞任务 {blocked} 个",
                detail="依赖失败导致下游任务被 BLOCKED（失败传播正常工作）",
                action="1. 检查失败的上游任务日志\n2. 确认工具/检索链路可用性",
                priority=7,
            ))
        if failed >= 3:
            self._add_suggestion(Suggestion(
                title=f"任务执行失败 {failed} 个",
                detail="任务 DAG 中出现失败节点，检查模型/工具调用",
                action="1. 查看 Trace 定位失败任务\n2. 检查对应工具成功率与熔断状态",
                priority=8,
            ))
        for flag, count in verifier_flags.items():
            if count >= 5:
                self._add_suggestion(Suggestion(
                    title=f"出口校验标记 {flag} × {count}",
                    detail="Verifier 反复标记同一风险（只标注不阻断）",
                    action="1. 检查对应工具证据/引用链路\n2. 确认是否需要补充检索",
                    priority=6,
                ))

        self._generate_profile_suggestions(agent_stats)

    @staticmethod
    def _fast_health(agent_stats: Dict[str, Any]) -> bool:
        """
        Fast profile 健康判定（有限反馈）：按 profile 字段识别 Fast。

        每 Profile 单执行实例（key 为 fast / deep）；样本不足（total < 10）
        视为健康（不干预）；成功率 < 0.85 视为不健康（临时升级 Deep）。
        只做这一条规则，不引入复杂在线学习。
        """
        from agents.profiles import ProfileName
        fast_keys = [
            k for k, s in agent_stats.items()
            if s.get("profile") == ProfileName.FAST.value
        ]
        if not fast_keys:
            return True
        for key in fast_keys:
            s = agent_stats[key]
            if s["total"] >= 10 and s["success_rate"] < 0.85:
                logger.warning(f"Fast profile 不健康（{key} 成功率 {s['success_rate']:.1%}），临时升级 Deep")
                return False
        return True

    def _check_threshold(self, metric: str, value: float, label: str) -> None:
        if metric not in self.THRESHOLDS:
            return
        threshold, severity, operator = self.THRESHOLDS[metric]
        triggered = (operator == "less_than" and value < threshold) or \
                    (operator == "greater_than" and value > threshold)
        key = f"{metric}:{label}"
        # 已存在未解决的同指标告警：异常持续中不重复告警（去重），恢复时标记 resolved
        active = next((a for a in self._alerts if a.metric == key and not a.resolved), None)
        if not triggered:
            if active is not None:
                active.resolved = True
                logger.info(f"告警恢复: {key} = {value:.3f}")
            return
        if active is not None:
            return
        alert = Alert(
            severity=severity,
            metric=key,
            message=f"{label} 的 {metric} = {value:.3f}，阈值 {threshold}",
            value=value,
            threshold=threshold,
        )
        self._alerts.append(alert)
        logger.warning(f"[{severity.value.upper()}] {alert.message}")
        # 异步发送 Webhook（不阻塞采集循环）
        if self._webhook:
            asyncio.create_task(self._send_webhook(alert))

    def _generate_profile_suggestions(self, agent_stats: Dict[str, Any]) -> None:
        """
        基于 Profile 在线表现生成优化建议。
        这是 Monitor → Orchestrator 有限反馈（Fast 不健康 → 升级 Deep）的补充。
        """
        for profile_key, s in agent_stats.items():
            if s["success_rate"] < 0.85 and s["total"] > 10:
                self._add_suggestion(Suggestion(
                    title=f"Profile {profile_key} 成功率偏低",
                    detail=f"成功率 {s['success_rate']:.1%}，P95 延迟 {s.get('p95_ms', 0)}ms",
                    action=(
                        "Monitor 已把本应走 Fast 的请求临时升级 Deep。\n"
                        "建议：1. 检查该 Profile 的模型/端点配置与限流状态\n"
                        "      2. 查看错误日志与工具成功率\n"
                        "      3. 确认是否需要调整模型或端点（异构路由）"
                    ),
                    priority=8,
                ))

    def _add_suggestion(self, s: Suggestion) -> None:
        # 去重：相同 title 不重复添加
        if not any(x.title == s.title for x in self._suggestions):
            self._suggestions.append(s)
            logger.info(f"优化建议 [P{s.priority}]: {s.title}")

    async def _send_webhook(self, alert: Alert) -> None:
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                await c.post(self._webhook, json=asdict(alert))  # type: ignore
        except Exception as ex:
            logger.error(f"Webhook 发送失败: {ex}")

    # ── 查询接口 ──────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """返回当前监控摘要，供 API 层暴露。"""
        obs = getattr(self._orchestrator, "observability_counts", lambda: {})()
        return {
            "agent_stats":   self._orchestrator.get_stats(),
            "tool_stats":    self._tool_manager.get_stats(),
            "task_status":   obs.get("task_status") or {},
            "runtime_counts": obs.get("runtime") or {},
            "verifier_flags": obs.get("verification") or {},
            "active_alerts": [asdict(a) for a in self._alerts if not a.resolved][-10:],
            "suggestions":   [
                {"title": s.title, "action": s.action, "priority": s.priority}
                for s in sorted(self._suggestions, key=lambda x: -x.priority)[:5]
            ],
        }
