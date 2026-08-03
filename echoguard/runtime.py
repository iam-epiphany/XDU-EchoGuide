"""Thread-safe, expiring trace-context storage."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional, Sequence

from .models import Actor, TraceContext


@dataclass
class _Entry:
    context: TraceContext
    expires_at: float


class TraceRegistry:
    """In-memory TTL registry safe for concurrent web/agent worker threads.

    Returned contexts are snapshots.  Mutations must go through the registry so
    a caller cannot race another request or accidentally extend a trace's life.
    """

    def __init__(
        self,
        ttl_seconds: float = 900.0,
        *,
        ttl_s: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_s is not None:
            ttl_seconds = ttl_s
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _actor(value: Actor | Mapping[str, Any] | str | None) -> Actor:
        if value is None:
            return Actor()
        if isinstance(value, Actor):
            return value
        if isinstance(value, str):
            return Actor(role=value)
        if isinstance(value, Mapping):
            scope = value.get("scope") or {}
            if not scope and ("tenant" in value or "tenant_id" in value):
                scope = {"tenant": value.get("tenant", value.get("tenant_id"))}
            return Actor(
                sub=str(value.get("sub", value.get("actor_id", "anonymous"))),
                role=str(value.get("role", "unknown")),
                team=str(value.get("team", "")),
                scope=scope,
                capabilities=tuple(value.get("capabilities") or ()),
                allowed_tools=tuple(value.get("allowed_tools") or ()),
            )
        raise TypeError("actor must be an Actor, mapping, string, or None")

    def _purge_locked(self, now: float) -> int:
        expired = [trace_id for trace_id, entry in self._entries.items() if entry.expires_at <= now]
        for trace_id in expired:
            del self._entries[trace_id]
        return len(expired)

    def upsert(
        self,
        trace_id: str,
        actor: Actor | Mapping[str, Any] | str | None = None,
        prompt: Optional[str] = None,
        selected_skill: Optional[str] = None,
        skill_allowed_tools: Optional[Sequence[str]] = None,
    ) -> TraceContext:
        trace_id = str(trace_id)
        if not trace_id:
            raise ValueError("trace_id must not be empty")
        now = self._clock()
        with self._lock:
            self._purge_locked(now)
            entry = self._entries.get(trace_id)
            if entry is None:
                context = TraceContext(
                    trace_id=trace_id,
                    actor=self._actor(actor),
                    prompt=prompt,
                    selected_skill=selected_skill,
                    skill_allowed_tools=tuple(skill_allowed_tools or ()),
                )
            else:
                context = copy.deepcopy(entry.context)
                if actor is not None:
                    context.actor = self._actor(actor)
                if prompt is not None:
                    context.prompt = str(prompt)
                if selected_skill is not None:
                    context.selected_skill = str(selected_skill)
                if skill_allowed_tools is not None:
                    context.skill_allowed_tools = tuple(str(item) for item in skill_allowed_tools)
                context.updated_at = datetime.now(timezone.utc)
            self._entries[trace_id] = _Entry(context=context, expires_at=now + self.ttl_seconds)
            return copy.deepcopy(context)

    def get(self, trace_id: str) -> Optional[TraceContext]:
        now = self._clock()
        with self._lock:
            self._purge_locked(now)
            entry = self._entries.get(str(trace_id))
            return None if entry is None else copy.deepcopy(entry.context)

    def add_taint(self, trace_id: str, label: str) -> TraceContext:
        now = self._clock()
        with self._lock:
            self._purge_locked(now)
            entry = self._entries.get(str(trace_id))
            if entry is None:
                raise KeyError(f"unknown or expired trace: {trace_id}")
            entry.context.taint_labels.add(str(label))
            entry.context.updated_at = datetime.now(timezone.utc)
            entry.expires_at = now + self.ttl_seconds
            return copy.deepcopy(entry.context)

    def mark_blocked(self, trace_id: str) -> TraceContext:
        now = self._clock()
        with self._lock:
            self._purge_locked(now)
            entry = self._entries.get(str(trace_id))
            if entry is None:
                raise KeyError(f"unknown or expired trace: {trace_id}")
            entry.context.blocked = True
            entry.context.updated_at = datetime.now(timezone.utc)
            entry.expires_at = now + self.ttl_seconds
            return copy.deepcopy(entry.context)

    def remove(self, trace_id: str) -> bool:
        with self._lock:
            return self._entries.pop(str(trace_id), None) is not None

    def purge_expired(self) -> int:
        with self._lock:
            return self._purge_locked(self._clock())

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            self._purge_locked(self._clock())
            return len(self._entries)
