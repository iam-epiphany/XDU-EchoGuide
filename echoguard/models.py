"""Core value objects shared by the EchoGuard policy boundary.

The models deliberately contain no AgentRange challenge identifiers.  A policy
decision must be explainable from the authenticated actor, selected skill and
tool call itself, rather than from a trace name supplied by a caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _as_tuple(values: Optional[Sequence[str]]) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values)


class PolicyDecision(str, Enum):
    """The three outcomes understood by the tool-call boundary."""

    ALLOW = "allow"
    ASK = "ask"
    BLOCK = "block"


@dataclass(frozen=True)
class Actor:
    """Authenticated workload identity.

    ``scope`` is intentionally a small, generic mapping.  The current policy
    consumes ``scope["tenant"]`` while leaving room for future project or
    environment boundaries without changing this public model.
    """

    sub: str = "anonymous"
    role: str = "unknown"
    team: str = ""
    scope: Mapping[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    allowed_tools: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sub", str(self.sub or "anonymous"))
        object.__setattr__(self, "role", str(self.role or "unknown"))
        object.__setattr__(self, "team", str(self.team or ""))
        object.__setattr__(self, "scope", dict(self.scope or {}))
        object.__setattr__(self, "capabilities", _as_tuple(self.capabilities))
        object.__setattr__(self, "allowed_tools", _as_tuple(self.allowed_tools))

    @property
    def actor_id(self) -> str:
        """Compatibility alias for audit/event consumers."""

        return self.sub

    @property
    def tenant_id(self) -> Optional[str]:
        tenant = self.scope.get("tenant", self.scope.get("tenant_id"))
        return None if tenant is None else str(tenant)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sub": self.sub,
            "role": self.role,
            "team": self.team,
            "scope": dict(self.scope),
            "capabilities": list(self.capabilities),
            "allowed_tools": list(self.allowed_tools),
        }


@dataclass
class TraceContext:
    """Security context propagated through one agent execution trace."""

    trace_id: str
    actor: Actor = field(default_factory=Actor)
    prompt: Optional[str] = None
    selected_skill: Optional[str] = None
    skill_allowed_tools: tuple[str, ...] = field(default_factory=tuple)
    taint_labels: set[str] = field(default_factory=set)
    blocked: bool = False
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.trace_id = str(self.trace_id)
        self.skill_allowed_tools = _as_tuple(self.skill_allowed_tools)
        self.taint_labels = {str(label) for label in self.taint_labels}

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "actor": self.actor.to_dict(),
            "prompt": self.prompt,
            "selected_skill": self.selected_skill,
            "skill_allowed_tools": list(self.skill_allowed_tools),
            "taint_labels": sorted(self.taint_labels),
            "blocked": self.blocked,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(frozen=True)
class PolicyVerdict:
    """Deterministic result returned before a tool is invoked."""

    decision: PolicyDecision
    reason: str
    rule_id: str
    labels: tuple[str, ...] = field(default_factory=tuple)
    redacted_arguments: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PolicyDecision):
            object.__setattr__(self, "decision", PolicyDecision(self.decision))
        object.__setattr__(self, "labels", _as_tuple(self.labels))

    @property
    def allowed(self) -> bool:
        return self.decision is PolicyDecision.ALLOW

    @property
    def requires_approval(self) -> bool:
        return self.decision is PolicyDecision.ASK

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "message": self.reason,
            "rule_id": self.rule_id,
            "reason_codes": [self.rule_id],
            "risk_score": {
                PolicyDecision.ALLOW: 0.0,
                PolicyDecision.ASK: 50.0,
                PolicyDecision.BLOCK: 100.0,
            }[self.decision],
            "labels": list(self.labels),
            "redacted_arguments": self.redacted_arguments,
        }


@dataclass(frozen=True)
class AuditEvent:
    """Serializable audit record for one policy evaluation."""

    trace_id: str
    actor: Actor
    server: str
    tool: str
    decision: PolicyDecision
    reason: str
    rule_id: str
    arguments: Any = None
    labels: tuple[str, ...] = field(default_factory=tuple)
    event_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PolicyDecision):
            object.__setattr__(self, "decision", PolicyDecision(self.decision))
        object.__setattr__(self, "labels", _as_tuple(self.labels))

    @classmethod
    def from_verdict(
        cls,
        *,
        context: TraceContext,
        server: str,
        tool: str,
        verdict: PolicyVerdict,
    ) -> "AuditEvent":
        return cls(
            trace_id=context.trace_id,
            actor=context.actor,
            server=server,
            tool=tool,
            decision=verdict.decision,
            reason=verdict.reason,
            rule_id=verdict.rule_id,
            arguments=verdict.redacted_arguments,
            labels=verdict.labels,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.timestamp(),
            "trace_id": self.trace_id,
            "actor": self.actor.to_dict(),
            "server": self.server,
            "tool": self.tool,
            "decision": self.decision.value,
            "reason": self.reason,
            "rule_id": self.rule_id,
            "labels": list(self.labels),
            "arguments": self.arguments,
        }
