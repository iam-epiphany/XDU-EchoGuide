"""Deterministic least-privilege policy for EchoGuide tool calls."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Iterable, Optional, Sequence

from .models import Actor, PolicyDecision, PolicyVerdict, TraceContext
from .redaction import find_sensitive_labels, redact_data


def _normal(value: Any) -> str:
    return str(value or "").strip().casefold().replace("_", "-")


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _strings(item)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            yield from _strings(item)
    elif value is not None:
        yield str(value)


class PolicyEngine:
    """Evaluate identity, capability and content rules in a stable order."""

    ROLE_TOOL_ALLOWLIST: dict[str, frozenset[str] | None] = {
        # 校园角色：学生可查课表/成绩，辅导员可查学籍，IT 人员可查网络/监控。
        "student": frozenset({"jwxt/query-schedule", "jwxt/query-score"}),
        "counselor": frozenset({"student-db/query", "jwxt/query-score"}),
        "it-staff": frozenset({"network/query-status", "monitoring/query-metric"}),
        # Administrative breadth does not bypass argument/content safeguards.
        "admin": None,
    }

    _TENANT_KEYS = frozenset({"tenant", "tenant-id", "tenantid"})
    _WILDCARD_TENANTS = frozenset({"*", "all", "any", "global"})
    _SENSITIVE_PATH_RE = re.compile(
        r"(?:^|[\\/])(?:\.env(?:\.[^\\/\s]+)?|\.ssh|\.aws|\.kube|credentials?)(?:$|[\\/\s])"
        r"|(?:^|[\\/])etc[\\/](?:shadow|passwd|sudoers)(?:$|[\\/\s])"
        r"|(?:^|[\\/])proc[\\/](?:self|\d+)[\\/]environ(?:$|\s)",
        re.IGNORECASE,
    )
    _REMOTE_PIPE_RE = re.compile(
        r"\b(?:curl|wget)\b[\s\S]{0,500}?\|\s*(?:/usr/bin/|/bin/)?(?:ba|z|k)?sh\b",
        re.IGNORECASE,
    )
    _HIDDEN_DEBUG_TOOL_RE = re.compile(
        r"(?:^|[-_.])(?:hidden[-_.]?)?debug(?:[-_.]?(?:exec|run|shell))?(?:$|[-_.])",
        re.IGNORECASE,
    )
    _INJECTION_RE = re.compile(
        r"(?:ignore|disregard|override)\s+(?:all\s+)?(?:previous|prior|system|developer)\s+instructions?"
        r"|(?:hidden|covert)\s+debug"
        r"|<\s*(?:system|developer|hidden[_-]?debug)\b",
        re.IGNORECASE,
    )
    _EGRESS_HINT_RE = re.compile(
        r"(?:sync|send|post|upload|publish|export|webhook|notify|email|http|curl|wget|external)",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        role_tool_allowlist: Optional[Mapping[str, Optional[Sequence[str]]]] = None,
    ) -> None:
        matrix: dict[str, frozenset[str] | None] = dict(self.ROLE_TOOL_ALLOWLIST)
        if role_tool_allowlist:
            for role, tools in role_tool_allowlist.items():
                matrix[_normal(role)] = None if tools is None else frozenset(_normal(item) for item in tools)
        self._role_tool_allowlist = matrix

    @staticmethod
    def _verdict(
        decision: PolicyDecision,
        rule_id: str,
        reason: str,
        arguments: Any,
        labels: Iterable[str] = (),
    ) -> PolicyVerdict:
        return PolicyVerdict(
            decision=decision,
            rule_id=rule_id,
            reason=reason,
            labels=tuple(sorted(set(labels))),
            redacted_arguments=redact_data(arguments),
        )

    @staticmethod
    def _tool_id(server: str, tool: str) -> str:
        return f"{_normal(server)}/{_normal(tool)}"

    @staticmethod
    def _matches_capability(allowed: Sequence[str], server: str, tool: str) -> bool:
        server_name = _normal(server)
        tool_name = _normal(tool)
        pair = f"{server_name}/{tool_name}"
        choices = {_normal(item).replace(":", "/") for item in allowed}
        return bool({"*", server_name, tool_name, pair} & choices)

    @classmethod
    def _tenant_values(cls, value: Any) -> list[str]:
        found: list[str] = []
        if isinstance(value, Mapping):
            for key, item in value.items():
                if _normal(key) in cls._TENANT_KEYS:
                    if isinstance(item, (list, tuple, set, frozenset)):
                        found.extend(str(element).strip() for element in item)
                    else:
                        found.append(str(item).strip())
                found.extend(cls._tenant_values(item))
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                found.extend(cls._tenant_values(item))
        elif isinstance(value, str):
            for match in re.finditer(
                r"\btenant(?:_id)?\b\s*(?:=|:|\bin\b)\s*[\[(' \"`]*([\w*.-]+)",
                value,
                re.IGNORECASE,
            ):
                found.append(match.group(1))
        return found

    @classmethod
    def _has_tenant_wildcard(cls, arguments: Any, server: str, tool: str) -> bool:
        if any(_normal(value) in cls._WILDCARD_TENANTS for value in cls._tenant_values(arguments)):
            return True
        if _normal(server) == "student-db" and _normal(tool) == "query":
            if isinstance(arguments, Mapping):
                for key in ("query", "filter", "where"):
                    value = arguments.get(key)
                    if isinstance(value, str) and value.strip().casefold() in {"*", "all", "any"}:
                        return True
        return False

    @classmethod
    def _is_egress(cls, server: str, tool: str, arguments: Any) -> bool:
        identity = f"{server}/{tool}"
        if cls._EGRESS_HINT_RE.search(identity):
            return True
        for text in _strings(arguments):
            if re.search(r"\bhttps?://", text, re.IGNORECASE):
                return True
        return False

    def evaluate(
        self,
        context: TraceContext,
        server: str,
        tool: str,
        arguments: Any,
    ) -> PolicyVerdict:
        if not isinstance(context, TraceContext):
            raise TypeError("context must be a TraceContext")
        server = str(server or "")
        tool = str(tool or "")
        arguments = {} if arguments is None else arguments

        if context.blocked:
            return self._verdict(
                PolicyDecision.BLOCK,
                "trace.previously_blocked",
                "This trace was already blocked by an earlier security decision.",
                arguments,
            )

        if context.taint_labels.intersection({"untrusted_instruction", "prompt_injection"}):
            return self._verdict(
                PolicyDecision.BLOCK,
                "trace.untrusted_instruction",
                "An untrusted instruction was observed earlier in this trace.",
                arguments,
                context.taint_labels,
            )

        actor_tenant = context.actor.tenant_id
        tenant_values = self._tenant_values(arguments)
        if actor_tenant and self._has_tenant_wildcard(arguments, server, tool):
            return self._verdict(
                PolicyDecision.BLOCK,
                "tenant.wildcard_scope",
                "A tenant-scoped actor cannot query a wildcard tenant.",
                arguments,
            )
        if actor_tenant and any(
            value and _normal(value) != _normal(actor_tenant) for value in tenant_values
        ):
            return self._verdict(
                PolicyDecision.BLOCK,
                "tenant.cross_boundary",
                "The requested tenant is outside the actor's authenticated scope.",
                arguments,
            )

        combined_text = " ".join(_strings(arguments))
        if self._SENSITIVE_PATH_RE.search(combined_text):
            return self._verdict(
                PolicyDecision.BLOCK,
                "filesystem.sensitive_path",
                "Reading credential, environment, or operating-system identity files is forbidden.",
                arguments,
                ("secret",),
            )
        if self._REMOTE_PIPE_RE.search(combined_text):
            return self._verdict(
                PolicyDecision.BLOCK,
                "command.remote_pipe",
                "Piping remote content directly to a shell is forbidden.",
                arguments,
            )
        if self._HIDDEN_DEBUG_TOOL_RE.search(_normal(tool)) or self._HIDDEN_DEBUG_TOOL_RE.search(_normal(server)):
            return self._verdict(
                PolicyDecision.BLOCK,
                "tool.hidden_debug",
                "Hidden or debug execution tools are not permitted through the agent boundary.",
                arguments,
            )
        if isinstance(arguments, Mapping) and any(
            _normal(key) in {"hidden-debug", "debug-exec", "internal-debug"} and bool(value)
            for key, value in arguments.items()
        ):
            return self._verdict(
                PolicyDecision.BLOCK,
                "tool.hidden_debug",
                "A hidden debug execution flag was requested.",
                arguments,
            )

        sensitive = find_sensitive_labels(arguments) | {
            label for label in context.taint_labels if label in {"secret", "pii", "environment"}
        }
        if sensitive and self._is_egress(server, tool, arguments):
            return self._verdict(
                PolicyDecision.BLOCK,
                "data.sensitive_exfiltration",
                "Secrets, PII, or environment data cannot be sent to an egress-capable tool.",
                arguments,
                sensitive,
            )

        if context.selected_skill and context.skill_allowed_tools:
            if not self._matches_capability(context.skill_allowed_tools, server, tool):
                return self._verdict(
                    PolicyDecision.BLOCK,
                    "skill.tool_not_allowed",
                    f"The selected skill {context.selected_skill!r} does not grant this tool capability.",
                    arguments,
                )

        if context.actor.allowed_tools and not self._matches_capability(
            context.actor.allowed_tools, server, tool
        ):
            return self._verdict(
                PolicyDecision.BLOCK,
                "actor.tool_not_allowed",
                "The authenticated actor does not grant this tool capability.",
                arguments,
            )

        role = _normal(context.actor.role)
        if role not in self._role_tool_allowlist:
            return self._verdict(
                PolicyDecision.BLOCK,
                "role.unmanaged",
                "The authenticated role has no registered tool policy.",
                arguments,
            )
        role_allowlist = self._role_tool_allowlist[role]
        if role_allowlist is not None and self._tool_id(server, tool) not in role_allowlist:
            return self._verdict(
                PolicyDecision.BLOCK,
                "role.tool_not_allowed",
                f"Role {context.actor.role!r} is not permitted to call this tool.",
                arguments,
            )

        return self._verdict(
            PolicyDecision.ALLOW,
            "policy.allow",
            "The tool call satisfies tenant, capability, and content policy.",
            arguments,
        )

    def inspect_text(self, text: str, source: str = "content") -> PolicyVerdict:
        """Inspect untrusted prompts/tool output before adding it to context.

        Sensitive content returns ``ASK`` so the runtime can taint the trace and
        continue internal processing; an actual egress attempt is still blocked
        by :meth:`evaluate`.
        """

        text = str(text or "")
        source = str(source or "content")
        if self._REMOTE_PIPE_RE.search(text):
            return self._verdict(
                PolicyDecision.BLOCK,
                "content.remote_pipe",
                f"{source} contains a remote-content-to-shell command.",
                {"text": text},
            )
        if self._INJECTION_RE.search(text):
            return self._verdict(
                PolicyDecision.BLOCK,
                "content.instruction_override",
                f"{source} contains a hidden instruction-override pattern.",
                {"text": text},
            )
        labels = find_sensitive_labels(text)
        if labels:
            return self._verdict(
                PolicyDecision.ASK,
                "data.sensitive_content",
                f"{source} contains sensitive data; taint the trace before continuing.",
                {"text": text},
                labels,
            )
        return self._verdict(
            PolicyDecision.ALLOW,
            "content.allow",
            f"{source} passed content inspection.",
            {"text": text},
        )
