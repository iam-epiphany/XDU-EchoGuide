"""Content-driven static asset and risk discovery for EchoGuard.

This module deliberately uses only the standard library.  It recognizes common
Agent deployment artifacts without relying on a demo/scenario name, directory
label, or hand-authored risk tag.
"""

from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .source import ScanSource, SourceFile, open_source


RULES: Dict[str, Dict[str, str]] = {
    "EG-JWT-001": {
        "title": "Weak JWT verification or signing secret",
        "severity": "high",
        "category": "authentication",
        "remediation": "Use a random signing secret of at least 32 bytes, reject alg=none, and always verify signatures.",
    },
    "EG-LANGFLOW-001": {
        "title": "Langflow automatic login is enabled",
        "severity": "high",
        "category": "authentication",
        "remediation": "Disable automatic login and require an authenticated identity provider in non-local deployments.",
    },
    "EG-IAM-001": {
        "title": "Service account has all-access privileges",
        "severity": "critical",
        "category": "authorization",
        "remediation": "Replace wildcard or all-access grants with the minimum actions and resources the service account needs.",
    },
    "EG-SKILL-001": {
        "title": "Skill contains prompt-injection instructions",
        "severity": "critical",
        "category": "agent-supply-chain",
        "remediation": "Quarantine the Skill, review its provenance, and remove instructions that override policy or exfiltrate data.",
    },
    "EG-SKILL-002": {
        "title": "Skill script decodes and dynamically executes content",
        "severity": "critical",
        "category": "agent-supply-chain",
        "remediation": "Remove encoded payload execution and require reviewable, signed source for every Skill script.",
    },
    "EG-MCP-001": {
        "title": "MCP tool description contains poisoned instructions",
        "severity": "critical",
        "category": "mcp-tool-poisoning",
        "remediation": "Treat tool metadata as untrusted, remove model-directed hidden instructions, and pin the reviewed server version.",
    },
    "EG-MCP-002": {
        "title": "MCP tool is hidden from normal discovery",
        "severity": "high",
        "category": "mcp-tool-integrity",
        "remediation": "Expose the tool in the reviewed manifest or remove it; deny undeclared tools at the MCP client boundary.",
    },
    "EG-MCP-003": {
        "title": "MCP server executes a subprocess through a shell",
        "severity": "critical",
        "category": "command-execution",
        "remediation": "Use an argument vector with shell disabled and validate every tool argument against a strict schema.",
    },
    "EG-MCP-004": {
        "title": "MCP tool can read an arbitrary caller-supplied path",
        "severity": "high",
        "category": "data-access",
        "remediation": "Resolve paths under an allow-listed root, reject traversal and symlinks, and return only approved file types.",
    },
    "EG-MCP-005": {
        "title": "MCP server starts an outbound connection",
        "severity": "high",
        "category": "network-egress",
        "remediation": "Remove startup egress or restrict destinations with an explicit allow-list and network policy.",
    },
    "EG-TENANT-001": {
        "title": "Cross-tenant wildcard access is enabled",
        "severity": "critical",
        "category": "tenant-isolation",
        "remediation": "Require a concrete tenant identifier derived from the authenticated principal and enforce it in every data query.",
    },
}


_COMPOSE_NAMES = {
    "compose.yml",
    "compose.yaml",
    "docker-compose.yml",
    "docker-compose.yaml",
}
_SCRIPT_SUFFIXES = {".py", ".js", ".cjs", ".mjs", ".ts", ".sh", ".ps1"}
_MCP_MARKERS = re.compile(
    r"(?i)(mcpservers|fastmcp|mcpserver|@(?:mcp|server)\.tool|register_?tool|server\.tool\s*\()"
)
_POISONED_DESCRIPTION = re.compile(
    r"(?is)(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|system|developer)\s+(?:instructions?|rules?|prompts?)|"
    r"disregard\s+(?:the\s+)?(?:system|developer|previous|prior)\s+(?:instructions?|rules?|prompts?)|"
    r"(?:always|must)\s+(?:call|invoke|select|use)\s+(?:this|the)\s+tool|"
    r"(?:reveal|print|return|send|upload|exfiltrat\w*)\s+(?:the\s+)?(?:system\s+prompt|secrets?|credentials?|api\s*keys?|environment\s+variables?)|"
    r"do\s+not\s+(?:tell|mention|show|reveal)\s+(?:the\s+)?user|"
    r"忽略.{0,20}(?:之前|以上|系统).{0,20}(?:指令|提示|规则)|"
    r"(?:泄露|上传|发送).{0,20}(?:密钥|凭据|系统提示|环境变量)"
    r")"
)
_SKILL_INJECTION = re.compile(
    r"(?is)(?:"
    r"ignore\s+(?:all\s+)?(?:previous|prior|above|system|developer)\s+(?:instructions?|rules?|prompts?)|"
    r"disregard\s+(?:the\s+)?(?:system|developer|previous|prior)\s+(?:instructions?|rules?|prompts?)|"
    r"override\s+(?:the\s+)?(?:system|developer|safety)\s+(?:instructions?|rules?|policy)|"
    r"(?:reveal|print|return|send|upload|exfiltrat\w*)\s+(?:all\s+)?(?:secrets?|credentials?|api\s*keys?|system\s+prompt|environment\s+variables?)|"
    r"忽略.{0,20}(?:之前|以上|系统).{0,20}(?:指令|提示|规则)|"
    r"(?:绕过|覆盖).{0,20}(?:安全|系统).{0,20}(?:规则|策略|指令)|"
    r"(?:泄露|上传|发送).{0,20}(?:密钥|凭据|系统提示|环境变量)"
    r")"
)


class StaticScanner:
    """Scan an Agent project directory or ZIP and return JSON-ready dictionaries."""

    def __init__(self, source_path: Union[str, ScanSource], **source_limits: int) -> None:
        self.source = open_source(source_path, **source_limits)
        self._assets: Dict[str, Dict[str, Any]] = {}
        self._edges: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._findings: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}
        self._path_assets: Dict[str, List[str]] = {}

    def scan(self) -> Dict[str, Any]:
        """Run a fresh scan; calling the same scanner twice is deterministic."""

        self._assets.clear()
        self._edges.clear()
        self._findings.clear()
        self._path_assets.clear()

        files = [item for item in self.source.iter_files() if self._looks_textual(item)]
        texts = {item.path: item.read_text() for item in files}

        for item in files:
            if PurePosixPath(item.path).name.lower() in _COMPOSE_NAMES:
                self._scan_compose(item.path, texts[item.path])

        skill_roots = self._scan_skills(files, texts)
        for item in files:
            self._scan_skill_script(item.path, texts[item.path], skill_roots)

        for item in files:
            text = texts[item.path]
            if self._is_mcp_candidate(item.path, text):
                self._scan_mcp(item.path, text)

        for item in files:
            self._scan_configuration_risks(item.path, texts[item.path])

        assets = sorted(self._assets.values(), key=lambda a: (a["type"], a["path"], a["name"]))
        edges = sorted(self._edges.values(), key=lambda e: (e["type"], e["source"], e["target"]))
        findings = sorted(
            self._findings.values(),
            key=lambda f: (f["path"], f["line"], f["rule_id"], f["id"]),
        )
        severity_counts = Counter(item["severity"] for item in findings)
        rule_counts = Counter(item["rule_id"] for item in findings)
        asset_counts = Counter(item["type"] for item in assets)

        source = self.source.describe()
        source["file_count"] = len(files)
        return {
            "source": source,
            "scanned_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "assets": assets,
            "edges": edges,
            "findings": findings,
            "summary": {
                "asset_count": len(assets),
                "edge_count": len(edges),
                "finding_count": len(findings),
                "total_assets": len(assets),
                "total_edges": len(edges),
                "total_findings": len(findings),
                "assets_by_type": dict(sorted(asset_counts.items())),
                "findings_by_severity": dict(sorted(severity_counts.items())),
                "findings_by_rule": dict(sorted(rule_counts.items())),
            },
        }

    @staticmethod
    def _looks_textual(item: SourceFile) -> bool:
        sample = item.content[:4096]
        if not sample:
            return True
        if b"\x00" in sample and not sample.startswith((b"\xff\xfe", b"\xfe\xff")):
            return False
        control = sum(byte < 9 or 13 < byte < 32 for byte in sample)
        return control / len(sample) < 0.08

    @staticmethod
    def _stable_id(kind: str, path: str, name: str) -> str:
        digest = hashlib.sha256(f"{kind}\0{path}\0{name}".encode("utf-8")).hexdigest()[:16]
        return f"{kind}:{digest}"

    def _add_asset(
        self,
        kind: str,
        name: str,
        path: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> str:
        asset_id = self._stable_id(kind, path, name)
        clean_metadata = self._json_safe(dict(metadata or {}))
        if asset_id not in self._assets:
            self._assets[asset_id] = {
                "id": asset_id,
                "type": kind,
                "name": str(name),
                "path": path,
                "metadata": clean_metadata,
            }
            self._path_assets.setdefault(path, []).append(asset_id)
        elif clean_metadata:
            self._assets[asset_id]["metadata"].update(clean_metadata)
        return asset_id

    def _add_edge(
        self,
        source: str,
        target: str,
        kind: str,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        key = (source, target, kind)
        self._edges[key] = {
            "source": source,
            "target": target,
            "type": kind,
            "metadata": self._json_safe(dict(metadata or {})),
        }

    def _add_finding(
        self,
        rule_id: str,
        path: str,
        line: int,
        evidence: str,
        *,
        asset_id: Optional[str] = None,
        detail: str = "",
    ) -> None:
        rule = RULES[rule_id]
        evidence = self._redact_evidence(evidence.strip())[:280]
        key = (rule_id, path, max(1, line), asset_id or "")
        digest = hashlib.sha256("\0".join(map(str, key)).encode("utf-8")).hexdigest()[:16]
        self._findings[key] = {
            "id": f"finding:{digest}",
            "rule_id": rule_id,
            "title": rule["title"],
            "severity": rule["severity"],
            "category": rule["category"],
            "asset_id": asset_id,
            "path": path,
            "line": max(1, int(line)),
            "evidence": evidence,
            "detail": detail,
            "remediation": rule["remediation"],
        }

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(k): StaticScanner._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [StaticScanner._json_safe(item) for item in value]
        return str(value)

    @staticmethod
    def _line_number(text: str, position: int) -> int:
        return text.count("\n", 0, max(0, position)) + 1

    @staticmethod
    def _line_text(text: str, line: int) -> str:
        lines = text.splitlines()
        return lines[line - 1].strip() if 0 < line <= len(lines) else ""

    @staticmethod
    def _redact_evidence(evidence: str) -> str:
        return re.sub(
            r"(?i)((?:jwt[_-]?)?(?:secret|signing[_-]?key)\s*[\"'\]]*\s*[:=]\s*)"
            r"(?:[\"'][^\"']*[\"']|[^\s,}\]]+)",
            r"\1<redacted>",
            evidence,
        )

    def _asset_for_path(self, path: str, preferred: Sequence[str] = ()) -> Optional[str]:
        candidates = self._path_assets.get(path, [])
        for kind in preferred:
            for asset_id in candidates:
                if self._assets[asset_id]["type"] == kind:
                    return asset_id
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------ Compose

    def _scan_compose(self, path: str, text: str) -> None:
        data = self._load_compose(text)
        services = data.get("services") if isinstance(data, Mapping) else None
        if not isinstance(services, Mapping):
            return
        declared = data.get("networks", {})
        network_names = set(str(name) for name in declared) if isinstance(declared, Mapping) else set()
        service_networks: Dict[str, List[str]] = {}
        service_depends: Dict[str, List[str]] = {}
        service_ids: Dict[str, str] = {}

        for raw_name, raw_config in services.items():
            name = str(raw_name)
            config = raw_config if isinstance(raw_config, Mapping) else {}
            networks = self._names_from_compose_value(config.get("networks"))
            if not networks:
                networks = ["default"]
            service_networks[name] = networks
            network_names.update(networks)
            service_depends[name] = self._names_from_compose_value(config.get("depends_on"))
            metadata: Dict[str, Any] = {}
            for key in ("image", "build", "container_name"):
                if key in config:
                    metadata[key] = config[key]
            if "ports" in config:
                metadata["ports"] = config["ports"]
            service_ids[name] = self._add_asset("compose_service", name, path, metadata)

        network_ids = {
            name: self._add_asset("compose_network", name, path)
            for name in sorted(network_names)
        }
        for service, networks in service_networks.items():
            for network in networks:
                self._add_edge(service_ids[service], network_ids[network], "uses_network")
        for service, dependencies in service_depends.items():
            for dependency in dependencies:
                if dependency in service_ids:
                    self._add_edge(service_ids[service], service_ids[dependency], "depends_on")

    @staticmethod
    def _names_from_compose_value(value: Any) -> List[str]:
        if isinstance(value, Mapping):
            return [str(name) for name in value]
        if isinstance(value, (list, tuple)):
            output = []
            for item in value:
                if isinstance(item, str):
                    output.append(item)
                elif isinstance(item, Mapping):
                    output.extend(str(name) for name in item)
            return output
        if isinstance(value, str):
            return [value]
        return []

    @staticmethod
    def _load_compose(text: str) -> Mapping[str, Any]:
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(text)
            if isinstance(loaded, Mapping):
                return loaded
        except Exception:
            # A malformed Compose document should not abort discovery of the
            # remaining files.  The focused fallback still recovers ordinary
            # service/network declarations.
            pass
        return StaticScanner._minimal_compose(text)

    @staticmethod
    def _minimal_compose(text: str) -> Mapping[str, Any]:
        """Small dependency-free parser for ordinary Compose indentation."""

        result: Dict[str, Dict[str, Any]] = {"services": {}, "networks": {}}
        section: Optional[str] = None
        current_service: Optional[str] = None
        subsection: Optional[str] = None

        for raw_line in text.splitlines():
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            line = raw_line.strip()
            if indent == 0 and re.fullmatch(r"(?:services|networks)\s*:", line):
                section = line.split(":", 1)[0]
                current_service = None
                subsection = None
                continue
            if section == "services" and indent == 2:
                match = re.match(r"([^:#]+)\s*:\s*(?:\{\s*\})?\s*$", line)
                if match:
                    current_service = match.group(1).strip(" \"'")
                    result["services"][current_service] = {}
                    subsection = None
                continue
            if section == "networks" and indent == 2:
                match = re.match(r"([^:#]+)\s*:", line)
                if match:
                    result["networks"][match.group(1).strip(" \"'")] = {}
                continue
            if section != "services" or current_service is None:
                continue
            config = result["services"][current_service]
            if indent == 4:
                pair = re.match(r"([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
                if not pair:
                    continue
                key, value = pair.group(1), pair.group(2).strip()
                subsection = key if not value else None
                if key in {"networks", "depends_on", "ports"}:
                    config[key] = StaticScanner._inline_yaml_list(value) if value else []
                elif value:
                    config[key] = value.strip(" \"'")
                continue
            if indent >= 6 and subsection in {"networks", "depends_on", "ports"}:
                if line.startswith("-"):
                    value = line[1:].strip().strip(" \"'")
                else:
                    value = line.split(":", 1)[0].strip(" \"'")
                if value:
                    config[subsection].append(value)
        return result

    @staticmethod
    def _inline_yaml_list(value: str) -> List[str]:
        value = value.strip()
        if not value:
            return []
        if value.startswith("[") and value.endswith("]"):
            return [item.strip(" \"'") for item in value[1:-1].split(",") if item.strip()]
        return [value.strip(" \"'")]

    # ------------------------------------------------------------------- Skills

    def _scan_skills(
        self,
        files: Iterable[SourceFile],
        texts: Mapping[str, str],
    ) -> Dict[str, str]:
        roots: Dict[str, str] = {}
        for item in files:
            pure = PurePosixPath(item.path)
            if pure.name.lower() != "skill.md":
                continue
            text = texts[item.path]
            name, description = self._skill_metadata(pure, text)
            asset_id = self._add_asset(
                "skill",
                name,
                item.path,
                {"description": description} if description else {},
            )
            roots[pure.parent.as_posix()] = asset_id
            match = self._malicious_skill_match(text)
            if match:
                line = self._line_number(text, match.start())
                self._add_finding(
                    "EG-SKILL-001",
                    item.path,
                    line,
                    self._line_text(text, line),
                    asset_id=asset_id,
                    detail="The instruction attempts to override higher-priority policy or disclose protected context.",
                )
        return roots

    @staticmethod
    def _skill_metadata(path: PurePosixPath, text: str) -> Tuple[str, str]:
        name = path.parent.name or "skill"
        description = ""
        stripped = text.lstrip("\ufeff \t\r\n")
        if stripped.startswith("---"):
            lines = stripped.splitlines()
            for line in lines[1:]:
                if line.strip() == "---":
                    break
                match = re.match(r"\s*(name|description)\s*:\s*(.*?)\s*$", line, re.I)
                if match:
                    value = match.group(2).strip(" \"'")
                    if match.group(1).lower() == "name" and value:
                        name = value
                    elif value:
                        description = value
        if name == (path.parent.name or "skill"):
            heading = re.search(r"(?m)^\s*#\s+(.+?)\s*$", text)
            if heading:
                name = heading.group(1).strip()
        return name, description

    @staticmethod
    def _malicious_skill_match(text: str) -> Optional[re.Match[str]]:
        for match in _SKILL_INJECTION.finditer(text):
            prefix = text[max(0, match.start() - 24):match.start()].lower()
            if re.search(r"(?:do\s+not|never|must\s+not|不要|禁止)\s*$", prefix):
                continue
            return match
        return None

    def _scan_skill_script(self, path: str, text: str, roots: Mapping[str, str]) -> None:
        pure = PurePosixPath(path)
        if pure.suffix.lower() not in _SCRIPT_SUFFIXES:
            return
        owner: Optional[str] = None
        parent = pure.parent
        while parent.parts:
            key = parent.as_posix()
            if key in roots:
                owner = roots[key]
                break
            parent = parent.parent
        if owner is None:
            return
        script_id = self._add_asset("skill_script", pure.name, path, {"language": pure.suffix.lstrip(".")})
        self._add_edge(owner, script_id, "contains")
        base64_match = re.search(
            r"(?i)(?:base64\s*\.\s*b64decode|base64\s+(?:--decode|-d)|frombase64string|atob\s*\(|encodedcommand)",
            text,
        )
        execution_match = re.search(
            r"(?i)(?:\bexec\s*\(|\beval\s*\(|os\s*\.\s*system\s*\(|subprocess\s*\.|child_process\s*\.\s*exec|invoke-expression|\biex\b|bash\s+-c)",
            text,
        )
        if base64_match and execution_match:
            position = min(base64_match.start(), execution_match.start())
            line = self._line_number(text, position)
            self._add_finding(
                "EG-SKILL-002",
                path,
                line,
                self._line_text(text, line),
                asset_id=script_id,
                detail="The same Skill script contains both encoded-payload decoding and dynamic execution primitives.",
            )

    # ---------------------------------------------------------------------- MCP

    @staticmethod
    def _is_mcp_candidate(path: str, text: str) -> bool:
        pure = PurePosixPath(path)
        path_hint = "mcp" in pure.name.lower() or any(part.lower() == "mcp" for part in pure.parts)
        return path_hint or bool(_MCP_MARKERS.search(text))

    def _scan_mcp(self, path: str, text: str) -> None:
        suffix = PurePosixPath(path).suffix.lower()
        if suffix == ".json":
            try:
                data = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                data = None
            if data is not None:
                self._scan_mcp_json(path, text, data)
        elif suffix in {".yml", ".yaml"}:
            try:
                import yaml  # type: ignore

                data = yaml.safe_load(text)
            except Exception:
                data = None
            if data is not None:
                self._scan_mcp_json(path, text, data)
        if suffix == ".py":
            self._scan_mcp_python(path, text)
        elif suffix in {".js", ".cjs", ".mjs", ".ts", ".tsx"}:
            self._scan_mcp_javascript(path, text)
        self._scan_mcp_text_risks(path, text)

    def _scan_mcp_json(self, path: str, text: str, data: Any) -> None:
        found_server = False

        def walk(value: Any, key_hint: str = "") -> None:
            nonlocal found_server
            if isinstance(value, Mapping):
                for key, child in value.items():
                    lowered = str(key).lower().replace("_", "")
                    if lowered == "mcpservers" and isinstance(child, Mapping):
                        for server_name, config in child.items():
                            found_server = True
                            self._register_json_server(path, text, str(server_name), config)
                    else:
                        walk(child, str(key))
            elif isinstance(value, list):
                for child in value:
                    walk(child, key_hint)

        walk(data)
        if not found_server and isinstance(data, Mapping):
            servers = data.get("servers")
            if isinstance(servers, Mapping):
                for server_name, config in servers.items():
                    found_server = True
                    self._register_json_server(path, text, str(server_name), config)
        if not found_server and isinstance(data, Mapping) and "tools" in data:
            server_name = str(data.get("name") or PurePosixPath(path).stem)
            server_id = self._add_asset("mcp_server", server_name, path, {"format": "manifest"})
            self._register_json_tools(server_id, path, text, data.get("tools"))

    def _register_json_server(self, path: str, text: str, name: str, config: Any) -> None:
        mapping = config if isinstance(config, Mapping) else {}
        metadata = {
            key: mapping[key]
            for key in ("command", "transport", "url")
            if key in mapping
        }
        server_id = self._add_asset("mcp_server", name, path, metadata)
        self._register_json_tools(server_id, path, text, mapping.get("tools"))

    def _register_json_tools(
        self,
        server_id: str,
        path: str,
        text: str,
        tools: Any,
    ) -> None:
        entries: List[Tuple[str, Mapping[str, Any]]] = []
        if isinstance(tools, Mapping):
            entries = [
                (str(name), config if isinstance(config, Mapping) else {})
                for name, config in tools.items()
            ]
        elif isinstance(tools, list):
            for item in tools:
                if isinstance(item, Mapping) and item.get("name"):
                    entries.append((str(item["name"]), item))
        for name, config in entries:
            description = str(config.get("description") or "")
            hidden = self._truthy(config.get("hidden")) or str(config.get("visibility", "")).lower() in {
                "hidden",
                "private",
                "internal",
            }
            line = self._find_name_line(text, name)
            tool_id = self._add_asset(
                "mcp_tool",
                name,
                path,
                {"description": description[:500], "hidden": hidden},
            )
            self._add_edge(server_id, tool_id, "exposes")
            self._check_tool_metadata(path, text, line, tool_id, name, description, hidden)

    def _scan_mcp_python(self, path: str, text: str) -> None:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            return
        server_vars: Dict[str, str] = {}
        tool_functions: Dict[str, Tuple[Union[ast.FunctionDef, ast.AsyncFunctionDef], str]] = {}
        handler_tools: Dict[str, str] = {}

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            call_name = self._ast_name(value.func).lower()
            if call_name.split(".")[-1] not in {"fastmcp", "mcpserver", "server"}:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            variable = next((target.id for target in targets if isinstance(target, ast.Name)), "server")
            name = self._ast_string(value.args[0]) if value.args else None
            name = name or self._ast_keyword_string(value, "name") or variable
            server_vars[variable] = self._add_asset("mcp_server", name, path, {"implementation": call_name})

        if not server_vars:
            server_vars["server"] = self._add_asset(
                "mcp_server",
                PurePosixPath(path).stem,
                path,
                {"implementation": "python"},
            )
        default_server = next(iter(server_vars.values()))

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            decorator_call: Optional[ast.Call] = None
            decorator_name = ""
            for decorator in node.decorator_list:
                candidate = decorator if isinstance(decorator, ast.Call) else None
                name = self._ast_name(candidate.func if candidate else decorator)
                if name.lower().split(".")[-1] in {"tool", "register_tool"}:
                    decorator_call = candidate
                    decorator_name = name
                    break
            if not decorator_name:
                continue
            name = self._ast_keyword_string(decorator_call, "name") if decorator_call else None
            name = name or node.name
            description = (
                self._ast_keyword_string(decorator_call, "description") if decorator_call else None
            ) or ast.get_docstring(node) or ""
            hidden = (
                self._ast_keyword_bool(decorator_call, "hidden") if decorator_call else False
            ) or name.startswith("_")
            receiver = decorator_name.split(".")[0] if "." in decorator_name else ""
            server_id = server_vars.get(receiver, default_server)
            tool_id = self._add_asset(
                "mcp_tool",
                name,
                path,
                {"description": description[:500], "hidden": hidden, "handler": node.name},
            )
            self._add_edge(server_id, tool_id, "exposes")
            tool_functions[node.name] = (node, tool_id)
            self._check_tool_metadata(path, text, node.lineno, tool_id, name, description, hidden)

        # Dataclass-style Tool(name=..., handler=...) registrations are common.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            call_name = self._ast_name(node.func)
            if call_name.lower().split(".")[-1] != "tool":
                continue
            name = self._ast_keyword_string(node, "name")
            if not name:
                continue
            description = self._ast_keyword_string(node, "description") or ""
            hidden = self._ast_keyword_bool(node, "hidden") or name.startswith("_")
            tool_id = self._add_asset(
                "mcp_tool",
                name,
                path,
                {"description": description[:500], "hidden": hidden},
            )
            self._add_edge(default_server, tool_id, "exposes")
            self._check_tool_metadata(path, text, node.lineno, tool_id, name, description, hidden)
            for keyword in node.keywords:
                if keyword.arg in {"handler", "callback", "func"} and isinstance(keyword.value, ast.Name):
                    handler_tools[keyword.value.id] = tool_id

        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for handler, tool_id in handler_tools.items():
            if handler in functions and handler not in tool_functions:
                tool_functions[handler] = (functions[handler], tool_id)

        for node, tool_id in tool_functions.values():
            if self._function_reads_parameter(node):
                line = self._arbitrary_read_line(node) or node.lineno
                self._add_finding(
                    "EG-MCP-004",
                    path,
                    line,
                    self._line_text(text, line),
                    asset_id=tool_id,
                    detail="A public tool parameter flows directly into a filesystem read API.",
                )

        outbound_line = self._python_startup_egress_line(tree)
        if outbound_line:
            self._add_finding(
                "EG-MCP-005",
                path,
                outbound_line,
                self._line_text(text, outbound_line),
                asset_id=default_server,
                detail="Network I/O occurs at module load or in a startup/lifespan hook.",
            )

    def _scan_mcp_javascript(self, path: str, text: str) -> None:
        server_match = re.search(
            r"(?is)new\s+(?:McpServer|Server)\s*\(\s*\{.{0,300}?name\s*:\s*['\"]([^'\"]+)",
            text,
        )
        name = server_match.group(1) if server_match else PurePosixPath(path).stem
        server_id = self._add_asset("mcp_server", name, path, {"implementation": "javascript"})
        tool_pattern = re.compile(
            r"(?is)(?:registerTool|\.tool)\s*\(\s*['\"](?P<name>[^'\"]+)['\"](?P<body>.{0,700}?)(?=\)\s*[;,])"
        )
        for match in tool_pattern.finditer(text):
            tool_name = match.group("name")
            body = match.group("body")
            desc_match = re.search(r"description\s*:\s*['\"]([^'\"]*)", body, re.I)
            if not desc_match:
                desc_match = re.search(r"^\s*,\s*['\"]([^'\"]*)", body)
            description = desc_match.group(1) if desc_match else ""
            hidden = tool_name.startswith("_") or bool(
                re.search(r"(?:hidden\s*:\s*true|visibility\s*:\s*['\"](?:hidden|private|internal))", body, re.I)
            )
            line = self._line_number(text, match.start())
            tool_id = self._add_asset(
                "mcp_tool",
                tool_name,
                path,
                {"description": description[:500], "hidden": hidden},
            )
            self._add_edge(server_id, tool_id, "exposes")
            self._check_tool_metadata(path, text, line, tool_id, tool_name, description, hidden)
        startup = re.search(
            r"(?is)(?:on\s*\(\s*['\"](?:start|startup)['\"]|onStart|startup|initialize)"
            r".{0,1200}?(?:fetch\s*\(|https?\.request\s*\(|axios\s*\.|WebSocket\s*\()",
            text,
        )
        if startup:
            line = self._line_number(text, startup.start())
            self._add_finding("EG-MCP-005", path, line, self._line_text(text, line), asset_id=server_id)

    def _check_tool_metadata(
        self,
        path: str,
        text: str,
        line: int,
        tool_id: str,
        name: str,
        description: str,
        hidden: bool,
    ) -> None:
        poison = _POISONED_DESCRIPTION.search(description)
        if poison:
            self._add_finding(
                "EG-MCP-001",
                path,
                line,
                self._line_text(text, line) or description,
                asset_id=tool_id,
                detail="The tool description directs model behavior beyond describing the tool capability.",
            )
        if hidden or name.startswith("_"):
            self._add_finding(
                "EG-MCP-002",
                path,
                line,
                self._line_text(text, line) or name,
                asset_id=tool_id,
            )

    def _scan_mcp_text_risks(self, path: str, text: str) -> None:
        shell = re.search(r"(?i)\bshell\s*(?:=|:)\s*(?:True|true|1)\b", text)
        if shell:
            line = self._line_number(text, shell.start())
            self._add_finding(
                "EG-MCP-003",
                path,
                line,
                self._line_text(text, line),
                asset_id=self._asset_for_path(path, ("mcp_tool", "mcp_server")),
            )
        # Fallback for non-Python tool implementations.
        arbitrary_read = re.search(
            r"(?i)(?:readFile(?:Sync)?\s*\(\s*(?:path|file(?:name|_path)?)\b|"
            r"Deno\.readTextFile\s*\(\s*(?:path|file)|"
            r"open\s*\(\s*(?:path|file_path|filename)\b|"
            r"Path\s*\(\s*(?:path|file_path|filename)\s*\)\s*\.\s*read_(?:text|bytes))",
            text,
        )
        if arbitrary_read and not any(
            finding["rule_id"] == "EG-MCP-004" and finding["path"] == path
            for finding in self._findings.values()
        ):
            line = self._line_number(text, arbitrary_read.start())
            self._add_finding(
                "EG-MCP-004",
                path,
                line,
                self._line_text(text, line),
                asset_id=self._asset_for_path(path, ("mcp_tool", "mcp_server")),
            )

    @staticmethod
    def _find_name_line(text: str, name: str) -> int:
        match = re.search(rf"(?m)[\"']{re.escape(name)}[\"']", text)
        return StaticScanner._line_number(text, match.start()) if match else 1

    @staticmethod
    def _truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on", "hidden", "private"}

    @staticmethod
    def _ast_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = StaticScanner._ast_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    @staticmethod
    def _ast_string(node: ast.AST) -> Optional[str]:
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    @staticmethod
    def _ast_keyword_string(call: Optional[ast.Call], name: str) -> Optional[str]:
        if call is None:
            return None
        for keyword in call.keywords:
            if keyword.arg == name:
                return StaticScanner._ast_string(keyword.value)
        return None

    @staticmethod
    def _ast_keyword_bool(call: Optional[ast.Call], name: str) -> bool:
        if call is None:
            return False
        for keyword in call.keywords:
            if keyword.arg == name and isinstance(keyword.value, ast.Constant):
                return bool(keyword.value.value)
        return False

    @staticmethod
    def _function_reads_parameter(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> bool:
        return StaticScanner._arbitrary_read_line(node) is not None

    @staticmethod
    def _arbitrary_read_line(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> Optional[int]:
        parameters = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
        for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
            name = StaticScanner._ast_name(call.func).lower()
            if name in {"open", "io.open", "os.open"} and call.args:
                if isinstance(call.args[0], ast.Name) and call.args[0].id in parameters:
                    return call.lineno
            if name.endswith((".read_text", ".read_bytes")) and isinstance(call.func, ast.Attribute):
                owner = call.func.value
                if isinstance(owner, ast.Call) and owner.args:
                    constructor = StaticScanner._ast_name(owner.func).lower()
                    if constructor.endswith("path") and isinstance(owner.args[0], ast.Name):
                        if owner.args[0].id in parameters:
                            return call.lineno
        return None

    @staticmethod
    def _python_startup_egress_line(tree: ast.Module) -> Optional[int]:
        startup_names = {"startup", "on_startup", "lifespan", "initialize", "init_server", "main"}
        for statement in tree.body:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = " ".join(StaticScanner._ast_name(item.func if isinstance(item, ast.Call) else item) for item in statement.decorator_list).lower()
                if statement.name.lower() not in startup_names and not any(
                    marker in decorators for marker in ("startup", "lifespan", "on_event")
                ):
                    continue
                calls = [item for item in ast.walk(statement) if isinstance(item, ast.Call)]
            elif isinstance(statement, (ast.ClassDef, ast.Import, ast.ImportFrom)):
                continue
            else:
                calls = [item for item in ast.walk(statement) if isinstance(item, ast.Call)]
            for call in calls:
                name = StaticScanner._ast_name(call.func).lower()
                network_suffixes = (
                    "requests.get", "requests.post", "requests.request",
                    "httpx.get", "httpx.post", "httpx.request",
                    "urllib.request.urlopen", "socket.create_connection",
                    "websockets.connect", "aiohttp.clientsession",
                )
                if any(name.endswith(suffix) for suffix in network_suffixes):
                    return call.lineno
                if name.endswith((".get", ".post", ".put", ".patch", ".request", ".send")):
                    literal_args = [
                        item.value
                        for item in call.args
                        if isinstance(item, ast.Constant) and isinstance(item.value, str)
                    ]
                    if any(re.match(r"(?i)(?:https?|wss?)://", item) for item in literal_args):
                        return call.lineno
                if name.endswith(("subprocess.run", "subprocess.popen", "subprocess.call")):
                    rendered = ast.unparse(call) if hasattr(ast, "unparse") else ""
                    if re.search(r"(?i)\b(?:curl|wget)\b", rendered):
                        return call.lineno
        return None

    # ---------------------------------------------------------- generic config

    def _scan_configuration_risks(self, path: str, text: str) -> None:
        self._scan_weak_jwt(path, text)
        self._scan_langflow_auto_login(path, text)
        self._scan_service_account(path, text)
        self._scan_tenant_wildcard(path, text)

    def _scan_weak_jwt(self, path: str, text: str) -> None:
        secret_pattern = re.compile(
            r"(?im)(?<![A-Za-z0-9_])(?P<key>JWT[_-]?(?:SECRET(?:[_-]?KEY)?|SIGNING[_-]?KEY)|TOKEN[_-]?SIGNING[_-]?KEY)"
            r"[\"'\]]*\s*[:=]\s*(?:(?P<quote>[\"'])(?P<quoted>[^\"'\r\n]*)(?P=quote)|(?P<bare>[^\s,#}\]]+))"
        )
        for match in secret_pattern.finditer(text):
            value = (match.group("quoted") if match.group("quote") else match.group("bare")) or ""
            value = value.strip()
            if not value or re.fullmatch(r"\$\{[^}:]+\}", value):
                continue
            default_match = re.search(r":-([^}]+)", value)
            effective = default_match.group(1) if default_match else value
            weak_words = {
                "secret", "changeme", "change-me", "default", "password", "jwtsecret",
                "dev-secret", "test-secret", "insecure", "your-secret-key", "123456",
            }
            if len(effective.encode("utf-8")) < 32 or effective.lower() in weak_words:
                line = self._line_number(text, match.start())
                self._add_finding(
                    "EG-JWT-001",
                    path,
                    line,
                    self._line_text(text, line),
                    asset_id=self._asset_for_path(path, ("compose_service", "mcp_server")),
                    detail="The configured JWT signing material is a short or predictable literal/default.",
                )
        for pattern, detail in (
            (r"(?i)(?:JWT[_-]?ALGORITHM|algorithms?)\s*[\"'\]]*\s*[:=]\s*(?:\[\s*)?[\"']?none\b", "JWT accepts the unsigned 'none' algorithm."),
            (r"(?is)verify[_-]?signature[\"']?\s*[:=]\s*False\b", "JWT signature verification is explicitly disabled."),
        ):
            for match in re.finditer(pattern, text):
                line = self._line_number(text, match.start())
                self._add_finding("EG-JWT-001", path, line, self._line_text(text, line), detail=detail)

    def _scan_langflow_auto_login(self, path: str, text: str) -> None:
        patterns = [
            re.compile(r"(?i)LANGFLOW[_-]AUTO[_-]LOGIN[\"'\]]*\s*[:=]\s*[\"']?(?:true|1|yes|on)\b"),
            re.compile(r"(?i)\blangflow\b[^\r\n]{0,180}\s--auto-login\b"),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                line = self._line_number(text, match.start())
                self._add_finding(
                    "EG-LANGFLOW-001",
                    path,
                    line,
                    self._line_text(text, line),
                    asset_id=self._asset_for_path(path, ("compose_service",)),
                )

    def _scan_service_account(self, path: str, text: str) -> None:
        context_pattern = re.compile(
            r"(?is)(?:service[_-]?account|serviceAccount).{0,260}?"
            r"(?:permissions?|roles?|scopes?|access).{0,80}?"
            r"(?:[\"']?\s*\*\s*[\"']?|all[-_\s]?access|superuser)|"
            r"(?:permissions?|roles?|scopes?|access).{0,80}?"
            r"(?:[\"']?\s*\*\s*[\"']?|all[-_\s]?access|superuser).{0,260}?"
            r"(?:service[_-]?account|serviceAccount)"
        )
        for match in context_pattern.finditer(text):
            line = self._line_number(text, match.start())
            self._add_finding(
                "EG-IAM-001",
                path,
                line,
                self._line_text(text, line),
                asset_id=self._asset_for_path(path, ("compose_service", "mcp_server")),
            )
        # Kubernetes wildcard ClusterRole bound to a ServiceAccount.
        if re.search(r"(?i)kind\s*:\s*ServiceAccount", text) and re.search(
            r"(?is)(?:verbs|resources)\s*:\s*\[?\s*[\"']?\*[\"']?", text
        ):
            match = re.search(r"(?is)(?:verbs|resources)\s*:\s*\[?\s*[\"']?\*[\"']?", text)
            assert match is not None
            line = self._line_number(text, match.start())
            self._add_finding("EG-IAM-001", path, line, self._line_text(text, line))

    def _scan_tenant_wildcard(self, path: str, text: str) -> None:
        pattern = re.compile(
            r"(?im)(?<![A-Za-z0-9_])(?:allowed[_-]?tenant(?:s|[_-]?ids?)?|tenant[_-]?(?:ids?|scope|filter|access))"
            r"[\"'\]]*\s*[:=]\s*(?:\[\s*)?[\"']?\*[\"']?"
        )
        for match in pattern.finditer(text):
            line = self._line_number(text, match.start())
            self._add_finding(
                "EG-TENANT-001",
                path,
                line,
                self._line_text(text, line),
                asset_id=self._asset_for_path(path, ("compose_service", "mcp_server", "mcp_tool")),
            )


def scan_source(source_path: Union[str, ScanSource], **source_limits: int) -> Dict[str, Any]:
    """Convenience API equivalent to ``StaticScanner(path).scan()``."""

    return StaticScanner(source_path, **source_limits).scan()


# Short aliases keep integration code readable and support older prototypes.
scan = scan_source
EchoGuardStaticScanner = StaticScanner


__all__ = [
    "EchoGuardStaticScanner",
    "RULES",
    "StaticScanner",
    "scan",
    "scan_source",
]
