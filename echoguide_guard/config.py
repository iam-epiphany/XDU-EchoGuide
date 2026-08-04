"""Environment-backed configuration for the EchoGuide Guard sidecar."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


DEFAULT_MCP_UPSTREAMS = {
    # 西电校园场景的 MCP 上游（占位地址，可按实际部署覆盖）
    "jwxt": "http://mcp-jwxt:8000/mcp",              # 教务系统
    "student-db": "http://mcp-student-db:8000/mcp",  # 学籍数据库
    "network": "http://mcp-network:8000/mcp",        # 校园网
    "calendar": "http://mcp-calendar:8000/mcp",      # 校历
    "notice": "http://mcp-notice:8000/mcp",          # 通知公告
    "monitoring": "http://mcp-monitoring:8000/mcp",  # 校园系统监控
    "knowledge": "http://mcp-knowledge:8000/mcp",    # 校园知识库
    "shell-runner": "http://mcp-shell-runner:8000/mcp",  # 沙箱执行（保留）
}


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8200
    agent_upstream: str = "http://opspilot-app:8000"
    llm_upstream: str = "http://llm-stub:8000"
    langflow_upstream: str = "http://langflow:7860"
    mcp_upstreams: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MCP_UPSTREAMS))
    target_path: Path = Path("/target")
    audit_db: Path = Path("/data/echoguide_guard.sqlite3")
    jwt_secret: str = "change-me-weak-secret"
    request_timeout_s: float = 15.0
    trace_ttl_s: int = 3600

    @property
    def target_source(self) -> str:
        """Compatibility name used by the ASGI control plane."""
        return str(self.target_path)

    @property
    def database_path(self) -> Path:
        return self.audit_db

    @property
    def jwt_algorithm(self) -> str:
        return "HS256"

    def ensure_runtime_directory(self) -> None:
        if str(self.audit_db) != ":memory:":
            self.audit_db.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Settings":
        import logging

        mcp_upstreams = dict(DEFAULT_MCP_UPSTREAMS)
        raw = os.getenv("ECHOGUIDE_MCP_UPSTREAMS", "").strip()
        if raw:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("ECHOGUIDE_MCP_UPSTREAMS must be a JSON object")
            mcp_upstreams.update({str(k): str(v) for k, v in parsed.items()})

        jwt_secret = os.getenv("ECHOGUIDE_JWT_SECRET", "change-me-weak-secret")
        if jwt_secret == "change-me-weak-secret":
            logging.getLogger(__name__).warning(
                "EchoGuard 使用默认 JWT 密钥，仅限演示环境；生产环境请设置 ECHOGUIDE_JWT_SECRET"
            )

        return cls(
            host=os.getenv("ECHOGUIDE_HOST", "0.0.0.0"),
            port=int(os.getenv("ECHOGUIDE_PORT", "8200")),
            agent_upstream=os.getenv("ECHOGUIDE_AGENT_UPSTREAM", "http://opspilot-app:8000").rstrip("/"),
            llm_upstream=os.getenv("ECHOGUIDE_LLM_UPSTREAM", "http://llm-stub:8000").rstrip("/"),
            langflow_upstream=os.getenv("ECHOGUIDE_LANGFLOW_UPSTREAM", "http://langflow:7860").rstrip("/"),
            mcp_upstreams=mcp_upstreams,
            target_path=Path(os.getenv("ECHOGUIDE_TARGET", "/target")),
            audit_db=Path(os.getenv("ECHOGUIDE_AUDIT_DB", "/data/echoguide_guard.sqlite3")),
            jwt_secret=jwt_secret,
            request_timeout_s=float(os.getenv("ECHOGUIDE_REQUEST_TIMEOUT", "15")),
            trace_ttl_s=int(os.getenv("ECHOGUIDE_TRACE_TTL", "3600")),
        )

    def mcp_upstream(self, server: str) -> str:
        try:
            return self.mcp_upstreams[server].rstrip("/")
        except KeyError as exc:
            raise KeyError(f"unknown MCP server: {server}") from exc
