"""Small SQLite audit store used by the sidecar control plane."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def _as_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported audit value: {type(value).__name__}")


class AuditStore:
    def __init__(self, path: Path | str):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    ts REAL NOT NULL,
                    phase TEXT NOT NULL,
                    source TEXT NOT NULL,
                    action TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    matched_rules TEXT NOT NULL,
                    details TEXT NOT NULL,
                    duration_ms REAL NOT NULL
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_trace ON audit_events(trace_id, ts)")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scan_reports (
                    scan_id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    report TEXT NOT NULL
                )
                """
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def append(self, event: Any) -> str:
        data = _as_dict(event)
        event_id = str(data.get("event_id") or uuid.uuid4())
        trace_id = str(data.get("trace_id") or "unknown")
        matched = data.get("matched_rules") or data.get("rules") or []
        details = data.get("details") or {}
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO audit_events
                (event_id, trace_id, ts, phase, source, action, decision,
                 risk_score, matched_rules, details, duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    trace_id,
                    float(data.get("timestamp", data.get("created_at", data.get("ts", time.time())))),
                    str(data.get("phase", "runtime")),
                    str(data.get("source", "echoguard")),
                    str(data.get("action", "observe")),
                    str(getattr(data.get("decision"), "value", data.get("decision", "allow"))),
                    float(data.get("risk_score", 0.0)),
                    json.dumps(matched, ensure_ascii=False, sort_keys=True),
                    json.dumps(details, ensure_ascii=False, sort_keys=True),
                    float(data.get("duration_ms", 0.0)),
                ),
            )
        return event_id

    def append_event(self, event: Any) -> str:
        """Backward-compatible, explicit name for callers at the proxy boundary."""
        return self.append(event)

    def list_traces(self, limit: int = 100) -> List[Dict[str, Any]]:
        limit = min(max(int(limit), 1), 1000)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT trace_id, MIN(ts) AS started_at, MAX(ts) AS updated_at,
                       COUNT(*) AS event_count, MAX(risk_score) AS max_risk,
                       SUM(CASE WHEN decision = 'block' THEN 1 ELSE 0 END) AS blocked_events,
                       SUM(CASE WHEN decision = 'ask' THEN 1 ELSE 0 END) AS pending_events
                FROM audit_events GROUP BY trace_id
                ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM audit_events WHERE trace_id = ? ORDER BY ts, event_id",
                (trace_id,),
            ).fetchall()
        if not rows:
            return None
        events = []
        for row in rows:
            item = dict(row)
            item["matched_rules"] = json.loads(item["matched_rules"])
            item["details"] = json.loads(item["details"])
            events.append(item)
        return {"trace_id": trace_id, "events": events}

    def save_scan(self, report: Dict[str, Any]) -> str:
        scan_id = str(uuid.uuid4())
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO scan_reports(scan_id, ts, report) VALUES (?, ?, ?)",
                (scan_id, time.time(), json.dumps(report, ensure_ascii=False, sort_keys=True)),
            )
        return scan_id

    def latest_scan(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT scan_id, ts, report FROM scan_reports ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        report = json.loads(row["report"])
        report["scan_id"] = row["scan_id"]
        report["stored_at"] = row["ts"]
        return report
