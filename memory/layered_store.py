"""
分层记忆存储层（L0/L1/L3 的数据底座，SQLite 零第三方依赖）。

对应 Hierarchical Long-term Memory with Provenance 思路，落库部分：

  L0 raw_messages     —— 原始对话全量（永不删除，证据链锚点 turn_id）
  L1 facts            —— 原子事实（结构化，source_conv/source_turn 指向 L0）
  L3 profile_history  —— 用户画像版本历史（可回滚，治理用）
  refs                —— 上下文卸载：工具完整结果落盘（上下文只留摘要行）

L2 场景块（Scenario）不在此落库：复用 ChromaDB 情景记忆 collection，
按 metadata layer="scenario" 标记（见 conversation_memory）。
Working Memory（当前会话近期上下文）存 Redis，不计入 L0-L3。

实现说明：
  - stdlib sqlite3，零第三方依赖；路径由 ECHOGUIDE_MEMORY_DB 配置，
    默认 <项目根>/data/memory.db（与 echoguide.db 分离，避免互相干扰）。
  - 所有方法 async，内部经 asyncio.to_thread 执行同步实现，避免阻塞事件循环
    （与 personal/store.py 同模式）。
  - 每次操作新建连接（本地文件开销可忽略），天然规避线程共享连接问题。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 上下文卸载阈值：工具结果超过该字符数时落 refs，上下文只留摘要行
OFFLOAD_CHARS = 1500
# 卸载摘要保留长度（保留"是什么"的概览，细节从 refs 取）
OFFLOAD_SUMMARY_CHARS = 200

_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_messages (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  TEXT NOT NULL,
    conv_id  TEXT NOT NULL,
    role     TEXT NOT NULL,             -- user / assistant / system
    content  TEXT NOT NULL,
    turn_id  INTEGER NOT NULL,          -- 会话内轮次序号（证据链锚点）
    ts       TEXT NOT NULL,
    meta     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_raw_user_conv ON raw_messages(user_id, conv_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_raw_user_ts   ON raw_messages(user_id, ts);

CREATE TABLE IF NOT EXISTS facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    fact         TEXT NOT NULL,         -- 原子事实文本
    category     TEXT NOT NULL DEFAULT 'preference',  -- preference / entity / decision / status
    source_conv  TEXT NOT NULL DEFAULT '',            -- 证据链：来源会话
    source_turn  INTEGER NOT NULL DEFAULT 0,          -- 证据链：来源轮次 → raw_messages.turn_id
    confidence   REAL NOT NULL DEFAULT 1.0,
    ts           TEXT NOT NULL,
    active       INTEGER NOT NULL DEFAULT 1           -- 治理：失效标记而非物理删除
);
CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, category);

CREATE TABLE IF NOT EXISTS profile_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    profile_json TEXT NOT NULL,         -- 整份画像快照（可回滚到任意版本）
    reason       TEXT NOT NULL DEFAULT '',
    ts           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profile_user ON profile_history(user_id);

CREATE TABLE IF NOT EXISTS refs (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  TEXT NOT NULL,
    conv_id  TEXT NOT NULL DEFAULT '',
    tool     TEXT NOT NULL DEFAULT '',
    content  TEXT NOT NULL,             -- 工具完整结果
    char_len INTEGER NOT NULL DEFAULT 0,
    ts       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_refs_user ON refs(user_id);

CREATE TABLE IF NOT EXISTS extract_marks (
    user_id    TEXT NOT NULL,
    conv_id    TEXT NOT NULL,
    last_turn  INTEGER NOT NULL DEFAULT 0,   -- 上次提炼时的最大 turn_id（增量提炼水位）
    ts         TEXT NOT NULL,
    PRIMARY KEY (user_id, conv_id)
);
"""


def _default_db_path() -> str:
    root = pathlib.Path(__file__).parent.parent.resolve()
    return os.getenv("ECHOGUIDE_MEMORY_DB", str(root / "data" / "memory.db"))


def estimate_tokens(text: str) -> int:
    """
    Token 估算（离线口径，供卸载对比与简历数据引用）：
      中文等宽字符按 1 字符 ≈ 1 token；ASCII 按 4 字符 ≈ 1 token。
    不依赖外部 tokenizer，确定性可复现；精确值以模型 API usage 为准。
    """
    if not text:
        return 0
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    wide_chars = len(text) - ascii_chars
    return wide_chars + (ascii_chars + 3) // 4


class LayeredStore:
    """分层记忆存储：L0 原文 / L1 事实 / L3 画像历史 / refs 卸载盘（按 user_id 隔离）。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        pathlib.Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── 基础 ──────────────────────────────────────────────────────────────────

    @contextmanager
    def _connect(self):
        """
        短生命周期连接：退出时自动 commit（异常则 rollback）并关闭。
        与 sqlite3.Connection 原生 context manager 语义一致，且确保 Windows
        下无文件锁残留（防泄漏）。
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    async def _run(self, fn, *args, **kwargs):
        """同步实现包装为异步，避免阻塞事件循环。"""
        return await asyncio.to_thread(fn, *args, **kwargs)

    # ── L0 原始对话 ──────────────────────────────────────────────────────────

    async def append_raw(
        self,
        user_id: str,
        conv_id: str,
        role: str,
        content: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        写入一条原始消息，返回 raw_messages.id。
        turn_id 在事务内自增（会话内轮次序号），保证同一会话内单调且无并发冲突。
        """
        return await self._run(
            self._append_raw_sync, user_id, conv_id, role, content, meta or {}
        )

    def _append_raw_sync(
        self,
        user_id: str,
        conv_id: str,
        role: str,
        content: str,
        meta: Dict[str, Any],
    ) -> int:
        now = datetime.now().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """SELECT COALESCE(MAX(turn_id), 0) + 1
                   FROM raw_messages WHERE user_id = ? AND conv_id = ?""",
                (user_id, conv_id),
            )
            turn_id = cur.fetchone()[0]
            cur = conn.execute(
                """INSERT INTO raw_messages (user_id, conv_id, role, content, turn_id, ts, meta)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, conv_id, role, content, turn_id, now, json.dumps(meta)),
            )
            return cur.lastrowid

    async def get_raw_range(
        self, user_id: str, conv_id: str, start_turn: int = 0, limit: int = 500
    ) -> List[Dict[str, Any]]:
        """按轮次区间取 L0 原文（证据链下钻的读取端）。"""
        return await self._run(
            self._get_raw_range_sync, user_id, conv_id, start_turn, limit
        )

    def _get_raw_range_sync(
        self, user_id: str, conv_id: str, start_turn: int, limit: int
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM raw_messages
                   WHERE user_id = ? AND conv_id = ? AND turn_id >= ?
                   ORDER BY turn_id ASC LIMIT ?""",
                (user_id, conv_id, start_turn, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    async def get_raw_by_turns(
        self, user_id: str, conv_id: str, turns: List[int]
    ) -> Dict[int, str]:
        """按 turn_id 批量取原文（L1 事实 → L0 溯源的核心查询）。"""
        return await self._run(self._get_raw_by_turns_sync, user_id, conv_id, turns)

    def _get_raw_by_turns_sync(
        self, user_id: str, conv_id: str, turns: List[int]
    ) -> Dict[int, str]:
        if not turns:
            return {}
        marks = ",".join("?" * len(turns))
        with self._connect() as conn:
            rows = conn.execute(
                f"""SELECT turn_id, content FROM raw_messages
                    WHERE user_id = ? AND conv_id = ? AND turn_id IN ({marks})""",
                [user_id, conv_id, *turns],
            ).fetchall()
        return {r["turn_id"]: r["content"] for r in rows}

    async def get_last_turn(self, user_id: str, conv_id: str) -> int:
        """会话内当前最大 turn_id（提炼事实时作为证据链锚点）。"""
        return await self._run(self._get_last_turn_sync, user_id, conv_id)

    def _get_last_turn_sync(self, user_id: str, conv_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT COALESCE(MAX(turn_id), 0) FROM raw_messages
                   WHERE user_id = ? AND conv_id = ?""",
                (user_id, conv_id),
            ).fetchone()
        return row[0]

    # ── 增量提炼水位（对齐 TencentDB-Agent-Memory：只提炼上次之后的新消息）──

    async def get_extract_mark(self, user_id: str, conv_id: str) -> int:
        """读取会话提炼水位（上次提炼时的最大 turn_id），无记录返回 0（首次全量预热）。"""
        return await self._run(self._get_extract_mark_sync, user_id, conv_id)

    def _get_extract_mark_sync(self, user_id: str, conv_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT last_turn FROM extract_marks
                   WHERE user_id = ? AND conv_id = ?""",
                (user_id, conv_id),
            ).fetchone()
        return row[0] if row else 0

    async def set_extract_mark(self, user_id: str, conv_id: str, turn: int) -> None:
        """推进提炼水位（提炼成功后才调用；失败不推进，下次幂等重试）。"""
        await self._run(self._set_extract_mark_sync, user_id, conv_id, turn)

    def _set_extract_mark_sync(self, user_id: str, conv_id: str, turn: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO extract_marks (user_id, conv_id, last_turn, ts)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, conv_id) DO UPDATE SET
                       last_turn = excluded.last_turn, ts = excluded.ts""",
                (user_id, conv_id, turn, datetime.now().isoformat()),
            )

    async def count_raw(self, user_id: Optional[str] = None) -> int:
        return await self._run(self._count_raw_sync, user_id)

    def _count_raw_sync(self, user_id: Optional[str]) -> int:
        with self._connect() as conn:
            if user_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM raw_messages WHERE user_id = ?", (user_id,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM raw_messages").fetchone()
        return row[0]

    # ── L1 原子事实 ──────────────────────────────────────────────────────────

    async def add_facts(
        self, user_id: str, facts: List[Dict[str, Any]]
    ) -> int:
        """
        批量写入原子事实（新提炼结果）。
        与既有 active 事实按文本去重（LLM 合并后的重复提炼不落库）。
        返回新增条数。
        """
        return await self._run(self._add_facts_sync, user_id, facts)

    def _add_facts_sync(self, user_id: str, facts: List[Dict[str, Any]]) -> int:
        if not facts:
            return 0
        now = datetime.now().isoformat()
        with self._connect() as conn:
            existing = {
                r["fact"] for r in conn.execute(
                    "SELECT fact FROM facts WHERE user_id = ? AND active = 1",
                    (user_id,),
                ).fetchall()
            }
            added = 0
            for f in facts:
                text = str(f.get("fact") or "").strip()
                if not text or text in existing:
                    continue
                conn.execute(
                    """INSERT INTO facts
                       (user_id, fact, category, source_conv, source_turn, confidence, ts)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        user_id,
                        text,
                        str(f.get("category") or "preference"),
                        str(f.get("source_conv") or ""),
                        int(f.get("source_turn") or 0),
                        float(f.get("confidence") or 1.0),
                        now,
                    ),
                )
                existing.add(text)
                added += 1
        return added

    async def list_facts(
        self, user_id: str, category: Optional[str] = None, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """列出用户事实（默认只取 active，按时间倒序）。"""
        return await self._run(self._list_facts_sync, user_id, category, active_only)

    def _list_facts_sync(
        self, user_id: str, category: Optional[str], active_only: bool
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM facts WHERE user_id = ?"
        args: List[Any] = [user_id]
        if category:
            sql += " AND category = ?"
            args.append(category)
        if active_only:
            sql += " AND active = 1"
        sql += " ORDER BY ts DESC, id DESC LIMIT 100"
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [dict(r) for r in rows]

    async def deactivate_fact(self, user_id: str, fact_id: int) -> bool:
        """治理：失效标记（不物理删除，保留审计痕迹）。"""
        return await self._run(self._deactivate_fact_sync, user_id, fact_id)

    def _deactivate_fact_sync(self, user_id: str, fact_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE facts SET active = 0 WHERE id = ? AND user_id = ?",
                (fact_id, user_id),
            )
        return cur.rowcount > 0

    async def count_facts(self, user_id: Optional[str] = None) -> int:
        return await self._run(self._count_facts_sync, user_id)

    def _count_facts_sync(self, user_id: Optional[str]) -> int:
        with self._connect() as conn:
            if user_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE user_id = ? AND active = 1",
                    (user_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE active = 1"
                ).fetchone()
        return row[0]

    # ── L3 画像版本历史（治理：可回滚）──────────────────────────────────────

    async def save_profile_version(
        self, user_id: str, profile_json: str, reason: str = ""
    ) -> int:
        return await self._run(
            self._save_profile_version_sync, user_id, profile_json, reason
        )

    def _save_profile_version_sync(
        self, user_id: str, profile_json: str, reason: str
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO profile_history (user_id, profile_json, reason, ts)
                   VALUES (?, ?, ?, ?)""",
                (user_id, profile_json, reason, datetime.now().isoformat()),
            )
            return cur.lastrowid

    async def list_profile_versions(
        self, user_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """画像版本列表（倒序：v0 最新）。"""
        return await self._run(self._list_profile_versions_sync, user_id, limit)

    def _list_profile_versions_sync(
        self, user_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM profile_history
                   WHERE user_id = ? ORDER BY id DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    async def get_profile_version(
        self, user_id: str, version_id: int
    ) -> Optional[Dict[str, Any]]:
        """回滚读取：取指定版本画像快照。"""
        return await self._run(self._get_profile_version_sync, user_id, version_id)

    def _get_profile_version_sync(
        self, user_id: str, version_id: int
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profile_history WHERE id = ? AND user_id = ?",
                (version_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    async def count_profile_versions(self, user_id: Optional[str] = None) -> int:
        return await self._run(self._count_profile_versions_sync, user_id)

    def _count_profile_versions_sync(self, user_id: Optional[str]) -> int:
        with self._connect() as conn:
            if user_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM profile_history WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM profile_history"
                ).fetchone()
        return row[0]

    # ── 上下文卸载（refs 落盘）───────────────────────────────────────────────

    async def save_ref(
        self, user_id: str, conv_id: str, tool: str, content: str
    ) -> int:
        """工具完整结果落盘，返回 refs.id 供上下文索引引用。"""
        return await self._run(
            self._save_ref_sync, user_id, conv_id, tool, content
        )

    def _save_ref_sync(
        self, user_id: str, conv_id: str, tool: str, content: str
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO refs (user_id, conv_id, tool, content, char_len, ts)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, conv_id, tool, content, len(content),
                 datetime.now().isoformat()),
            )
            return cur.lastrowid

    async def get_ref(self, user_id: str, ref_id: int) -> Optional[Dict[str, Any]]:
        """按 id 取卸载的完整结果（100% 找回的读取端）。"""
        return await self._run(self._get_ref_sync, user_id, ref_id)

    def _get_ref_sync(
        self, user_id: str, ref_id: int
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM refs WHERE id = ? AND user_id = ?",
                (ref_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    async def count_refs(self, user_id: Optional[str] = None) -> int:
        return await self._run(self._count_refs_sync, user_id)

    def _count_refs_sync(self, user_id: Optional[str]) -> int:
        with self._connect() as conn:
            if user_id:
                row = conn.execute(
                    "SELECT COUNT(*) FROM refs WHERE user_id = ?", (user_id,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM refs").fetchone()
        return row[0]

    # ── 治理：生命周期清理（prune）──────────────────────────────────────────

    async def prune(
        self,
        user_id: Optional[str] = None,
        raw_ttl_days: int = 60,
        ref_ttl_days: int = 7,
        fact_ttl_days: int = 30,
        max_profile_versions: int = 20,
    ) -> Dict[str, int]:
        """
        生命周期治理（简单版，无定时任务：启动/空闲时调用一次即可）：
          - L0 原文：超过 raw_ttl_days 的旧轮次删除（原文虽珍贵，也要控制存储增长）
          - refs：超过 ref_ttl_days 的卸载盘删除
          - facts：active=0 且超过 fact_ttl_days 的失效事实物理清理
          - profile_history：每人只保留最近 max_profile_versions 版
        返回各表清理条数统计。
        """
        return await self._run(
            self._prune_sync, user_id, raw_ttl_days, ref_ttl_days,
            fact_ttl_days, max_profile_versions,
        )

    def _prune_sync(
        self,
        user_id: Optional[str],
        raw_ttl_days: int,
        ref_ttl_days: int,
        fact_ttl_days: int,
        max_profile_versions: int,
    ) -> Dict[str, int]:
        stats = {"raw": 0, "refs": 0, "facts": 0, "profiles": 0}
        raw_cut = (datetime.now() - timedelta(days=raw_ttl_days)).isoformat()
        ref_cut = (datetime.now() - timedelta(days=ref_ttl_days)).isoformat()
        fact_cut = (datetime.now() - timedelta(days=fact_ttl_days)).isoformat()
        with self._connect() as conn:
            base = "WHERE user_id = ?" if user_id else ""

            cur = conn.execute(
                f"DELETE FROM raw_messages {base} AND ts < ?" if base
                else "DELETE FROM raw_messages WHERE ts < ?",
                ([user_id, raw_cut] if base else [raw_cut]),
            )
            stats["raw"] = cur.rowcount

            cur = conn.execute(
                f"DELETE FROM refs {base} AND ts < ?" if base
                else "DELETE FROM refs WHERE ts < ?",
                ([user_id, ref_cut] if base else [ref_cut]),
            )
            stats["refs"] = cur.rowcount

            cur = conn.execute(
                f"DELETE FROM facts {base} AND active = 0 AND ts < ?" if base
                else "DELETE FROM facts WHERE active = 0 AND ts < ?",
                ([user_id, fact_cut] if base else [fact_cut]),
            )
            stats["facts"] = cur.rowcount

            # profile_history：每人保留最近 max_profile_versions 版
            rows = conn.execute(
                "SELECT user_id FROM profile_history" + (f" WHERE user_id = ?" if user_id else "")
                + " GROUP BY user_id",
                ([user_id] if user_id else []),
            ).fetchall()
            for row in rows:
                uid = row["user_id"]
                keep = conn.execute(
                    """SELECT id FROM profile_history WHERE user_id = ?
                       ORDER BY id DESC LIMIT ?""",
                    (uid, max_profile_versions),
                ).fetchall()
                keep_ids = [k["id"] for k in keep]
                if keep_ids:
                    marks = ",".join("?" * len(keep_ids))
                    cur = conn.execute(
                        f"""DELETE FROM profile_history
                            WHERE user_id = ? AND id NOT IN ({marks})""",
                        [uid, *keep_ids],
                    )
                    stats["profiles"] += cur.rowcount
        return stats
