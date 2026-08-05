"""
个人数据中心 —— SQLite 存储层。

承载用户个人数据，全部按 user_id 隔离：
  - schedule 表：课程表（day_of_week 0=周一；weeks 为逗号分隔的教学周列表）
  - todos 表：待办 / DDL / 考试（kind 区分），due_at 为 "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM"

实现说明：
  - stdlib sqlite3，零第三方依赖；数据库路径由 ECHOGUIDE_DB_PATH 配置，
    默认 <项目根>/data/echoguide.db（Docker 部署时挂载持久化）。
  - 所有方法 async，内部经 asyncio.to_thread 执行同步实现，避免阻塞事件循环。
  - 每次操作新建连接（本地文件开销可忽略），天然规避线程共享连接问题。
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import sqlite3
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedule (
    user_id     TEXT NOT NULL,
    course      TEXT NOT NULL,
    day_of_week INTEGER NOT NULL,          -- 0=周一 … 6=周日
    start_time  TEXT NOT NULL,             -- "08:30"
    end_time    TEXT NOT NULL,             -- "10:05"
    location    TEXT NOT NULL DEFAULT '',
    weeks       TEXT NOT NULL DEFAULT '',  -- 教学周列表 "1,3,5-8"（空=所有周）
    note        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_schedule_user ON schedule(user_id);

CREATE TABLE IF NOT EXISTS todos (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      TEXT NOT NULL,
    content      TEXT NOT NULL,
    kind         TEXT NOT NULL DEFAULT 'todo',  -- todo / ddl / exam
    due_at       TEXT,                          -- "YYYY-MM-DD" 或 "YYYY-MM-DD HH:MM"
    done         INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_todos_user ON todos(user_id);
"""


def _default_db_path() -> str:
    root = pathlib.Path(__file__).parent.parent.resolve()
    return os.getenv("ECHOGUIDE_DB_PATH", str(root / "data" / "echoguide.db"))


class PersonalStore:
    """SQLite 个人数据存储（按 user_id 隔离的课程表 / 待办 / DDL）。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _default_db_path()
        pathlib.Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── 基础 ──────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    async def _run(self, fn, *args, **kwargs):
        """同步实现包装为异步，避免阻塞事件循环。"""
        return await asyncio.to_thread(fn, *args, **kwargs)

    # ── 课程表 ────────────────────────────────────────────────────────────────

    async def replace_schedule(self, user_id: str, courses: List[Dict[str, Any]]) -> int:
        """整表替换：清空该用户旧课表后写入新课表（重导语义）。"""
        return await self._run(self._replace_schedule_sync, user_id, courses)

    def _replace_schedule_sync(self, user_id: str, courses: List[Dict[str, Any]]) -> int:
        with self._connect() as conn:
            conn.execute("DELETE FROM schedule WHERE user_id = ?", (user_id,))
            conn.executemany(
                """INSERT INTO schedule
                   (user_id, course, day_of_week, start_time, end_time, location, weeks)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        user_id,
                        c["course"],
                        int(c["day_of_week"]),
                        c["start_time"],
                        c["end_time"],
                        c.get("location", ""),
                        c.get("weeks", ""),
                    )
                    for c in courses
                ],
            )
        return len(courses)

    async def get_schedule(self, user_id: str) -> List[Dict[str, Any]]:
        """返回该用户全部课程，按（星期, 开始时间）排序。"""
        return await self._run(self._get_schedule_sync, user_id)

    def _get_schedule_sync(self, user_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM schedule WHERE user_id = ? ORDER BY day_of_week, start_time",
                (user_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    async def count_schedule(self, user_id: str) -> int:
        return await self._run(self._count_schedule_sync, user_id)

    def _count_schedule_sync(self, user_id: str) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM schedule WHERE user_id = ?", (user_id,)
            ).fetchone()[0]

    async def clear_schedule(self, user_id: str) -> None:
        await self._run(self._clear_schedule_sync, user_id)

    def _clear_schedule_sync(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM schedule WHERE user_id = ?", (user_id,))

    # ── 待办 / DDL / 考试 ─────────────────────────────────────────────────────

    async def add_todo(
        self,
        user_id: str,
        content: str,
        kind: str = "todo",
        due_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._run(self._add_todo_sync, user_id, content, kind, due_at)

    def _add_todo_sync(
        self, user_id: str, content: str, kind: str, due_at: Optional[str]
    ) -> Dict[str, Any]:
        created = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO todos (user_id, content, kind, due_at, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, content, kind, due_at or None, created),
            )
            return self._todo_row_to_dict(
                conn.execute(
                    "SELECT * FROM todos WHERE id = ?", (cur.lastrowid,)
                ).fetchone()
            )

    async def list_todos(
        self,
        user_id: str,
        status: str = "open",          # open / done / all
        kinds: Optional[List[str]] = None,
        due_before: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        return await self._run(self._list_todos_sync, user_id, status, kinds, due_before)

    def _list_todos_sync(
        self,
        user_id: str,
        status: str,
        kinds: Optional[List[str]],
        due_before: Optional[date],
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM todos WHERE user_id = ?"
        args: List[Any] = [user_id]
        if status == "open":
            sql += " AND done = 0"
        elif status == "done":
            sql += " AND done = 1"
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        if due_before is not None:
            sql += " AND due_at IS NOT NULL AND substr(due_at, 1, 10) <= ?"
            args.append(due_before.isoformat())
        sql += " ORDER BY done ASC, due_at IS NULL, due_at ASC, id DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._todo_row_to_dict(r) for r in rows]

    async def get_todo(self, user_id: str, todo_id: int) -> Optional[Dict[str, Any]]:
        return await self._run(self._get_todo_sync, user_id, todo_id)

    def _get_todo_sync(self, user_id: str, todo_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM todos WHERE id = ? AND user_id = ?", (todo_id, user_id)
            ).fetchone()
        return self._todo_row_to_dict(row) if row else None

    async def set_todo_done(self, user_id: str, todo_id: int, done: bool = True) -> bool:
        """完成/恢复待办，返回是否成功（不存在或不属于该用户时返回 False）。"""
        return await self._run(self._set_todo_done_sync, user_id, todo_id, done)

    def _set_todo_done_sync(self, user_id: str, todo_id: int, done: bool) -> bool:
        completed = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if done else None
        with self._connect() as conn:
            cur = conn.execute(
                """UPDATE todos SET done = ?, completed_at = ?
                   WHERE id = ? AND user_id = ?""",
                (1 if done else 0, completed, todo_id, user_id),
            )
        return cur.rowcount > 0

    async def delete_todo(self, user_id: str, todo_id: int) -> bool:
        return await self._run(self._delete_todo_sync, user_id, todo_id)

    def _delete_todo_sync(self, user_id: str, todo_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM todos WHERE id = ? AND user_id = ?", (todo_id, user_id)
            )
        return cur.rowcount > 0

    @staticmethod
    def _todo_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["done"] = bool(d["done"])
        return d
