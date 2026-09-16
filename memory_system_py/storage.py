"""
模块: memory_system_py.storage
职责: 记忆条目的 SQLite 持久化与 CRUD。
输入: MemoryRecord 或 List[MemoryRecord]；root_dir 路径。
输出: 查询返回 List[MemoryRecord]；写操作无返回值。
副作用: 在 root_dir 下创建 memory.db 并读写磁盘。
失败处理: SQLite 异常向上抛出；add_many 对空列表直接返回。
"""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import List

from .models import MemoryRecord, utc_now_iso


class MemoryStore:
    """注释：记忆库唯一持久化入口；压缩流程通过 replace_all 整库替换。"""

    def __init__(self, root_dir: Path) -> None:
        """注释：确保目录存在并初始化表结构。"""
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.db_file = self.root_dir / "memory.db"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """注释：统一连接入口，便于后续切换 WAL/超时等参数。"""
        return sqlite3.connect(self.db_file)

    def _init_db(self) -> None:
        """注释：首次启动自动建表 memories。"""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    score REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _row_to_record(self, row: sqlite3.Row) -> MemoryRecord:
        """注释：将 SQLite 行反序列化为 MemoryRecord。"""
        return MemoryRecord(
            id=row[0],
            category=row[1],
            content=row[2],
            source=row[3],
            tags=json.loads(row[4]),
            score=float(row[5]),
            created_at=row[6],
            updated_at=row[7],
            metadata=json.loads(row[8]),
        )

    def all(self) -> List[MemoryRecord]:
        """注释：按 created_at 升序返回全部记忆。"""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, category, content, source, tags, score, created_at, updated_at, metadata
                FROM memories
                ORDER BY created_at ASC
                """
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def add_many(self, records: List[MemoryRecord]) -> None:
        """注释：批量 INSERT OR REPLACE；空列表无操作。"""
        if not records:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO memories (
                    id, category, content, source, tags, score, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.id,
                        record.category,
                        record.content,
                        record.source,
                        json.dumps(record.tags, ensure_ascii=False),
                        record.score,
                        record.created_at,
                        record.updated_at,
                        json.dumps(record.metadata, ensure_ascii=False),
                    )
                    for record in records
                ],
            )
            conn.commit()

    def upsert(self, record: MemoryRecord) -> None:
        """注释：单条 upsert；已存在时保留 created_at 并刷新 updated_at。"""
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT created_at FROM memories WHERE id = ?",
                (record.id,),
            ).fetchone()
            if exists:
                record.created_at = exists[0]
                record.updated_at = utc_now_iso()
            conn.execute(
                """
                INSERT OR REPLACE INTO memories (
                    id, category, content, source, tags, score, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.category,
                    record.content,
                    record.source,
                    json.dumps(record.tags, ensure_ascii=False),
                    record.score,
                    record.created_at,
                    record.updated_at,
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )
            conn.commit()

    def replace_all(self, records: List[MemoryRecord]) -> None:
        """注释：DELETE 后全量插入，供 compact 等整库重写场景使用。"""
        with self._connect() as conn:
            conn.execute("DELETE FROM memories")
            conn.executemany(
                """
                INSERT INTO memories (
                    id, category, content, source, tags, score, created_at, updated_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.id,
                        record.category,
                        record.content,
                        record.source,
                        json.dumps(record.tags, ensure_ascii=False),
                        record.score,
                        record.created_at,
                        record.updated_at,
                        json.dumps(record.metadata, ensure_ascii=False),
                    )
                    for record in records
                ],
            )
            conn.commit()
