"""
模块: memory_system_py.system
职责: 记忆系统统一门面，编排提取/存储/检索/压缩/同步与事件记录。
输入: 会话轮次、手工记忆字段、查询字符串、压缩上限等。
输出: MemoryRecord 列表、单条 MemoryRecord、同步批次 dict，或 OperationResult。
副作用: 通过 MemoryStore 读写 SQLite（root_dir/memory.db）。
失败处理: 提取为空时不写库；各子模块失败策略见各模块 docstring。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .compact import MemoryCompactor
from .events import EventSink, NullEventSink, OperationEvent, OperationResult
from .extractor import MemoryExtractor
from .models import ConversationTurn, MemoryRecord
from .retriever import MemoryRetriever
from .storage import MemoryStore
from .sync import TeamMemorySync


class MemorySystem:
    """
    注释：外部唯一推荐入口类；组合各子模块，隐藏内部实现细节。
    使用: MemorySystem(root_dir) 或 create_memory_system(root_dir)。
    """

    def __init__(self, root_dir: str = "./runtime_memory", event_sink: EventSink | None = None) -> None:
        """注释：初始化存储目录、各子模块单例与可选事件 sink。"""
        self.store = MemoryStore(Path(root_dir))
        self.extractor = MemoryExtractor()
        self.retriever = MemoryRetriever()
        self.syncer = TeamMemorySync()
        self.compactor = MemoryCompactor()
        self.event_sink = event_sink or NullEventSink()

    def _result(self, name: str, status: str, value, count: int = 0, message: str = "", **metadata) -> OperationResult:
        """注释：统一创建 OperationResult 并发射事件，避免各方法重复拼装。"""
        event = OperationEvent(
            name=name,
            status=status,
            count=count,
            message=message,
            metadata=metadata,
        )
        self.event_sink.emit(event)
        return OperationResult(ok=status != "error", value=value, event=event)

    def ingest_session(self, turns: List[ConversationTurn]) -> List[MemoryRecord]:
        """
        注释：从会话提取长期记忆并批量入库。
        输入: turns。
        输出: 本次新提取的 MemoryRecord 列表（可能为空）。
        副作用: 有提取结果时调用 store.add_many。
        """
        extracted = self.extractor.extract(turns, source="session_ingest")
        if extracted:
            self.store.add_many(extracted)
        return extracted

    def ingest_session_with_result(self, turns: List[ConversationTurn]) -> OperationResult[List[MemoryRecord]]:
        """
        注释：结构化版本的 ingest_session，返回提取结果与审计事件。
        输入: turns。
        输出: OperationResult[List[MemoryRecord]]。
        副作用: 有提取结果时写入 SQLite，并发射 ingest_session 事件。
        """
        extracted = self.ingest_session(turns)
        status = "success" if extracted else "empty"
        return self._result(
            "ingest_session",
            status,
            extracted,
            count=len(extracted),
            message="extracted memories from conversation turns",
            turn_count=len(turns),
        )

    def remember(
        self,
        category: str,
        content: str,
        source: str = "manual",
        tags: List[str] | None = None,
        record_id: str | None = None,
        metadata: Dict[str, str] | None = None,
        score: float = 0.5,
    ) -> MemoryRecord:
        """
        注释：手工或桥接写入单条记忆。
        输入: category, content, source, tags；可选 record_id/metadata/score。
        输出: 已持久化的 MemoryRecord。
        副作用: store.upsert。
        """
        record = MemoryRecord.create(
            category=category,
            content=content,
            source=source,
            tags=tags,
            record_id=record_id,
            metadata=metadata,
            score=score,
        )
        self.store.upsert(record)
        return record

    def remember_with_result(
        self,
        category: str,
        content: str,
        source: str = "manual",
        tags: List[str] | None = None,
        record_id: str | None = None,
        metadata: Dict[str, str] | None = None,
        score: float = 0.5,
    ) -> OperationResult[MemoryRecord]:
        """
        注释：结构化版本的 remember，返回写入记录与审计事件。
        输入: category, content, source, tags；可选 record_id/metadata/score。
        输出: OperationResult[MemoryRecord]。
        副作用: 写入 SQLite，并发射 remember 事件。
        """
        record = self.remember(
            category=category,
            content=content,
            source=source,
            tags=tags,
            record_id=record_id,
            metadata=metadata,
            score=score,
        )
        return self._result(
            "remember",
            "success",
            record,
            count=1,
            message="stored one memory record",
            category=category,
            source=source,
            stable_id=record_id is not None,
        )

    def recall(self, query: str, top_k: int = 5) -> List[MemoryRecord]:
        """
        注释：在当前全库上检索相关记忆。
        输入: query, top_k。
        输出: 相关度最高的前 top_k 条。
        """
        return self.retriever.search(self.store.all(), query, top_k=top_k)

    def recall_with_result(self, query: str, top_k: int = 5) -> OperationResult[List[MemoryRecord]]:
        """
        注释：结构化版本的 recall，返回命中列表与查询事件。
        输入: query, top_k。
        输出: OperationResult[List[MemoryRecord]]。
        副作用: 发射 recall 事件；不写库。
        """
        records = self.recall(query, top_k=top_k)
        status = "success" if records else "empty"
        return self._result(
            "recall",
            status,
            records,
            count=len(records),
            message="retrieved memories for query",
            query=query,
            top_k=top_k,
        )

    def compact(self, max_items: int = 200) -> int:
        """
        注释：压缩记忆库并写回磁盘。
        输入: max_items 保留上限。
        输出: 压缩后条目总数。
        副作用: store.replace_all。
        """
        compacted = self.compactor.compact(self.store.all(), max_items=max_items)
        self.store.replace_all(compacted)
        return len(compacted)

    def compact_with_result(self, max_items: int = 200) -> OperationResult[int]:
        """
        注释：结构化版本的 compact，返回压缩后数量与压缩事件。
        输入: max_items。
        输出: OperationResult[int]。
        副作用: 整库 replace_all，并发射 compact 事件。
        """
        before = len(self.store.all())
        total = self.compact(max_items=max_items)
        return self._result(
            "compact",
            "success",
            total,
            count=total,
            message="compacted memory store",
            before=before,
            after=total,
            max_items=max_items,
        )

    def export_sync_batch(self) -> dict:
        """
        注释：导出团队同步批次，含审计用 blocked 列表。
        输入: 无（使用当前 store.all()）。
        输出: {"allowed": [...], "blocked": [...]}。
        副作用: 无。
        """
        allowed, blocked = self.syncer.prepare_sync_batch(self.store.all())
        return {
            "allowed": allowed,
            "blocked": blocked,
        }

    def export_sync_batch_with_result(self) -> OperationResult[dict]:
        """
        注释：结构化版本的 export_sync_batch，返回批次与同步审计事件。
        输入: 无。
        输出: OperationResult[dict]，value 含 allowed / blocked。
        副作用: 发射 export_sync_batch 事件；不执行网络同步。
        """
        batch = self.export_sync_batch()
        return self._result(
            "export_sync_batch",
            "success",
            batch,
            count=len(batch["allowed"]),
            message="prepared team sync batch",
            allowed_count=len(batch["allowed"]),
            blocked_count=len(batch["blocked"]),
        )
