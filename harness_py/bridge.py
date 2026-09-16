"""
模块: harness_py.bridge
职责: 将 TaskCluster 桥接写入 memory_system_py，并支持稳定 ID 与 dry-run 拟写入预览。
输入: List[TaskCluster]，已构造的 MemorySystem 实例。
输出: 写入条数（int）或 MemoryWritePreview 列表。
副作用: 通过 MemorySystem.remember 写入 SQLite；重复 Harness 运行会 upsert 同一稳定 ID。
失败处理: 单簇写入失败会向上抛出（由 MemoryStore 层处理）；空 clusters 返回 0。
"""

from __future__ import annotations

from hashlib import sha256
from typing import Iterable, List

from memory_system_py import MemorySystem

from .types import MemoryPayload, MemoryWritePreview, TaskCluster


class MemorySystemBridge:
    """注释：每个簇写 1 条摘要 + 每文件 1 条 ingest 记忆。"""

    def __init__(self, memory_system: MemorySystem) -> None:
        self.memory_system = memory_system

    def ingest_clusters(self, clusters: List[TaskCluster]) -> int:
        """
        注释：遍历簇与文件，调用 remember 入库。
        输入: clusters。
        输出: 新增记忆条数（摘要 + 文件）。
        """
        count = 0
        for payload in self.iter_memory_payloads(clusters):
            self.memory_system.remember(
                category=payload.category,
                content=payload.content,
                source=payload.source,
                tags=payload.tags,
                record_id=payload.record_id,
                metadata=payload.metadata,
            )
            count += 1
        return count

    @classmethod
    def preview_clusters(cls, clusters: List[TaskCluster], preview_chars: int = 240) -> List[MemoryWritePreview]:
        """
        注释：生成拟写入记忆预览；不创建 MemoryRecord，不写 SQLite。
        输入: clusters, preview_chars。
        输出: MemoryWritePreview 列表。
        """
        previews: List[MemoryWritePreview] = []
        for payload in cls.iter_memory_payloads(clusters):
            previews.append(
                MemoryWritePreview(
                    planned_id=payload.record_id,
                    category=payload.category,
                    source=payload.source,
                    tags=payload.tags,
                    metadata=payload.metadata,
                    content_preview=cls._preview_text(payload.content, preview_chars),
                    content_length=len(payload.content),
                )
            )
        return previews

    @classmethod
    def iter_memory_payloads(cls, clusters: List[TaskCluster]) -> Iterable[MemoryPayload]:
        """
        注释：统一生成写入载荷，供正式 ingest 与 dry-run preview 复用。
        输出: MemoryPayload 迭代器。
        """
        for cluster in clusters:
            cluster_metadata = {
                "task_name": cluster.task_name,
                "file_count": str(len(cluster.files)),
                "content_sha256": cls._hash_text(cluster.description),
            }
            yield MemoryPayload(
                record_id=cls._stable_id("cluster", cluster.task_name),
                category=cluster.task_name,
                content=cluster.description,
                source="harness_cluster_summary",
                tags=["cluster", "summary"],
                metadata=cluster_metadata,
            )

            for snippet in cluster.files:
                payload = cls._build_file_payload(
                    cluster.task_name,
                    snippet.file.relative_path,
                    snippet.content,
                    snippet.truncated,
                )
                file_metadata = {
                    "doc_id": snippet.document_id,
                    "source_kind": snippet.source_kind,
                    "task_name": cluster.task_name,
                    "relative_path": snippet.file.relative_path,
                    "extension": snippet.file.extension,
                    "truncated": str(snippet.truncated).lower(),
                    "summary_hint": snippet.summary_hint,
                    "content_sha256": cls._hash_text(snippet.content),
                    "byte_size": snippet.metadata.get("byte_size", ""),
                    "line_count": snippet.metadata.get("line_count", ""),
                }
                yield MemoryPayload(
                    record_id=cls._stable_id("file", cluster.task_name, snippet.file.relative_path),
                    category=cluster.task_name,
                    content=payload,
                    source="harness_file_ingest",
                    tags=["file", snippet.file.extension or "unknown"],
                    metadata=file_metadata,
                )

    @staticmethod
    def _build_file_payload(task_name: str, relative_path: str, content: str, truncated: bool) -> str:
        """注释：统一文件记忆格式，便于 recall 时解析 Task/Path/Mode。"""
        marker = "truncated" if truncated else "full"
        return (
            f"[Task={task_name}] [Path={relative_path}] [Mode={marker}]\n"
            f"{content}"
        )

    @staticmethod
    def _preview_text(content: str, max_chars: int) -> str:
        """注释：生成单行预览文本，超长内容以省略标记收尾。"""
        normalized = " ".join(content.split())
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[:max_chars].rstrip()}..."

    @staticmethod
    def _stable_id(kind: str, *parts: str) -> str:
        """注释：基于 Harness 节点身份生成稳定 ID，用于重复运行时 upsert。"""
        raw = "::".join([kind, *parts])
        digest = sha256(raw.encode("utf-8")).hexdigest()[:24]
        return f"harness_{kind}_{digest}"

    @staticmethod
    def _hash_text(content: str) -> str:
        """注释：记录内容哈希，便于后续增量扫描判断内容是否变化。"""
        return sha256(content.encode("utf-8")).hexdigest()
