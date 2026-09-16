"""
模块: harness_py.types
职责: 定义 Harness 流水线各阶段的领域数据结构与 Harness IR。
输入: 扫描/读取/分诊阶段构造参数。
输出: ScannedFile, MemoryInputDocument, FileSnippet, TaskCluster, MemoryPayload, MemoryWritePreview 数据类。
副作用: 无。
失败处理: 无；由上游保证字段合法。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class ScannedFile:
    """注释：扫描阶段产物，含绝对路径与相对路径。"""

    path: Path
    relative_path: str
    extension: str


@dataclass
class FileSnippet:
    """注释：读取阶段兼容视图，供旧 triage/bridge 继续消费。"""

    file: ScannedFile
    content: str
    truncated: bool = False
    summary_hint: str = ""
    document_id: str = ""
    source_kind: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryInputDocument:
    """注释：Harness IR，把原始文件统一成 LLM 可读、可追踪的中间表示。"""

    doc_id: str
    file: ScannedFile
    source_kind: str
    content: str
    truncated: bool = False
    summary_hint: str = ""
    content_sha256: str = ""
    byte_size: int = 0
    line_count: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def relative_path(self) -> str:
        """注释：转发扫描阶段相对路径，方便后续 operator 使用。"""
        return self.file.relative_path

    @property
    def extension(self) -> str:
        """注释：转发文件扩展名。"""
        return self.file.extension

    def to_snippet(self) -> FileSnippet:
        """注释：生成旧 FileSnippet 视图，保持现有 triage 与 bridge 兼容。"""
        return FileSnippet(
            file=self.file,
            content=self.content,
            truncated=self.truncated,
            summary_hint=self.summary_hint,
            document_id=self.doc_id,
            source_kind=self.source_kind,
            metadata={
                **self.metadata,
                "doc_id": self.doc_id,
                "source_kind": self.source_kind,
                "content_sha256": self.content_sha256,
                "byte_size": str(self.byte_size),
                "line_count": str(self.line_count),
            },
        )


@dataclass
class TaskCluster:
    """注释：分诊阶段产物，表示同一任务名下的文件集合。"""

    task_name: str
    description: str
    files: List[FileSnippet] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryPayload:
    """注释：Harness 到 MemorySystem 的标准写入载荷，支持稳定 ID 与 metadata。"""

    record_id: str
    category: str
    content: str
    source: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class MemoryWritePreview:
    """注释：dry-run 阶段的拟写入记忆预览，使用稳定 planned_id 但不触碰存储。"""

    category: str
    source: str
    planned_id: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    content_preview: str = ""
    content_length: int = 0
