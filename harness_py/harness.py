"""
模块: harness_py.harness
职责: 混乱工作台一站式流水线：扫描 → Harness IR → 分诊 → 桥接入库/预览。
输入: workspace_root, memory_root；可选 gist_fn、limit_files、dry_run。
输出: HarnessResult 统计快照与可选 dry-run 预览。
副作用: 非 dry-run 时写入 memory_root 下 SQLite；dry-run 只读 workspace。
失败处理: 单文件读取失败由 Reader 降级；limit_files 截断扫描结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from memory_system_py import MemorySystem

from .bridge import MemorySystemBridge
from .reader import HeterogeneousReader
from .scanner import FileScanner
from .triage import FastPassTriageRouter, GistFn
from .types import MemoryInputDocument, MemoryWritePreview, TaskCluster


@dataclass
class HarnessResult:
    """注释：run() 执行结果，供测试与 CLI 审计。"""

    scanned_count: int
    document_count: int
    snippet_count: int
    cluster_count: int
    memory_written: int
    clusters: List[TaskCluster]
    documents: List[MemoryInputDocument]
    dry_run: bool = False
    planned_memory_count: int = 0
    memory_previews: List[MemoryWritePreview] | None = None


class MessyWorkspaceHarness:
    """注释：组合 Scanner / Reader / Triage / Bridge 与 MemorySystem。"""

    def __init__(self, workspace_root: str, memory_root: str, gist_fn: GistFn | None = None) -> None:
        self.workspace_root = Path(workspace_root)
        self.memory_root = Path(memory_root)
        self._memory_system: MemorySystem | None = None
        self._bridge: MemorySystemBridge | None = None
        self.scanner = FileScanner(self.workspace_root)
        self.reader = HeterogeneousReader()
        self.triage = FastPassTriageRouter(gist_fn=gist_fn)

    @property
    def memory_system(self) -> MemorySystem:
        """注释：懒加载记忆系统，dry-run 不访问时不会创建 SQLite。"""
        if self._memory_system is None:
            self._memory_system = MemorySystem(root_dir=str(self.memory_root))
        return self._memory_system

    @property
    def bridge(self) -> MemorySystemBridge:
        """注释：懒加载桥接器，绑定同一个 MemorySystem 实例。"""
        if self._bridge is None:
            self._bridge = MemorySystemBridge(self.memory_system)
        return self._bridge

    def run(self, limit_files: int | None = None, dry_run: bool = False, preview_limit: int = 20) -> HarnessResult:
        """
        注释：执行完整 ingest 流水线。
        输入: limit_files 可选，限制扫描文件数以加速测试；dry_run 为 True 时不写库。
        输出: HarnessResult。
        """
        scanned = self.scanner.scan()
        if limit_files is not None:
            scanned = scanned[:limit_files]
        documents = [self.reader.read_document(item) for item in scanned]
        snippets = [document.to_snippet() for document in documents]
        gists = self.triage.extract_gists(snippets)
        clusters = self.triage.cluster(gists)
        previews = MemorySystemBridge.preview_clusters(clusters)
        if dry_run:
            written = 0
            selected_previews: List[MemoryWritePreview] | None = previews[:preview_limit]
        else:
            written = self.bridge.ingest_clusters(clusters)
            selected_previews = None

        return HarnessResult(
            scanned_count=len(scanned),
            document_count=len(documents),
            snippet_count=len(snippets),
            cluster_count=len(clusters),
            memory_written=written,
            clusters=clusters,
            documents=documents,
            dry_run=dry_run,
            planned_memory_count=len(previews),
            memory_previews=selected_previews,
        )
