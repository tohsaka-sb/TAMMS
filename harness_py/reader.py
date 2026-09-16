"""
模块: harness_py.reader
职责: 将异构文件读成 Harness IR，并提供 FileSnippet 兼容视图。
输入: ScannedFile。
输出: MemoryInputDocument 或 FileSnippet。
副作用: 只读源文件。
失败处理: 非文本/解码失败时返回占位摘要，不中断 Harness 主流程。
"""

from __future__ import annotations

import csv
from hashlib import sha256
from pathlib import Path

from .types import FileSnippet, MemoryInputDocument, ScannedFile


class HeterogeneousReader:
    """注释：按扩展名路由文本直读、CSV 摘要或兜底策略。"""

    TEXT_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".sql", ".sh"}
    CSV_EXTENSIONS = {".csv"}
    CODE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx"}
    CONFIG_EXTENSIONS = {".json", ".yaml", ".yml", ".toml", ".ini"}

    def __init__(self, max_chars: int = 32000, head_lines: int = 200, tail_lines: int = 200) -> None:
        self.max_chars = max_chars
        self.head_lines = head_lines
        self.tail_lines = tail_lines

    def read(self, scanned_file: ScannedFile) -> FileSnippet:
        """
        注释：兼容旧接口，根据 IR 生成 FileSnippet。
        输出: 带 truncated / summary_hint 的 FileSnippet。
        """
        return self.read_document(scanned_file).to_snippet()

    def read_document(self, scanned_file: ScannedFile) -> MemoryInputDocument:
        """
        注释：根据扩展名生成统一 Harness IR，供 LLM/压缩/聚类算子消费。
        输出: MemoryInputDocument。
        """
        ext = scanned_file.extension
        if ext in self.CSV_EXTENSIONS:
            content = self._read_csv_summary(scanned_file.path)
            return self._document(scanned_file, content, truncated=False, summary_hint="csv_schema_preview")
        if ext in self.TEXT_EXTENSIONS or ext == "":
            content, truncated = self._read_text_with_truncation(scanned_file.path)
            return self._document(scanned_file, content, truncated=truncated, summary_hint="text_preview")

        try:
            content, truncated = self._read_text_with_truncation(scanned_file.path)
            return self._document(scanned_file, content, truncated=truncated, summary_hint="fallback_text_preview")
        except UnicodeDecodeError:
            return self._document(
                scanned_file,
                content="[Unsupported binary-like content]",
                truncated=False,
                summary_hint="unsupported",
            )

    def _read_text_with_truncation(self, path: Path) -> tuple[str, bool]:
        """注释：超长文本保留 head+tail，中间以 Truncated 标记。"""
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if len(raw) <= self.max_chars:
            return raw, False

        lines = raw.splitlines()
        head = "\n".join(lines[: self.head_lines])
        tail = "\n".join(lines[-self.tail_lines :])
        truncated = f"{head}\n\n...[Truncated]...\n\n{tail}"
        return truncated, True

    def _read_csv_summary(self, path: Path) -> str:
        """注释：仅读取表头与前 5 行，避免大 CSV 撑爆内存。"""
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as fp:
            reader = csv.reader(fp)
            for idx, row in enumerate(reader):
                rows.append(row)
                if idx >= 5:
                    break

        if not rows:
            return "CSV: <empty>"
        header = rows[0]
        preview = rows[1:6]
        output = ["CSV Header:", ", ".join(header), "", "Top 5 Rows:"]
        output.extend(", ".join(r) for r in preview)
        return "\n".join(output)

    def _document(
        self,
        scanned_file: ScannedFile,
        content: str,
        truncated: bool,
        summary_hint: str,
    ) -> MemoryInputDocument:
        """注释：统一填充 IR 元数据，保持 reader 输出可追踪。"""
        byte_size = self._byte_size(scanned_file.path)
        content_sha = sha256(content.encode("utf-8")).hexdigest()
        source_kind = self._source_kind(scanned_file)
        return MemoryInputDocument(
            doc_id=self._doc_id(scanned_file.relative_path),
            file=scanned_file,
            source_kind=source_kind,
            content=content,
            truncated=truncated,
            summary_hint=summary_hint,
            content_sha256=content_sha,
            byte_size=byte_size,
            line_count=self._line_count(content),
            metadata={
                "reader_version": "harness_ir_v1",
                "relative_path": scanned_file.relative_path,
                "extension": scanned_file.extension,
                "llm_ready": "true",
            },
        )

    def _source_kind(self, scanned_file: ScannedFile) -> str:
        """注释：把扩展名归一到后续 operator 可用的来源类型。"""
        ext = scanned_file.extension
        if ext in self.CODE_EXTENSIONS:
            return "code"
        if ext == ".md":
            return "markdown"
        if ext == ".csv":
            return "csv"
        if ext in self.CONFIG_EXTENSIONS:
            return "config"
        if ext == ".sql":
            return "sql"
        if ext == ".sh":
            return "shell"
        if ext in {".txt", ""}:
            return "text"
        return "unknown"

    def _doc_id(self, relative_path: str) -> str:
        """注释：基于路径生成稳定文档 ID；内容变化由 content_sha256 表达。"""
        digest = sha256(relative_path.encode("utf-8")).hexdigest()[:24]
        return f"harness_doc_{digest}"

    def _line_count(self, content: str) -> int:
        """注释：统计 LLM 可读内容行数，空内容为 0。"""
        if not content:
            return 0
        return len(content.splitlines()) or 1

    def _byte_size(self, path: Path) -> int:
        """注释：读取原始文件大小；失败时返回 0，不中断主流程。"""
        try:
            return path.stat().st_size
        except OSError:
            return 0
