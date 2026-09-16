"""
模块: formation_py.meta_memory
职责: 加载、索引、去重并安全写回 formation meta-memory Markdown。
输入: 可选 meta memory 文件路径、MetaMemoryUpdateCandidate。
输出: Markdown 文本、preview、apply result。
副作用: provider 只读；writer apply 时备份并写回 Markdown。
失败处理: 文件不存在时返回空字符串；重复/超限更新返回未应用结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re

from .meta_formation import MetaMemoryUpdateCandidate


DEFAULT_META_MEMORY_FILE = Path(__file__).resolve().parent / "meta" / "formation_meta_memory.md"
DEFAULT_META_MEMORY_SECTIONS = [
    "General Principles",
    "Intent Strategies",
    "Operator Guidelines",
    "Selector Guidelines",
    "Successful Patterns",
    "Failure Lessons",
    "Anti-Patterns",
    "Update History",
]


class FormationMetaMemoryProvider:
    """注释：加载 formation meta-memory Markdown。"""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_META_MEMORY_FILE

    def load_reference(self) -> str:
        """注释：加载 Markdown 元记忆；缺失时返回空字符串。"""
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8", errors="ignore")


@dataclass
class MetaMemoryApplyResult:
    """注释：writer.apply_update 的结构化结果。"""

    applied: bool
    reason: str = ""
    memory_id: str = ""
    history_id: str = ""
    backup_path: str = ""
    warnings: list[str] = field(default_factory=list)


class MetaMemoryIndex:
    """注释：轻量 Markdown 索引，用于 section、ID、去重和限长。"""

    def __init__(self, markdown: str) -> None:
        self.markdown = markdown or ""
        self.sections = _parse_sections(self.markdown)

    def section_exists(self, section: str) -> bool:
        """注释：判断 section 是否存在。"""
        return section in self.sections

    def count_items(self, section: str) -> int:
        """注释：统计某 section 下 bullet 数量。"""
        return sum(1 for line in self.sections.get(section, []) if line.strip().startswith("- "))

    def memory_ids(self) -> set[str]:
        """注释：提取 MM/UH 等 ID。"""
        return set(re.findall(r"\[(MM-\d{8}-\d{4}|UH-\d{8}-\d{4})\]", self.markdown))

    def has_memory_id(self, memory_id: str) -> bool:
        """注释：检查 ID 是否已存在。"""
        return memory_id in self.memory_ids()

    def has_duplicate_content(self, section: str, content: str) -> bool:
        """注释：同 section 下 normalized content 去重。"""
        target = _normalize_for_dedup(content)
        if not target:
            return False
        for line in self.sections.get(section, []):
            normalized = _normalize_for_dedup(_strip_markdown_id(line))
            if normalized and (target == normalized or target in normalized or normalized in target):
                return True
        return False


class MetaMemoryDeduper:
    """注释：候选去重封装，便于后续替换为 embedding/统计版本。"""

    def is_duplicate(self, candidate: MetaMemoryUpdateCandidate, index: MetaMemoryIndex) -> bool:
        """注释：ID 或同 section 内容重复则视为重复。"""
        return index.has_memory_id(candidate.proposed_memory_id) or index.has_duplicate_content(
            candidate.section,
            candidate.content,
        )


class FormationMetaMemoryWriter:
    """注释：安全写回 formation_meta_memory.md；默认追加 bullet，不自动压缩。"""

    def __init__(
        self,
        path: str | None = None,
        max_items_per_section: int = 30,
        deduper: MetaMemoryDeduper | None = None,
    ) -> None:
        self.path = Path(path) if path else DEFAULT_META_MEMORY_FILE
        self.max_items_per_section = max_items_per_section
        self.deduper = deduper or MetaMemoryDeduper()

    def preview_update(self, candidate: MetaMemoryUpdateCandidate) -> str:
        """注释：返回候选将写入的 Markdown，不修改文件。"""
        return f"## {candidate.section}\n\n{candidate.to_markdown_bullet()}"

    def apply_update(self, candidate: MetaMemoryUpdateCandidate, create_backup: bool = True) -> MetaMemoryApplyResult:
        """注释：备份后写入 section，并追加 Update History。"""
        if candidate.status == "rejected":
            return MetaMemoryApplyResult(False, reason="candidate_rejected", memory_id=candidate.proposed_memory_id)
        text = self._load_or_template()
        index = MetaMemoryIndex(text)
        if candidate.section not in DEFAULT_META_MEMORY_SECTIONS:
            return MetaMemoryApplyResult(False, reason="unknown_section", memory_id=candidate.proposed_memory_id)
        if self.deduper.is_duplicate(candidate, index):
            return MetaMemoryApplyResult(False, reason="duplicate", memory_id=candidate.proposed_memory_id)
        if index.count_items(candidate.section) >= self.max_items_per_section:
            return MetaMemoryApplyResult(
                False,
                reason="section_capacity_exceeded",
                memory_id=candidate.proposed_memory_id,
                warnings=[f"{candidate.section} has reached max_items_per_section={self.max_items_per_section}"],
            )

        backup_path = ""
        if create_backup and self.path.exists():
            backup_path = str(self.backup_current_memory())
        history_id = _next_history_id(text)
        updated = _ensure_sections(text, DEFAULT_META_MEMORY_SECTIONS)
        updated = _append_to_section(updated, candidate.section, candidate.to_markdown_bullet())
        history = (
            f"- [{history_id}] Applied {candidate.proposed_memory_id} to `{candidate.section}`; "
            f"stage={candidate.stage}; source={candidate.episode_id}; candidate={candidate.candidate_id}."
        )
        updated = _append_to_section(updated, "Update History", history)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(updated, encoding="utf-8")
        return MetaMemoryApplyResult(
            True,
            reason="applied",
            memory_id=candidate.proposed_memory_id,
            history_id=history_id,
            backup_path=backup_path,
        )

    def backup_current_memory(self) -> Path:
        """注释：写入前创建时间戳备份。"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup = self.path.with_name(f"{self.path.name}.bak.{timestamp}")
        suffix = 1
        while backup.exists():
            backup = self.path.with_name(f"{self.path.name}.bak.{timestamp}.{suffix}")
            suffix += 1
        backup.write_text(self.path.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        return backup

    def _load_or_template(self) -> str:
        if self.path.exists():
            return self.path.read_text(encoding="utf-8", errors="ignore")
        return "# Formation Meta Memory\n\n" + "\n\n".join(f"## {section}\n" for section in DEFAULT_META_MEMORY_SECTIONS)


def _parse_sections(markdown: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = ""
    for line in markdown.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections.setdefault(current, [])
            continue
        if current:
            sections.setdefault(current, []).append(line)
    return sections


def _ensure_sections(markdown: str, sections: list[str]) -> str:
    text = markdown.rstrip() or "# Formation Meta Memory"
    existing = MetaMemoryIndex(text)
    for section in sections:
        if not existing.section_exists(section):
            text += f"\n\n## {section}\n"
    return text.rstrip() + "\n"


def _append_to_section(markdown: str, section: str, bullet: str) -> str:
    lines = markdown.rstrip().splitlines()
    header = f"## {section}"
    try:
        start = lines.index(header)
    except ValueError:
        lines.extend(["", header, bullet])
        return "\n".join(lines).rstrip() + "\n"

    insert_at = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("## "):
            insert_at = idx
            break
    while insert_at > start + 1 and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    prefix = [""] if insert_at > start + 1 else []
    lines[insert_at:insert_at] = prefix + [bullet]
    return "\n".join(lines).rstrip() + "\n"


def _next_history_id(markdown: str) -> str:
    date = datetime.now().strftime("%Y%m%d")
    ids = [int(match) for match in re.findall(rf"\[UH-{date}-(\d{{4}})\]", markdown)]
    return f"UH-{date}-{(max(ids) + 1 if ids else 1):04d}"


def _strip_markdown_id(text: str) -> str:
    return re.sub(r"^-\s+\[[A-Z]+-\d{8}-\d{4}\]\s*", "", text.strip())


def _normalize_for_dedup(text: str) -> str:
    stripped = _strip_markdown_id(text)
    stripped = re.sub(r"\*\*|`|\[|\]|\(|\)|:|;|,|\.|\"|'", " ", stripped.lower())
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped
