"""
模块: formation_py.profile
职责: 分析 Harness IR 的工作区特征与文档角色，供 meta-controller 规划。
输入: List[MemoryInputDocument]。
输出: WorkspaceProfile 与 DocumentRoleProfile。
副作用: 无。
失败处理: 空输入返回零值 profile，不抛错。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from harness_py import MemoryInputDocument


@dataclass
class WorkspaceProfile:
    """注释：工作区级统计和启发式信号。"""

    document_count: int = 0
    source_kind_distribution: dict[str, int] = field(default_factory=dict)
    top_path_distribution: dict[str, int] = field(default_factory=dict)
    total_line_count: int = 0
    total_byte_size: int = 0
    has_code: bool = False
    has_markdown: bool = False
    has_config: bool = False
    has_tests: bool = False
    has_build_files: bool = False
    has_scripts: bool = False
    has_logs: bool = False
    has_data_files: bool = False
    dominant_source_kind: str = ""
    dominant_language_hint: str = ""
    complexity_level: str = "small"
    organization_level: str = "flat"


@dataclass
class DocumentRoleProfile:
    """注释：单文档角色与 intent hint，用于 document selection。"""

    doc_id: str
    relative_path: str
    source_kind: str
    content_roles: list[str] = field(default_factory=list)
    intent_hints: list[str] = field(default_factory=list)
    importance_score: float = 0.5


class WorkspaceProfiler:
    """注释：根据 IR 文档集合生成工作区画像。"""

    def profile(self, documents: list[MemoryInputDocument]) -> WorkspaceProfile:
        """注释：统计 source kind、路径组织、复杂度和语言提示。"""
        source_counts = Counter(document.source_kind for document in documents)
        top_counts = Counter(_top_path(document.relative_path) for document in documents)
        total_lines = sum(document.line_count for document in documents)
        total_bytes = sum(document.byte_size for document in documents)
        paths = [document.relative_path.lower() for document in documents]
        extensions = [document.extension.lower() for document in documents]

        return WorkspaceProfile(
            document_count=len(documents),
            source_kind_distribution=dict(source_counts),
            top_path_distribution=dict(top_counts),
            total_line_count=total_lines,
            total_byte_size=total_bytes,
            has_code=source_counts.get("code", 0) > 0,
            has_markdown=source_counts.get("markdown", 0) > 0,
            has_config=source_counts.get("config", 0) > 0,
            has_tests=any("test" in path or "spec" in path for path in paths),
            has_build_files=any(_is_build_path(path) for path in paths),
            has_scripts=any(kind in source_counts for kind in ["shell", "sql"]),
            has_logs=any(path.endswith((".log", ".out", ".err")) or "log" in path for path in paths),
            has_data_files=source_counts.get("csv", 0) > 0,
            dominant_source_kind=_dominant(source_counts),
            dominant_language_hint=_language_hint(extensions, paths),
            complexity_level=_complexity(len(documents), total_lines),
            organization_level=_organization(top_counts, len(documents)),
        )


class DocumentRoleProfiler:
    """注释：给单个 IR 文档打输入角色和意图提示。"""

    def profile(self, documents: list[MemoryInputDocument]) -> list[DocumentRoleProfile]:
        """注释：为每个文档生成角色画像，并把角色写回 document.metadata。"""
        profiles = []
        for document in documents:
            roles = _content_roles(document)
            hints = _intent_hints(document, roles)
            score = _importance_score(document, roles)
            document.metadata["content_roles"] = roles
            document.metadata["intent_hints"] = hints
            document.metadata["importance_score"] = score
            profiles.append(
                DocumentRoleProfile(
                    doc_id=document.doc_id,
                    relative_path=document.relative_path,
                    source_kind=document.source_kind,
                    content_roles=roles,
                    intent_hints=hints,
                    importance_score=score,
                )
            )
        return profiles


def _top_path(relative_path: str) -> str:
    parts = relative_path.split("/")
    return parts[0] if len(parts) > 1 else "root"


def _dominant(counter: Counter) -> str:
    return counter.most_common(1)[0][0] if counter else ""


def _complexity(document_count: int, total_lines: int) -> str:
    if document_count >= 200 or total_lines >= 50000:
        return "large"
    if document_count >= 30 or total_lines >= 5000:
        return "medium"
    return "small"


def _organization(top_counts: Counter, document_count: int) -> str:
    if document_count <= 3:
        return "flat"
    non_root = [path for path in top_counts if path != "root"]
    if len(non_root) >= 3:
        return "structured"
    if non_root:
        return "semi_structured"
    return "flat"


def _language_hint(extensions: list[str], paths: list[str]) -> str:
    if any(ext in {".cpp", ".cc", ".c", ".h", ".hpp"} for ext in extensions) or any("cmakelists" in path for path in paths):
        return "cpp"
    if ".py" in extensions:
        return "python"
    if any(ext in {".ts", ".tsx", ".js", ".jsx"} for ext in extensions):
        return "javascript"
    return ""


def _is_build_path(path: str) -> bool:
    return any(marker in path for marker in ["cmakelists.txt", "makefile", "package.json", "pyproject.toml", "setup.py", "dockerfile"])


def _content_roles(document: MemoryInputDocument) -> list[str]:
    path = document.relative_path.lower()
    roles: list[str] = []
    if document.source_kind == "code":
        roles.append("implementation")
    if "test" in path or "spec" in path:
        roles.append("test")
    if document.source_kind == "markdown":
        roles.append("documentation")
    if document.source_kind == "config":
        roles.append("configuration")
    if _is_build_path(path):
        roles.append("build_script")
    if document.source_kind == "shell":
        roles.append("run_script")
    if path.endswith((".log", ".out", ".err")) or "log" in path:
        roles.append("log")
    if document.source_kind == "csv":
        roles.append("dataset")
    return roles or ["unknown"]


def _intent_hints(document: MemoryInputDocument, roles: list[str]) -> list[str]:
    hints = {"overview"}
    if {"documentation", "build_script", "run_script"}.intersection(roles):
        hints.add("reproduce")
    if {"test", "log", "configuration"}.intersection(roles):
        hints.add("debug")
    if {"implementation", "test"}.intersection(roles):
        hints.add("locate")
        hints.add("graph")
    if document.line_count > 80:
        hints.add("summarize")
    return sorted(hints)


def _importance_score(document: MemoryInputDocument, roles: list[str]) -> float:
    score = 0.4
    if "documentation" in roles:
        score += 0.15
    if "implementation" in roles:
        score += 0.15
    if "test" in roles or "build_script" in roles:
        score += 0.15
    if document.line_count > 80:
        score += 0.1
    return min(score, 1.0)
