"""
模块: harness_py.scanner
职责: 递归扫描工作区，过滤二进制、缓存目录与 .gitignore 命中项。
输入: workspace 根目录 Path。
输出: List[ScannedFile]。
副作用: 只读磁盘遍历。
失败处理: .gitignore 缺失时仅用内置规则；pathspec 缺失时使用内置简易匹配器。
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable, List, Protocol

try:
    import pathspec
except ModuleNotFoundError:
    pathspec = None

from .types import ScannedFile


class IgnoreSpec(Protocol):
    """注释：抽象 gitignore 匹配接口，兼容 pathspec 与内置 fallback。"""

    def match_file(self, rel: str) -> bool:
        """注释：判断相对路径是否被忽略。"""


class SimpleGitIgnoreSpec:
    """注释：pathspec 不可用时的简易 .gitignore 匹配器，覆盖常见文件/目录/glob 规则。"""

    def __init__(self, lines: List[str]) -> None:
        self.patterns = [line.strip() for line in lines if self._is_active_pattern(line)]

    def match_file(self, rel: str) -> bool:
        """注释：按常见 .gitignore 规则做保守匹配；不支持否定规则。"""
        rel = rel.replace("\\", "/")
        parts = rel.split("/")
        for pattern in self.patterns:
            normalized = pattern.lstrip("/")
            if normalized.endswith("/"):
                directory = normalized.rstrip("/")
                if directory in parts:
                    return True
                continue
            if "/" not in normalized and any(fnmatch(part, normalized) for part in parts):
                return True
            if fnmatch(rel, normalized) or fnmatch(rel, f"*/{normalized}"):
                return True
        return False

    def _is_active_pattern(self, line: str) -> bool:
        """注释：过滤空行、注释和否定规则；否定规则在 fallback 中保守跳过。"""
        stripped = line.strip()
        return bool(stripped) and not stripped.startswith("#") and not stripped.startswith("!")


class FileScanner:
    """注释：支持 .gitignore + 内置 skip 规则的递归扫描器。"""

    DEFAULT_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".pytest_cache", ".idea", ".vscode"}
    DEFAULT_BINARY_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".pdf",
        ".zip",
        ".7z",
        ".tar",
        ".gz",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".mp3",
        ".mp4",
    }
    DEFAULT_SKIP_FILES = {".gitignore", ".gitattributes", ".gitmodules"}

    def __init__(self, root: Path) -> None:
        """注释：绑定根目录并预加载 .gitignore 规则。"""
        self.root = root
        self.spec = self._load_gitignore_spec()

    def scan(self) -> List[ScannedFile]:
        """
        注释：rglob 全目录，跳过目录/二进制/gitignore 后返回文件清单。
        输出: 通过过滤的 ScannedFile 列表（可为空）。
        """
        results: List[ScannedFile] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            if self._should_skip(path, rel):
                continue
            results.append(
                ScannedFile(
                    path=path,
                    relative_path=rel,
                    extension=path.suffix.lower(),
                )
            )
        return results

    def _load_gitignore_spec(self) -> IgnoreSpec | None:
        """注释：解析根目录 .gitignore；不存在则返回 None。"""
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            return None
        lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
        if pathspec is None:
            return SimpleGitIgnoreSpec(lines)
        return pathspec.PathSpec.from_lines("gitignore", lines)

    def _should_skip(self, path: Path, rel: str) -> bool:
        """注释：综合目录名、扩展名、gitignore 判断是否跳过。"""
        parts = set(path.parts)
        if self.DEFAULT_SKIP_DIRS.intersection(parts):
            return True
        if path.name in self.DEFAULT_SKIP_FILES:
            return True
        if path.suffix.lower() in self.DEFAULT_BINARY_EXTENSIONS:
            return True
        if self.spec and self.spec.match_file(rel):
            return True
        return False

    @staticmethod
    def filter_by_extensions(files: Iterable[ScannedFile], allowed: set[str]) -> List[ScannedFile]:
        """注释：后置按扩展名过滤，便于针对性 ingest。"""
        return [f for f in files if f.extension in allowed]
