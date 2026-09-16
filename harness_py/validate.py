"""
模块: harness_py.validate
职责: 提供混乱工作台 Harness 的命令行集成校验入口。
输入: workspace 路径、memory-root 路径、limit-files、dry-run/write 模式。
输出: 控制台统计摘要，进程退出码 0/1。
副作用: 默认 dry-run 只读；传入 --write 时写入 memory-root 下 SQLite。
失败处理: 工作区不存在、扫描为空或写入数量不一致时返回非零退出码。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .harness import HarnessResult, MessyWorkspaceHarness


def validate_workspace(
    workspace_root: str,
    memory_root: str,
    limit_files: int | None = None,
    dry_run: bool = True,
    preview_limit: int = 10,
) -> HarnessResult:
    """
    注释：运行 Harness 并做最小集成断言，供 CLI 与测试复用。
    输入: workspace_root, memory_root, limit_files, dry_run, preview_limit。
    输出: HarnessResult；校验失败时抛出 ValueError。
    """
    workspace = Path(workspace_root)
    if not workspace.exists() or not workspace.is_dir():
        raise ValueError(f"workspace does not exist or is not a directory: {workspace}")

    harness = MessyWorkspaceHarness(
        workspace_root=str(workspace),
        memory_root=memory_root,
    )
    result = harness.run(limit_files=limit_files, dry_run=dry_run, preview_limit=preview_limit)
    if result.scanned_count == 0:
        raise ValueError("workspace scan produced no readable files")
    if result.planned_memory_count == 0:
        raise ValueError("harness produced no memory write plan")

    if not dry_run:
        records = harness.memory_system.store.all()
        if result.memory_written != result.planned_memory_count:
            raise ValueError("memory_written does not match planned_memory_count")
        if len(records) < result.memory_written:
            raise ValueError("stored record count is lower than memory_written")

    return result


def format_result(result: HarnessResult) -> str:
    """
    注释：将校验结果格式化为稳定的多行摘要。
    输入: HarnessResult。
    输出: 可打印字符串。
    """
    lines = [
        "Harness validation completed",
        f"mode={'dry-run' if result.dry_run else 'write'}",
        f"scanned_count={result.scanned_count}",
        f"document_count={result.document_count}",
        f"snippet_count={result.snippet_count}",
        f"cluster_count={result.cluster_count}",
        f"planned_memory_count={result.planned_memory_count}",
        f"memory_written={result.memory_written}",
    ]
    if result.memory_previews:
        lines.append("previews:")
        for idx, preview in enumerate(result.memory_previews, start=1):
            tags = ",".join(preview.tags)
            lines.append(
                f"{idx}. id={preview.planned_id} category={preview.category} source={preview.source} "
                f"tags={tags} chars={preview.content_length} text={preview.content_preview}"
            )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """注释：构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(description="Validate the messy workspace memory harness.")
    parser.add_argument("--workspace", default=".", help="Workspace directory to scan.")
    parser.add_argument("--memory-root", default="./.tamms_memory_validate", help="SQLite memory output root.")
    parser.add_argument("--limit-files", type=int, default=None, help="Optional max file count for faster checks.")
    parser.add_argument("--preview-limit", type=int, default=10, help="Max dry-run preview entries to print.")
    parser.add_argument("--write", action="store_true", help="Write memories to SQLite instead of dry-run.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """注释：CLI 主函数，返回进程退出码。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = validate_workspace(
            workspace_root=args.workspace,
            memory_root=args.memory_root,
            limit_files=args.limit_files,
            dry_run=not args.write,
            preview_limit=args.preview_limit,
        )
    except ValueError as exc:
        print(f"Harness validation failed: {exc}")
        return 1

    print(format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
