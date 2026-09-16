"""注释：真实目录扫描与转换测试，验证 Harness 能处理实际项目文件夹。"""

from pathlib import Path

from harness_py import MessyWorkspaceHarness
from harness_py.scanner import FileScanner


def test_scan_real_memory_system_py_folder() -> None:
    """注释：扫描真实的 memory_system_py 目录，而非仅使用临时模拟目录。"""
    repo_root = Path(__file__).resolve().parents[2]
    target_dir = repo_root / "memory_system_py"
    assert target_dir.exists() and target_dir.is_dir()

    scanner = FileScanner(target_dir)
    files = scanner.scan()

    # 注释：至少应扫描到多个文件，确保“全目录扫描”真实发生。
    assert len(files) > 0

    rel_paths = {f.relative_path for f in files}
    # 注释：核心文件应出现在扫描结果中，证明不是空跑。
    assert "system.py" in rel_paths
    assert "storage.py" in rel_paths


def test_convert_real_memory_system_py_folder_to_memory_nodes(tmp_path: Path) -> None:
    """注释：真实目录端到端转换测试，验证能生成可检索记忆节点。"""
    repo_root = Path(__file__).resolve().parents[2]
    target_dir = repo_root / "memory_system_py"
    memory_root = tmp_path / "runtime_memory"

    harness = MessyWorkspaceHarness(
        workspace_root=str(target_dir),
        memory_root=str(memory_root),
    )
    result = harness.run(limit_files=30)

    assert result.scanned_count > 0
    assert result.memory_written > 0

    records = harness.memory_system.store.all()
    assert len(records) == result.memory_written

    # 注释：验证桥接写入格式包含任务名、文件路径与截断模式。
    file_nodes = [r for r in records if r.source == "harness_file_ingest"]
    assert len(file_nodes) > 0
    assert any("[Task=" in r.content and "[Path=" in r.content and "[Mode=" in r.content for r in file_nodes)
