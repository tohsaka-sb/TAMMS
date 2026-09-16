"""注释：Harness 端到端测试，验证能把混乱目录写入记忆系统。"""

from pathlib import Path

from harness_py import MessyWorkspaceHarness
from harness_py.validate import validate_workspace


def test_harness_pipeline_writes_memories(tmp_path: Path) -> None:
    """注释：构造混乱工作区后，检查产出统计与入库结果。"""
    workspace = tmp_path / "workspace"
    memory_root = tmp_path / "memory_runtime"
    workspace.mkdir()

    (workspace / ".gitignore").write_text("skip.log\n", encoding="utf-8")
    (workspace / "main.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    (workspace / "notes.md").write_text("# doc\nproject plan\n", encoding="utf-8")
    (workspace / "data.csv").write_text("name,age\nalice,20\nbob,30\n", encoding="utf-8")
    (workspace / "skip.log").write_text("do not read", encoding="utf-8")

    harness = MessyWorkspaceHarness(
        workspace_root=str(workspace),
        memory_root=str(memory_root),
    )
    result = harness.run()

    assert result.scanned_count == 3
    assert result.document_count == 3
    assert result.snippet_count == 3
    assert result.cluster_count >= 1
    assert result.memory_written >= 4
    assert result.planned_memory_count == result.memory_written
    assert {document.relative_path for document in result.documents} == {"main.py", "notes.md", "data.csv"}


def test_harness_dry_run_returns_preview_without_creating_memory_db(tmp_path: Path) -> None:
    """注释：dry-run 只生成拟写入预览，不初始化 SQLite 目录。"""
    workspace = tmp_path / "workspace"
    memory_root = tmp_path / "memory_runtime"
    workspace.mkdir()

    (workspace / "main.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    (workspace / "notes.md").write_text("# plan\n项目目标是整理记忆框架\n", encoding="utf-8")

    harness = MessyWorkspaceHarness(
        workspace_root=str(workspace),
        memory_root=str(memory_root),
    )
    result = harness.run(dry_run=True, preview_limit=2)

    assert result.dry_run is True
    assert result.memory_written == 0
    assert result.planned_memory_count >= 2
    assert result.document_count == result.snippet_count
    assert result.memory_previews is not None
    assert len(result.memory_previews) == 2
    assert result.memory_previews[0].category.startswith("Task_")
    assert result.memory_previews[0].planned_id.startswith("harness_")
    assert memory_root.exists() is False


def test_validate_workspace_write_mode_checks_persisted_records(tmp_path: Path) -> None:
    """注释：集成校验函数在 write 模式下会确认实际入库数量。"""
    workspace = tmp_path / "workspace"
    memory_root = tmp_path / "memory_runtime"
    workspace.mkdir()

    (workspace / "main.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    (workspace / "README.md").write_text("# docs\n记忆系统说明\n", encoding="utf-8")

    result = validate_workspace(
        workspace_root=str(workspace),
        memory_root=str(memory_root),
        dry_run=False,
    )

    assert result.dry_run is False
    assert result.memory_written == result.planned_memory_count
    assert (memory_root / "memory.db").exists()


def test_harness_repeated_run_upserts_stable_memory_nodes(tmp_path: Path) -> None:
    """注释：重复导入同一工作区时使用稳定 ID 更新，不重复追加记忆节点。"""
    workspace = tmp_path / "workspace"
    memory_root = tmp_path / "memory_runtime"
    workspace.mkdir()

    target = workspace / "main.py"
    target.write_text("def run():\n    return 'v1'\n", encoding="utf-8")

    harness = MessyWorkspaceHarness(
        workspace_root=str(workspace),
        memory_root=str(memory_root),
    )
    first = harness.run()
    records_after_first = harness.memory_system.store.all()
    file_records = [record for record in records_after_first if record.source == "harness_file_ingest"]
    assert len(records_after_first) == first.memory_written
    assert len(file_records) == 1
    original_id = file_records[0].id
    original_created_at = file_records[0].created_at

    target.write_text("def run():\n    return 'v2'\n", encoding="utf-8")
    second = harness.run()
    records_after_second = harness.memory_system.store.all()
    updated_file_records = [record for record in records_after_second if record.source == "harness_file_ingest"]

    assert second.memory_written == first.memory_written
    assert len(records_after_second) == len(records_after_first)
    assert len(updated_file_records) == 1
    assert updated_file_records[0].id == original_id
    assert updated_file_records[0].created_at == original_created_at
    assert "v2" in updated_file_records[0].content
    assert updated_file_records[0].metadata["doc_id"].startswith("harness_doc_")
    assert updated_file_records[0].metadata["source_kind"] == "code"
    assert updated_file_records[0].metadata["relative_path"] == "main.py"
