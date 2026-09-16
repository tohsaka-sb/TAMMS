"""注释：核心流程单元测试，覆盖提取、存储、召回、压缩与同步。"""

from pathlib import Path

from memory_system_py import ConversationTurn, InMemoryEventSink, MemorySystem, create_memory_system


def test_ingest_and_recall_and_compact(tmp_path: Path) -> None:
    """注释：验证会话提取入库，并能被检索与压缩。"""
    system = MemorySystem(root_dir=str(tmp_path / "runtime"))
    turns = [
        ConversationTurn(role="user", content="我喜欢把任务拆分成小步骤。"),
        ConversationTurn(role="user", content="项目目标是本周完成交付。"),
    ]

    extracted = system.ingest_session(turns)
    assert len(extracted) >= 1

    recalled = system.recall("项目 目标 交付", top_k=3)
    assert len(recalled) >= 1

    total = system.compact(max_items=1)
    assert total == 1


def test_sync_secret_guard_blocks_sensitive_content(tmp_path: Path) -> None:
    """注释：验证同步时会拦截包含敏感词的记忆内容。"""
    system = MemorySystem(root_dir=str(tmp_path / "runtime"))
    system.remember("manual", "这个方案使用 password 作为示例", tags=["security"])
    system.remember("manual", "项目目标是提升性能", tags=["plan"])

    batch = system.export_sync_batch()
    assert len(batch["allowed"]) == 1
    assert len(batch["blocked"]) == 1


def test_structured_operation_results_emit_events(tmp_path: Path) -> None:
    """注释：验证第三阶段结构化结果会记录可审计事件，且不破坏原有 API。"""
    sink = InMemoryEventSink()
    system = create_memory_system(root_dir=str(tmp_path / "runtime"), event_sink=sink)

    ingest_result = system.ingest_session_with_result(
        [ConversationTurn(role="user", content="项目目标是把记忆系统整理成可审计版本。")]
    )
    assert ingest_result.ok is True
    assert ingest_result.event.name == "ingest_session"
    assert ingest_result.event.count == len(ingest_result.value)

    recall_result = system.recall_with_result("项目 目标", top_k=3)
    assert recall_result.ok is True
    assert recall_result.event.metadata["top_k"] == 3

    sync_result = system.export_sync_batch_with_result()
    assert sync_result.ok is True
    assert "allowed" in sync_result.value
    assert [event.name for event in sink.events] == [
        "ingest_session",
        "recall",
        "export_sync_batch",
    ]


def test_empty_structured_result_is_not_error(tmp_path: Path) -> None:
    """注释：空命中是正常状态，事件 status 为 empty 但 ok 仍为 True。"""
    system = MemorySystem(root_dir=str(tmp_path / "runtime"))

    result = system.recall_with_result("完全不存在的查询词", top_k=3)

    assert result.ok is True
    assert result.value == []
    assert result.event.status == "empty"
