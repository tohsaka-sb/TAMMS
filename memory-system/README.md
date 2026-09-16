# Unified Memory System (Python)

<!-- 注释：文档入口，与可执行代码 memory_system_py / harness_py 对应。 -->

## 快速使用

```python
from memory_system_py import ConversationTurn, InMemoryEventSink, create_memory_system

sink = InMemoryEventSink()
system = create_memory_system("./runtime_memory", event_sink=sink)

result = system.ingest_session_with_result([
    ConversationTurn(role="user", content="我喜欢分步交付。"),
])
hits = system.recall("分步")
```

## 端到端流程

1. **会话路径**：`ConversationTurn` → `MemoryExtractor` → `MemoryStore`
2. **检索**：`MemoryRetriever.search`
3. **压缩**：`MemoryCompactor.compact`
4. **同步**：`TeamMemorySync.prepare_sync_batch`（含敏感词拦截）
5. **工作台路径**：`MessyWorkspaceHarness` → `MemorySystemBridge` → `MemorySystem.remember`
6. **预览/校验路径**：`MessyWorkspaceHarness.run(dry_run=True)` 或 `python -m harness_py.validate`

## 结构化结果

<!-- 注释：第三阶段新增，旧 API 不变；需要审计时使用 with_result 方法。 -->

- `OperationEvent`：记录操作名、状态、数量、消息、metadata 与时间戳。
- `OperationResult[T]`：包装业务返回值、事件与 ok 状态。
- `*_with_result`：`ingest_session_with_result`、`remember_with_result`、`recall_with_result`、`compact_with_result`、`export_sync_batch_with_result`。

详见 `workflow.md` 与 `COMMENT_TEMPLATE.md`。

## Harness Dry-run

```python
from harness_py import MessyWorkspaceHarness

harness = MessyWorkspaceHarness("./workspace", "./runtime_memory")
preview = harness.run(dry_run=True, preview_limit=5)
```

dry-run 返回 `planned_memory_count` 与 `memory_previews`，不会创建或写入 SQLite。
