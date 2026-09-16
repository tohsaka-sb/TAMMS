# Memory Workflow

<!-- 注释：触发点、产物与失败策略（Python 实现版）。 -->

## 触发点

| 事件 | 模块 | 行为 |
|------|------|------|
| 会话推进 | `ingest_session` | 提取并入库 |
| 手动/policy | `remember` | 单条 upsert |
| 查询 | `recall` | top_k 检索 |
| 库过大 | `compact` | 保留高分条目 |
| 团队同步 | `export_sync_batch` | allowed / blocked 分组 |
| 目录导入 | `MessyWorkspaceHarness.run` | 扫描→分诊→桥接 |
| 目录预览 | `MessyWorkspaceHarness.run(dry_run=True)` | 扫描→分诊→拟写入预览 |
| 集成校验 | `python -m harness_py.validate` | dry-run 或写入模式校验 |

## 事件层

<!-- 注释：第三阶段新增事件钩子，供测试、日志和后续 UI 使用。 -->

| 方法 | 事件名 | 关键 metadata |
|------|--------|---------------|
| `ingest_session_with_result` | `ingest_session` | `turn_count` |
| `remember_with_result` | `remember` | `category`, `source` |
| `recall_with_result` | `recall` | `query`, `top_k` |
| `compact_with_result` | `compact` | `before`, `after`, `max_items` |
| `export_sync_batch_with_result` | `export_sync_batch` | `allowed_count`, `blocked_count` |

## 失败处理

- 提取无命中：返回 `[]`，不抛错
- 存储空批量：`add_many` 直接返回
- 同步敏感命中：进入 `blocked`，不中断主流程
- Harness 读二进制失败：`HeterogeneousReader` 返回占位摘要
- 结构化空结果：`OperationEvent.status = "empty"`，但 `OperationResult.ok = True`
- Harness dry-run：不初始化 `memory_root`，只返回 `MemoryWritePreview`
- Harness 重复导入：cluster/file 使用稳定 `harness_*` ID，通过 upsert 更新既有记忆
- 集成校验：工作区不存在、扫描为空或写入数量不一致时返回非零退出码
