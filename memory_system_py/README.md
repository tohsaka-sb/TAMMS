# memory_system_py

<!-- 注释：可独立运行的 Python 记忆系统；Phase 2 已补齐模块五要素注释与 create_memory_system 入口。 -->

## 设计目标

- 独立运行：不依赖原 TS 调用链
- 接口简化：统一通过 `MemorySystem` / `create_memory_system()` 操作
- 能力完整：提取、存储、检索、压缩、同步防护

## 存储说明

- 默认 SQLite，数据库文件：`{root_dir}/memory.db`

## 核心接口

| 方法 | 说明 |
|------|------|
| `ingest_session(turns)` | 会话提取并入库 |
| `remember(...)` | 手工/桥接写入 |
| `recall(query, top_k)` | 检索 |
| `compact(max_items)` | 压缩并写回 |
| `export_sync_batch()` | 同步批次（allowed / blocked） |

## 快速运行

```bash
python -m memory_system_py.demo
```

## 测试

```bash
pytest -q
```

注释规范见 `memory-system/COMMENT_TEMPLATE.md`。
