# harness_py

<!-- 注释：本目录用于把“混乱物理工作台”转换为 memory_system_py 可消费的记忆输入。 -->

## 功能链路

1. `scanner.py`：扫描工作区并应用 `.gitignore` 与内置二进制过滤
2. `reader.py`：读取异构文件，生成 `MemoryInputDocument`，并进行“防爆”截断/摘要
3. `triage.py`：给文件打一句话 gist 并聚类成任务簇
4. `bridge.py`：把任务簇写入 `memory_system_py`
5. `harness.py`：提供一体化执行入口

## 快速使用

```python
from harness_py import MessyWorkspaceHarness

h = MessyWorkspaceHarness(
    workspace_root="./some_messy_workspace",
    memory_root="./memory_system_py/.runtime_harness",
)
result = h.run(limit_files=200)
print(result.scanned_count, result.cluster_count, result.memory_written)
```

## Harness IR

`HeterogeneousReader.read_document()` 会把文件系统输入转换为 `MemoryInputDocument`：

- `doc_id`：基于相对路径的稳定文档 ID
- `source_kind`：`code` / `markdown` / `csv` / `config` / `sql` / `shell` / `text` / `unknown`
- `content`：LLM 可读文本或结构化预览
- `content_sha256`：当前 LLM 可读内容 hash
- `byte_size` / `line_count`：原始大小与可读内容行数
- `truncated` / `summary_hint`：截断与读取策略标记

`MessyWorkspaceHarness.run()` 返回 `documents` 与 `document_count`，后续 formation operator 可以直接消费这层 IR。

## Dry-run 预览

```python
result = h.run(limit_files=200, dry_run=True, preview_limit=5)
print(result.planned_memory_count)
for item in result.memory_previews:
    print(item.category, item.source, item.content_preview)
```

`dry_run=True` 不会创建或写入 `memory_root`，适合先审计混乱工作台会产生哪些记忆节点。

## 稳定 ID 与重复运行

Harness 写入的 cluster/file 记忆使用稳定 `harness_*` ID。重复运行同一工作区时，已有记忆会被 upsert 更新，而不是重复追加。

文件记忆 metadata 会记录：

- `relative_path`
- `doc_id`
- `source_kind`
- `extension`
- `truncated`
- `summary_hint`
- `content_sha256`

## 集成校验

```bash
python -m harness_py.validate --workspace ./some_messy_workspace --limit-files 200
python -m harness_py.validate --workspace ./some_messy_workspace --memory-root ./runtime_memory --write
```

## 可替换的 LLM 打标器

- `MessyWorkspaceHarness(..., gist_fn=your_func)` 支持注入外部模型函数
- 函数签名：`(FileSnippet) -> str`
