"""注释：扫描与读取模块测试，验证过滤与截断策略。"""

from pathlib import Path

from harness_py.reader import HeterogeneousReader
from harness_py.scanner import FileScanner


def test_scanner_respects_gitignore_and_binary_filter(tmp_path: Path) -> None:
    """注释：验证 .gitignore 和二进制扩展过滤生效。"""
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    (tmp_path / "ok.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    scanner = FileScanner(tmp_path)
    files = scanner.scan()
    rels = {f.relative_path for f in files}

    assert "ok.py" in rels
    assert "ignored.txt" not in rels
    assert "image.png" not in rels


def test_reader_truncates_long_text(tmp_path: Path) -> None:
    """注释：验证超长文本会被头尾切片并插入 Truncated 标记。"""
    target = tmp_path / "big.md"
    content = "\n".join(f"line-{i}" for i in range(1000))
    target.write_text(content, encoding="utf-8")

    scanner = FileScanner(tmp_path)
    scanned = scanner.scan()[0]
    reader = HeterogeneousReader(max_chars=200, head_lines=3, tail_lines=3)
    snippet = reader.read(scanned)

    assert snippet.truncated is True
    assert "...[Truncated]..." in snippet.content


def test_reader_builds_memory_input_document_ir(tmp_path: Path) -> None:
    """注释：验证 reader 会先生成统一 Harness IR，再兼容转换为 FileSnippet。"""
    target = tmp_path / "main.py"
    target.write_text("def run():\n    return 'ok'\n", encoding="utf-8")

    scanned = FileScanner(tmp_path).scan()[0]
    document = HeterogeneousReader().read_document(scanned)
    snippet = document.to_snippet()

    assert document.doc_id.startswith("harness_doc_")
    assert document.source_kind == "code"
    assert document.relative_path == "main.py"
    assert document.content_sha256
    assert document.byte_size > 0
    assert document.line_count == 2
    assert document.metadata["llm_ready"] == "true"
    assert snippet.document_id == document.doc_id
    assert snippet.metadata["content_sha256"] == document.content_sha256
