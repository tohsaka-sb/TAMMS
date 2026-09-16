"""
模块: formation_py.operators
职责: 提供记忆形成算子的基础协议与确定性初版实现。
输入: List[MemoryInputDocument] 与可选 context。
输出: FormationResult。
副作用: 无；不调用网络、不写存储。
失败处理: 内置算子捕获可预期空输入并返回 ok=False 或空产物。
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Dict, List, Protocol

from harness_py import MemoryInputDocument

from .types import EvidenceSpan, FormationArtifact, FormationContext, FormationResult, FormationTrace, OperatorSpec


class FormationOperator(Protocol):
    """注释：所有记忆形成算子需要实现的最小接口。"""

    name: str
    output_type: str
    spec: OperatorSpec

    def run(self, documents: List[MemoryInputDocument], context: FormationContext | None = None) -> FormationResult:
        """注释：执行算子并返回结构化结果。"""


class GistOperator:
    """注释：低成本 gist 算子，基于 IR 类型、路径和内容片段形成文件级摘要。"""

    name = "GistOperator"
    output_type = "gist"
    spec = OperatorSpec(
        name=name,
        output_type=output_type,
        input_scope="documents",
        supported_intents=["overview", "summarize", "preview", "locate"],
        cost_level="low",
        produces_schema=["flat"],
    )

    def run(self, documents: List[MemoryInputDocument], context: FormationContext | None = None) -> FormationResult:
        """注释：为每个 MemoryInputDocument 生成一个 gist artifact。"""
        artifacts: List[FormationArtifact] = []
        for document in documents:
            content = self._gist(document)
            artifacts.append(
                FormationArtifact(
                    artifact_type=self.output_type,
                    content=content,
                    source_ids=[document.doc_id],
                    evidence=[_head_span(document)],
                    metadata={
                        "relative_path": document.relative_path,
                        "source_kind": document.source_kind,
                        "confidence_policy": "fixed_placeholder",
                    },
                    confidence=0.62,
                )
            )
        return FormationResult(
            ok=True,
            artifacts=artifacts,
            trace=FormationTrace(
                operator_name=self.name,
                input_ids=[document.doc_id for document in documents],
                strategy="heuristic_path_type_content",
            ),
            status="ok" if artifacts else "empty",
        )

    def _gist(self, document: MemoryInputDocument) -> str:
        """注释：基于 source_kind 生成一句话文件摘要。"""
        path = document.relative_path
        if document.source_kind == "code":
            facts = _code_facts(document.content)[:4]
            detail = "; ".join(facts) if facts else f"{document.line_count} readable lines"
            return f"{path}: code file; {detail}"
        if document.source_kind == "markdown":
            heading = _first_markdown_heading(document.content)
            if heading:
                return f"{path}: markdown document about {heading}"
            return f"{path}: markdown document"
        if document.source_kind == "csv":
            return f"{path}: csv-like structured data preview"
        if document.source_kind == "config":
            return f"{path}: configuration file"
        if document.source_kind in {"sql", "shell"}:
            return f"{path}: {document.source_kind} script"
        return f"{path}: {document.source_kind} workspace document"


class HeuristicGistOperator(GistOperator):
    """注释：兼容旧命名；行为等同 GistOperator。"""

    name = "HeuristicGistOperator"
    spec = OperatorSpec(
        name=name,
        output_type=GistOperator.output_type,
        input_scope="documents",
        supported_intents=["overview", "summarize", "preview", "locate"],
        cost_level="low",
        produces_schema=["flat"],
    )


class AMemNoteOperator:
    """注释：A-Mem 风格 note 构造算子，生成 context/keywords/tags/link hints。"""

    name = "AMemNoteOperator"
    output_type = "amem_note"
    spec = OperatorSpec(
        name=name,
        output_type=output_type,
        input_scope="documents",
        supported_intents=["overview", "link", "evolve", "graph"],
        cost_level="low",
        produces_schema=["graph", "hybrid"],
    )

    def run(self, documents: List[MemoryInputDocument], context: FormationContext | None = None) -> FormationResult:
        """注释：每个文档生成一个原子 note artifact。"""
        artifacts: List[FormationArtifact] = []
        for document in documents:
            keywords = _keywords(document.content, limit=8)
            tags = _tags_for(document)
            related = _related_paths(document, documents)
            content = (
                f"Context: {document.relative_path} ({document.source_kind})\n"
                f"Keywords: {', '.join(keywords)}\n"
                f"Tags: {', '.join(tags)}\n"
                f"LinkHints: {', '.join(related) if related else 'none'}"
            )
            artifacts.append(
                FormationArtifact(
                    artifact_type=self.output_type,
                    content=content,
                    source_ids=[document.doc_id],
                    evidence=[_head_span(document)],
                    metadata={
                        "relative_path": document.relative_path,
                        "source_kind": document.source_kind,
                        "keywords": keywords,
                        "tags": tags,
                        "link_hint_count": str(len(related)),
                        "confidence_policy": "fixed_placeholder",
                    },
                    confidence=0.57,
                )
            )
        return FormationResult(
            ok=True,
            artifacts=artifacts,
            trace=FormationTrace(
                operator_name=self.name,
                input_ids=[document.doc_id for document in documents],
                strategy="amem_like_note_construction_heuristic",
            ),
            status="ok" if artifacts else "empty",
        )


class CAMClusterOperator:
    """注释：CAM 风格重叠聚类骨架，按 source_kind 与路径目录生成多视角 cluster。"""

    name = "CAMClusterOperator"
    output_type = "cam_cluster"
    spec = OperatorSpec(
        name=name,
        output_type=output_type,
        input_scope="documents",
        supported_intents=["overview", "cluster", "summarize", "graph"],
        cost_level="low",
        produces_schema=["hierarchical", "hybrid"],
    )

    def run(self, documents: List[MemoryInputDocument], context: FormationContext | None = None) -> FormationResult:
        """注释：输出 source_kind 与目录两类重叠 cluster。"""
        buckets: dict[str, list[MemoryInputDocument]] = defaultdict(list)
        for document in documents:
            buckets[f"type:{document.source_kind}"].append(document)
            buckets[f"path:{_top_path(document.relative_path)}"].append(document)

        artifacts: List[FormationArtifact] = []
        for key, items in sorted(buckets.items()):
            if not items:
                continue
            paths = ", ".join(document.relative_path for document in items[:6])
            artifacts.append(
                FormationArtifact(
                    artifact_type=self.output_type,
                    content=f"CAMCluster[{key}]: {len(items)} document(s): {paths}",
                    source_ids=[document.doc_id for document in items],
                    evidence=[_head_span(document) for document in items[:3]],
                    metadata={
                        "cluster_key": key,
                        "cluster_size": str(len(items)),
                        "overlap_policy": "source_kind+top_path",
                        "confidence_policy": "fixed_placeholder",
                    },
                    confidence=0.56,
                )
            )
        return FormationResult(
            ok=True,
            artifacts=artifacts,
            trace=FormationTrace(
                operator_name=self.name,
                input_ids=[document.doc_id for document in documents],
                strategy="cam_like_overlapping_buckets",
            ),
            status="ok" if artifacts else "empty",
        )


class EvidencePreservingCompressor:
    """注释：保留头尾证据的确定性压缩算子，适合长文本进入 LLM 前降噪。"""

    name = "EvidencePreservingCompressor"
    output_type = "compressed_summary"
    spec = OperatorSpec(
        name=name,
        output_type=output_type,
        input_scope="documents",
        supported_intents=["summarize", "compress", "debug", "overview"],
        cost_level="low",
        produces_schema=["flat"],
    )

    def __init__(self, max_chars: int = 1200, head_lines: int = 12, tail_lines: int = 8) -> None:
        self.max_chars = max_chars
        self.head_lines = head_lines
        self.tail_lines = tail_lines

    def run(self, documents: List[MemoryInputDocument], context: FormationContext | None = None) -> FormationResult:
        """注释：对每个文档生成压缩产物，短文档也会输出摘要壳。"""
        artifacts: List[FormationArtifact] = []
        for document in documents:
            summary, evidence = self._compress(document)
            artifacts.append(
                FormationArtifact(
                    artifact_type=self.output_type,
                    content=summary,
                    source_ids=[document.doc_id],
                    evidence=evidence,
                    metadata={
                        "relative_path": document.relative_path,
                        "source_kind": document.source_kind,
                        "compressed": str(len(document.content) > self.max_chars).lower(),
                        "compression_policy": "head_salient_tail",
                        "confidence_policy": "fixed_placeholder",
                    },
                    confidence=0.58,
                )
            )
        return FormationResult(
            ok=True,
            artifacts=artifacts,
            trace=FormationTrace(
                operator_name=self.name,
                input_ids=[document.doc_id for document in documents],
                strategy="deterministic_head_salient_tail",
            ),
            status="ok" if artifacts else "empty",
        )

    def _compress(self, document: MemoryInputDocument) -> tuple[str, List[EvidenceSpan]]:
        """注释：生成头尾压缩摘要和证据 span。"""
        lines = document.content.splitlines()
        if len(document.content) <= self.max_chars:
            return (
                f"[Path={document.relative_path}] [Kind={document.source_kind}] [Mode=full]\n{document.content}",
                [_head_span(document)],
            )

        spans = _dedupe_spans(
            [
                (1, len(lines[: self.head_lines])),
                *_salient_spans(document, max_spans=6),
                (max(1, len(lines) - self.tail_lines + 1), len(lines)) if self.tail_lines else (0, 0),
            ]
        )
        sections = []
        evidence = []
        for start_line, end_line in spans:
            if start_line <= 0 or end_line <= 0:
                continue
            evidence_span = _span(document, start_line, end_line)
            evidence.append(evidence_span)
            sections.append(f"[Lines {start_line}-{end_line}]\n{evidence_span.excerpt}")
        summary = (
            f"[Path={document.relative_path}] [Kind={document.source_kind}] [Mode=compressed]\n"
            + "\n\n...[EvidencePreservingCompressor selected evidence spans]...\n\n".join(sections)
        )
        return summary, evidence


class SourceKindClusterOperator:
    """注释：规则聚类算子，先按 source_kind 分桶，为 CAM 式聚类预留接口形状。"""

    name = "SourceKindClusterOperator"
    output_type = "cluster"
    spec = OperatorSpec(
        name=name,
        output_type=output_type,
        input_scope="documents",
        supported_intents=["overview", "cluster"],
        cost_level="low",
        produces_schema=["flat"],
    )

    def run(self, documents: List[MemoryInputDocument], context: FormationContext | None = None) -> FormationResult:
        """注释：按 source_kind 聚合文档，输出 cluster artifact。"""
        buckets: dict[str, list[MemoryInputDocument]] = defaultdict(list)
        for document in documents:
            buckets[document.source_kind].append(document)

        artifacts: List[FormationArtifact] = []
        for source_kind, items in sorted(buckets.items()):
            paths = ", ".join(document.relative_path for document in items[:5])
            if len(items) > 5:
                paths += f", +{len(items) - 5} more"
            artifacts.append(
                FormationArtifact(
                    artifact_type=self.output_type,
                    content=f"Cluster[{source_kind}]: {len(items)} document(s): {paths}",
                    source_ids=[document.doc_id for document in items],
                    evidence=[_head_span(document) for document in items[:3]],
                    metadata={
                        "cluster_key": source_kind,
                        "cluster_size": str(len(items)),
                        "confidence_policy": "fixed_placeholder",
                    },
                    confidence=0.55,
                )
            )
        return FormationResult(
            ok=True,
            artifacts=artifacts,
            trace=FormationTrace(
                operator_name=self.name,
                input_ids=[document.doc_id for document in documents],
                strategy="source_kind_bucket",
            ),
            status="ok" if artifacts else "empty",
        )


class ProceduralExtractor:
    """注释：提取运行、测试、安装、调试等过程性记忆的规则算子。"""

    name = "ProceduralExtractor"
    output_type = "procedural_memory"

    PROCEDURE_PATTERNS = (
        "install",
        "run",
        "test",
        "pytest",
        "build",
        "deploy",
        "debug",
        "reproduce",
        "pip ",
        "npm ",
        "docker",
        "conda",
        "python -m",
        "make",
        "cmake",
        "ctest",
        "bash",
        " sh ",
        "./",
        "go test",
        "cargo",
        "mvn",
        "gradle",
        "requirements.txt",
        "environment.yml",
    )
    spec = OperatorSpec(
        name=name,
        output_type=output_type,
        input_scope="documents",
        supported_intents=["reproduce", "debug", "run", "test"],
        cost_level="low",
        produces_schema=["procedural"],
    )

    def run(self, documents: List[MemoryInputDocument], context: FormationContext | None = None) -> FormationResult:
        """注释：从文档行中抽取过程性提示。"""
        artifacts: List[FormationArtifact] = []
        for document in documents:
            matches = _matching_lines(document, self.PROCEDURE_PATTERNS)
            if not matches:
                continue
            steps = [f"- line {line_no}: {line.strip()}" for line_no, line in matches[:8]]
            artifacts.append(
                FormationArtifact(
                    artifact_type=self.output_type,
                    content=f"Procedure hints from {document.relative_path}:\n" + "\n".join(steps),
                    source_ids=[document.doc_id],
                    evidence=[_span(document, line_no, line_no) for line_no, _ in matches[:5]],
                    metadata={
                        "relative_path": document.relative_path,
                        "step_count": str(len(matches)),
                        "procedure_type": _procedure_type(matches),
                        "confidence_policy": "fixed_placeholder",
                    },
                    confidence=0.6,
                )
            )
        return FormationResult(
            ok=True,
            artifacts=artifacts,
            trace=FormationTrace(
                operator_name=self.name,
                input_ids=[document.doc_id for document in documents],
                strategy="procedure_keyword_lines",
            ),
            status="ok" if artifacts else "empty",
        )


class SemanticFactExtractor:
    """注释：从代码、Markdown、配置和 CSV 预览中抽取确定性事实记忆。"""

    name = "SemanticFactExtractor"
    output_type = "semantic_fact"
    spec = OperatorSpec(
        name=name,
        output_type=output_type,
        input_scope="documents",
        supported_intents=["overview", "locate", "debug", "summarize"],
        cost_level="low",
        produces_schema=["flat"],
    )

    def run(self, documents: List[MemoryInputDocument], context: FormationContext | None = None) -> FormationResult:
        """注释：按 source_kind 抽取事实 artifact。"""
        artifacts: List[FormationArtifact] = []
        for document in documents:
            facts = self._facts(document)
            if not facts:
                continue
            artifacts.append(
                FormationArtifact(
                    artifact_type=self.output_type,
                    content=f"Facts from {document.relative_path}:\n" + "\n".join(f"- {fact}" for fact in facts),
                    source_ids=[document.doc_id],
                    evidence=[_head_span(document)],
                    metadata={
                        "relative_path": document.relative_path,
                        "source_kind": document.source_kind,
                        "fact_count": str(len(facts)),
                        "confidence_policy": "fixed_placeholder",
                    },
                    confidence=0.61,
                )
            )
        return FormationResult(
            ok=True,
            artifacts=artifacts,
            trace=FormationTrace(
                operator_name=self.name,
                input_ids=[document.doc_id for document in documents],
                strategy="source_kind_structural_facts",
            ),
            status="ok" if artifacts else "empty",
        )

    def _facts(self, document: MemoryInputDocument) -> List[str]:
        """注释：根据 IR 类型选择轻量事实抽取规则。"""
        if document.source_kind == "code":
            return _code_facts(document.content)
        if document.source_kind == "markdown":
            headings = [_clean_heading(line) for line in document.content.splitlines() if line.strip().startswith("#")]
            return [f"markdown heading: {heading}" for heading in headings[:8] if heading]
        if document.source_kind == "csv":
            for line in document.content.splitlines():
                if line.startswith("CSV Header:"):
                    continue
                if line.strip():
                    return [f"csv header/preview: {line.strip()}"]
        if document.source_kind == "config":
            return _config_facts(document.content)
        return []


class LinkEvolutionOperator:
    """注释：A-Mem 风格链接演化骨架，给相似文档生成关系建议。"""

    name = "LinkEvolutionOperator"
    output_type = "link_suggestion"
    spec = OperatorSpec(
        name=name,
        output_type=output_type,
        input_scope="documents",
        supported_intents=["link", "graph", "evolve", "debug"],
        cost_level="medium",
        produces_schema=["graph", "hybrid"],
    )

    def __init__(self, top_k_per_document: int = 5) -> None:
        self.top_k_per_document = top_k_per_document

    def run(self, documents: List[MemoryInputDocument], context: FormationContext | None = None) -> FormationResult:
        """注释：基于 source_kind、路径和关键词重叠生成链接建议。"""
        artifacts: List[FormationArtifact] = []
        seen: set[tuple[str, str]] = set()
        for left in documents:
            candidates = []
            for right in documents:
                if left.doc_id == right.doc_id:
                    continue
                relation = self._link_relation(left, right)
                if relation["score"] <= 0:
                    continue
                candidates.append((relation["score"], right, relation))
            candidates.sort(key=lambda item: item[0], reverse=True)
            for _, right, relation in candidates[: self.top_k_per_document]:
                key = tuple(sorted([left.doc_id, right.doc_id]))
                if key in seen:
                    continue
                seen.add(key)
                artifacts.append(
                    FormationArtifact(
                        artifact_type=self.output_type,
                        content=f"{left.relative_path} -> {right.relative_path}: {relation['relation_type']}={relation['relation_value']}",
                        source_ids=[left.doc_id, right.doc_id],
                        evidence=[_head_span(left), _head_span(right)],
                        metadata={
                            "left_path": left.relative_path,
                            "right_path": right.relative_path,
                            "relation": relation,
                            "confidence_policy": "heuristic_link_score",
                        },
                        confidence=float(relation["score"]),
                    )
                )
        return FormationResult(
            ok=True,
            artifacts=artifacts,
            trace=FormationTrace(
                operator_name=self.name,
                input_ids=[document.doc_id for document in documents],
                strategy="candidate_topk_link_evolution",
            ),
            status="ok" if artifacts else "empty",
        )

    def _link_relation(self, left: MemoryInputDocument, right: MemoryInputDocument) -> Dict[str, object]:
        """注释：给两个文档判断一个结构化关系。"""
        if left.source_kind == right.source_kind:
            return {"relation_type": "same_source_kind", "relation_value": left.source_kind, "score": 0.52}
        if _top_path(left.relative_path) == _top_path(right.relative_path):
            return {"relation_type": "same_top_path", "relation_value": _top_path(left.relative_path), "score": 0.5}
        overlap = set(_keywords(left.content, limit=12)).intersection(_keywords(right.content, limit=12))
        if len(overlap) >= 2:
            return {"relation_type": "keyword_overlap", "relation_value": sorted(overlap)[:4], "score": min(0.8, 0.45 + len(overlap) * 0.05)}
        return {"relation_type": "", "relation_value": "", "score": 0.0}


class ClusterFusionOperator:
    """注释：把文档集合融合成项目级高层摘要的 CAM/GraphRAG 风格算子。"""

    name = "ClusterFusionOperator"
    output_type = "cluster_fusion_summary"
    spec = OperatorSpec(
        name=name,
        output_type=output_type,
        input_scope="documents",
        supported_intents=["overview", "summarize", "cluster"],
        cost_level="low",
        produces_schema=["hierarchical", "hybrid"],
    )

    def run(self, documents: List[MemoryInputDocument], context: FormationContext | None = None) -> FormationResult:
        """注释：输出一个按类型统计的高层 summary artifact。"""
        if not documents:
            return FormationResult(
                ok=True,
                artifacts=[],
                trace=FormationTrace(operator_name=self.name, strategy="empty_input"),
            )
        counts = Counter(document.source_kind for document in documents)
        count_text = ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items()))
        sample_paths = ", ".join(document.relative_path for document in documents[:8])
        artifact = FormationArtifact(
            artifact_type=self.output_type,
            content=(
                f"Workspace fusion summary: {len(documents)} document(s); "
                f"type distribution: {count_text}; sample paths: {sample_paths}"
            ),
            source_ids=[document.doc_id for document in documents],
            evidence=[_head_span(document) for document in documents[:5]],
            metadata={
                "document_count": str(len(documents)),
                "type_distribution": count_text,
                "confidence_policy": "fixed_placeholder",
            },
            confidence=0.54,
        )
        return FormationResult(
            ok=True,
            artifacts=[artifact],
            trace=FormationTrace(
                operator_name=self.name,
                input_ids=[document.doc_id for document in documents],
                strategy="type_distribution_fusion",
            ),
        )


class OperatorRegistry:
    """注释：轻量算子注册表，供后续 meta-controller 按名称选择算子。"""

    def __init__(self, operators: List[FormationOperator] | None = None) -> None:
        self._operators: Dict[str, FormationOperator] = {}
        for operator in operators or []:
            self.register(operator)

    def register(self, operator: FormationOperator) -> None:
        """注释：按 operator.name 注册或覆盖算子。"""
        self._operators[operator.name] = operator

    def get(self, name: str) -> FormationOperator:
        """注释：按名称取出算子；不存在时抛 KeyError。"""
        return self._operators[name]

    def names(self) -> List[str]:
        """注释：返回已注册算子名。"""
        return sorted(self._operators)


def create_default_registry() -> OperatorRegistry:
    """注释：创建 Version 3 默认确定性算子集合。"""
    return OperatorRegistry(
        [
            GistOperator(),
            AMemNoteOperator(),
            CAMClusterOperator(),
            EvidencePreservingCompressor(),
            ProceduralExtractor(),
            SemanticFactExtractor(),
            LinkEvolutionOperator(),
            ClusterFusionOperator(),
            SourceKindClusterOperator(),
        ]
    )


def _first_markdown_heading(content: str) -> str:
    """注释：提取首个 Markdown 标题文本。"""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""


def _clean_heading(line: str) -> str:
    """注释：清理 Markdown 标题标记。"""
    return line.strip().lstrip("#").strip()


def _keywords(content: str, limit: int = 8) -> List[str]:
    """注释：简单关键词抽取，过滤短词和常见结构词。"""
    stopwords = {
        "from",
        "import",
        "return",
        "with",
        "this",
        "that",
        "true",
        "false",
        "none",
        "def",
        "class",
        "the",
        "and",
        "for",
        "you",
    }
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", content.lower())
    counts = Counter(word for word in words if word not in stopwords)
    return [word for word, _ in counts.most_common(limit)]


def _tags_for(document: MemoryInputDocument) -> List[str]:
    """注释：为 AMem note 生成基础 tags。"""
    tags = [document.source_kind]
    if document.truncated:
        tags.append("truncated")
    if "test" in document.relative_path.lower():
        tags.append("test")
    return tags


def _related_paths(document: MemoryInputDocument, documents: List[MemoryInputDocument]) -> List[str]:
    """注释：基于同类型和同目录生成链接提示路径。"""
    related = []
    for other in documents:
        if other.doc_id == document.doc_id:
            continue
        if other.source_kind == document.source_kind or _top_path(other.relative_path) == _top_path(document.relative_path):
            related.append(other.relative_path)
        if len(related) >= 5:
            break
    return related


def _top_path(relative_path: str) -> str:
    """注释：取相对路径的顶层目录；根文件归为 root。"""
    parts = relative_path.split("/")
    return parts[0] if len(parts) > 1 else "root"


def _matching_lines(document: MemoryInputDocument, patterns: tuple[str, ...]) -> List[tuple[int, str]]:
    """注释：找出包含任一 pattern 的行。"""
    matches: List[tuple[int, str]] = []
    for index, line in enumerate(document.content.splitlines(), start=1):
        lower = line.lower()
        if any(pattern in lower for pattern in patterns):
            matches.append((index, line))
    return matches


def _salient_spans(document: MemoryInputDocument, max_spans: int = 6, radius: int = 1) -> List[tuple[int, int]]:
    """注释：按文件类型保留关键行附近证据，补足 head/tail 压缩的盲区。"""
    lines = document.content.splitlines()
    spans: List[tuple[int, int]] = []
    for index, line in enumerate(lines, start=1):
        if _is_salient_line(document, line):
            start = max(1, index - radius)
            end = min(len(lines), index + radius)
            spans.append((start, end))
        if len(spans) >= max_spans:
            break
    return spans


def _is_salient_line(document: MemoryInputDocument, line: str) -> bool:
    """注释：判断一行是否值得在压缩摘要中强制保留。"""
    stripped = line.strip()
    lower = stripped.lower()
    if not stripped:
        return False
    if document.source_kind == "code":
        return (
            stripped.startswith(("import ", "from ", "def ", "class ", "#include"))
            or bool(re.match(r"^(int|void|float|double|char|bool|auto|static)\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", stripped))
            or stripped.startswith(("add_executable", "target_link_libraries"))
            or "todo" in lower
        )
    if document.source_kind == "markdown":
        return stripped.startswith("#") or stripped.startswith(("```", "- ", "* ")) or _looks_like_command(stripped)
    if document.source_kind in {"shell", "sql"}:
        return _looks_like_command(stripped) or lower.startswith(("select ", "insert ", "update ", "delete ", "create "))
    if document.source_kind == "config":
        return ":" in stripped or "=" in stripped or stripped.startswith("[")
    return any(marker in lower for marker in ("error", "traceback", "exception", "failed", "fatal", "warning"))


def _looks_like_command(line: str) -> bool:
    """注释：识别 README/Shell 中常见命令行。"""
    lower = line.lower().lstrip("$ ").strip()
    return lower.startswith(
        (
            "python ",
            "python -m",
            "pytest",
            "pip ",
            "conda ",
            "npm ",
            "make",
            "cmake",
            "docker",
            "bash ",
            "sh ",
            "./",
        )
    )


def _dedupe_spans(spans: List[tuple[int, int]]) -> List[tuple[int, int]]:
    """注释：合并重叠/相邻 span，保持原始顺序。"""
    normalized = sorted((start, end) for start, end in spans if start > 0 and end >= start)
    merged: List[tuple[int, int]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _procedure_type(matches: List[tuple[int, str]]) -> str:
    """注释：根据命中行粗分 procedure 类型。"""
    text = "\n".join(line.lower() for _, line in matches)
    if "test" in text or "pytest" in text or "ctest" in text or "go test" in text:
        return "test"
    if "install" in text or "pip " in text or "npm " in text or "conda" in text:
        return "install"
    if "build" in text or "make" in text or "cmake" in text or "gradle" in text or "mvn" in text:
        return "build"
    if "debug" in text or "traceback" in text or "error" in text:
        return "debug"
    return "run"


def _code_facts(content: str) -> List[str]:
    """注释：从代码文本中抽取 import/function/class 事实。"""
    facts: List[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            facts.append(f"dependency/import: {stripped}")
        elif stripped.startswith("#include"):
            facts.append(f"dependency/include: {stripped}")
        elif stripped.startswith("def "):
            facts.append(f"function: {stripped.split('(')[0].replace('def ', '').strip()}")
        elif stripped.startswith("class "):
            facts.append(f"class: {stripped.split('(')[0].replace('class ', '').replace(':', '').strip()}")
        elif re.match(r"^(int|void|float|double|char|bool|auto|static)\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", stripped):
            facts.append(f"function: {stripped.split('(')[0].split()[-1]}")
        elif stripped.startswith("add_executable") or stripped.startswith("target_link_libraries"):
            facts.append(f"build target: {stripped}")
        if len(facts) >= 12:
            break
    return facts


def _config_facts(content: str) -> List[str]:
    """注释：从配置文本中抽取顶层 key 或 section。"""
    facts: List[str] = []
    for line in content.splitlines():
        stripped = line.strip().strip(",")
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped:
            facts.append(f"config key: {_clean_config_key(stripped.split(':', 1)[0])}")
        elif "=" in stripped:
            facts.append(f"config key: {_clean_config_key(stripped.split('=', 1)[0])}")
        elif stripped.startswith("[") and stripped.endswith("]"):
            facts.append(f"config section: {stripped.strip('[]')}")
        if len(facts) >= 12:
            break
    return facts


def _clean_config_key(raw: str) -> str:
    """注释：清理 JSON/YAML/TOML/INI key 外层符号。"""
    return raw.strip().strip("{").strip("}").strip().strip('"').strip("'")


def _head_span(document: MemoryInputDocument) -> EvidenceSpan:
    """注释：取文档开头作为默认证据 span。"""
    return _span(document, 1, min(5, max(1, document.line_count)))


def _span(document: MemoryInputDocument, start_line: int, end_line: int) -> EvidenceSpan:
    """注释：根据行号构造证据片段。"""
    lines = document.content.splitlines()
    start_index = max(0, start_line - 1)
    end_index = max(start_index, end_line)
    excerpt = "\n".join(lines[start_index:end_index])
    return EvidenceSpan(
        doc_id=document.doc_id,
        relative_path=document.relative_path,
        start_line=start_line,
        end_line=end_line,
        excerpt=excerpt,
    )
