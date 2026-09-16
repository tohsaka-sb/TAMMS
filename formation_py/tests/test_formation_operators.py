"""注释：formation operator 基础层测试，覆盖 gist、压缩、聚类与 registry。"""

from pathlib import Path

from formation_py import (
    AMemNoteOperator,
    CAMClusterOperator,
    ClusterFusionOperator,
    EvidencePreservingCompressor,
    GistOperator,
    HeuristicGistOperator,
    LinkEvolutionOperator,
    ProceduralExtractor,
    SemanticFactExtractor,
    SourceKindClusterOperator,
    create_default_registry,
)
from formation_py import FormationPipeline, FormationPlan, FormationStep, debug_plan, graph_plan, overview_plan
from harness_py.reader import HeterogeneousReader
from harness_py.scanner import FileScanner


def _documents(tmp_path: Path):
    """注释：构造跨类型 Harness IR 文档。"""
    (tmp_path / "main.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Project Notes\n\nrun tests with pytest\n", encoding="utf-8")
    (tmp_path / "config.json").write_text('{"model": "demo", "enabled": true}\n', encoding="utf-8")
    scanned = FileScanner(tmp_path).scan()
    reader = HeterogeneousReader()
    return [reader.read_document(item) for item in scanned]


def test_heuristic_gist_operator_outputs_traceable_artifacts(tmp_path: Path) -> None:
    """注释：gist 算子为每个 IR 文档输出可追踪 artifact。"""
    documents = _documents(tmp_path)
    result = HeuristicGistOperator().run(documents)

    assert result.ok is True
    assert result.trace is not None
    assert result.trace.operator_name == "HeuristicGistOperator"
    assert len(result.artifacts) == len(documents)
    assert all(artifact.artifact_type == "gist" for artifact in result.artifacts)
    assert all(artifact.source_ids for artifact in result.artifacts)
    assert all(artifact.evidence for artifact in result.artifacts)

    formal = GistOperator().run(documents)
    assert formal.trace is not None
    assert formal.trace.operator_name == "GistOperator"


def test_evidence_preserving_compressor_marks_compressed_long_document(tmp_path: Path) -> None:
    """注释：压缩算子对长文档保留头尾证据 span。"""
    target = tmp_path / "big.md"
    target.write_text("\n".join(f"line-{idx}" for idx in range(100)), encoding="utf-8")
    document = HeterogeneousReader(max_chars=10000).read_document(FileScanner(tmp_path).scan()[0])

    result = EvidencePreservingCompressor(max_chars=120, head_lines=3, tail_lines=2).run([document])

    assert result.ok is True
    artifact = result.artifacts[0]
    assert artifact.artifact_type == "compressed_summary"
    assert artifact.metadata["compressed"] == "true"
    assert len(artifact.evidence) == 2
    assert "line-0" in artifact.content
    assert "line-99" in artifact.content


def test_evidence_preserving_compressor_keeps_salient_middle_code(tmp_path: Path) -> None:
    """注释：压缩算子不仅保留头尾，也保留中间关键代码 span。"""
    target = tmp_path / "model.py"
    lines = [f"# filler {idx}" for idx in range(40)]
    lines.insert(22, "def important_model_step(x):")
    lines.insert(23, "    return x + 1")
    target.write_text("\n".join(lines), encoding="utf-8")
    document = HeterogeneousReader(max_chars=10000).read_document(FileScanner(tmp_path).scan()[0])

    result = EvidencePreservingCompressor(max_chars=120, head_lines=2, tail_lines=2).run([document])

    artifact = result.artifacts[0]
    assert artifact.metadata["compression_policy"] == "head_salient_tail"
    assert "def important_model_step" in artifact.content
    assert len(artifact.evidence) >= 3


def test_source_kind_cluster_operator_groups_documents(tmp_path: Path) -> None:
    """注释：规则聚类算子按 source_kind 分桶。"""
    documents = _documents(tmp_path)

    result = SourceKindClusterOperator().run(documents)

    assert result.ok is True
    cluster_keys = {artifact.metadata["cluster_key"] for artifact in result.artifacts}
    assert {"code", "markdown"}.issubset(cluster_keys)


def test_amem_note_operator_outputs_keywords_tags_and_link_hints(tmp_path: Path) -> None:
    """注释：A-Mem note 算子输出 context/keywords/tags/link hints。"""
    documents = _documents(tmp_path)

    result = AMemNoteOperator().run(documents)

    assert result.ok is True
    assert len(result.artifacts) == len(documents)
    assert all(artifact.artifact_type == "amem_note" for artifact in result.artifacts)
    assert all("Keywords:" in artifact.content for artifact in result.artifacts)
    assert all("tags" in artifact.metadata for artifact in result.artifacts)


def test_cam_cluster_operator_outputs_overlapping_cluster_shapes(tmp_path: Path) -> None:
    """注释：CAM 聚类算子输出 type/path 两类重叠 cluster。"""
    documents = _documents(tmp_path)

    result = CAMClusterOperator().run(documents)

    assert result.ok is True
    assert all(artifact.artifact_type == "cam_cluster" for artifact in result.artifacts)
    cluster_keys = {artifact.metadata["cluster_key"] for artifact in result.artifacts}
    assert "path:root" in cluster_keys
    assert any(key.startswith("type:") for key in cluster_keys)


def test_procedural_extractor_finds_run_and_test_steps(tmp_path: Path) -> None:
    """注释：过程抽取算子从 README 等文本中抽取运行/测试步骤。"""
    documents = _documents(tmp_path)

    result = ProceduralExtractor().run(documents)

    assert result.ok is True
    assert result.artifacts
    assert all(artifact.artifact_type == "procedural_memory" for artifact in result.artifacts)
    assert any("pytest" in artifact.content for artifact in result.artifacts)


def test_empty_operator_result_sets_empty_status(tmp_path: Path) -> None:
    """注释：无产物不是失败，但应以 empty status 暴露给 pipeline。"""
    (tmp_path / "plain.txt").write_text("ordinary notes without command hints\n", encoding="utf-8")
    document = HeterogeneousReader().read_document(FileScanner(tmp_path).scan()[0])

    result = ProceduralExtractor().run([document])

    assert result.ok is True
    assert result.artifacts == []
    assert result.status == "empty"


def test_semantic_fact_extractor_finds_code_and_config_facts(tmp_path: Path) -> None:
    """注释：事实抽取算子提取代码函数和配置 key。"""
    documents = _documents(tmp_path)

    result = SemanticFactExtractor().run(documents)

    assert result.ok is True
    assert result.artifacts
    contents = "\n".join(artifact.content for artifact in result.artifacts)
    assert "function: run" in contents
    assert "config key: model" in contents


def test_link_evolution_operator_suggests_pairwise_links(tmp_path: Path) -> None:
    """注释：链接演化算子为相关文档生成关系建议。"""
    documents = _documents(tmp_path)

    result = LinkEvolutionOperator().run(documents)

    assert result.ok is True
    assert result.artifacts
    assert all(artifact.artifact_type == "link_suggestion" for artifact in result.artifacts)
    assert all(len(artifact.source_ids) == 2 for artifact in result.artifacts)
    assert all(isinstance(artifact.metadata["relation"], dict) for artifact in result.artifacts)
    assert all("relation_type" in artifact.metadata["relation"] for artifact in result.artifacts)


def test_cluster_fusion_operator_outputs_workspace_summary(tmp_path: Path) -> None:
    """注释：cluster fusion 算子输出项目级高层摘要。"""
    documents = _documents(tmp_path)

    result = ClusterFusionOperator().run(documents)

    assert result.ok is True
    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.artifact_type == "cluster_fusion_summary"
    assert artifact.metadata["document_count"] == str(len(documents))


def test_operator_specs_expose_controller_metadata() -> None:
    """注释：operator spec 提供 meta-controller 选择所需的能力信息。"""
    registry = create_default_registry()

    for name in registry.names():
        operator = registry.get(name)
        assert operator.spec.name == name
        assert operator.spec.output_type == operator.output_type
        assert operator.spec.cost_level in {"low", "medium", "high"}
        assert operator.spec.requires_llm is False


def test_default_registry_exposes_version_3_operators() -> None:
    """注释：默认 registry 暴露 operator_list 中的完整确定性算子集合。"""
    registry = create_default_registry()

    assert registry.names() == [
        "AMemNoteOperator",
        "CAMClusterOperator",
        "ClusterFusionOperator",
        "EvidencePreservingCompressor",
        "GistOperator",
        "LinkEvolutionOperator",
        "ProceduralExtractor",
        "SemanticFactExtractor",
        "SourceKindClusterOperator",
    ]
    assert registry.get("GistOperator").name == "GistOperator"


def test_formation_pipeline_runs_overview_plan(tmp_path: Path) -> None:
    """注释：pipeline 可按固定 overview plan 顺序执行多个 operator。"""
    documents = _documents(tmp_path)

    run = FormationPipeline().run(documents, overview_plan())

    assert run.ok is True
    assert run.status == "ok"
    assert run.plan.plan_id == "overview_v1"
    assert run.metrics["step_count"] == 4
    assert len(run.traces) == 4
    artifact_types = {artifact.artifact_type for artifact in run.artifacts}
    assert {"gist", "semantic_fact", "cam_cluster", "cluster_fusion_summary"}.issubset(artifact_types)


def test_formation_pipeline_runs_debug_and_graph_plans(tmp_path: Path) -> None:
    """注释：debug/graph 固定计划能生成不同 artifact 组合。"""
    documents = _documents(tmp_path)
    pipeline = FormationPipeline()

    debug_run = pipeline.run(documents, debug_plan())
    graph_run = pipeline.run(documents, graph_plan())

    assert debug_run.ok is True
    assert graph_run.ok is True
    assert any(artifact.artifact_type == "compressed_summary" for artifact in debug_run.artifacts)
    assert any(artifact.artifact_type == "amem_note" for artifact in graph_run.artifacts)


def test_formation_pipeline_stops_on_missing_required_operator(tmp_path: Path) -> None:
    """注释：required step 不存在时 pipeline 返回 failed run。"""
    documents = _documents(tmp_path)
    plan = FormationPlan(
        plan_id="broken",
        target_schema="test",
        task_intent="test",
        steps=[FormationStep("MissingOperator", required=True)],
    )

    run = FormationPipeline().run(documents, plan)

    assert run.ok is False
    assert run.status == "failed"
    assert run.errors == ["unknown operator: MissingOperator"]
