"""注释：Version 4 规则型 meta-controller 测试。"""

from pathlib import Path

from formation_py import FormationPipeline, RuleBasedMetaController, WorkspaceBriefBuilder
from formation_py.profile import DocumentRoleProfiler, WorkspaceProfiler
from harness_py.reader import HeterogeneousReader
from harness_py.scanner import FileScanner


def _workspace_documents(tmp_path: Path):
    """注释：构造带 README、代码、测试、配置和构建文件的工作区。"""
    (tmp_path / "README.md").write_text("# Demo\n\nRun tests with pytest.\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("import os\n\ndef run():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")
    (tmp_path / "config.json").write_text('{"enabled": true}\n', encoding="utf-8")
    (tmp_path / "CMakeLists.txt").write_text("add_executable(demo src/main.cpp)\n", encoding="utf-8")

    reader = HeterogeneousReader()
    return [reader.read_document(item) for item in FileScanner(tmp_path).scan()]


def test_workspace_profiler_detects_workspace_signals(tmp_path: Path) -> None:
    """注释：WorkspaceProfiler 输出工作区级统计和启发式信号。"""
    documents = _workspace_documents(tmp_path)

    profile = WorkspaceProfiler().profile(documents)

    assert profile.document_count == len(documents)
    assert profile.has_code is True
    assert profile.has_markdown is True
    assert profile.has_config is True
    assert profile.has_tests is True
    assert profile.has_build_files is True
    assert profile.dominant_language_hint in {"python", "cpp"}
    assert profile.organization_level in {"semi_structured", "structured"}


def test_document_role_profiler_tags_roles_and_hints(tmp_path: Path) -> None:
    """注释：DocumentRoleProfiler 给文档打角色，并写回 IR metadata。"""
    documents = _workspace_documents(tmp_path)

    roles = DocumentRoleProfiler().profile(documents)

    by_path = {role.relative_path: role for role in roles}
    assert "documentation" in by_path["README.md"].content_roles
    assert "implementation" in by_path["src/main.py"].content_roles
    assert "test" in by_path["tests/test_main.py"].content_roles
    assert "configuration" in by_path["config.json"].content_roles
    assert documents[0].metadata["content_roles"]


def test_meta_controller_generates_plans_for_core_intents(tmp_path: Path) -> None:
    """注释：规则型 controller 为核心 intent 生成不同 FormationPlan。"""
    documents = _workspace_documents(tmp_path)
    controller = RuleBasedMetaController()

    overview = controller.plan(documents, "overview")
    debug = controller.plan(documents, "debug")
    reproduce = controller.plan(documents, "reproduce")
    graph = controller.plan(documents, "graph")

    assert overview.plan.target_schema == "hybrid_file_task_graph"
    assert [step.operator_name for step in overview.plan.steps] == [
        "GistOperator",
        "SemanticFactExtractor",
        "CAMClusterOperator",
        "ClusterFusionOperator",
    ]
    assert debug.plan.target_schema == "debug_trace"
    assert reproduce.plan.target_schema == "procedural_trace"
    assert graph.plan.target_schema == "memory_graph"
    assert "AMemNoteOperator" in graph.decision_trace.selected_operators
    assert graph.decision_trace.reasons


def test_meta_controller_selection_params_are_executable(tmp_path: Path) -> None:
    """注释：controller 输出的 document selection 能被 FormationPipeline 执行。"""
    documents = _workspace_documents(tmp_path)
    result = RuleBasedMetaController().plan(documents, "debug")

    debug_steps = {step.operator_name: step for step in result.plan.steps}
    assert debug_steps["EvidencePreservingCompressor"].document_selector is not None
    assert debug_steps["EvidencePreservingCompressor"].document_selector.doc_ids
    assert "test" in debug_steps["EvidencePreservingCompressor"].document_selector.path_contains

    run = FormationPipeline().run(documents, result.plan)

    assert run.ok is True
    assert run.artifacts
    assert run.metrics["step_count"] == len(result.plan.steps)


def test_meta_controller_unknown_intent_falls_back_to_overview(tmp_path: Path) -> None:
    """注释：未知 intent 归一为 overview，并在 decision trace 记录 warning。"""
    documents = _workspace_documents(tmp_path)

    result = RuleBasedMetaController().plan(documents, "please help me")

    assert result.plan.task_intent == "overview"
    assert result.decision_trace.warnings
    assert result.decision_trace.selected_target_schema == "hybrid_file_task_graph"


def test_meta_controller_trace_records_selection_details(tmp_path: Path) -> None:
    """注释：PlanDecisionTrace 记录 schema/operator/selection 解释。"""
    documents = _workspace_documents(tmp_path)

    result = RuleBasedMetaController().plan(documents, "debug")
    trace = result.decision_trace

    assert trace.original_intent == "debug"
    assert trace.normalized_intent == "debug"
    assert trace.selected_schema_family == "debug"
    assert "EvidencePreservingCompressor" in trace.operator_reasons
    assert trace.selection_reasons["EvidencePreservingCompressor"]
    assert trace.selected_doc_counts["EvidencePreservingCompressor"] > 0


def test_meta_controller_plan_from_query_uses_rule_normalizer(tmp_path: Path) -> None:
    """注释：规则 query normalizer 支持自然语言入口，不调用 LLM。"""
    documents = _workspace_documents(tmp_path)
    controller = RuleBasedMetaController()

    reproduce = controller.plan_from_query(documents, "这个 lab 怎么跑？")
    debug = controller.plan_from_query(documents, "这个报错怎么解决？")
    locate = controller.plan_from_query(documents, "函数 run 在哪实现？")

    assert reproduce.plan.task_intent == "reproduce"
    assert debug.plan.task_intent == "debug"
    assert locate.plan.task_intent == "locate"
    assert reproduce.plan.metadata["workspace_brief"]["query"] == "这个 lab 怎么跑？"
    assert reproduce.decision_trace.warnings


def test_workspace_brief_builder_omits_full_content_and_includes_operator_catalog(tmp_path: Path) -> None:
    """注释：WorkspaceBriefBuilder 为 LLM planner 准备结构化摘要而非全文。"""
    documents = _workspace_documents(tmp_path)
    controller = RuleBasedMetaController()
    result = controller.plan(documents, "overview")

    brief = WorkspaceBriefBuilder().build("summarize this repo", result.workspace_profile, result.document_roles)

    assert brief["query"] == "summarize this repo"
    assert "supported_intents" in brief
    assert "schema_vocabulary" in brief
    assert any(operator["name"] == "GistOperator" for operator in brief["operators"])
    assert brief["documents"]
    assert "content" not in brief["documents"][0]
