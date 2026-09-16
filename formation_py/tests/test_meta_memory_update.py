"""注释：Version 6.1 staged meta-formation memory update 测试。"""

from __future__ import annotations

import json
from pathlib import Path

from formation_py import FormationPipeline
from formation_py.llm_planner import LLMMetaController, LLMPlannerConfig
from formation_py.meta_formation import (
    FormationEpisodeBuilder,
    MetaFormationIdFactory,
    MetaMemoryUpdateCandidate,
    MetaMemoryUpdateDecision,
    RuleBasedMetaMemoryReflector,
    STAGE1_ENGINEERING,
    STAGE3_HUMAN_PREFERENCE,
    apply_review_decision,
)
from formation_py.meta_memory import FormationMetaMemoryWriter, MetaMemoryIndex
from harness_py.reader import HeterogeneousReader
from harness_py.scanner import FileScanner


class FakeLLMClient:
    """注释：顺序返回 mock LLM 输出。"""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def complete(self, messages, model: str, temperature: float = 0.0) -> str:
        self.calls += 1
        if self.calls > len(self.outputs):
            raise RuntimeError("no more fake outputs")
        return self.outputs[self.calls - 1]


def _documents(tmp_path: Path):
    (tmp_path / "README.md").write_text("# Demo\n\nRun tests with pytest.\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    reader = HeterogeneousReader()
    return [reader.read_document(item) for item in FileScanner(tmp_path).scan()]


def _valid_plan(documents) -> str:
    return json.dumps(
        {
            "primary_intent": "summarize",
            "secondary_intents": [],
            "target_schema": "hierarchical_summary",
            "steps": [
                {
                    "step_id": "gist",
                    "operator_name": "GistOperator",
                    "required": True,
                    "document_selector": {"doc_ids": [documents[0].doc_id], "max_documents": 1},
                    "reason": "Summarize entry docs.",
                },
                {
                    "step_id": "cluster",
                    "operator_name": "CAMClusterOperator",
                    "required": True,
                    "document_selector": {"max_documents": 2},
                    "reason": "Cluster workspace structure.",
                },
            ],
            "assumptions": [],
            "reasons": [],
        }
    )


def test_success_episode_generates_successful_pattern_candidate(tmp_path: Path) -> None:
    """注释：成功 planning + pipeline 会生成 Successful Patterns 候选。"""
    documents = _documents(tmp_path)
    planning = LLMMetaController(
        config=LLMPlannerConfig(planner_mode="llm_preferred", model="fake"),
        llm_client=FakeLLMClient([_valid_plan(documents)]),
    ).plan_from_query(documents, "总结这个项目结构")
    run = FormationPipeline().run(documents, planning.plan)
    id_factory = MetaFormationIdFactory(date="20260705")

    episode = FormationEpisodeBuilder(id_factory=id_factory).build(planning, run, source="test")
    candidates = RuleBasedMetaMemoryReflector(id_factory=id_factory).propose_updates(episode)

    assert episode.episode_id == "FE-20260705-0001"
    assert episode.primary_intent == "summarize"
    assert episode.artifact_count and episode.artifact_count > 0
    assert len(candidates) == 1
    assert candidates[0].candidate_id == "MMC-20260705-0001"
    assert candidates[0].proposed_memory_id == "MM-20260705-0001"
    assert candidates[0].section == "Successful Patterns"
    assert "GistOperator -> CAMClusterOperator" in candidates[0].content


def test_failed_planning_episode_generates_failure_lesson(tmp_path: Path) -> None:
    """注释：三次 selector 错误 abort 仍可形成失败 episode 和 Failure Lessons 候选。"""
    documents = _documents(tmp_path)
    bad = json.dumps(
        {
            "primary_intent": "debug",
            "secondary_intents": [],
            "target_schema": "debug_trace",
            "steps": [
                {
                    "step_id": "missing",
                    "operator_name": "EvidencePreservingCompressor",
                    "required": True,
                    "document_selector": {"doc_ids": ["missing-doc"]},
                    "reason": "Use missing evidence.",
                }
            ],
            "assumptions": [],
            "reasons": [],
        }
    )
    planning = LLMMetaController(
        config=LLMPlannerConfig(planner_mode="llm_preferred", max_replan_attempts=3),
        llm_client=FakeLLMClient([bad, bad, bad]),
    ).plan_from_query(documents, "debug")
    id_factory = MetaFormationIdFactory(date="20260705")

    episode = FormationEpisodeBuilder(id_factory=id_factory).build(planning, None)
    candidates = RuleBasedMetaMemoryReflector(id_factory=id_factory).propose_updates(episode)

    assert planning.ok is False
    assert episode.issue_for_meta_memory is True
    assert any(candidate.section == "Failure Lessons" for candidate in candidates)


def test_writer_preview_apply_backup_history_and_dedup(tmp_path: Path) -> None:
    """注释：preview 不写文件；apply 备份、写 section、写 history，并阻止重复写入。"""
    memory_path = tmp_path / "formation_meta_memory.md"
    memory_path.write_text("# Formation Meta Memory\n\n## Successful Patterns\n\n## Update History\n", encoding="utf-8")
    candidate = MetaMemoryUpdateCandidate(
        candidate_id="MMC-20260705-0001",
        proposed_memory_id="MM-20260705-0001",
        episode_id="FE-20260705-0001",
        stage=STAGE1_ENGINEERING,
        update_type="successful_pattern",
        section="Successful Patterns",
        content="**summarize / hierarchical_summary**: Use `GistOperator -> CAMClusterOperator` for compact summaries.",
        evidence=["episode=FE-20260705-0001"],
        confidence=0.72,
    )
    writer = FormationMetaMemoryWriter(str(memory_path))

    preview = writer.preview_update(candidate)
    assert "MM-20260705-0001" in preview
    assert "MM-20260705-0001" not in memory_path.read_text(encoding="utf-8")

    result = writer.apply_update(candidate)
    updated = memory_path.read_text(encoding="utf-8")

    assert result.applied is True
    assert Path(result.backup_path).exists()
    assert "MM-20260705-0001" in updated
    assert "Applied MM-20260705-0001" in updated

    duplicate = writer.apply_update(candidate)
    assert duplicate.applied is False
    assert duplicate.reason == "duplicate"
    assert updated == memory_path.read_text(encoding="utf-8")


def test_writer_section_capacity_limit(tmp_path: Path) -> None:
    """注释：section 超限时拒绝写入，不修改文件。"""
    memory_path = tmp_path / "formation_meta_memory.md"
    memory_path.write_text("# Formation Meta Memory\n\n## Failure Lessons\n\n## Update History\n", encoding="utf-8")
    candidate = MetaMemoryUpdateCandidate(
        candidate_id="MMC-20260705-0002",
        proposed_memory_id="MM-20260705-0002",
        episode_id="FE-20260705-0002",
        stage=STAGE1_ENGINEERING,
        update_type="failure_lesson",
        section="Failure Lessons",
        content="**debug / failure**: Avoid missing document selectors.",
        evidence=[],
    )
    writer = FormationMetaMemoryWriter(str(memory_path), max_items_per_section=0)

    result = writer.apply_update(candidate)

    assert result.applied is False
    assert result.reason == "section_capacity_exceeded"
    assert "MM-20260705-0002" not in memory_path.read_text(encoding="utf-8")


def test_stage3_review_decision_revises_candidate() -> None:
    """注释：Stage 3 人工偏好接口预留 accept/reject/revise。"""
    candidate = MetaMemoryUpdateCandidate(
        candidate_id="MMC-20260705-0003",
        proposed_memory_id="MM-20260705-0003",
        episode_id="FE-20260705-0003",
        stage=STAGE3_HUMAN_PREFERENCE,
        update_type="operator_guideline",
        section="Operator Guidelines",
        content="old content",
        requires_human_review=True,
        review_status="pending",
    )
    decision = MetaMemoryUpdateDecision(
        candidate_id="MMC-20260705-0003",
        action="revise",
        revised_content="revised content",
        reason="more general",
        reviewer="tester",
    )

    revised = apply_review_decision(candidate, decision)

    assert revised.status == "accepted"
    assert revised.review_status == "revise"
    assert revised.content == "revised content"
    assert revised.human_feedback["reviewer"] == "tester"


def test_meta_memory_index_extracts_ids_and_sections() -> None:
    """注释：MetaMemoryIndex 支持审计 ID 和 section 计数。"""
    markdown = """# Formation Meta Memory

## Successful Patterns

- [MM-20260705-0001] content

## Update History

- [UH-20260705-0001] Applied MM-20260705-0001
"""

    index = MetaMemoryIndex(markdown)

    assert index.section_exists("Successful Patterns")
    assert index.count_items("Successful Patterns") == 1
    assert "MM-20260705-0001" in index.memory_ids()
