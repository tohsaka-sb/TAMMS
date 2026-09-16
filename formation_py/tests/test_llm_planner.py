"""注释：Version 5 LLM planner 测试，全部使用 mock client，不依赖真实 API。"""

from __future__ import annotations

import json
from pathlib import Path

from formation_py import FormationPipeline
from formation_py.llm_planner import LLMMetaController, LLMPlanParser, LLMPlannerConfig
from harness_py.reader import HeterogeneousReader
from harness_py.scanner import FileScanner


class FakeLLMClient:
    """注释：顺序返回预置 LLM 输出，记录调用次数。"""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0
        self.messages = []

    def complete(self, messages, model: str, temperature: float = 0.0) -> str:
        self.calls += 1
        self.messages.append(messages)
        if self.calls > len(self.outputs):
            raise RuntimeError("no more fake outputs")
        return self.outputs[self.calls - 1]


def _documents(tmp_path: Path):
    """注释：构造 LLM planner 测试工作区。"""
    (tmp_path / "README.md").write_text("# Demo\n\nRun tests with pytest.\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def run():\n    return 'ok'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_main.py").write_text("def test_run():\n    assert True\n", encoding="utf-8")
    reader = HeterogeneousReader()
    return [reader.read_document(item) for item in FileScanner(tmp_path).scan()]


def _proposal(documents, operator_name: str = "GistOperator", max_documents: int = 10) -> str:
    doc_ids = [document.doc_id for document in documents[:2]]
    return json.dumps(
        {
            "primary_intent": "overview",
            "secondary_intents": ["summarize"],
            "target_schema": "hybrid_file_task_graph",
            "steps": [
                {
                    "step_id": "step_gist",
                    "operator_name": operator_name,
                    "required": True,
                    "document_selector": {
                        "doc_ids": doc_ids,
                        "max_documents": max_documents,
                    },
                    "reason": "Need file gists for overview.",
                }
            ],
            "assumptions": ["Workspace brief is sufficient for planning."],
            "reasons": ["User asks for overview."],
        },
        ensure_ascii=False,
    )


def test_llm_planner_config_reads_environment(monkeypatch) -> None:
    """注释：真实 API 配置从环境变量读取，避免把 key 固定在代码里。"""
    monkeypatch.setenv("TAMMS_LLM_API_KEY", "test-key")
    monkeypatch.setenv("TAMMS_LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("TAMMS_LLM_MODEL", "test-model")

    config = LLMPlannerConfig()

    assert config.api_key == "test-key"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "test-model"


def test_llm_plan_parser_handles_fenced_json_and_scalar_fields() -> None:
    """注释：真实 LLM 可能返回 fenced JSON、字符串列表字段和字符串数字。"""
    raw = """```json
{
  "primary_intent": "overview",
  "secondary_intents": "summarize",
  "target_schema": "hybrid_file_task_graph",
  "steps": [
    {
      "step_id": "step_docs",
      "operator_name": "GistOperator",
      "required": "false",
      "document_selector": {
        "content_roles": "documentation",
        "max_documents": "3",
        "min_role_score": "0.5"
      },
      "reason": "Summarize docs."
    }
  ],
  "assumptions": "brief is enough",
  "reasons": "smoke"
}
```"""

    proposal = LLMPlanParser().parse(raw)

    assert proposal.secondary_intents == ["summarize"]
    assert proposal.assumptions == ["brief is enough"]
    assert proposal.reasons == ["smoke"]
    assert proposal.steps[0].required is False
    assert proposal.steps[0].document_selector.content_roles == ["documentation"]
    assert proposal.steps[0].document_selector.max_documents == 3
    assert proposal.steps[0].document_selector.min_role_score == 0.5


def test_rule_only_mode_does_not_call_llm(tmp_path: Path) -> None:
    """注释：rule_only 不调用 LLM，保持离线 baseline。"""
    documents = _documents(tmp_path)
    client = FakeLLMClient([])
    controller = LLMMetaController(
        config=LLMPlannerConfig(planner_mode="rule_only"),
        llm_client=client,
    )

    result = controller.plan_from_query(documents, "这个项目怎么跑？")

    assert result.ok is True
    assert result.plan is not None
    assert result.plan.task_intent == "reproduce"
    assert client.calls == 0
    assert result.decision_trace.planner_mode == "rule_only"


def test_llm_preferred_valid_plan_can_execute(tmp_path: Path) -> None:
    """注释：合法 LLM plan 会转成 FormationPlan，并可由 pipeline 执行。"""
    documents = _documents(tmp_path)
    client = FakeLLMClient([_proposal(documents)])
    controller = LLMMetaController(
        config=LLMPlannerConfig(planner_mode="llm_preferred", model="fake-model"),
        llm_client=client,
    )

    result = controller.plan_from_query(documents, "Summarize this workspace")

    assert result.ok is True
    assert result.plan is not None
    assert result.decision_trace.llm_called is True
    assert result.decision_trace.llm_model == "fake-model"
    assert result.decision_trace.final_plan_source == "llm"
    assert "formation_meta_memory" in client.messages[0][1]["content"]

    run = FormationPipeline().run(documents, result.plan)
    assert run.ok is True
    assert run.artifacts


def test_llm_preferred_retries_after_invalid_operator(tmp_path: Path) -> None:
    """注释：非法 operator 会反馈给 LLM 重试，第二次合法则成功。"""
    documents = _documents(tmp_path)
    client = FakeLLMClient([
        _proposal(documents, operator_name="InventedOperator"),
        _proposal(documents),
    ])
    controller = LLMMetaController(
        config=LLMPlannerConfig(planner_mode="llm_preferred", model="fake-model", max_replan_attempts=3),
        llm_client=client,
    )

    result = controller.plan_from_query(documents, "overview")

    assert result.ok is True
    assert client.calls == 2
    assert result.decision_trace.planner_retry_count == 2
    assert result.decision_trace.final_plan_source == "llm"
    assert "invalid operator" in client.messages[1][1]["content"]


def test_llm_preferred_clips_max_documents(tmp_path: Path) -> None:
    """注释：validator 会按预算裁剪过大的 max_documents。"""
    documents = _documents(tmp_path)
    client = FakeLLMClient([_proposal(documents, max_documents=999)])
    controller = LLMMetaController(
        config=LLMPlannerConfig(planner_mode="llm_preferred", max_documents_per_step=2),
        llm_client=client,
    )

    result = controller.plan_from_query(documents, "overview")

    assert result.ok is True
    assert result.plan is not None
    assert result.plan.steps[0].document_selector is not None
    assert result.plan.steps[0].document_selector.max_documents == 2
    assert result.decision_trace.llm_repairs == ["step_gist max_documents clipped to 2"]
    assert result.decision_trace.final_plan_source == "llm_repaired"


def test_llm_preferred_allows_repeated_operator_with_unique_step_ids(tmp_path: Path) -> None:
    """注释：同一 operator 可重复出现，只要 step_id 唯一且 selector/purpose 可区分。"""
    documents = _documents(tmp_path)
    proposal = {
        "primary_intent": "overview",
        "secondary_intents": [],
        "target_schema": "hybrid_file_task_graph",
        "steps": [
            {
                "step_id": "gist_docs",
                "operator_name": "GistOperator",
                "required": True,
                "document_selector": {"content_roles": ["documentation"], "max_documents": 5},
                "reason": "Summarize docs.",
            },
            {
                "step_id": "gist_code",
                "operator_name": "GistOperator",
                "required": True,
                "document_selector": {"content_roles": ["implementation"], "max_documents": 5},
                "reason": "Summarize code.",
            },
        ],
        "assumptions": [],
        "reasons": [],
    }
    client = FakeLLMClient([json.dumps(proposal)])
    controller = LLMMetaController(
        config=LLMPlannerConfig(planner_mode="llm_preferred"),
        llm_client=client,
    )

    result = controller.plan_from_query(documents, "overview")

    assert result.ok is True
    assert result.plan is not None
    assert [step.operator_name for step in result.plan.steps] == ["GistOperator", "GistOperator"]
    assert [step.step_id for step in result.plan.steps] == ["gist_docs", "gist_code"]
    assert "gist_docs" in result.decision_trace.operator_reasons


def test_llm_preferred_warns_on_duplicate_operator_selector_and_reason(tmp_path: Path) -> None:
    """注释：重复 step 不阻断 plan，但进入 soft warning 供后续元记忆分析。"""
    documents = _documents(tmp_path)
    doc_ids = [document.doc_id for document in documents[:2]]
    proposal = {
        "primary_intent": "overview",
        "secondary_intents": [],
        "target_schema": "hybrid_file_task_graph",
        "steps": [
            {
                "step_id": "gist_a",
                "operator_name": "GistOperator",
                "required": True,
                "document_selector": {"doc_ids": doc_ids, "max_documents": 2},
                "reason": "Summarize the same files.",
            },
            {
                "step_id": "gist_b",
                "operator_name": "GistOperator",
                "required": True,
                "document_selector": {"doc_ids": doc_ids, "max_documents": 2},
                "reason": "Summarize the same files.",
            },
        ],
        "assumptions": [],
        "reasons": [],
    }
    client = FakeLLMClient([json.dumps(proposal)])
    controller = LLMMetaController(
        config=LLMPlannerConfig(planner_mode="llm_preferred"),
        llm_client=client,
    )

    result = controller.plan_from_query(documents, "overview")

    assert result.ok is True
    assert result.decision_trace.llm_validation_errors == []
    assert any("duplicates operator/selector/reason" in warning for warning in result.decision_trace.llm_soft_warnings)


def test_llm_preferred_aborts_after_repeated_invalid_selection(tmp_path: Path) -> None:
    """注释：三次 selector 仍选不到文档时彻底 abort，不返回 partial plan。"""
    documents = _documents(tmp_path)
    bad = json.dumps(
        {
            "primary_intent": "debug",
            "secondary_intents": [],
            "target_schema": "debug_trace",
            "steps": [
                {
                    "step_id": "missing_doc",
                    "operator_name": "EvidencePreservingCompressor",
                    "required": True,
                    "document_selector": {"doc_ids": ["does_not_exist"]},
                    "reason": "Use missing doc.",
                }
            ],
            "assumptions": [],
            "reasons": [],
        }
    )
    client = FakeLLMClient([bad, bad, bad])
    controller = LLMMetaController(
        config=LLMPlannerConfig(planner_mode="llm_preferred", max_replan_attempts=3),
        llm_client=client,
    )

    result = controller.plan_from_query(documents, "debug")

    assert result.ok is False
    assert result.plan is None
    assert client.calls == 3
    assert result.decision_trace.final_plan_source == "abort"
    assert result.decision_trace.issue_for_meta_memory is True
    assert "missing doc_ids" in "\n".join(result.decision_trace.llm_validation_errors)
