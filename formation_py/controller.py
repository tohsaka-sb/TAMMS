"""
模块: formation_py.controller
职责: 根据 workspace/profile/intent/query 生成可解释的 FormationPlan，并预留 LLM planner 接口。
输入: MemoryInputDocument 列表、task_intent 或 query、OperatorRegistry。
输出: MetaControllerResult，包含 FormationPlan 与 PlanDecisionTrace。
副作用: 无；只规划不执行 pipeline。
失败处理: 未知 intent 归一为 overview，并在 trace warnings 中记录。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from harness_py import MemoryInputDocument

from .operators import OperatorRegistry, create_default_registry
from .profile import DocumentRoleProfile, DocumentRoleProfiler, WorkspaceProfile, WorkspaceProfiler
from .types import DocumentSelector, FormationPlan, FormationStep, OperatorSpec


SUPPORTED_INTENTS = {"overview", "debug", "reproduce", "locate", "graph", "summarize"}

SCHEMA_BY_INTENT = {
    "overview": "hybrid_file_task_graph",
    "debug": "debug_trace",
    "reproduce": "procedural_trace",
    "locate": "flat_relation_index",
    "graph": "memory_graph",
    "summarize": "hierarchical_summary",
}

SCHEMA_FAMILY_BY_SCHEMA = {
    "hybrid_file_task_graph": "hybrid",
    "debug_trace": "debug",
    "procedural_trace": "procedural",
    "flat_relation_index": "flat",
    "memory_graph": "planar",
    "hierarchical_summary": "hierarchical",
}

OPERATOR_ORDER = {
    "overview": ["GistOperator", "SemanticFactExtractor", "CAMClusterOperator", "ClusterFusionOperator"],
    "debug": ["EvidencePreservingCompressor", "ProceduralExtractor", "SemanticFactExtractor", "LinkEvolutionOperator"],
    "reproduce": ["EvidencePreservingCompressor", "ProceduralExtractor", "SemanticFactExtractor"],
    "locate": ["GistOperator", "SemanticFactExtractor", "LinkEvolutionOperator"],
    "graph": ["AMemNoteOperator", "SemanticFactExtractor", "LinkEvolutionOperator"],
    "summarize": ["GistOperator", "CAMClusterOperator", "ClusterFusionOperator"],
}

REQUIRED_BY_INTENT = {
    "overview": {"GistOperator"},
    "debug": {"EvidencePreservingCompressor"},
    "reproduce": {"ProceduralExtractor"},
    "locate": {"SemanticFactExtractor"},
    "graph": {"AMemNoteOperator", "LinkEvolutionOperator"},
    "summarize": {"GistOperator", "ClusterFusionOperator"},
}


@dataclass
class PlanDecisionTrace:
    """注释：记录 meta-controller 为什么选择该 plan。"""

    task_intent: str
    original_intent: str = ""
    normalized_intent: str = ""
    workspace_signals: dict[str, Any] = field(default_factory=dict)
    selected_target_schema: str = ""
    selected_schema_family: str = ""
    selected_operators: list[str] = field(default_factory=list)
    skipped_operators: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    operator_reasons: dict[str, list[str]] = field(default_factory=dict)
    selection_reasons: dict[str, list[str]] = field(default_factory=dict)
    selected_doc_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    planner_mode: str = "rule_only"
    llm_called: bool = False
    llm_model: str = ""
    llm_prompt_messages: list[list[dict[str, str]]] = field(default_factory=list)
    llm_raw_outputs: list[str] = field(default_factory=list)
    llm_parse_ok: bool = False
    llm_validation_errors: list[str] = field(default_factory=list)
    llm_soft_warnings: list[str] = field(default_factory=list)
    llm_repairs: list[str] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""
    final_plan_source: str = "rule"
    planner_retry_count: int = 0
    issue_type: str = ""
    issue_for_meta_memory: bool = False


@dataclass
class MetaControllerResult:
    """注释：规则型 meta-controller 的完整规划结果。"""

    plan: FormationPlan
    decision_trace: PlanDecisionTrace
    workspace_profile: WorkspaceProfile
    document_roles: list[DocumentRoleProfile]


class IntentNormalizer:
    """注释：把枚举型 intent 或简单自然语言 query 规整到固定集合。"""

    def normalize(self, task_intent: str) -> tuple[str, str]:
        """注释：返回 (normalized_intent, warning)。"""
        normalized = (task_intent or "overview").strip().lower()
        if normalized in SUPPORTED_INTENTS:
            return normalized, ""
        return "overview", f"unknown task_intent={task_intent}; fallback to overview"

    def normalize_query(self, query: str) -> tuple[str, str]:
        """注释：规则版 query normalizer；真实 LLM parser 留到后续版本。"""
        text = (query or "").strip().lower()
        if not text:
            return "overview", "empty query; fallback to overview"
        if any(word in text for word in ["报错", "失败", "debug", "error", "traceback", "fail"]):
            return "debug", ""
        if any(word in text for word in ["在哪", "哪里", "实现", "函数", "locate", "where"]):
            return "locate", ""
        if any(word in text for word in ["怎么跑", "运行", "复现", "install", "run", "reproduce", "build"]):
            return "reproduce", ""
        if any(word in text for word in ["关系", "图", "依赖", "graph", "link"]):
            return "graph", ""
        if any(word in text for word in ["总结", "摘要", "summarize", "summary"]):
            return "summarize", ""
        return "overview", ""


class LLMIntentParser(Protocol):
    """注释：未来 LLM query parser 接口；Version 4.2 不提供真实实现。"""

    def parse_intent(self, query: str, workspace_brief: dict[str, Any]) -> dict[str, Any]:
        """注释：返回 primary_intent / secondary_intents 等结构化字段。"""


class LLMPlanParser(Protocol):
    """注释：未来 LLM plan parser 接口；Version 4.2 不提供真实实现。"""

    def parse_plan(self, query: str, workspace_brief: dict[str, Any]) -> dict[str, Any]:
        """注释：返回可校验的 plan JSON。"""


class WorkspaceBriefBuilder:
    """注释：构造给 LLM planner 的结构化 brief，不包含完整文件内容。"""

    def __init__(self, registry: OperatorRegistry | None = None) -> None:
        self.registry = registry or create_default_registry()

    def build(
        self,
        query: str,
        workspace_profile: WorkspaceProfile,
        document_roles: list[DocumentRoleProfile],
    ) -> dict[str, Any]:
        """注释：生成 query + intents + schemas + operator catalog + workspace/doc summaries。"""
        return {
            "query": query,
            "supported_intents": {
                "overview": "understand the overall workspace",
                "debug": "find evidence related to errors, tests, failures, logs",
                "reproduce": "extract install/build/run/test procedures",
                "locate": "find implementation locations or relevant files",
                "graph": "build relation graph among files/modules/concepts",
                "summarize": "produce concise or hierarchical summaries",
            },
            "schema_vocabulary": list(SCHEMA_FAMILY_BY_SCHEMA),
            "operators": [
                {
                    "name": self.registry.get(name).spec.name,
                    "output_type": self.registry.get(name).spec.output_type,
                    "supported_intents": self.registry.get(name).spec.supported_intents,
                    "cost_level": self.registry.get(name).spec.cost_level,
                    "requires_llm": self.registry.get(name).spec.requires_llm,
                    "produces_schema": self.registry.get(name).spec.produces_schema,
                }
                for name in self.registry.names()
            ],
            "workspace_profile": _workspace_signals(workspace_profile),
            "documents": [
                {
                    "doc_id": role.doc_id,
                    "path": role.relative_path,
                    "source_kind": role.source_kind,
                    "content_roles": role.content_roles,
                    "intent_hints": role.intent_hints,
                    "importance_score": role.importance_score,
                }
                for role in document_roles
            ],
        }


class RuleBasedMetaController:
    """注释：规则 + 打分的 Version 4 meta-controller。"""

    def __init__(self, registry: OperatorRegistry | None = None) -> None:
        self.registry = registry or create_default_registry()
        self.workspace_profiler = WorkspaceProfiler()
        self.role_profiler = DocumentRoleProfiler()
        self.intent_normalizer = IntentNormalizer()
        self.brief_builder = WorkspaceBriefBuilder(self.registry)

    def plan(self, documents: list[MemoryInputDocument], task_intent: str = "overview") -> MetaControllerResult:
        """注释：生成 FormationPlan，不执行 pipeline。"""
        normalized_intent, warning = self.intent_normalizer.normalize(task_intent)
        return self._plan_for_intent(documents, normalized_intent, original_intent=task_intent, warning=warning)

    def plan_from_query(self, documents: list[MemoryInputDocument], query: str) -> MetaControllerResult:
        """注释：规则版 query -> intent -> plan；LLM planner 后续替换此入口。"""
        normalized_intent, warning = self.intent_normalizer.normalize_query(query)
        if not warning:
            warning = "query normalized by rule-based parser; LLM parser not enabled"
        result = self._plan_for_intent(documents, normalized_intent, original_intent=query, warning=warning)
        workspace_brief = self.brief_builder.build(query, result.workspace_profile, result.document_roles)
        result.plan.metadata["workspace_brief"] = workspace_brief
        result.plan.metadata["query"] = query
        return result

    def _plan_for_intent(
        self,
        documents: list[MemoryInputDocument],
        normalized_intent: str,
        original_intent: str,
        warning: str = "",
    ) -> MetaControllerResult:
        """注释：共享的 intent -> plan 实现。"""
        workspace_profile = self.workspace_profiler.profile(documents)
        document_roles = self.role_profiler.profile(documents)
        target_schema = SCHEMA_BY_INTENT[normalized_intent]
        schema_family = SCHEMA_FAMILY_BY_SCHEMA[target_schema]
        scores = self._score_operators(normalized_intent, target_schema, workspace_profile)
        ordered_names = OPERATOR_ORDER[normalized_intent]
        selected = [name for name in ordered_names if name in self.registry.names()]
        skipped = [name for name in self.registry.names() if name not in selected]
        reasons = self._reasons(normalized_intent, target_schema, selected, scores, workspace_profile)
        operator_reasons = {name: self._operator_reasons(normalized_intent, name, scores[name]) for name in selected}
        warnings = [warning] if warning else []
        if workspace_profile.document_count == 0:
            warnings.append("workspace has no documents")

        steps = []
        selection_reasons = {}
        selected_doc_counts = {}
        for name in selected:
            selector, selector_reasons, selected_count = self._document_selector(normalized_intent, name, document_roles)
            steps.append(
                FormationStep(
                    operator_name=name,
                    required=name in REQUIRED_BY_INTENT.get(normalized_intent, set()),
                    document_selector=selector,
                    params=selector.to_params() if selector else {},
                )
            )
            selection_reasons[name] = selector_reasons
            selected_doc_counts[name] = selected_count
        plan = FormationPlan(
            plan_id=f"meta_{normalized_intent}_v1",
            target_schema=target_schema,
            task_intent=normalized_intent,
            steps=steps,
            metadata={
                "controller": "RuleBasedMetaController",
                "document_selection": "document_selector_v1",
                "schema_family": schema_family,
            },
        )
        trace = PlanDecisionTrace(
            task_intent=normalized_intent,
            original_intent=original_intent,
            normalized_intent=normalized_intent,
            workspace_signals=_workspace_signals(workspace_profile),
            selected_target_schema=target_schema,
            selected_schema_family=schema_family,
            selected_operators=selected,
            skipped_operators=skipped,
            reasons=reasons,
            operator_reasons=operator_reasons,
            selection_reasons=selection_reasons,
            selected_doc_counts=selected_doc_counts,
            warnings=warnings,
            scores={name: scores[name] for name in selected},
        )
        return MetaControllerResult(
            plan=plan,
            decision_trace=trace,
            workspace_profile=workspace_profile,
            document_roles=document_roles,
        )

    def _score_operators(self, intent: str, target_schema: str, profile: WorkspaceProfile) -> dict[str, float]:
        """注释：planning score，用于解释 operator 选择，不代表 memory 质量。"""
        scores: dict[str, float] = {}
        for name in self.registry.names():
            spec: OperatorSpec = self.registry.get(name).spec
            score = 0.0
            if intent in spec.supported_intents:
                score += 3.0
            if target_schema in spec.produces_schema:
                score += 2.0
            if spec.cost_level == "low":
                score += 0.5
            if profile.has_code and name == "SemanticFactExtractor":
                score += 1.0
            if profile.has_markdown and name == "ProceduralExtractor":
                score += 1.0
            if intent == "debug" and name == "EvidencePreservingCompressor":
                score += 3.0
            if intent == "graph" and name in {"AMemNoteOperator", "LinkEvolutionOperator"}:
                score += 2.0
            scores[name] = score
        return scores

    def _document_selector(
        self,
        intent: str,
        operator_name: str,
        roles: list[DocumentRoleProfile],
    ) -> tuple[DocumentSelector | None, list[str], int]:
        """注释：根据 intent/operator 生成 DocumentSelector。"""
        selected_roles: set[str] = set()
        path_contains: list[str] = []
        max_documents = 0
        if intent == "debug":
            selected_roles = {"test", "log", "configuration", "build_script", "documentation"}
            path_contains = ["readme", "cmakelists", "makefile", "test", "log", "config"]
            max_documents = 40
        elif intent == "reproduce":
            selected_roles = {"documentation", "build_script", "run_script", "configuration"}
            path_contains = ["readme", "docs", "cmakelists", "makefile", "requirements", "environment", "run", "script"]
            max_documents = 20
        elif intent == "locate":
            selected_roles = {"implementation", "test"}
            path_contains = ["src", "include", "test", "tests"]
            max_documents = 60
        elif intent == "graph":
            selected_roles = {"implementation", "test", "configuration", "documentation"}
            max_documents = 80
        elif intent == "summarize":
            selected_roles = {"documentation", "implementation", "configuration", "dataset"}
            max_documents = 60

        if operator_name in {"CAMClusterOperator", "ClusterFusionOperator", "LinkEvolutionOperator"}:
            selected_roles = set()
            path_contains = []
        doc_ids = _doc_ids_for_selector(roles, selected_roles, path_contains, max_documents)
        if not doc_ids and not selected_roles and not path_contains and not max_documents:
            return None, ["no document selector; operator receives all documents"], len(roles)
        selector = DocumentSelector(
            doc_ids=doc_ids,
            content_roles=sorted(selected_roles),
            path_contains=path_contains,
            max_documents=max_documents,
        )
        reasons = []
        if selected_roles:
            reasons.append(f"selected content_roles={sorted(selected_roles)}")
        if path_contains:
            reasons.append(f"selected path_contains={path_contains}")
        if max_documents:
            reasons.append(f"limited max_documents={max_documents}")
        if not doc_ids:
            reasons.append("selector matched no documents; pipeline will pass empty document set")
        return selector, reasons, len(doc_ids)

    def _operator_reasons(self, intent: str, operator_name: str, score: float) -> list[str]:
        """注释：生成 operator 级别选择原因。"""
        reasons = [f"planning_score={score:.1f}"]
        if intent in self.registry.get(operator_name).spec.supported_intents:
            reasons.append(f"operator supports intent={intent}")
        if operator_name in REQUIRED_BY_INTENT.get(intent, set()):
            reasons.append("operator is required by hard rule")
        return reasons

    def _reasons(
        self,
        intent: str,
        target_schema: str,
        selected: list[str],
        scores: dict[str, float],
        profile: WorkspaceProfile,
    ) -> list[str]:
        """注释：生成可读决策理由。"""
        reasons = [f"selected target_schema={target_schema} for task_intent={intent}"]
        for name in selected:
            reasons.append(f"selected {name} with planning_score={scores.get(name, 0):.1f}")
        if profile.has_tests:
            reasons.append("workspace has tests; debug/locate plans can use test evidence")
        if profile.has_build_files:
            reasons.append("workspace has build files; reproduce/debug plans can use procedural evidence")
        if profile.dominant_language_hint:
            reasons.append(f"dominant_language_hint={profile.dominant_language_hint}")
        return reasons


def _doc_ids_for_selector(
    roles: list[DocumentRoleProfile],
    selected_roles: set[str],
    path_contains: list[str],
    max_documents: int,
) -> list[str]:
    if not selected_roles and not path_contains:
        return []
    matched = []
    lowered_paths = {role.doc_id: role.relative_path.lower() for role in roles}
    for role in sorted(roles, key=lambda item: item.importance_score, reverse=True):
        role_match = not selected_roles or bool(selected_roles.intersection(role.content_roles))
        path_match = not path_contains or any(marker in lowered_paths[role.doc_id] for marker in path_contains)
        if role_match or path_match:
            matched.append(role.doc_id)
        if max_documents and len(matched) >= max_documents:
            break
    return matched


def _workspace_signals(profile: WorkspaceProfile) -> dict[str, Any]:
    return {
        "document_count": profile.document_count,
        "source_kind_distribution": profile.source_kind_distribution,
        "top_path_distribution": profile.top_path_distribution,
        "has_code": profile.has_code,
        "has_markdown": profile.has_markdown,
        "has_config": profile.has_config,
        "has_tests": profile.has_tests,
        "has_build_files": profile.has_build_files,
        "has_scripts": profile.has_scripts,
        "has_logs": profile.has_logs,
        "has_data_files": profile.has_data_files,
        "dominant_source_kind": profile.dominant_source_kind,
        "dominant_language_hint": profile.dominant_language_hint,
        "complexity_level": profile.complexity_level,
        "organization_level": profile.organization_level,
    }
