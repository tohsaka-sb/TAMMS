"""
模块: formation_py.pipeline
职责: 按 FormationPlan 顺序执行 formation operators，并汇总 FormationRun。
输入: MemoryInputDocument 列表、FormationPlan、OperatorRegistry。
输出: FormationRun。
副作用: 无；不调用网络、不写存储。
失败处理: required step 失败时停止；非 required step 失败时记录错误并继续。
"""

from __future__ import annotations

from uuid import uuid4

from harness_py import MemoryInputDocument

from .operators import OperatorRegistry, create_default_registry
from .types import FormationArtifact, FormationContext, FormationPlan, FormationRun, FormationStep


class FormationPipeline:
    """注释：执行 FormationPlan 的最小 pipeline，供后续 meta-controller 复用。"""

    def __init__(self, registry: OperatorRegistry | None = None) -> None:
        self.registry = registry or create_default_registry()

    def run(self, documents: list[MemoryInputDocument], plan: FormationPlan) -> FormationRun:
        """注释：按 plan.steps 执行 operator，并返回统一 FormationRun。"""
        run_id = f"formation_run_{uuid4()}"
        context = FormationContext(
            run_id=run_id,
            task_intent=plan.task_intent,
            target_schema=plan.target_schema,
            metadata=plan.metadata,
        )
        artifacts: list[FormationArtifact] = []
        traces = []
        warnings: list[str] = []
        errors: list[str] = []
        step_metrics = []

        for step in plan.steps:
            try:
                operator = self.registry.get(step.operator_name)
            except KeyError:
                message = f"unknown operator: {step.operator_name}"
                errors.append(message)
                if step.required:
                    return _run_result(run_id, plan, False, artifacts, traces, warnings, errors, step_metrics, "failed")
                continue

            selected_documents = self._select_documents(step, documents, artifacts)
            result = operator.run(selected_documents, context=context)
            if result.trace:
                traces.append(result.trace)
            artifacts.extend(result.artifacts)
            warnings.extend(result.warnings)
            errors.extend(result.errors)
            if result.error:
                errors.append(result.error)
            step_metrics.append(
                {
                    "step_id": step.step_id,
                    "operator_name": step.operator_name,
                    "status": result.status,
                    "artifact_count": len(result.artifacts),
                    "ok": result.ok,
                }
            )
            if not result.artifacts:
                warnings.append(f"{step.operator_name} produced no artifacts")
            if not result.ok and step.required:
                return _run_result(run_id, plan, False, artifacts, traces, warnings, errors, step_metrics, "failed")

        status = "ok" if not errors else "partial"
        return _run_result(run_id, plan, not errors, artifacts, traces, warnings, errors, step_metrics, status)

    def _select_documents(
        self,
        step: FormationStep,
        documents: list[MemoryInputDocument],
        artifacts: list[FormationArtifact],
    ) -> list[MemoryInputDocument]:
        """注释：根据 step.params 做轻量 document selection。"""
        params = {**(step.document_selector.to_params() if step.document_selector else {}), **step.params}
        selected = documents
        doc_ids = set(params.get("doc_ids", []))
        if doc_ids:
            selected = [document for document in selected if document.doc_id in doc_ids]

        source_kinds = set(params.get("source_kinds", []))
        if source_kinds:
            selected = [document for document in selected if document.source_kind in source_kinds]

        content_roles = set(params.get("content_roles", []))
        if content_roles:
            selected = [
                document
                for document in selected
                if content_roles.intersection(set(document.metadata.get("content_roles", [])))
            ]

        path_contains = [item.lower() for item in params.get("path_contains", [])]
        if path_contains:
            selected = [document for document in selected if any(item in document.relative_path.lower() for item in path_contains)]

        path_prefixes = [item.lower() for item in params.get("path_prefixes", [])]
        if path_prefixes:
            selected = [document for document in selected if any(document.relative_path.lower().startswith(item) for item in path_prefixes)]

        exclude_path_contains = [item.lower() for item in params.get("exclude_path_contains", [])]
        if exclude_path_contains:
            selected = [document for document in selected if not any(item in document.relative_path.lower() for item in exclude_path_contains)]

        min_role_score = float(params.get("min_role_score", 0) or 0)
        if min_role_score > 0:
            selected = [document for document in selected if float(document.metadata.get("importance_score", 0) or 0) >= min_role_score]

        max_documents = int(params.get("max_documents", 0) or 0)
        if max_documents > 0:
            selected = selected[:max_documents]

        return selected


def overview_plan() -> FormationPlan:
    """注释：工作区概览固定计划。"""
    return FormationPlan(
        plan_id="overview_v1",
        target_schema="hybrid_file_task_graph",
        task_intent="overview",
        steps=[
            FormationStep("GistOperator"),
            FormationStep("SemanticFactExtractor"),
            FormationStep("CAMClusterOperator"),
            FormationStep("ClusterFusionOperator"),
        ],
    )


def debug_plan() -> FormationPlan:
    """注释：调试/复现固定计划。"""
    return FormationPlan(
        plan_id="debug_v1",
        target_schema="debug_trace",
        task_intent="debug",
        steps=[
            FormationStep("EvidencePreservingCompressor"),
            FormationStep("ProceduralExtractor", required=False),
            FormationStep("SemanticFactExtractor", required=False),
            FormationStep("LinkEvolutionOperator", required=False),
        ],
    )


def graph_plan() -> FormationPlan:
    """注释：关系图谱固定计划。"""
    return FormationPlan(
        plan_id="graph_v1",
        target_schema="memory_graph",
        task_intent="graph",
        steps=[
            FormationStep("AMemNoteOperator"),
            FormationStep("SemanticFactExtractor", required=False),
            FormationStep("LinkEvolutionOperator", required=False),
        ],
    )


def _run_result(
    run_id: str,
    plan: FormationPlan,
    ok: bool,
    artifacts: list[FormationArtifact],
    traces,
    warnings: list[str],
    errors: list[str],
    step_metrics,
    status: str,
) -> FormationRun:
    """注释：统一创建 FormationRun，避免 run() 中重复拼 metrics。"""
    return FormationRun(
        run_id=run_id,
        plan=plan,
        ok=ok,
        artifacts=artifacts,
        traces=traces,
        warnings=warnings,
        errors=errors,
        metrics={
            "artifact_count": len(artifacts),
            "trace_count": len(traces),
            "step_count": len(plan.steps),
            "steps": step_metrics,
        },
        status=status,
    )
