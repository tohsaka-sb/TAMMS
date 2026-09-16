"""
模块: formation_py.types
职责: 定义记忆形成算子的输入、输出、trace、plan/run 与轻量证据片段。
输入: Harness IR、算子名称、产物内容与 metadata。
输出: OperatorSpec、FormationContext、DocumentSelector、FormationArtifact、FormationResult、FormationTrace、FormationPlan、FormationRun、EvidenceSpan。
副作用: 无。
失败处理: 算子失败时通过 FormationResult.ok/error 表达。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class OperatorSpec:
    """注释：算子能力描述，供后续 meta-controller 选择 operator。"""

    name: str
    output_type: str
    input_scope: str = "documents"
    required_source_kinds: List[str] = field(default_factory=list)
    supported_intents: List[str] = field(default_factory=list)
    cost_level: str = "low"
    requires_llm: bool = False
    produces_schema: List[str] = field(default_factory=list)


@dataclass
class FormationContext:
    """注释：一次 formation 执行的上下文，替代弱 Dict[str, str]。"""

    run_id: str = ""
    task_intent: str = "overview"
    target_schema: str = "hybrid"
    workspace_profile: Dict[str, Any] = field(default_factory=dict)
    budget: Dict[str, Any] = field(default_factory=dict)
    existing_memory_refs: List[str] = field(default_factory=list)
    allow_llm: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceSpan:
    """注释：指向输入文档中的证据片段，当前以行号和摘录表示。"""

    doc_id: str
    relative_path: str
    start_line: int = 1
    end_line: int = 1
    excerpt: str = ""


@dataclass
class FormationTrace:
    """注释：记录一个产物由哪个算子、哪些输入和何种策略形成。"""

    operator_name: str
    input_ids: List[str] = field(default_factory=list)
    strategy: str = "heuristic"
    llm_used: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormationArtifact:
    """注释：算子产物，可表达 gist、压缩摘要、聚类、事实或链接。"""

    artifact_type: str
    content: str
    source_ids: List[str] = field(default_factory=list)
    evidence: List[EvidenceSpan] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5


@dataclass
class FormationResult:
    """注释：统一算子执行结果，保留产物、trace 与错误信息。"""

    ok: bool
    artifacts: List[FormationArtifact] = field(default_factory=list)
    trace: FormationTrace | None = None
    status: str = "ok"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    error: str = ""


@dataclass
class DocumentSelector:
    """注释：FormationStep 的文档选择器，避免 selection magic keys 散落。"""

    doc_ids: List[str] = field(default_factory=list)
    source_kinds: List[str] = field(default_factory=list)
    content_roles: List[str] = field(default_factory=list)
    path_contains: List[str] = field(default_factory=list)
    path_prefixes: List[str] = field(default_factory=list)
    exclude_path_contains: List[str] = field(default_factory=list)
    max_documents: int = 0
    min_role_score: float = 0.0

    def to_params(self) -> Dict[str, Any]:
        """注释：序列化为 pipeline 可消费的 params。"""
        return {
            "doc_ids": self.doc_ids,
            "source_kinds": self.source_kinds,
            "content_roles": self.content_roles,
            "path_contains": self.path_contains,
            "path_prefixes": self.path_prefixes,
            "exclude_path_contains": self.exclude_path_contains,
            "max_documents": self.max_documents,
            "min_role_score": self.min_role_score,
        }


@dataclass
class FormationStep:
    """注释：FormationPlan 中的单个执行步骤。"""

    operator_name: str
    step_id: str = ""
    input_selector: str = "documents"
    params: Dict[str, Any] = field(default_factory=dict)
    required: bool = True
    document_selector: DocumentSelector | None = None


@dataclass
class FormationPlan:
    """注释：可执行的 formation 计划，由 meta-controller 或固定模板生成。"""

    plan_id: str
    target_schema: str
    task_intent: str
    steps: List[FormationStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FormationRun:
    """注释：FormationPipeline 的统一执行结果。"""

    run_id: str
    plan: FormationPlan
    ok: bool
    artifacts: List[FormationArtifact] = field(default_factory=list)
    traces: List[FormationTrace] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
