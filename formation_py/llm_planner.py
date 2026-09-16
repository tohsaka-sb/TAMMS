"""
模块: formation_py.llm_planner
职责: LLM Planner MetaController，把 workspace brief + query 转成可校验 FormationPlan。
输入: MemoryInputDocument 列表、自然语言 query、LLMPlannerConfig。
输出: LLMPlanningResult，成功时含 FormationPlan，失败时含 issue trace。
副作用: llm_preferred 模式会通过 LLMClient 调用外部 API；不写缓存、不写 SQLite。
失败处理: LLM 输出最多重试 max_replan_attempts 次，失败后彻底 abort。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol

from harness_py import MemoryInputDocument

from .controller import (
    PlanDecisionTrace,
    RuleBasedMetaController,
    SCHEMA_BY_INTENT,
    SCHEMA_FAMILY_BY_SCHEMA,
    SUPPORTED_INTENTS,
    WorkspaceBriefBuilder,
    _workspace_signals,
)
from .meta_memory import FormationMetaMemoryProvider
from .operators import OperatorRegistry, create_default_registry
from .profile import DocumentRoleProfile, DocumentRoleProfiler, WorkspaceProfile, WorkspaceProfiler
from .types import DocumentSelector, FormationPlan, FormationStep


@dataclass
class LLMPlannerConfig:
    """注释：LLM planner 配置；可由构造参数或环境变量填写。"""

    planner_mode: str = "rule_only"
    api_key: str = field(default_factory=lambda: os.getenv("TAMMS_LLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("TAMMS_LLM_BASE_URL", ""))
    model: str = field(default_factory=lambda: os.getenv("TAMMS_LLM_MODEL", ""))
    max_replan_attempts: int = 3
    max_steps: int = 6
    max_documents_per_step: int = 40
    timeout_seconds: int = 60
    temperature: float = 0.0
    meta_memory_path: str = ""
    save_prompt_messages: bool = True
    save_raw_output: bool = True


@dataclass
class LLMPlanStepProposal:
    """注释：LLM 输出的单个 step proposal。"""

    step_id: str
    operator_name: str
    required: bool = True
    document_selector: DocumentSelector = field(default_factory=DocumentSelector)
    reason: str = ""


@dataclass
class LLMPlanProposal:
    """注释：LLM 输出的结构化 plan proposal。"""

    primary_intent: str
    secondary_intents: list[str] = field(default_factory=list)
    target_schema: str = ""
    steps: list[LLMPlanStepProposal] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class LLMPlanningResult:
    """注释：LLM MetaController 的规划结果；失败时 plan 为 None。"""

    ok: bool
    plan: FormationPlan | None
    decision_trace: PlanDecisionTrace
    workspace_profile: WorkspaceProfile
    document_roles: list[DocumentRoleProfile]
    workspace_brief: dict[str, Any] = field(default_factory=dict)
    error: str = ""


class LLMClient(Protocol):
    """注释：LLM client 协议，测试可注入 mock。"""

    def complete(self, messages: list[dict[str, str]], model: str, temperature: float = 0.0) -> str:
        """注释：返回模型原始文本输出。"""


class OpenAICompatibleLLMClient:
    """注释：OpenAI-compatible chat completions client，使用 api_key/base_url。"""

    def __init__(self, api_key: str, base_url: str, timeout_seconds: int = 60) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def complete(self, messages: list[dict[str, str]], model: str, temperature: float = 0.0) -> str:
        """注释：调用 /chat/completions，返回 message.content。"""
        if not self.api_key:
            raise ValueError("api_key is required for OpenAICompatibleLLMClient")
        if not self.base_url:
            raise ValueError("base_url is required for OpenAICompatibleLLMClient")
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc
        return data["choices"][0]["message"]["content"]


class LLMPlannerPromptBuilder:
    """注释：构造强约束 JSON planner prompt。"""

    def build_messages(
        self,
        query: str,
        workspace_brief: dict[str, Any],
        meta_memory: str,
        validation_errors: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """注释：返回 chat messages；validation_errors 用于 replan。"""
        system = (
            "You are a memory formation planner. "
            "Choose only from the provided intents, schemas, operators, and documents. "
            "Do not execute operators. Do not invent operators. "
            "Do not request full file contents. Output valid JSON only."
        )
        schema = {
            "primary_intent": "one supported intent",
            "secondary_intents": ["zero or more supported intents"],
            "target_schema": "one schema from schema_vocabulary",
            "steps": [
                {
                    "step_id": "unique id",
                    "operator_name": "operator from catalog",
                    "required": True,
                    "document_selector": {
                        "doc_ids": [],
                        "source_kinds": [],
                        "content_roles": [],
                        "path_contains": [],
                        "path_prefixes": [],
                        "exclude_path_contains": [],
                        "max_documents": 20,
                        "min_role_score": 0.0,
                    },
                    "reason": "why this step is useful",
                }
            ],
            "assumptions": [],
            "reasons": [],
        }
        user_payload = {
            "query": query,
            "formation_meta_memory": meta_memory,
            "workspace_brief": workspace_brief,
            "required_json_schema": schema,
            "validation_errors_to_fix": validation_errors or [],
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
        ]


class LLMPlanParser:
    """注释：解析 LLM JSON 输出为 LLMPlanProposal。"""

    def parse(self, raw_output: str) -> LLMPlanProposal:
        """注释：宽解析真实 LLM 输出；最终合法性仍由 validator 决定。"""
        data = json.loads(_extract_json_object(raw_output))
        steps = []
        step_items = data.get("steps", [])
        if not isinstance(step_items, list):
            step_items = []
        for idx, item in enumerate(step_items, start=1):
            if not isinstance(item, dict):
                item = {}
            selector = item.get("document_selector", {}) or {}
            if not isinstance(selector, dict):
                selector = {}
            steps.append(
                LLMPlanStepProposal(
                    step_id=str(item.get("step_id") or f"step_{idx}"),
                    operator_name=str(item.get("operator_name", "")),
                    required=_as_bool(item.get("required", True)),
                    document_selector=DocumentSelector(
                        doc_ids=_as_str_list(selector.get("doc_ids", [])),
                        source_kinds=_as_str_list(selector.get("source_kinds", [])),
                        content_roles=_as_str_list(selector.get("content_roles", [])),
                        path_contains=_as_str_list(selector.get("path_contains", [])),
                        path_prefixes=_as_str_list(selector.get("path_prefixes", [])),
                        exclude_path_contains=_as_str_list(selector.get("exclude_path_contains", [])),
                        max_documents=_as_int(selector.get("max_documents", 0)),
                        min_role_score=_as_float(selector.get("min_role_score", 0)),
                    ),
                    reason=str(item.get("reason", "")),
                )
            )
        secondary_intents = _as_str_list(data.get("secondary_intents", []))
        return LLMPlanProposal(
            primary_intent=str(data.get("primary_intent", "")),
            secondary_intents=secondary_intents,
            target_schema=str(data.get("target_schema", "")),
            steps=steps,
            assumptions=_as_str_list(data.get("assumptions", [])),
            reasons=_as_str_list(data.get("reasons", [])),
        )


@dataclass
class ValidationOutcome:
    """注释：LLM plan 校验结果。"""

    ok: bool
    errors: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)


class LLMPlanValidator:
    """注释：校验 LLMPlanProposal 是否能安全转成 FormationPlan。"""

    def __init__(self, registry: OperatorRegistry, config: LLMPlannerConfig) -> None:
        self.registry = registry
        self.config = config

    def validate(self, proposal: LLMPlanProposal, document_roles: list[DocumentRoleProfile]) -> ValidationOutcome:
        """注释：校验 intent/schema/operator/selector/预算。"""
        errors: list[str] = []
        soft_warnings: list[str] = []
        repairs: list[str] = []
        doc_ids = {role.doc_id for role in document_roles}
        source_kinds = {role.source_kind for role in document_roles}
        allowed_roles = {
            "implementation",
            "test",
            "documentation",
            "configuration",
            "build_script",
            "run_script",
            "log",
            "dataset",
            "unknown",
        }
        if proposal.primary_intent not in SUPPORTED_INTENTS:
            errors.append(f"invalid primary_intent: {proposal.primary_intent}")
        invalid_secondary = [intent for intent in proposal.secondary_intents if intent not in SUPPORTED_INTENTS]
        if invalid_secondary:
            errors.append(f"invalid secondary_intents: {invalid_secondary}")
        if proposal.target_schema not in SCHEMA_FAMILY_BY_SCHEMA:
            errors.append(f"invalid target_schema: {proposal.target_schema}")
        if not proposal.steps:
            errors.append("proposal has no steps")
        if len(proposal.steps) > self.config.max_steps:
            errors.append(f"too many steps: {len(proposal.steps)} > {self.config.max_steps}")

        seen_step_ids = set()
        seen_step_signatures: dict[tuple[str, tuple[tuple[str, Any], ...], str], str] = {}
        for step in proposal.steps:
            if not step.step_id:
                errors.append("step missing step_id")
            if step.step_id in seen_step_ids:
                errors.append(f"duplicate step_id: {step.step_id}")
            seen_step_ids.add(step.step_id)
            if step.operator_name not in self.registry.names():
                errors.append(f"invalid operator: {step.operator_name}")
            elif self.registry.get(step.operator_name).spec.requires_llm:
                errors.append(f"{step.step_id} uses LLM-enhanced operator not allowed in Version 5: {step.operator_name}")
            selector = step.document_selector
            missing_doc_ids = [doc_id for doc_id in selector.doc_ids if doc_id not in doc_ids]
            if missing_doc_ids:
                errors.append(f"{step.step_id} references missing doc_ids: {missing_doc_ids}")
            invalid_source_kinds = [kind for kind in selector.source_kinds if kind not in source_kinds]
            if invalid_source_kinds:
                errors.append(f"{step.step_id} has invalid source_kinds: {invalid_source_kinds}")
            invalid_roles = [role for role in selector.content_roles if role not in allowed_roles]
            if invalid_roles:
                errors.append(f"{step.step_id} has invalid content_roles: {invalid_roles}")
            if selector.max_documents > self.config.max_documents_per_step:
                selector.max_documents = self.config.max_documents_per_step
                repairs.append(f"{step.step_id} max_documents clipped to {self.config.max_documents_per_step}")
            selected_count = _count_selected_docs(selector, document_roles)
            if step.required and selected_count == 0:
                errors.append(f"{step.step_id} required selector matched zero documents")
            if not step.required and selected_count == 0:
                soft_warnings.append(f"{step.step_id} optional selector matched zero documents")
            if _selector_is_broad(selector) and selected_count > self.config.max_documents_per_step:
                soft_warnings.append(f"{step.step_id} uses broad selector over {selected_count} documents")
            signature = (step.operator_name, _selector_signature(selector), _normalize_reason(step.reason))
            previous_step_id = seen_step_signatures.get(signature)
            if previous_step_id:
                soft_warnings.append(
                    f"{step.step_id} duplicates operator/selector/reason of {previous_step_id}; consider merging"
                )
            else:
                seen_step_signatures[signature] = step.step_id
        return ValidationOutcome(ok=not errors, errors=errors, soft_warnings=soft_warnings, repairs=repairs)


class LLMMetaController:
    """注释：Version 5 LLM Planner MetaController；只规划，不执行 pipeline。"""

    def __init__(
        self,
        config: LLMPlannerConfig | None = None,
        llm_client: LLMClient | None = None,
        registry: OperatorRegistry | None = None,
    ) -> None:
        self.config = config or LLMPlannerConfig()
        self.registry = registry or create_default_registry()
        self.rule_controller = RuleBasedMetaController(self.registry)
        self.workspace_profiler = WorkspaceProfiler()
        self.role_profiler = DocumentRoleProfiler()
        self.brief_builder = WorkspaceBriefBuilder(self.registry)
        self.meta_memory_provider = FormationMetaMemoryProvider(self.config.meta_memory_path or None)
        self.prompt_builder = LLMPlannerPromptBuilder()
        self.parser = LLMPlanParser()
        self.validator = LLMPlanValidator(self.registry, self.config)
        self.llm_client = llm_client or OpenAICompatibleLLMClient(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout_seconds=self.config.timeout_seconds,
        )

    def plan_from_query(self, documents: list[MemoryInputDocument], query: str) -> LLMPlanningResult:
        """注释：按 planner_mode 生成 plan；llm_preferred 失败后 abort。"""
        if self.config.planner_mode == "rule_only":
            rule_result = self.rule_controller.plan_from_query(documents, query)
            rule_result.decision_trace.planner_mode = "rule_only"
            return LLMPlanningResult(
                ok=True,
                plan=rule_result.plan,
                decision_trace=rule_result.decision_trace,
                workspace_profile=rule_result.workspace_profile,
                document_roles=rule_result.document_roles,
                workspace_brief=rule_result.plan.metadata.get("workspace_brief", {}),
            )
        if self.config.planner_mode != "llm_preferred":
            raise ValueError(f"unsupported planner_mode: {self.config.planner_mode}")

        workspace_profile = self.workspace_profiler.profile(documents)
        document_roles = self.role_profiler.profile(documents)
        workspace_brief = self.brief_builder.build(query, workspace_profile, document_roles)
        meta_memory = self.meta_memory_provider.load_reference()
        raw_outputs: list[str] = []
        prompt_messages: list[list[dict[str, str]]] = []
        validation_errors: list[str] = []
        soft_warnings: list[str] = []
        repairs: list[str] = []
        parse_ok = False

        for attempt in range(1, self.config.max_replan_attempts + 1):
            try:
                messages = self.prompt_builder.build_messages(query, workspace_brief, meta_memory, validation_errors)
                if self.config.save_prompt_messages:
                    prompt_messages.append(messages)
                raw_output = self.llm_client.complete(
                    messages=messages,
                    model=self.config.model,
                    temperature=self.config.temperature,
                )
                if self.config.save_raw_output:
                    raw_outputs.append(raw_output)
                proposal = self.parser.parse(raw_output)
                parse_ok = True
            except Exception as exc:
                validation_errors = [f"LLM parse/call failed on attempt {attempt}: {exc}"]
                continue

            outcome = self.validator.validate(proposal, document_roles)
            repairs.extend(outcome.repairs)
            soft_warnings.extend(outcome.soft_warnings)
            if outcome.ok:
                plan = _proposal_to_plan(proposal)
                trace = _llm_trace(
                    proposal=proposal,
                    workspace_profile=workspace_profile,
                    prompt_messages=prompt_messages,
                    raw_outputs=raw_outputs,
                    validation_errors=[],
                    soft_warnings=soft_warnings,
                    repairs=repairs,
                    retry_count=attempt,
                    config=self.config,
                    final_plan_source="llm" if not repairs else "llm_repaired",
                )
                return LLMPlanningResult(
                    ok=True,
                    plan=plan,
                    decision_trace=trace,
                    workspace_profile=workspace_profile,
                    document_roles=document_roles,
                    workspace_brief=workspace_brief,
                )
            validation_errors = outcome.errors

        trace = PlanDecisionTrace(
            task_intent="",
            original_intent=query,
            normalized_intent="",
            workspace_signals=_workspace_signals(workspace_profile),
            warnings=["LLM planner failed; abort without executing partial plan"],
            planner_mode="llm_preferred",
            llm_called=True,
            llm_model=self.config.model,
            llm_prompt_messages=prompt_messages,
            llm_raw_outputs=raw_outputs,
            llm_parse_ok=parse_ok,
            llm_validation_errors=validation_errors,
            llm_soft_warnings=soft_warnings,
            llm_repairs=repairs,
            fallback_used=False,
            final_plan_source="abort",
            planner_retry_count=self.config.max_replan_attempts,
            issue_type="llm_planning_failed",
            issue_for_meta_memory=True,
        )
        return LLMPlanningResult(
            ok=False,
            plan=None,
            decision_trace=trace,
            workspace_profile=workspace_profile,
            document_roles=document_roles,
            workspace_brief=workspace_brief,
            error="LLM planner failed after max_replan_attempts",
        )


def _proposal_to_plan(proposal: LLMPlanProposal) -> FormationPlan:
    """注释：将已校验 proposal 转为 FormationPlan。"""
    return FormationPlan(
        plan_id=f"llm_{proposal.primary_intent}_v1",
        target_schema=proposal.target_schema,
        task_intent=proposal.primary_intent,
        steps=[
            FormationStep(
                operator_name=step.operator_name,
                step_id=step.step_id,
                required=step.required,
                document_selector=step.document_selector,
                params=step.document_selector.to_params(),
            )
            for step in proposal.steps
        ],
        metadata={
            "controller": "LLMMetaController",
            "primary_intent": proposal.primary_intent,
            "secondary_intents": proposal.secondary_intents,
            "assumptions": proposal.assumptions,
            "reasons": proposal.reasons,
            "schema_family": SCHEMA_FAMILY_BY_SCHEMA.get(proposal.target_schema, ""),
        },
    )


def _llm_trace(
    proposal: LLMPlanProposal,
    workspace_profile: WorkspaceProfile,
    prompt_messages: list[list[dict[str, str]]],
    raw_outputs: list[str],
    validation_errors: list[str],
    soft_warnings: list[str],
    repairs: list[str],
    retry_count: int,
    config: LLMPlannerConfig,
    final_plan_source: str,
) -> PlanDecisionTrace:
    """注释：构造成功 LLM planning trace。"""
    selected = [step.operator_name for step in proposal.steps]
    return PlanDecisionTrace(
        task_intent=proposal.primary_intent,
        original_intent=proposal.primary_intent,
        normalized_intent=proposal.primary_intent,
        workspace_signals=_workspace_signals(workspace_profile),
        selected_target_schema=proposal.target_schema,
        selected_schema_family=SCHEMA_FAMILY_BY_SCHEMA.get(proposal.target_schema, ""),
        selected_operators=selected,
        skipped_operators=[],
        reasons=proposal.reasons,
        operator_reasons={step.step_id: [f"{step.operator_name}: {step.reason}"] for step in proposal.steps},
        selection_reasons={step.step_id: ["selected by LLM proposal"] for step in proposal.steps},
        selected_doc_counts={step.step_id: len(step.document_selector.doc_ids) for step in proposal.steps},
        warnings=[],
        planner_mode=config.planner_mode,
        llm_called=True,
        llm_model=config.model,
        llm_prompt_messages=prompt_messages,
        llm_raw_outputs=raw_outputs,
        llm_parse_ok=True,
        llm_validation_errors=validation_errors,
        llm_soft_warnings=soft_warnings,
        llm_repairs=repairs,
        fallback_used=False,
        final_plan_source=final_plan_source,
        planner_retry_count=retry_count,
    )


def _extract_json_object(raw_output: str) -> str:
    """注释：从 LLM 输出中提取首个 JSON object。"""
    text = _strip_code_fence(raw_output.strip())
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in LLM output")
    return text[start : end + 1]


def _strip_code_fence(text: str) -> str:
    """注释：兼容 ```json ... ``` 包裹的模型输出。"""
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return text


def _as_str_list(value: Any) -> list[str]:
    """注释：把真实 LLM 常见的 scalar/null/list 输出规整为字符串列表。"""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _as_int(value: Any) -> int:
    """注释：宽解析整数；非法值留给预算/selector 逻辑处理为 0。"""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    """注释：宽解析浮点数。"""
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _as_bool(value: Any) -> bool:
    """注释：兼容 required=false/\"false\" 等常见 LLM 输出。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "none", ""}
    return bool(value)


def _selector_signature(selector: DocumentSelector) -> tuple[tuple[str, Any], ...]:
    """注释：生成可比较 selector 签名，用于重复 step 检查。"""
    return tuple(
        sorted(
            {
                "doc_ids": tuple(selector.doc_ids),
                "source_kinds": tuple(selector.source_kinds),
                "content_roles": tuple(selector.content_roles),
                "path_contains": tuple(selector.path_contains),
                "path_prefixes": tuple(selector.path_prefixes),
                "exclude_path_contains": tuple(selector.exclude_path_contains),
                "max_documents": selector.max_documents,
                "min_role_score": selector.min_role_score,
            }.items()
        )
    )


def _normalize_reason(reason: str) -> str:
    """注释：重复检测中使用的轻量 purpose 规整。"""
    return " ".join((reason or "").strip().lower().split())


def _selector_is_broad(selector: DocumentSelector) -> bool:
    """注释：没有过滤条件且不设 max_documents 的 selector 属于宽泛选择。"""
    return not any(
        [
            selector.doc_ids,
            selector.source_kinds,
            selector.content_roles,
            selector.path_contains,
            selector.path_prefixes,
            selector.exclude_path_contains,
            selector.max_documents,
            selector.min_role_score,
        ]
    )


def _count_selected_docs(selector: DocumentSelector, roles: list[DocumentRoleProfile]) -> int:
    """注释：计算 selector 命中的文档数量。"""
    selected = roles
    if selector.doc_ids:
        doc_ids = set(selector.doc_ids)
        selected = [role for role in selected if role.doc_id in doc_ids]
    if selector.source_kinds:
        source_kinds = set(selector.source_kinds)
        selected = [role for role in selected if role.source_kind in source_kinds]
    if selector.content_roles:
        content_roles = set(selector.content_roles)
        selected = [role for role in selected if content_roles.intersection(role.content_roles)]
    path_contains = [item.lower() for item in selector.path_contains]
    if path_contains:
        selected = [role for role in selected if any(item in role.relative_path.lower() for item in path_contains)]
    path_prefixes = [item.lower() for item in selector.path_prefixes]
    if path_prefixes:
        selected = [role for role in selected if any(role.relative_path.lower().startswith(item) for item in path_prefixes)]
    excluded = [item.lower() for item in selector.exclude_path_contains]
    if excluded:
        selected = [role for role in selected if not any(item in role.relative_path.lower() for item in excluded)]
    if selector.min_role_score > 0:
        selected = [role for role in selected if role.importance_score >= selector.min_role_score]
    if selector.max_documents > 0:
        selected = selected[: selector.max_documents]
    return len(selected)
