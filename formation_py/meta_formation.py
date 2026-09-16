"""
模块: formation_py.meta_formation
职责: 将 formation planning/execution 过程压缩为 episode，并生成元记忆更新候选。
输入: LLMPlanningResult/MetaControllerResult、可选 FormationRun、当前 meta-memory 文本。
输出: FormationEpisode、MetaMemoryUpdateCandidate、review decision。
副作用: 无；只构造候选，不写文件。
失败处理: 缺少 plan/run 时生成 failure episode 或空候选，不抛错。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Protocol

from .types import FormationRun


STAGE1_ENGINEERING = "stage1_engineering"
STAGE2_QA_DATASET = "stage2_qa_dataset"
STAGE3_HUMAN_PREFERENCE = "stage3_human_preference"

VALID_STAGES = {STAGE1_ENGINEERING, STAGE2_QA_DATASET, STAGE3_HUMAN_PREFERENCE}


@dataclass
class FormationEpisode:
    """注释：一次 formation planning/execution 循环的可审计摘要。"""

    episode_id: str
    timestamp: str
    stage: str = STAGE1_ENGINEERING
    source: str = "formation_runtime"
    query: str = ""
    workspace_brief_hash: str = ""
    workspace_summary: dict[str, Any] = field(default_factory=dict)
    planner_mode: str = ""
    final_plan_source: str = ""
    primary_intent: str = ""
    secondary_intents: list[str] = field(default_factory=list)
    target_schema: str = ""
    selected_operators: list[str] = field(default_factory=list)
    planner_retry_count: int = 0
    llm_parse_ok: bool = False
    hard_errors: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)
    formation_run_ok: bool | None = None
    artifact_count: int | None = None
    warning_count: int | None = None
    error_count: int | None = None
    issue_type: str = ""
    issue_for_meta_memory: bool = False
    evaluation_signals: dict[str, Any] = field(default_factory=dict)
    qa_dataset_id: str = ""
    qa_item_ids: list[str] = field(default_factory=list)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    current_metrics: dict[str, Any] = field(default_factory=dict)
    human_feedback: dict[str, Any] = field(default_factory=dict)
    review_status: str = "not_required"
    reviewer_comment: str = ""


@dataclass
class MetaMemoryUpdateCandidate:
    """注释：准备写入 formation_meta_memory.md 的带 ID 候选更新。"""

    candidate_id: str
    proposed_memory_id: str
    episode_id: str
    stage: str
    update_type: str
    section: str
    content: str
    evidence: list[str] = field(default_factory=list)
    confidence: float = 0.5
    status: str = "proposed"
    requires_human_review: bool = False
    review_status: str = "not_required"
    reviewer_comment: str = ""
    human_feedback: dict[str, Any] = field(default_factory=dict)

    def to_markdown_bullet(self) -> str:
        """注释：统一 Markdown bullet 格式，便于审计和回滚。"""
        evidence = "; ".join(self.evidence) if self.evidence else "none"
        return (
            f"- [{self.proposed_memory_id}] {self.content}  \n"
            f"  Evidence: {evidence}; stage={self.stage}; "
            f"candidate={self.candidate_id}; confidence={self.confidence:.2f}."
        )


@dataclass
class MetaMemoryUpdateDecision:
    """注释：Stage 3 人工偏好接口占位。"""

    candidate_id: str
    action: str
    revised_content: str = ""
    reason: str = ""
    reviewer: str = ""


class MetaMemoryReflector(Protocol):
    """注释：元记忆候选生成器协议。"""

    def propose_updates(self, episode: FormationEpisode, current_meta_memory: str = "") -> list[MetaMemoryUpdateCandidate]:
        """注释：根据 episode 生成候选更新。"""


class MetaFormationIdFactory:
    """注释：生成 FE/MMC/MM/UH 风格 ID；测试可注入固定日期。"""

    def __init__(self, date: str | None = None) -> None:
        self.date = date or datetime.now().strftime("%Y%m%d")
        self._counters: dict[str, int] = {}

    def next_id(self, prefix: str) -> str:
        """注释：返回如 FE-20260705-0001 的 ID。"""
        current = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = current
        return f"{prefix}-{self.date}-{current:04d}"


class FormationEpisodeBuilder:
    """注释：从 planning result 和可选 FormationRun 构造 FormationEpisode。"""

    def __init__(self, id_factory: MetaFormationIdFactory | None = None) -> None:
        self.id_factory = id_factory or MetaFormationIdFactory()

    def build(
        self,
        planning_result: Any,
        formation_run: FormationRun | None = None,
        stage: str = STAGE1_ENGINEERING,
        source: str = "formation_runtime",
        evaluation_signals: dict[str, Any] | None = None,
        qa_dataset_id: str = "",
        qa_item_ids: list[str] | None = None,
        baseline_metrics: dict[str, Any] | None = None,
        current_metrics: dict[str, Any] | None = None,
        human_feedback: dict[str, Any] | None = None,
        review_status: str = "not_required",
        reviewer_comment: str = "",
    ) -> FormationEpisode:
        """注释：兼容 LLMPlanningResult 与规则 MetaControllerResult。"""
        stage = stage if stage in VALID_STAGES else STAGE1_ENGINEERING
        trace = getattr(planning_result, "decision_trace", None)
        plan = getattr(planning_result, "plan", None)
        workspace_profile = getattr(planning_result, "workspace_profile", None)
        workspace_brief = getattr(planning_result, "workspace_brief", {}) or {}
        plan_metadata = getattr(plan, "metadata", {}) or {}
        selected_operators = list(getattr(trace, "selected_operators", []) or [])
        if not selected_operators and plan:
            selected_operators = [step.operator_name for step in plan.steps]

        return FormationEpisode(
            episode_id=self.id_factory.next_id("FE"),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            stage=stage,
            source=source,
            query=str(getattr(trace, "original_intent", "") or workspace_brief.get("query", "")),
            workspace_brief_hash=_stable_hash(workspace_brief),
            workspace_summary=_workspace_summary(workspace_profile),
            planner_mode=str(getattr(trace, "planner_mode", "")),
            final_plan_source=str(getattr(trace, "final_plan_source", "")),
            primary_intent=str(getattr(plan, "task_intent", "") or getattr(trace, "task_intent", "")),
            secondary_intents=list(plan_metadata.get("secondary_intents", []) or []),
            target_schema=str(getattr(plan, "target_schema", "") or getattr(trace, "selected_target_schema", "")),
            selected_operators=selected_operators,
            planner_retry_count=int(getattr(trace, "planner_retry_count", 0) or 0),
            llm_parse_ok=bool(getattr(trace, "llm_parse_ok", False)),
            hard_errors=list(getattr(trace, "llm_validation_errors", []) or []),
            soft_warnings=list(getattr(trace, "llm_soft_warnings", []) or []),
            repairs=list(getattr(trace, "llm_repairs", []) or []),
            formation_run_ok=formation_run.ok if formation_run else None,
            artifact_count=len(formation_run.artifacts) if formation_run else None,
            warning_count=len(formation_run.warnings) if formation_run else None,
            error_count=len(formation_run.errors) if formation_run else None,
            issue_type=str(getattr(trace, "issue_type", "")),
            issue_for_meta_memory=bool(getattr(trace, "issue_for_meta_memory", False)),
            evaluation_signals=evaluation_signals or {},
            qa_dataset_id=qa_dataset_id,
            qa_item_ids=qa_item_ids or [],
            baseline_metrics=baseline_metrics or {},
            current_metrics=current_metrics or {},
            human_feedback=human_feedback or {},
            review_status=review_status,
            reviewer_comment=reviewer_comment,
        )


class RuleBasedMetaMemoryReflector:
    """注释：确定性 Stage 1 reflector，把 episode 转为可审核候选。"""

    def __init__(self, id_factory: MetaFormationIdFactory | None = None) -> None:
        self.id_factory = id_factory or MetaFormationIdFactory()

    def propose_updates(self, episode: FormationEpisode, current_meta_memory: str = "") -> list[MetaMemoryUpdateCandidate]:
        """注释：按成功、失败、warning/repair 触发候选。"""
        candidates: list[MetaMemoryUpdateCandidate] = []
        if _is_success_episode(episode):
            candidates.append(self._candidate(episode, "successful_pattern", "Successful Patterns", _success_content(episode), 0.72))
        if _is_failure_episode(episode):
            candidates.append(self._candidate(episode, "failure_lesson", "Failure Lessons", _failure_content(episode), 0.74))
        if episode.soft_warnings:
            candidates.append(self._candidate(episode, "anti_pattern", "Anti-Patterns", _warning_content(episode), 0.66))
        if episode.repairs:
            candidates.append(self._candidate(episode, "selector_guideline", "Selector Guidelines", _repair_content(episode), 0.68))
        return candidates

    def _candidate(
        self,
        episode: FormationEpisode,
        update_type: str,
        section: str,
        content: str,
        confidence: float,
    ) -> MetaMemoryUpdateCandidate:
        return MetaMemoryUpdateCandidate(
            candidate_id=self.id_factory.next_id("MMC"),
            proposed_memory_id=self.id_factory.next_id("MM"),
            episode_id=episode.episode_id,
            stage=episode.stage,
            update_type=update_type,
            section=section,
            content=content,
            evidence=_episode_evidence(episode),
            confidence=confidence,
            requires_human_review=episode.stage == STAGE3_HUMAN_PREFERENCE,
            review_status="pending" if episode.stage == STAGE3_HUMAN_PREFERENCE else "not_required",
        )


class LLMMetaMemoryReflector:
    """注释：LLM reflector 接口初稿；默认不在测试或 smoke 中调用真实 API。"""

    def __init__(self, llm_client: Any, model: str = "", temperature: float = 0.0, id_factory: MetaFormationIdFactory | None = None) -> None:
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.id_factory = id_factory or MetaFormationIdFactory()

    def propose_updates(self, episode: FormationEpisode, current_meta_memory: str = "") -> list[MetaMemoryUpdateCandidate]:
        """注释：要求 LLM 返回 candidates JSON；写回仍由 writer 审核处理。"""
        messages = [
            {"role": "system", "content": "Propose meta-memory update candidates as valid JSON only."},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "episode": episode.__dict__,
                        "current_meta_memory_excerpt": current_meta_memory[:4000],
                        "required_schema": {
                            "candidates": [
                                {
                                    "update_type": "successful_pattern | failure_lesson | selector_guideline | anti_pattern",
                                    "section": "Markdown section",
                                    "content": "one reusable strategy bullet without memory id",
                                    "evidence": ["short evidence item"],
                                    "confidence": 0.5,
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        raw = self.llm_client.complete(messages=messages, model=self.model, temperature=self.temperature)
        data = json.loads(_extract_json_object(raw))
        candidates = []
        for item in data.get("candidates", []) or []:
            candidates.append(
                MetaMemoryUpdateCandidate(
                    candidate_id=self.id_factory.next_id("MMC"),
                    proposed_memory_id=self.id_factory.next_id("MM"),
                    episode_id=episode.episode_id,
                    stage=episode.stage,
                    update_type=str(item.get("update_type", "operator_guideline")),
                    section=str(item.get("section", "Operator Guidelines")),
                    content=str(item.get("content", "")),
                    evidence=[str(value) for value in item.get("evidence", []) or []],
                    confidence=float(item.get("confidence", 0.5) or 0.5),
                    requires_human_review=episode.stage == STAGE3_HUMAN_PREFERENCE,
                    review_status="pending" if episode.stage == STAGE3_HUMAN_PREFERENCE else "not_required",
                )
            )
        return candidates


def apply_review_decision(candidate: MetaMemoryUpdateCandidate, decision: MetaMemoryUpdateDecision) -> MetaMemoryUpdateCandidate:
    """注释：Stage 3 人工反馈接口初稿，返回更新后的 candidate。"""
    if decision.candidate_id != candidate.candidate_id:
        raise ValueError("decision candidate_id does not match candidate")
    action = decision.action.strip().lower()
    if action not in {"accept", "reject", "revise"}:
        raise ValueError(f"unsupported review action: {decision.action}")
    if action == "reject":
        status = "rejected"
        content = candidate.content
    elif action == "revise":
        status = "accepted"
        content = decision.revised_content or candidate.content
    else:
        status = "accepted"
        content = candidate.content
    return replace(
        candidate,
        status=status,
        content=content,
        review_status=action,
        reviewer_comment=decision.reason,
        human_feedback={"reviewer": decision.reviewer, "action": action, "reason": decision.reason},
    )


def _is_success_episode(episode: FormationEpisode) -> bool:
    return bool(episode.formation_run_ok and (episode.artifact_count or 0) > 0 and episode.final_plan_source in {"llm", "llm_repaired"})


def _is_failure_episode(episode: FormationEpisode) -> bool:
    return bool(
        episode.issue_for_meta_memory
        or episode.planner_retry_count >= 2
        or episode.hard_errors
        or episode.formation_run_ok is False
    )


def _success_content(episode: FormationEpisode) -> str:
    chain = _operator_chain(episode)
    workspace = _workspace_label(episode)
    return (
        f"**{episode.primary_intent or 'unknown'} / {episode.target_schema or 'unknown'}**: "
        f"For {workspace} workspaces, `{chain}` is a reusable formation plan for this intent."
    )


def _failure_content(episode: FormationEpisode) -> str:
    issue = episode.issue_type or (episode.hard_errors[0] if episode.hard_errors else "planning_or_execution_failure")
    return (
        f"**{episode.primary_intent or 'unknown'} / failure**: When `{issue}` appears, revise the plan before execution; "
        f"prefer valid operators, existing document selectors, and evidence-preserving fallbacks."
    )


def _warning_content(episode: FormationEpisode) -> str:
    warning = episode.soft_warnings[0] if episode.soft_warnings else "soft warning"
    return (
        f"**{episode.primary_intent or 'unknown'} / anti-pattern**: Avoid plans that trigger `{warning}`; "
        f"make repeated operators differ by selector and purpose."
    )


def _repair_content(episode: FormationEpisode) -> str:
    repair = episode.repairs[0] if episode.repairs else "repair"
    return (
        f"**{episode.primary_intent or 'unknown'} / selector**: If planner validation requires `{repair}`, "
        f"prefer tighter selectors and explicit document budgets in future plans."
    )


def _episode_evidence(episode: FormationEpisode) -> list[str]:
    evidence = [
        f"episode={episode.episode_id}",
        f"query={episode.query!r}",
        f"plan={_operator_chain(episode)}",
    ]
    if episode.planner_retry_count:
        evidence.append(f"retry_count={episode.planner_retry_count}")
    if episode.artifact_count is not None:
        evidence.append(f"artifact_count={episode.artifact_count}")
    return evidence


def _operator_chain(episode: FormationEpisode) -> str:
    return " -> ".join(episode.selected_operators) if episode.selected_operators else "none"


def _workspace_label(episode: FormationEpisode) -> str:
    summary = episode.workspace_summary
    pieces = [
        summary.get("complexity_level", ""),
        summary.get("organization_level", ""),
        summary.get("dominant_language_hint", ""),
        summary.get("dominant_source_kind", ""),
    ]
    label = " ".join(piece for piece in pieces if piece)
    return label or "unknown"


def _workspace_summary(workspace_profile: Any) -> dict[str, Any]:
    if not workspace_profile:
        return {}
    return {
        "document_count": getattr(workspace_profile, "document_count", 0),
        "dominant_source_kind": getattr(workspace_profile, "dominant_source_kind", ""),
        "dominant_language_hint": getattr(workspace_profile, "dominant_language_hint", ""),
        "complexity_level": getattr(workspace_profile, "complexity_level", ""),
        "organization_level": getattr(workspace_profile, "organization_level", ""),
        "has_tests": getattr(workspace_profile, "has_tests", False),
        "has_build_files": getattr(workspace_profile, "has_build_files", False),
        "has_logs": getattr(workspace_profile, "has_logs", False),
    }


def _stable_hash(value: Any) -> str:
    data = json.dumps(value or {}, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def _extract_json_object(raw_output: str) -> str:
    text = raw_output.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in LLM output")
    return text[start : end + 1]
