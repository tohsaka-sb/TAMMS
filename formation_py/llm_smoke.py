"""
模块: formation_py.llm_smoke
职责: 手动真实 API smoke test，不纳入默认 pytest。
输入: workspace、query、可选 api/base/model 参数或 TAMMS_LLM_* 环境变量。
输出: LLM planner 连通性、JSON 解析与 FormationPlan 生成结果。
副作用: 会调用外部 LLM API；不写缓存、不写 SQLite，默认不执行 pipeline。
失败处理: 缺少 API 配置时跳过；规划失败时返回非零状态并打印 trace 摘要。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from harness_py.reader import HeterogeneousReader
from harness_py.scanner import FileScanner

from .llm_planner import LLMMetaController, LLMPlannerConfig
from .pipeline import FormationPipeline


def main(argv: Sequence[str] | None = None) -> int:
    """注释：命令行入口；真实 API smoke test 由用户手动触发。"""
    parser = argparse.ArgumentParser(description="Run a manual LLM planner smoke test.")
    parser.add_argument("--workspace", default="./memory_system_py", help="workspace to scan")
    parser.add_argument("--query", default="总结这个项目结构", help="planner query")
    parser.add_argument("--api-key", default=os.getenv("TAMMS_LLM_API_KEY", ""), help="OpenAI-compatible API key")
    parser.add_argument("--base-url", default=os.getenv("TAMMS_LLM_BASE_URL", ""), help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=os.getenv("TAMMS_LLM_MODEL", ""), help="model name")
    parser.add_argument("--limit-files", type=int, default=12, help="max files to include in smoke workspace")
    parser.add_argument("--attempts", type=int, default=1, help="max LLM planning attempts")
    parser.add_argument("--timeout-seconds", type=int, default=45, help="HTTP timeout")
    parser.add_argument("--execute-pipeline", action="store_true", help="also run deterministic FormationPipeline")
    args = parser.parse_args(argv)

    if not args.api_key or not args.base_url or not args.model:
        print("smoke_skipped=True")
        print("reason=missing api_key/base_url/model; set TAMMS_LLM_API_KEY, TAMMS_LLM_BASE_URL, TAMMS_LLM_MODEL")
        return 0

    workspace = Path(args.workspace)
    documents = _read_documents(workspace, args.limit_files)
    if not documents:
        print("smoke_ok=False")
        print(f"error=no readable documents under {workspace}")
        return 2

    config = LLMPlannerConfig(
        planner_mode="llm_preferred",
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_replan_attempts=args.attempts,
        timeout_seconds=args.timeout_seconds,
    )
    result = LLMMetaController(config=config).plan_from_query(documents, args.query)

    print(f"smoke_ok={result.ok}")
    print(f"document_count={len(documents)}")
    print(f"final_plan_source={result.decision_trace.final_plan_source}")
    print(f"planner_retry_count={result.decision_trace.planner_retry_count}")
    print(f"llm_parse_ok={result.decision_trace.llm_parse_ok}")
    if result.decision_trace.llm_validation_errors:
        print("validation_errors=" + " | ".join(result.decision_trace.llm_validation_errors))
    if result.decision_trace.llm_soft_warnings:
        print("soft_warnings=" + " | ".join(result.decision_trace.llm_soft_warnings))
    if result.decision_trace.llm_repairs:
        print("repairs=" + " | ".join(result.decision_trace.llm_repairs))

    if not result.ok or result.plan is None:
        return 1

    print(f"plan_id={result.plan.plan_id}")
    print(f"task_intent={result.plan.task_intent}")
    print(f"target_schema={result.plan.target_schema}")
    print("steps=" + ",".join(f"{step.step_id}:{step.operator_name}" for step in result.plan.steps))

    if args.execute_pipeline:
        run = FormationPipeline().run(documents, result.plan)
        print(f"pipeline_ok={run.ok}")
        print(f"artifact_count={len(run.artifacts)}")
        if run.errors:
            print("pipeline_errors=" + " | ".join(run.errors))
        return 0 if run.ok else 3
    return 0


def _read_documents(workspace: Path, limit_files: int):
    """注释：扫描少量文件，避免真实 API smoke prompt 过大。"""
    reader = HeterogeneousReader()
    documents = []
    for item in FileScanner(workspace).scan()[:limit_files]:
        documents.append(reader.read_document(item))
    return documents


if __name__ == "__main__":
    raise SystemExit(main())
