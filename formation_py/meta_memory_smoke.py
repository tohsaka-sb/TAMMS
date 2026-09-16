"""
模块: formation_py.meta_memory_smoke
职责: Version 6.1 元记忆更新 smoke：planning/execution -> episode -> candidate -> preview/apply。
输入: workspace、query、LLM API 配置、apply 开关。
输出: FormationEpisode 摘要、MetaMemoryUpdateCandidate preview、可选 apply 结果。
副作用: 默认不写文件；--apply 时备份并追加 formation_meta_memory.md。
失败处理: 缺少 API 配置时跳过；planning 失败时仍生成 failure episode 候选。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from harness_py.reader import HeterogeneousReader
from harness_py.scanner import FileScanner

from .llm_planner import LLMMetaController, LLMPlannerConfig
from .meta_formation import FormationEpisodeBuilder, RuleBasedMetaMemoryReflector, STAGE1_ENGINEERING
from .meta_memory import FormationMetaMemoryProvider, FormationMetaMemoryWriter
from .pipeline import FormationPipeline


def main(argv: Sequence[str] | None = None) -> int:
    """注释：命令行入口；默认 preview，不自动写回。"""
    parser = argparse.ArgumentParser(description="Preview or apply staged meta-memory updates.")
    parser.add_argument("--workspace", default="./memory_system_py", help="workspace to scan")
    parser.add_argument("--query", default="总结这个项目结构", help="planner query")
    parser.add_argument("--stage", default=STAGE1_ENGINEERING, help="update stage")
    parser.add_argument("--api-key", default=os.getenv("TAMMS_LLM_API_KEY", ""), help="OpenAI-compatible API key")
    parser.add_argument("--base-url", default=os.getenv("TAMMS_LLM_BASE_URL", ""), help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=os.getenv("TAMMS_LLM_MODEL", ""), help="model name")
    parser.add_argument("--limit-files", type=int, default=12, help="max files to include")
    parser.add_argument("--attempts", type=int, default=1, help="max LLM planning attempts")
    parser.add_argument("--timeout-seconds", type=int, default=45, help="HTTP timeout")
    parser.add_argument("--meta-memory-path", default="", help="override formation_meta_memory.md path")
    parser.add_argument("--apply", action="store_true", help="apply candidates to Markdown after preview")
    args = parser.parse_args(argv)

    if not args.api_key or not args.base_url or not args.model:
        print("meta_memory_smoke_skipped=True")
        print("reason=missing api_key/base_url/model; set TAMMS_LLM_API_KEY, TAMMS_LLM_BASE_URL, TAMMS_LLM_MODEL")
        return 0

    documents = _read_documents(Path(args.workspace), args.limit_files)
    if not documents:
        print("meta_memory_smoke_ok=False")
        print(f"error=no readable documents under {args.workspace}")
        return 2

    config = LLMPlannerConfig(
        planner_mode="llm_preferred",
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_replan_attempts=args.attempts,
        timeout_seconds=args.timeout_seconds,
        meta_memory_path=args.meta_memory_path,
    )
    planning_result = LLMMetaController(config=config).plan_from_query(documents, args.query)
    formation_run = FormationPipeline().run(documents, planning_result.plan) if planning_result.plan else None
    episode = FormationEpisodeBuilder().build(
        planning_result=planning_result,
        formation_run=formation_run,
        stage=args.stage,
        source="meta_memory_smoke",
    )
    provider = FormationMetaMemoryProvider(args.meta_memory_path or None)
    current_meta_memory = provider.load_reference()
    candidates = RuleBasedMetaMemoryReflector().propose_updates(episode, current_meta_memory)
    writer = FormationMetaMemoryWriter(args.meta_memory_path or None)

    print("meta_memory_smoke_ok=True")
    print(f"episode_id={episode.episode_id}")
    print(f"planning_ok={planning_result.ok}")
    print(f"formation_run_ok={episode.formation_run_ok}")
    print(f"artifact_count={episode.artifact_count}")
    print(f"candidate_count={len(candidates)}")
    for candidate in candidates:
        print(f"candidate={candidate.candidate_id} memory_id={candidate.proposed_memory_id} section={candidate.section}")
        print(writer.preview_update(candidate))
        if args.apply:
            result = writer.apply_update(candidate)
            print(
                f"apply_result applied={result.applied} reason={result.reason} "
                f"memory_id={result.memory_id} history_id={result.history_id} backup={result.backup_path}"
            )
    return 0 if candidates or planning_result.ok else 1


def _read_documents(workspace: Path, limit_files: int):
    """注释：扫描少量文件，避免 prompt 过大。"""
    reader = HeterogeneousReader()
    return [reader.read_document(item) for item in FileScanner(workspace).scan()[:limit_files]]


if __name__ == "__main__":
    raise SystemExit(main())
