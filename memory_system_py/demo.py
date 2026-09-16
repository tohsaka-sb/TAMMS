"""
模块: memory_system_py.demo
职责: 可独立运行的端到端演示脚本。
输入: 无（内置示例会话）。
输出: 控制台打印提取/召回/压缩/同步统计。
副作用: 清理并重建 ./memory_system_py/.runtime 目录。
失败处理: 依赖各 API 默认行为；脚本级异常向上抛出。
"""

from pathlib import Path
import shutil

from memory_system_py import ConversationTurn, create_memory_system


def main() -> None:
    """注释：演示 ingest → remember → recall → compact → export_sync_batch 全流程。"""
    runtime_dir = Path("./memory_system_py/.runtime")
    # 注释：每次演示前清理临时库，保证输出可重复。
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    system = create_memory_system(root_dir=str(runtime_dir))

    turns = [
        ConversationTurn(role="user", content="我喜欢把任务拆成最小可 review 的步骤。"),
        ConversationTurn(role="assistant", content="明白，我会分步输出。"),
        ConversationTurn(role="user", content="项目目标是在月底前交付独立记忆系统。"),
    ]
    extracted = system.ingest_session(turns)
    print(f"extracted={len(extracted)}")

    system.remember("manual", "不要在同步时上传包含密码的文本", tags=["policy"])
    recalled = system.recall("项目 目标 交付")
    print("recalled:", [m.content for m in recalled])

    compacted_size = system.compact(max_items=50)
    print(f"compacted_size={compacted_size}")

    batch = system.export_sync_batch()
    print(f"sync_allowed={len(batch['allowed'])}, sync_blocked={len(batch['blocked'])}")


if __name__ == "__main__":
    main()
