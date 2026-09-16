"""
模块: memory_system_py（包入口）
职责: 对外暴露记忆系统门面、事件类型与核心数据类型。
输入: 无（import 时加载）。
输出: MemorySystem, create_memory_system, ConversationTurn, MemoryRecord, OperationEvent, OperationResult。
副作用: 无。
失败处理: 无。
"""

from .events import InMemoryEventSink, OperationEvent, OperationResult
from .models import ConversationTurn, MemoryRecord
from .system import MemorySystem


def create_memory_system(root_dir: str = "./runtime_memory", event_sink=None) -> MemorySystem:
    """
    注释：工厂函数，统一创建记忆系统实例（Phase 2 轻量入口）。
    输入: root_dir 持久化目录，其下将生成 memory.db；event_sink 可选事件输出。
    输出: 已装配完成的 MemorySystem。
    """
    return MemorySystem(root_dir=root_dir, event_sink=event_sink)


__all__ = [
    "MemorySystem",
    "create_memory_system",
    "ConversationTurn",
    "MemoryRecord",
    "OperationEvent",
    "OperationResult",
    "InMemoryEventSink",
]
