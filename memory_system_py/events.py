"""
模块: memory_system_py.events
职责: 定义记忆系统操作事件、状态与结构化结果。
输入: 操作名、状态、计数、payload 与错误信息。
输出: OperationEvent、OperationResult，以及可插拔 EventSink。
副作用: InMemoryEventSink 会在进程内保存事件列表；默认 NullEventSink 无副作用。
失败处理: 事件记录失败不应影响主流程；当前内置 sink 不抛业务异常。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Protocol, TypeVar

from .models import utc_now_iso


T = TypeVar("T")


@dataclass
class OperationEvent:
    """注释：单次记忆系统操作的审计事件，便于测试、日志或后续 UI 展示。"""

    name: str
    status: str
    count: int = 0
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    ts: str = field(default_factory=utc_now_iso)


@dataclass
class OperationResult(Generic[T]):
    """注释：兼容业务返回值之外的结构化结果包装。"""

    ok: bool
    value: T
    event: OperationEvent
    error: str = ""


class EventSink(Protocol):
    """注释：事件输出协议，允许后续接日志、文件、队列或监控系统。"""

    def emit(self, event: OperationEvent) -> None:
        """注释：记录一个操作事件。"""


class NullEventSink:
    """注释：默认空 sink，保持 MemorySystem 零配置可用。"""

    def emit(self, event: OperationEvent) -> None:
        """注释：无操作实现。"""
        return None


class InMemoryEventSink:
    """注释：测试/调试用 sink，把事件保存在内存列表中。"""

    def __init__(self) -> None:
        self.events: List[OperationEvent] = []

    def emit(self, event: OperationEvent) -> None:
        """注释：追加事件到 events，供断言或调试查看。"""
        self.events.append(event)
