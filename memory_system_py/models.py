"""
模块: memory_system_py.models
职责: 定义记忆条目与会话轮次等核心数据结构。
输入: 构造参数或 MemoryRecord.create 的 category/content/source/tags/record_id/metadata。
输出: ConversationTurn、MemoryRecord 实例。
副作用: 无；纯数据层。
失败处理: create 时对 content 做 strip；空内容由调用方决定是否拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List
from uuid import uuid4


def utc_now_iso() -> str:
    """注释：统一 UTC ISO 时间戳，供存储与同步批次使用。"""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConversationTurn:
    """
    注释：单轮会话输入，作为 MemoryExtractor 的原始材料。
    输入: role, content, 可选 ts。
    输出: 不可变数据对象（dataclass）。
    """

    role: str
    content: str
    ts: str = field(default_factory=utc_now_iso)


@dataclass
class MemoryRecord:
    """
    注释：统一记忆实体，覆盖会话提取、手工写入与 Harness 桥接。
    输入: 全字段或 MemoryRecord.create 简化构造。
    输出: 可序列化到 SQLite 的记录。
    """

    id: str
    category: str
    content: str
    source: str
    tags: List[str] = field(default_factory=list)
    score: float = 0.5
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, str] = field(default_factory=dict)

    @staticmethod
    def create(
        category: str,
        content: str,
        source: str,
        tags: List[str] | None = None,
        record_id: str | None = None,
        metadata: Dict[str, str] | None = None,
        score: float = 0.5,
    ) -> "MemoryRecord":
        """注释：创建新记忆；未提供 record_id 时分配 UUID，content 自动 strip。"""
        return MemoryRecord(
            id=record_id or str(uuid4()),
            category=category,
            content=content.strip(),
            source=source,
            tags=tags or [],
            score=score,
            metadata=metadata or {},
        )
