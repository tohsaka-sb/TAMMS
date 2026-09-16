"""
模块: memory_system_py.compact
职责: 控制记忆库规模，保留高价值条目。
输入: List[MemoryRecord]，max_items 上限。
输出: 压缩后的 List[MemoryRecord]（长度 <= max_items）。
副作用: 无；由 MemorySystem.compact 调用 store.replace_all 写回磁盘。
失败处理: 输入为空时返回空列表；不抛错。
"""

from __future__ import annotations

from typing import List

from .models import MemoryRecord


class MemoryCompactor:
    """注释：按 (score, content 长度) 降序截取前 max_items 条。"""

    def compact(self, records: List[MemoryRecord], max_items: int = 200) -> List[MemoryRecord]:
        """
        注释：整库压缩排序策略，丢弃低分短内容条目。
        输入: records, max_items。
        输出: 截断后的新列表（不修改原列表对象）。
        """
        ranked = sorted(
            records,
            key=lambda r: (r.score, len(r.content)),
            reverse=True,
        )
        return ranked[:max_items]
