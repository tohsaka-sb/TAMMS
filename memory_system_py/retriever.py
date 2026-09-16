"""
模块: memory_system_py.retriever
职责: 按查询词从记忆列表中检索最相关条目。
输入: 全量或子集 List[MemoryRecord]、query 字符串、top_k。
输出: 排序后的前 top_k 条 MemoryRecord（可能少于 top_k）。
副作用: 无；纯内存计算。
失败处理: 无重叠词时返回空列表，不抛错。
"""

from __future__ import annotations

from typing import List

from .models import MemoryRecord


class MemoryRetriever:
    """注释：关键词重叠 + record.score 简易排序；可后续换 embedding 检索。"""

    def search(self, records: List[MemoryRecord], query: str, top_k: int = 5) -> List[MemoryRecord]:
        """
        注释：返回与 query 词重叠且总分最高的前 top_k 条。
        输入: records, query, top_k。
        输出: 按相关度降序的记忆列表。
        """
        q_words = self._words(query)
        scored = []
        for record in records:
            overlap = len(q_words.intersection(self._words(record.content)))
            total_score = overlap + record.score
            if overlap > 0:
                scored.append((total_score, record))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def _words(self, text: str) -> set[str]:
        """注释：词级 + 字符级 token，兼容中英文无空格场景。"""
        normalized = text.strip().lower()
        tokens = {w.strip() for w in normalized.replace(",", " ").split() if w.strip()}
        tokens.update({ch for ch in normalized if ch.strip()})
        return tokens
