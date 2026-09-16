"""
模块: memory_system_py.extractor
职责: 从用户会话轮次中抽取可长期保存的记忆候选。
输入: List[ConversationTurn]，可选 source 标记。
输出: List[MemoryRecord]（可能为空）。
副作用: 无；不写库，由 MemorySystem.ingest_session 持久化。
失败处理: 仅处理 role=user；无关键词匹配时跳过该句，不抛错。
"""

from __future__ import annotations

from typing import List

from .models import ConversationTurn, MemoryRecord


class MemoryExtractor:
    """注释：基于关键词启发式提取；后续可替换为 LLM 提取器。"""

    KEYWORDS = ("喜欢", "偏好", "习惯", "总是", "不要", "项目", "目标", "截止")

    def extract(self, turns: List[ConversationTurn], source: str = "session") -> List[MemoryRecord]:
        """
        注释：遍历用户轮次，按句切分并过滤关键词后生成 MemoryRecord。
        输入: turns, source。
        输出: 提取到的记忆列表（可为空）。
        """
        memories: List[MemoryRecord] = []
        for turn in turns:
            if turn.role.lower() != "user":
                continue
            for sentence in self._split_sentences(turn.content):
                if self._is_memory_worthy(sentence):
                    memories.append(
                        MemoryRecord.create(
                            category="long_term",
                            content=sentence,
                            source=source,
                            tags=["extracted", "user"],
                        )
                    )
        return memories

    def _split_sentences(self, text: str) -> List[str]:
        """注释：按句号/英文点分句，过滤过短片段。"""
        parts = [p.strip() for p in text.replace("。", ".").split(".")]
        return [p for p in parts if len(p) >= 6]

    def _is_memory_worthy(self, sentence: str) -> bool:
        """注释：句内包含任一 KEYWORDS 即视为值得记忆。"""
        return any(k in sentence for k in self.KEYWORDS)
