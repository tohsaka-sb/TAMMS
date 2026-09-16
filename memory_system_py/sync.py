"""
模块: memory_system_py.sync
职责: 团队记忆同步前的敏感信息过滤与批次划分。
输入: List[MemoryRecord]。
输出: (allowed, blocked) 二元组。
副作用: 无；不执行实际上传，由调用方处理 allowed/blocked。
失败处理: 命中敏感词时归入 blocked，不抛错、不中断批次构建。
"""

from __future__ import annotations

from typing import List, Tuple

from .models import MemoryRecord


class SecretGuard:
    """注释：基础敏感词扫描；大小写不敏感匹配英文，原文匹配中文。"""

    BLOCKED = ("password", "token", "secret", "apikey", "私钥", "密码")

    def check(self, text: str) -> bool:
        """
        注释：通过检查返回 True，命中敏感词返回 False。
        输入: 单条记忆的 content 文本。
        输出: 是否允许同步。
        """
        lower = text.lower()
        return not any(word in lower for word in self.BLOCKED)


class TeamMemorySync:
    """注释：将记忆库划分为可同步与需审计拦截两组。"""

    def __init__(self) -> None:
        self.guard = SecretGuard()

    def prepare_sync_batch(self, records: List[MemoryRecord]) -> Tuple[List[MemoryRecord], List[MemoryRecord]]:
        """
        注释：逐条扫描 content，分流到 allowed / blocked。
        输入: 当前记忆库全量或子集。
        输出: (可同步列表, 被拦截列表)。
        """
        allowed: List[MemoryRecord] = []
        blocked: List[MemoryRecord] = []
        for record in records:
            if self.guard.check(record.content):
                allowed.append(record)
            else:
                blocked.append(record)
        return allowed, blocked
