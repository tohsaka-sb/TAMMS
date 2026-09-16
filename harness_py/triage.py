"""
模块: harness_py.triage
职责: 为文件生成一句话 gist，并按任务名聚类。
输入: List[FileSnippet]；可选自定义 gist_fn。
输出: List[TaskCluster]。
副作用: 无。
失败处理: 空 gist 时使用默认任务名 Task_Misc；不抛错。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, List

from .types import FileSnippet, TaskCluster


GistFn = Callable[[FileSnippet], str]


class FastPassTriageRouter:
    """注释：默认启发式 gist；可注入 LLM 函数替换 _heuristic_gist。"""

    def __init__(self, gist_fn: GistFn | None = None) -> None:
        self.gist_fn = gist_fn or self._heuristic_gist

    def extract_gists(self, snippets: List[FileSnippet]) -> List[tuple[FileSnippet, str]]:
        """注释：为每个文件片段生成一句话概括。"""
        return [(snippet, self.gist_fn(snippet)) for snippet in snippets]

    def cluster(self, gists: List[tuple[FileSnippet, str]]) -> List[TaskCluster]:
        """
        注释：按任务路由名分桶，生成带描述与 metadata 的 TaskCluster。
        输出: 按 task_name 排序的簇列表。
        """
        buckets: dict[str, list[tuple[FileSnippet, str]]] = defaultdict(list)
        for snippet, gist in gists:
            task = self._route_task_name(gist, snippet)
            buckets[task].append((snippet, gist))

        clusters: List[TaskCluster] = []
        for task_name, items in buckets.items():
            description = self._describe_cluster(task_name, [g for _, g in items])
            clusters.append(
                TaskCluster(
                    task_name=task_name,
                    description=description,
                    files=[s for s, _ in items],
                    metadata={"file_count": str(len(items))},
                )
            )
        return sorted(clusters, key=lambda c: c.task_name)

    def _heuristic_gist(self, snippet: FileSnippet) -> str:
        """注释：基于路径与内容前 1000 字的规则 gist。"""
        rel = snippet.file.relative_path.lower()
        content = snippet.content[:1000].lower()

        if "test" in rel or "assert" in content:
            return "这是一个与测试验证相关的文件"
        if snippet.file.extension in {".py", ".ts", ".js", ".tsx", ".jsx"}:
            return "这是一个与应用代码实现相关的文件"
        if snippet.file.extension in {".md"}:
            return "这是一个与文档说明相关的文件"
        if snippet.file.extension in {".csv"}:
            return "这是一个与结构化数据记录相关的文件"
        if "docker" in rel or "compose" in rel:
            return "这是一个与部署运行环境相关的文件"
        return "这是一个与项目杂项配置相关的文件"

    def _route_task_name(self, gist: str, snippet: FileSnippet) -> str:
        """注释：将 gist + 路径映射到 Task_* 桶名。"""
        text = f"{gist} {snippet.file.relative_path}".lower()
        if "测试" in text or "test" in text:
            return "Task_Testing"
        if "文档" in text or snippet.file.extension == ".md":
            return "Task_Documentation"
        if "数据" in text or snippet.file.extension == ".csv":
            return "Task_Data"
        if "部署" in text or "docker" in text:
            return "Task_DevOps"
        if snippet.file.extension in {".py", ".ts", ".js", ".tsx", ".jsx"}:
            return "Task_ApplicationCode"
        return "Task_Misc"

    def _describe_cluster(self, task_name: str, gists: List[str]) -> str:
        """注释：生成簇级描述，含文件数与样例 gist。"""
        sample = gists[0] if gists else "无描述"
        return f"{task_name}: 聚合了 {len(gists)} 个文件；样例概括：{sample}"
