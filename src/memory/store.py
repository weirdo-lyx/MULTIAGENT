"""M4 共享记忆存储：ChromaDB 主路径，内存后端作离线兜底。"""
from __future__ import annotations

from pathlib import Path
import math
from typing import Any, Literal

from src.config import load_config
from src.eval.metrics import Metrics, get_metrics
from src.memory.unit import MemoryUnit

Backend = Literal["auto", "chroma", "memory"]


class MemoryStore:
    """共享记忆存储与三种检索接口。"""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = "multiagent_memory",
        metrics: Metrics | None = None,
        backend: Backend = "auto",
    ) -> None:
        cfg = load_config().get("memory", {})
        self.persist_dir = persist_dir or cfg.get("persist_dir", "data/chroma")
        self.collection_name = collection_name
        self.metrics = metrics or get_metrics()
        self.backend: Literal["chroma", "memory"] = "memory"
        self._units: dict[str, MemoryUnit] = {}
        self._collection: Any | None = None

        if backend in {"auto", "chroma"}:
            try:
                self._collection = self._make_chroma_collection()
                self.backend = "chroma"
            except Exception:
                if backend == "chroma":
                    raise
                self._collection = None

    def add(self, u: MemoryUnit) -> str:
        """添加或更新一条记忆，返回 memory_id。"""
        if self.backend == "chroma":
            assert self._collection is not None
            self._collection.upsert(
                ids=[u.memory_id],
                documents=[u.content],
                embeddings=[u.embedding],
                metadatas=[u.to_metadata()],
            )
        else:
            self._units[u.memory_id] = u
        return u.memory_id

    def search_semantic(
        self,
        query_emb: list[float],
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[MemoryUnit]:
        """按向量相似度检索记忆，并记录命中指标。"""
        if self.backend == "chroma":
            units = self._search_semantic_chroma(query_emb, top_k, filters)
        else:
            units = self._search_semantic_memory(query_emb, top_k, filters)
        self.metrics.record_memory_query(hit=bool(units))
        return units

    def search_keyword(self, kw: str, top_k: int = 5) -> list[MemoryUnit]:
        """按关键词检索 content/summary/topic。"""
        keyword = kw.strip().lower()
        if not keyword:
            self.metrics.record_memory_query(hit=False)
            return []

        if self.backend == "chroma":
            units = self._search_keyword_chroma(keyword, top_k)
        else:
            units = self._search_keyword_memory(keyword, top_k)
        self.metrics.record_memory_query(hit=bool(units))
        return units

    def search_tag(self, tag: str) -> list[MemoryUnit]:
        """按标签精确召回记忆。"""
        target = tag.strip()
        if not target:
            self.metrics.record_memory_query(hit=False)
            return []

        units = (
            self._all_chroma_units()
            if self.backend == "chroma"
            else list(self._units.values())
        )
        result = [u for u in units if target in u.tags]
        self.metrics.record_memory_query(hit=bool(result))
        return result

    def get(self, memory_id: str) -> MemoryUnit:
        """按 ID 获取单条记忆。"""
        if self.backend == "chroma":
            assert self._collection is not None
            data = self._collection.get(
                ids=[memory_id],
                include=["documents", "metadatas", "embeddings"],
            )
            units = self._units_from_chroma_result(data)
            if not units:
                raise KeyError(memory_id)
            return units[0]

        try:
            return self._units[memory_id]
        except KeyError as exc:
            raise KeyError(memory_id) from exc

    def _make_chroma_collection(self) -> Any:
        """创建 ChromaDB collection；导入放在运行期，便于缺依赖时内存兜底。"""
        import chromadb

        path = Path(self.persist_dir)
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(path))
        return client.get_or_create_collection(name=self.collection_name)

    def _search_semantic_chroma(
        self,
        query_emb: list[float],
        top_k: int,
        filters: dict | None,
    ) -> list[MemoryUnit]:
        assert self._collection is not None
        tags_filter = None
        where = None
        if filters:
            where = {k: v for k, v in filters.items() if k != "tags"}
            tags_filter = filters.get("tags")
            where = where or None

        # 若需要 tag 后过滤，先多取一些，避免 top_k 太早截断。
        n_results = max(top_k, top_k * 5 if tags_filter else top_k)
        data = self._collection.query(
            query_embeddings=[list(map(float, query_emb))],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "embeddings"],
        )
        units = self._units_from_chroma_query(data)
        if tags_filter is not None:
            expected = {tags_filter} if isinstance(tags_filter, str) else set(tags_filter)
            units = [u for u in units if expected.intersection(u.tags)]
        return units[:top_k]

    def _search_keyword_chroma(self, keyword: str, top_k: int) -> list[MemoryUnit]:
        # Chroma 的 where_document 在不同版本里大小写/包含语义不完全一致；
        # M4 规模较小，统一取回后用 Python 做稳定的大小写无关过滤。
        return self._search_keyword_memory_over(self._all_chroma_units(), keyword, top_k)

    def _all_chroma_units(self) -> list[MemoryUnit]:
        assert self._collection is not None
        data = self._collection.get(include=["documents", "metadatas", "embeddings"])
        return self._units_from_chroma_result(data)

    def _units_from_chroma_query(self, data: dict) -> list[MemoryUnit]:
        """Chroma query 返回嵌套列表，这里统一拍平为 MemoryUnit。"""
        ids = data.get("ids", [[]])[0]
        docs = data.get("documents", [[]])[0]
        metas = data.get("metadatas", [[]])[0]
        embeddings = data.get("embeddings", [[]])[0]
        return [
            MemoryUnit.from_record(memory_id, doc, meta or {}, emb)
            for memory_id, doc, meta, emb in zip(ids, docs, metas, embeddings)
        ]

    def _units_from_chroma_result(self, data: dict) -> list[MemoryUnit]:
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        embeddings = data.get("embeddings", [])
        return [
            MemoryUnit.from_record(memory_id, doc, meta or {}, emb)
            for memory_id, doc, meta, emb in zip(ids, docs, metas, embeddings)
        ]

    def _search_semantic_memory(
        self,
        query_emb: list[float],
        top_k: int,
        filters: dict | None,
    ) -> list[MemoryUnit]:
        units = [u for u in self._units.values() if _matches_filters(u, filters)]
        ranked = sorted(
            units,
            key=lambda u: _cosine_similarity(query_emb, u.embedding),
            reverse=True,
        )
        return ranked[:top_k]

    def _search_keyword_memory(self, keyword: str, top_k: int) -> list[MemoryUnit]:
        return self._search_keyword_memory_over(list(self._units.values()), keyword, top_k)

    @staticmethod
    def _search_keyword_memory_over(
        units: list[MemoryUnit],
        keyword: str,
        top_k: int,
    ) -> list[MemoryUnit]:
        result = [
            u
            for u in units
            if keyword in " ".join([u.content, u.summary, u.task_topic]).lower()
        ]
        return result[:top_k]


def _matches_filters(u: MemoryUnit, filters: dict | None) -> bool:
    """内存后端的简单元数据过滤，覆盖 M4/M6 常用字段。"""
    if not filters:
        return True
    for key, expected in filters.items():
        if key == "tags":
            expected_set = {expected} if isinstance(expected, str) else set(expected)
            if not expected_set.intersection(u.tags):
                return False
            continue
        if getattr(u, key, None) != expected:
            return False
    return True


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """纯 Python 余弦相似度，避免 M4 内存兜底依赖 numpy。"""
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)
