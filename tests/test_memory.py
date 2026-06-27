"""M4 共享记忆测试。"""
from __future__ import annotations

import pytest

from src.eval.metrics import get_metrics
from src.memory import MemoryStore, MemoryUnit


def make_memory(
    memory_id: str,
    content: str,
    embedding: list[float],
    tags: list[str],
    memory_type: str = "evidence",
) -> MemoryUnit:
    """构造测试记忆，保持元数据齐全。"""
    return MemoryUnit(
        memory_id=memory_id,
        source_agent="retriever",
        created_at="2026-06-27T15:30:00+08:00",
        task_topic="RAG 技术研究",
        summary=content[:30],
        tags=tags,
        memory_type=memory_type,
        content=content,
        embedding=embedding,
    )


def test_memory_unit_requires_complete_metadata() -> None:
    """缺关键元数据应直接失败，避免写入不可解释记忆。"""
    with pytest.raises(ValueError):
        MemoryUnit(
            memory_id="",
            source_agent="retriever",
            created_at="2026-06-27T15:30:00+08:00",
            task_topic="RAG",
            summary="summary",
            tags=["rag"],
            memory_type="evidence",
            content="content",
            embedding=[1.0, 0.0],
        )


def test_memory_store_add_and_get() -> None:
    store = MemoryStore(backend="memory")
    unit = make_memory(
        "mem_rag",
        "RAG 通过检索外部知识增强生成质量。",
        [1.0, 0.0, 0.0],
        ["rag", "retrieval"],
    )

    memory_id = store.add(unit)
    loaded = store.get(memory_id)

    assert memory_id == "mem_rag"
    assert loaded.content == unit.content
    assert loaded.tags == ["rag", "retrieval"]


def test_search_semantic_keyword_and_tag_all_hit_and_record_metrics() -> None:
    metrics = get_metrics()
    metrics.reset()
    store = MemoryStore(metrics=metrics, backend="memory")
    store.add(
        make_memory(
            "mem_rag",
            "RAG 适合需要外部知识支撑的问答任务。",
            [1.0, 0.0, 0.0],
            ["rag", "qa"],
        )
    )
    store.add(
        make_memory(
            "mem_sandbox",
            "CodeAct 沙箱需要限制 CPU、内存、网络和危险导入。",
            [0.0, 1.0, 0.0],
            ["sandbox", "codeact"],
            memory_type="strategy",
        )
    )

    semantic = store.search_semantic([0.95, 0.05, 0.0], top_k=1)
    keyword = store.search_keyword("CPU", top_k=3)
    tag = store.search_tag("rag")

    assert [u.memory_id for u in semantic] == ["mem_rag"]
    assert [u.memory_id for u in keyword] == ["mem_sandbox"]
    assert [u.memory_id for u in tag] == ["mem_rag"]

    summary = metrics.summary()
    assert summary["memory"]["queries"] == 3
    assert summary["memory"]["hits"] == 3
    assert summary["memory"]["hit_rate"] == 1.0


def test_search_semantic_supports_metadata_filters() -> None:
    store = MemoryStore(backend="memory")
    store.add(
        make_memory(
            "mem_evidence",
            "RAG 证据记忆。",
            [1.0, 0.0],
            ["rag"],
            memory_type="evidence",
        )
    )
    store.add(
        make_memory(
            "mem_strategy",
            "RAG 策略记忆。",
            [0.9, 0.1],
            ["rag", "plan"],
            memory_type="strategy",
        )
    )

    result = store.search_semantic(
        [1.0, 0.0],
        top_k=5,
        filters={"memory_type": "strategy", "tags": "plan"},
    )

    assert [u.memory_id for u in result] == ["mem_strategy"]
