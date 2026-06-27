"""M4 记忆单元定义：统一保存内容、嵌入和可检索元数据。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json


VALID_MEMORY_TYPES = {"evidence", "strategy", "conclusion", "experience"}


@dataclass
class MemoryUnit:
    """多智能体共享记忆的最小存储单元。"""

    memory_id: str
    source_agent: str
    created_at: str
    task_topic: str
    summary: str
    tags: list[str]
    memory_type: str
    content: str
    embedding: list[float]

    def __post_init__(self) -> None:
        """校验关键元数据，避免写入缺字段记忆后检索不可解释。"""
        required = {
            "memory_id": self.memory_id,
            "source_agent": self.source_agent,
            "created_at": self.created_at,
            "task_topic": self.task_topic,
            "summary": self.summary,
            "memory_type": self.memory_type,
            "content": self.content,
        }
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"MemoryUnit 缺少必填字段: {', '.join(missing)}")

        _parse_iso_datetime(self.created_at)
        if self.memory_type not in VALID_MEMORY_TYPES:
            raise ValueError(
                "memory_type 必须是 "
                + "|".join(sorted(VALID_MEMORY_TYPES))
            )
        if not isinstance(self.tags, list) or not all(isinstance(t, str) for t in self.tags):
            raise TypeError("tags 必须是 list[str]")
        if not self.embedding:
            raise ValueError("embedding 不能为空")

        self.tags = [tag.strip() for tag in self.tags if tag.strip()]
        self.embedding = [float(x) for x in self.embedding]

    def to_metadata(self) -> dict[str, str]:
        """转换为 ChromaDB 支持的标量 metadata。"""
        return {
            "memory_id": self.memory_id,
            "source_agent": self.source_agent,
            "created_at": self.created_at,
            "task_topic": self.task_topic,
            "summary": self.summary,
            "memory_type": self.memory_type,
            # Chroma metadata 不支持 list，存 JSON 便于跨版本兼容。
            "tags_json": json.dumps(self.tags, ensure_ascii=False),
        }

    @classmethod
    def from_record(
        cls,
        memory_id: str,
        document: str,
        metadata: dict,
        embedding: list[float],
    ) -> "MemoryUnit":
        """从向量库记录还原 MemoryUnit。"""
        tags_raw = metadata.get("tags_json", "[]")
        try:
            tags = json.loads(tags_raw) if isinstance(tags_raw, str) else list(tags_raw)
        except (TypeError, json.JSONDecodeError):
            tags = []
        return cls(
            memory_id=metadata.get("memory_id", memory_id),
            source_agent=metadata.get("source_agent", ""),
            created_at=metadata.get("created_at", ""),
            task_topic=metadata.get("task_topic", ""),
            summary=metadata.get("summary", ""),
            tags=tags,
            memory_type=metadata.get("memory_type", ""),
            content=document,
            embedding=list(embedding),
        )


def _parse_iso_datetime(value: str) -> datetime:
    """解析 ISO 时间；兼容常见的 Z 结尾 UTC 写法。"""
    normalized = value.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)
