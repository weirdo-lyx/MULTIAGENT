"""M1 起全程埋点：消息数/token/字节/状态传递/记忆命中率等。全局单例。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Metrics:
    """全局指标采集单例。通过 get_metrics() 获取实例。"""

    _instance: "Metrics | None" = None

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # 消息通信
    # ------------------------------------------------------------------
    def record_message(self, mode: str, bytes_: int, tokens: int) -> None:
        """记录一次消息传输。

        Args:
            mode: "text" 或 "struct"
            bytes_: 实际传输字节数
            tokens: tiktoken 计算的 token 数
        """
        bucket = self._messages.setdefault(mode, [])
        bucket.append({"bytes": bytes_, "tokens": tokens, "ts": time.time()})

    # ------------------------------------------------------------------
    # 非文本状态传递（共享内存）
    # ------------------------------------------------------------------
    def record_state_transfer(self, bytes_: int) -> None:
        """记录一次共享内存状态传递。"""
        self._state_transfers.append({"bytes": bytes_, "ts": time.time()})

    # ------------------------------------------------------------------
    # 记忆检索
    # ------------------------------------------------------------------
    def record_memory_query(self, hit: bool) -> None:
        """记录一次记忆检索。"""
        self._memory_queries += 1
        if hit:
            self._memory_hits += 1

    # ------------------------------------------------------------------
    # 任务计时
    # ------------------------------------------------------------------
    def start_task(self, task_id: str) -> None:
        self._task_starts[task_id] = time.time()

    def end_task(self, task_id: str) -> None:
        start = self._task_starts.get(task_id)
        if start is not None:
            elapsed = time.time() - start
            self._task_durations[task_id] = elapsed

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """汇总所有指标。"""
        text_msgs = self._messages.get("text", [])
        struct_msgs = self._messages.get("struct", [])

        text_tokens = sum(m["tokens"] for m in text_msgs)
        text_bytes = sum(m["bytes"] for m in text_msgs)
        struct_tokens = sum(m["tokens"] for m in struct_msgs)
        struct_bytes = sum(m["bytes"] for m in struct_msgs)

        state_count = len(self._state_transfers)
        state_bytes = sum(s["bytes"] for s in self._state_transfers)

        hit_rate = (
            self._memory_hits / self._memory_queries
            if self._memory_queries > 0
            else 0.0
        )

        # 性能提升百分比（token 节省）
        token_saving = 0.0
        if text_tokens > 0:
            token_saving = (1 - struct_tokens / text_tokens) * 100

        byte_saving = 0.0
        if text_bytes > 0:
            byte_saving = (1 - struct_bytes / text_bytes) * 100

        return {
            "message_count": {
                "text": len(text_msgs),
                "struct": len(struct_msgs),
            },
            "tokens": {
                "text": text_tokens,
                "struct": struct_tokens,
            },
            "bytes": {
                "text": text_bytes,
                "struct": struct_bytes,
            },
            "token_saving_pct": round(token_saving, 2),
            "byte_saving_pct": round(byte_saving, 2),
            "state_transfers": {
                "count": state_count,
                "bytes": state_bytes,
            },
            "memory": {
                "queries": self._memory_queries,
                "hits": self._memory_hits,
                "hit_rate": round(hit_rate, 4),
            },
            "task_durations": dict(self._task_durations),
        }

    def export(self, path: str) -> None:
        """导出指标到 JSON 文件。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)

    def reset(self) -> None:
        """重置所有指标。"""
        self._messages: dict[str, list[dict]] = {}
        self._state_transfers: list[dict] = []
        self._memory_queries: int = 0
        self._memory_hits: int = 0
        self._task_starts: dict[str, float] = {}
        self._task_durations: dict[str, float] = {}


def get_metrics() -> Metrics:
    """获取全局 Metrics 单例。"""
    if Metrics._instance is None:
        Metrics._instance = Metrics()
    return Metrics._instance
