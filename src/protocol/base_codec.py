"""M1 编解码器抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.protocol.message import Message


class Codec(ABC):
    """编解码器抽象基类。TextCodec 和 StructCodec 均实现此接口。"""

    @abstractmethod
    def encode(self, msg: Message) -> bytes:
        """将 Message 编码为字节。"""
        ...

    @abstractmethod
    def decode(self, raw: bytes) -> Message:
        """将字节解码为 Message。"""
        ...

    @abstractmethod
    def measure(self, msg: Message) -> tuple[int, int]:
        """计量实际传输开销。

        Returns:
            (字节数, token数) — 字节按编码后二进制大小，token 用 tiktoken cl100k_base。
        """
        ...
