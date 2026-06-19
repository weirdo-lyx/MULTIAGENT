"""M2 传输层抽象。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Protocol

from src.protocol.message import Message


class MessageHandler(Protocol):
    """可注册到 Transport 的消息处理器协议。"""

    def handle(self, msg: Message) -> Message:
        """处理消息并返回响应。"""
        ...


Handler = MessageHandler | Callable[[Message], Message]


class Transport(ABC):
    """跨 Agent 通信传输抽象。"""

    @abstractmethod
    def register(self, agent_id: str, handler: Handler) -> None:
        """注册 agent_id 对应的消息处理器。"""
        ...

    @abstractmethod
    def request(self, msg: Message) -> Message:
        """发送请求消息并返回响应消息。"""
        ...
