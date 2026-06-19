"""M2 Agent 基类。所有具体 Agent 都继承此抽象。"""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.protocol.message import Message


class BaseAgent(ABC):
    """多智能体系统的统一 Agent 抽象。

    Args:
        agent_id: Agent 唯一标识，例如 "planner"。
        role: Agent 角色描述，例如 "planner"。
        capabilities: 可处理的 action 列表。
    """

    agent_id: str
    role: str
    capabilities: list[str]

    def __init__(self, agent_id: str, role: str, capabilities: list[str]) -> None:
        self.agent_id = agent_id
        self.role = role
        self.capabilities = list(capabilities)

    def can_handle(self, action: str) -> bool:
        """返回当前 Agent 是否能处理指定 action。"""
        return action in self.capabilities

    @abstractmethod
    def handle(self, msg: Message) -> Message:
        """处理输入消息并返回响应消息。"""
        ...
