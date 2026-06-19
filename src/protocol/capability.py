"""M2 能力注册、发现与协议握手。"""
from __future__ import annotations

from dataclasses import dataclass, field


PROTOCOL_VERSION = 1


@dataclass
class CapabilityRegistry:
    """Agent 能力注册表。

    register() 记录 Agent 可处理的 action；discover() 根据 action 找到
    所有可处理的 Agent；handshake() 返回两个 Agent 的协议版本协商结果。
    """

    protocol_version: int = PROTOCOL_VERSION
    _capabilities: dict[str, list[str]] = field(default_factory=dict)

    def register(self, agent_id: str, capabilities: list[str]) -> None:
        """登记一个 Agent 的能力列表。"""
        if not agent_id:
            raise ValueError("agent_id 不能为空")
        self._capabilities[agent_id] = list(dict.fromkeys(capabilities))

    def discover(self, action: str) -> list[str]:
        """返回所有能处理 action 的 Agent ID，保持注册顺序。"""
        return [
            agent_id
            for agent_id, capabilities in self._capabilities.items()
            if action in capabilities
        ]

    def handshake(self, a: str, b: str) -> dict:
        """两个 Agent 做协议版本协商。

        当前 MVP 只有一个协议版本，因此协商结果是双方都存在时返回
        protocol_version；后续若引入多版本可在此扩展。
        """
        missing = [
            agent_id for agent_id in (a, b) if agent_id not in self._capabilities
        ]
        ok = not missing
        return {
            "ok": ok,
            "version": self.protocol_version if ok else None,
            "agents": [a, b],
            "capabilities": {
                a: self._capabilities.get(a, []),
                b: self._capabilities.get(b, []),
            },
            "missing": missing,
        }
