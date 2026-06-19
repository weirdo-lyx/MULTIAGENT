"""M2 Agent 基类、能力发现与进程内传输测试。"""
from __future__ import annotations

from src.agents.base import BaseAgent
from src.eval.metrics import get_metrics
from src.protocol.capability import CapabilityRegistry, PROTOCOL_VERSION
from src.protocol.message import Message
from src.transport.inprocess import InProcessTransport


class EchoAgent(BaseAgent):
    """测试用 Agent：把收到的 action/params 回显。"""

    def handle(self, msg: Message) -> Message:
        return Message(
            src=self.agent_id,
            dst=msg.src,
            msg_type="response",
            action=msg.action,
            params={},
            result={
                "handled_by": self.agent_id,
                "action": msg.action,
                "params": msg.params,
            },
            corr_id=msg.corr_id,
        )


def test_base_agent_can_handle() -> None:
    agent = EchoAgent("retriever", "retriever", ["search", "fetch"])

    assert agent.can_handle("search") is True
    assert agent.can_handle("plan") is False


def test_capability_registry_discover_and_handshake() -> None:
    registry = CapabilityRegistry()
    registry.register("planner", ["plan"])
    registry.register("retriever", ["search"])
    registry.register("executor", ["execute"])
    registry.register("summarizer", ["summarize"])

    assert registry.discover("search") == ["retriever"]
    assert registry.discover("summarize") == ["summarizer"]
    assert registry.discover("missing") == []

    result = registry.handshake("planner", "retriever")
    assert result["ok"] is True
    assert result["version"] == PROTOCOL_VERSION
    assert result["capabilities"]["planner"] == ["plan"]
    assert result["capabilities"]["retriever"] == ["search"]


def test_capability_registry_handshake_reports_missing_agent() -> None:
    registry = CapabilityRegistry()
    registry.register("planner", ["plan"])

    result = registry.handshake("planner", "ghost")
    assert result["ok"] is False
    assert result["version"] is None
    assert result["missing"] == ["ghost"]


def test_inprocess_transport_struct_records_metrics_and_routes() -> None:
    metrics = get_metrics()
    metrics.reset()
    transport = InProcessTransport(mode="struct", metrics=metrics)
    transport.register("retriever", EchoAgent("retriever", "retriever", ["search"]))

    response = transport.request(
        Message(
            src="planner",
            dst="retriever",
            msg_type="request",
            action="search",
            params={"query": "RAG", "top_k": 3},
        )
    )

    assert response.msg_type == "response"
    assert response.src == "retriever"
    assert response.dst == "planner"
    assert response.result["handled_by"] == "retriever"
    assert response.result["action"] == "search"
    assert response.result["params"] == {"query": "RAG", "top_k": 3}

    summary = metrics.summary()
    assert summary["message_count"]["struct"] == 2
    assert summary["tokens"]["struct"] > 0
    assert summary["bytes"]["struct"] > 0


def test_inprocess_transport_text_records_metrics_and_routes() -> None:
    metrics = get_metrics()
    metrics.reset()
    transport = InProcessTransport(mode="text", metrics=metrics)
    transport.register("planner", EchoAgent("planner", "planner", ["plan"]))

    response = transport.request(
        Message(
            src="user",
            dst="planner",
            msg_type="request",
            action="plan",
            params={"task": "分析多智能体通信"},
            context_ref="ctx_001",
            memory_refs=["mem_001"],
        )
    )

    assert response.msg_type == "response"
    assert response.src == "planner"
    assert response.result["action"] == "plan"
    assert response.result["params"] == {"task": "分析多智能体通信"}

    summary = metrics.summary()
    assert summary["message_count"]["text"] == 2
    assert summary["tokens"]["text"] > 0
    assert summary["bytes"]["text"] > 0


def test_inprocess_transport_unknown_agent_returns_error() -> None:
    metrics = get_metrics()
    metrics.reset()
    transport = InProcessTransport(mode="struct", metrics=metrics)

    response = transport.request(
        Message(src="planner", dst="missing", action="search", params={})
    )

    assert response.msg_type == "error"
    assert response.src == "transport"
    assert "未找到目标 Agent" in response.result
