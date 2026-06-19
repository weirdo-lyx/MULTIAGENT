"""M2 进程内传输实现。

即使 Agent 同进程运行，也强制经过 Codec 的 encode -> measure -> decode
流程，并将真实传输开销写入 Metrics，方便 text/struct 公平对比。
"""
from __future__ import annotations

from collections.abc import Callable

from src.eval.metrics import Metrics, get_metrics
from src.protocol.base_codec import Codec
from src.protocol.message import Message
from src.protocol.struct_codec import StructCodec
from src.protocol.text_codec import TextCodec
from src.transport.base import Handler, Transport


class InProcessTransport(Transport):
    """进程内消息传输。

    Args:
        mode: "text" 或 "struct"，决定默认 Codec 和 Metrics 记录桶。
        codec: 可注入自定义 Codec；为空时根据 mode 创建。
        metrics: 可注入 Metrics；为空时使用全局单例。
    """

    def __init__(
        self,
        mode: str = "struct",
        codec: Codec | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        if mode not in {"text", "struct"}:
            raise ValueError("mode 必须是 'text' 或 'struct'")
        self.mode = mode
        self.codec = codec or (TextCodec() if mode == "text" else StructCodec())
        self.metrics = metrics or get_metrics()
        self._handlers: dict[str, Handler] = {}

    def register(self, agent_id: str, handler: Handler) -> None:
        """注册 Agent 处理器。handler 可为 BaseAgent 实例或 callable。"""
        if not agent_id:
            raise ValueError("agent_id 不能为空")
        self._handlers[agent_id] = handler

    def request(self, msg: Message) -> Message:
        """发送请求并返回响应。

        请求和响应都会各自经过一次 Codec 过线流程，Metrics 因此能记录
        双向通信开销。
        """
        inbound = self._cross_boundary(msg)
        handler = self._handlers.get(inbound.dst)
        if handler is None:
            return self._cross_boundary(
                Message(
                    src="transport",
                    dst=msg.src,
                    msg_type="error",
                    action=msg.action,
                    params={},
                    result=f"未找到目标 Agent: {inbound.dst}",
                    corr_id=msg.corr_id,
                )
            )

        response = self._dispatch(handler, inbound)
        if not isinstance(response, Message):
            raise TypeError("handler 必须返回 Message")
        if not response.corr_id:
            response.corr_id = msg.corr_id
        return self._cross_boundary(response)

    def _cross_boundary(self, msg: Message) -> Message:
        """执行 encode -> measure -> decode，并记录 Metrics。"""
        raw = self.codec.encode(msg)
        bytes_, tokens = self.codec.measure(msg)
        self.metrics.record_message(self.mode, bytes_, tokens)
        return self.codec.decode(raw)

    @staticmethod
    def _dispatch(handler: Handler, msg: Message) -> Message:
        if hasattr(handler, "handle"):
            return handler.handle(msg)  # type: ignore[union-attr]
        if isinstance(handler, Callable):
            return handler(msg)
        raise TypeError("handler 必须实现 handle(msg) 或可直接调用")
