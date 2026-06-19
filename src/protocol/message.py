"""M1 统一消息结构 Message。所有跨模块通信必须走此结构。"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """多智能体通信的统一消息结构。

    Attributes:
        version: 协议版本号（当前为 1）
        src: 发送方 Agent ID
        dst: 接收方 Agent ID，"*" 表示广播
        msg_type: 消息类型 request|response|handshake|capability|error
        action: 请求动作 plan|search|execute|summarize 等
        params: 输入参数（可含引用，不含全文）
        result: 返回结果
        capabilities: Agent 能力列表
        context_ref: 上下文引用 ID（替代重传任务全文）
        memory_refs: 记忆 ID 列表
        state_handle: 非文本状态句柄 {"shm_name": str, "shape": tuple, "dtype": str}
        corr_id: 关联 ID（请求-响应配对）
        ts: 时间戳
    """

    version: int = 1
    src: str = ""
    dst: str = ""
    msg_type: str = "request"
    action: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    capabilities: list[str] = field(default_factory=list)
    context_ref: str = ""
    memory_refs: list[str] = field(default_factory=list)
    state_handle: dict[str, Any] | None = None
    corr_id: str = ""
    ts: float = 0.0

    def __post_init__(self) -> None:
        if self.ts == 0.0:
            self.ts = time.time()
        if not self.corr_id:
            self.corr_id = uuid.uuid4().hex[:12]
