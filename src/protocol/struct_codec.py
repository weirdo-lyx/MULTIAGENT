"""M1 结构化编解码器：msgpack 紧凑序列化，引用保持为 ID 不展开。"""
from __future__ import annotations

import json

import msgpack
import tiktoken

from src.protocol.base_codec import Codec
from src.protocol.message import Message

# tiktoken 编码器（cl100k_base，与 TextCodec 同一把尺）
_ENC = tiktoken.get_encoding("cl100k_base")


def _msg_to_compact_dict(msg: Message) -> dict:
    """将 Message 转为紧凑 dict，字段用短名。引用保持为 ID。"""
    d: dict = {
        "v": msg.version,
        "src": msg.src,
        "dst": msg.dst,
        "type": msg.msg_type,
        "act": msg.action,
    }
    if msg.params:
        d["p"] = msg.params
    if msg.result is not None:
        d["r"] = msg.result
    if msg.capabilities:
        d["cap"] = msg.capabilities
    if msg.context_ref:
        d["ctx"] = msg.context_ref  # 引用保持为 ID，不展开
    if msg.memory_refs:
        d["mem"] = msg.memory_refs  # 引用保持为 ID 列表，不展开
    if msg.state_handle is not None:
        d["sh"] = msg.state_handle  # 句柄保持为 dict，不展开
    d["cid"] = msg.corr_id
    d["ts"] = msg.ts
    return d


def _compact_dict_to_msg(d: dict) -> Message:
    """从紧凑 dict 还原 Message。"""
    return Message(
        version=d.get("v", 1),
        src=d.get("src", ""),
        dst=d.get("dst", ""),
        msg_type=d.get("type", "request"),
        action=d.get("act", ""),
        params=d.get("p", {}),
        result=d.get("r", None),
        capabilities=d.get("cap", []),
        context_ref=d.get("ctx", ""),
        memory_refs=d.get("mem", []),
        state_handle=d.get("sh", None),
        corr_id=d.get("cid", ""),
        ts=d.get("ts", 0.0),
    )


class StructCodec(Codec):
    """结构化编解码器。

    - encode: msgpack 二进制紧凑序列化，字段短名
    - decode: msgpack 反序列化
    - measure: 字节数=msgpack 二进制大小, token=tiktoken 对等价 JSON 文本计数

    **引用保持为 ID 不展开**：context_ref/memory_refs/state_handle 只传编号，
    接收方通过共享上下文/记忆库/共享内存解引用。
    """

    def encode(self, msg: Message) -> bytes:
        d = _msg_to_compact_dict(msg)
        return msgpack.packb(d, use_bin_type=True)

    def decode(self, raw: bytes) -> Message:
        d = msgpack.unpackb(raw, raw=False)
        return _compact_dict_to_msg(d)

    def measure(self, msg: Message) -> tuple[int, int]:
        """计量结构化传输开销。

        - 字节数：msgpack 编码后的二进制大小
        - token 数：对等价 JSON 文本用 tiktoken 计数（保证同一把尺）
        """
        packed = self.encode(msg)
        byte_count = len(packed)
        # 等价 JSON 文本用于 token 计数
        d = _msg_to_compact_dict(msg)
        json_text = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
        token_count = len(_ENC.encode(json_text))
        return byte_count, token_count
