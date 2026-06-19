"""M1 文本编解码器：渲染成自然语言，必须展开所有引用为实际文本。

核心设计原则（决定 25 分通信效率评分）：
- TextCodec 模拟真实纯文本系统的开销——纯文本 Agent 无法靠编号查内容，
  所以必须把 context_ref / memory_refs / state_handle 全部展开成原文内联。
- 这与 StructCodec 的"引用保持为 ID"形成公平对比：信息没有丢失，
  只是结构化模式下接收方可以自己按编号解引用，省下的是"重复传输"。

依赖注入设计：
- M3（StateChannel）/ M4（MemoryStore）尚未实现时，通过可注入的回调函数
  解析引用内容。生产环境接入真实服务，测试环境用 mock。
"""
from __future__ import annotations

import json
from typing import Any, Callable

import tiktoken

from src.protocol.base_codec import Codec
from src.protocol.message import Message

# tiktoken 编码器（cl100k_base，与 StructCodec 同一把尺）
_ENC = tiktoken.get_encoding("cl100k_base")


# ---- 引用解析回调类型 ----
# context_resolver(context_ref) -> str  （根据上下文 ID 返回完整任务描述文本）
ContextResolver = Callable[[str], str]
# memory_resolver(memory_id) -> str     （根据记忆 ID 返回完整记忆内容文本）
MemoryResolver = Callable[[str], str]
# state_resolver(state_handle) -> str   （根据状态句柄返回状态摘要文本）
StateResolver = Callable[[dict[str, Any]], str]


def _default_context_resolver(context_ref: str) -> str:
    """默认上下文解析器：返回占位文本（测试/开发用）。"""
    return f"[上下文 {context_ref} 内容]"


def _default_memory_resolver(memory_id: str) -> str:
    """默认记忆解析器：返回占位文本（测试/开发用）。"""
    return f"[记忆 {memory_id} 内容]"


def _default_state_resolver(state_handle: dict[str, Any]) -> str:
    """默认状态解析器：返回句柄描述文本（测试/开发用）。"""
    shape = state_handle.get("shape", "unknown")
    dtype = state_handle.get("dtype", "unknown")
    return f"[共享内存状态: shape={shape}, dtype={dtype}]"


class TextCodec(Codec):
    """文本编解码器。

    - encode: 渲染成自然语言文本（UTF-8 字节）
    - decode: 从文本解析回 Message（简化实现，保留核心字段）
    - measure: 字节数=UTF-8 编码大小, token=tiktoken 对渲染文本计数

    **核心要求**：必须把 context_ref / memory_refs / state_handle 全部展开
    成实际文本内容内联。纯文本 Agent 无法解引用，只能把原文塞进去——
    这是真实纯文本系统的开销。

    Args:
        context_resolver: 根据 context_ref 获取完整上下文文本的回调
        memory_resolver: 根据 memory_id 获取完整记忆文本的回调
        state_resolver: 根据 state_handle 获取状态描述文本的回调
    """

    def __init__(
        self,
        context_resolver: ContextResolver | None = None,
        memory_resolver: MemoryResolver | None = None,
        state_resolver: StateResolver | None = None,
    ) -> None:
        self._resolve_context = context_resolver or _default_context_resolver
        self._resolve_memory = memory_resolver or _default_memory_resolver
        self._resolve_state = state_resolver or _default_state_resolver

    def _render_text(self, msg: Message) -> str:
        """将 Message 渲染为完整自然语言文本（展开所有引用）。"""
        parts: list[str] = []

        # 文本模式仍然是自然语言传输，但保留一行可解析的协议元数据，
        # 让 Transport 完成 decode 后不会丢失 action/params 等路由字段。
        metadata = {
            "version": msg.version,
            "src": msg.src,
            "dst": msg.dst,
            "msg_type": msg.msg_type,
            "action": msg.action,
            "params": msg.params,
            "result": msg.result,
            "capabilities": msg.capabilities,
            "context_ref": msg.context_ref,
            "memory_refs": msg.memory_refs,
            "state_handle": msg.state_handle,
            "corr_id": msg.corr_id,
            "ts": msg.ts,
        }
        parts.append(
            "【协议元数据】"
            + json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        )

        # 1. 开头：发送方自我介绍 + 请求接收方
        role_map = {
            "planner": "策划Agent",
            "retriever": "资料员Agent",
            "executor": "执行者Agent",
            "summarizer": "总结者Agent",
        }
        src_name = role_map.get(msg.src, msg.src)
        dst_name = role_map.get(msg.dst, msg.dst)
        parts.append(
            f"你好，我是{src_name}({msg.src})，"
            f"请你({dst_name}{msg.dst})执行{msg.action}任务。"
        )

        # 2. 展开上下文（context_ref → 完整任务背景文本）
        if msg.context_ref:
            ctx_text = self._resolve_context(msg.context_ref)
            parts.append(f"【完整任务背景】{ctx_text}")

        # 3. 展开记忆引用（memory_refs → 每条记忆的完整内容）
        for mem_id in msg.memory_refs:
            mem_text = self._resolve_memory(mem_id)
            parts.append(f"【相关历史记忆 {mem_id} 全文】{mem_text}")

        # 4. 展开状态句柄（state_handle → 状态描述文本）
        if msg.state_handle is not None:
            state_text = self._resolve_state(msg.state_handle)
            parts.append(f"【共享状态数据】{state_text}")

        # 5. 请求参数
        if msg.params:
            params_text = json.dumps(msg.params, ensure_ascii=False, indent=2)
            parts.append(f"【请求参数】\n{params_text}")

        # 6. 结果（如果有）
        if msg.result is not None:
            result_text = (
                msg.result
                if isinstance(msg.result, str)
                else json.dumps(msg.result, ensure_ascii=False)
            )
            parts.append(f"【执行结果】{result_text}")

        # 7. 能力列表
        if msg.capabilities:
            caps = "、".join(msg.capabilities)
            parts.append(f"【能力列表】{caps}")

        # 8. 结尾
        parts.append(f"请{msg.action}上述内容并返回结果。")

        return "\n".join(parts)

    def encode(self, msg: Message) -> bytes:
        """编码为自然语言 UTF-8 字节。"""
        text = self._render_text(msg)
        return text.encode("utf-8")

    def decode(self, raw: bytes) -> Message:
        """从文本解码回 Message（简化实现，提取核心字段）。

        注意：文本模式的 decode 是尽力而为（best-effort），
        因为自然语言不像结构化格式能完美还原所有字段。
        主要用于 Transport 层的 encode→measure→decode 流程中
        验证消息可传输，不要求完全还原。
        """
        text = raw.decode("utf-8")
        for line in text.split("\n"):
            if line.startswith("【协议元数据】"):
                payload = line.removeprefix("【协议元数据】")
                try:
                    data = json.loads(payload)
                    return Message(
                        version=data.get("version", 1),
                        src=data.get("src", ""),
                        dst=data.get("dst", ""),
                        msg_type=data.get("msg_type", "request"),
                        action=data.get("action", ""),
                        params=data.get("params", {}),
                        result=data.get("result", None),
                        capabilities=data.get("capabilities", []),
                        context_ref=data.get("context_ref", ""),
                        memory_refs=data.get("memory_refs", []),
                        state_handle=data.get("state_handle", None),
                        corr_id=data.get("corr_id", ""),
                        ts=data.get("ts", 0.0),
                    )
                except json.JSONDecodeError:
                    break

        # 简化解析：提取能识别的字段
        msg = Message()
        for line in text.split("\n"):
            if line.startswith("你好，我是"):
                # 尝试提取 src
                import re
                m = re.search(r"\((\w+)\)", line)
                if m:
                    msg.src = m.group(1)
            elif "【请求参数】" in line:
                # 后续行是 JSON
                pass
        # 完整文本存入 result 供调试
        msg.result = text
        msg.action = "text_decoded"
        return msg

    def measure(self, msg: Message) -> tuple[int, int]:
        """计量文本传输开销。

        - 字节数：渲染文本的 UTF-8 编码大小
        - token 数：对渲染出的文字用 tiktoken 计数
        """
        text = self._render_text(msg)
        byte_count = len(text.encode("utf-8"))
        token_count = len(_ENC.encode(text))
        return byte_count, token_count
