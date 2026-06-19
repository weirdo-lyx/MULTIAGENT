"""M1 通信协议测试。核心验证：同消息下 struct 的 token 显著小于 text。"""
from __future__ import annotations

import pytest

from src.protocol.message import Message
from src.protocol.struct_codec import StructCodec
from src.protocol.text_codec import TextCodec


# ----------------------------------------------------------------
# 辅助：模拟引用解析器（M3/M4 尚未实现，用 mock 数据）
# ----------------------------------------------------------------

MOCK_CONTEXTS: dict[str, str] = {
    "ctx_001": (
        '用户希望系统研究“检索增强生成(RAG)”技术，重点包括以下几个方面：\n'
        '1) RAG 的基本原理与流程：检索增强生成是一种将信息检索与文本生成相结合的技术，'
        '通过在生成过程中动态检索外部知识库来增强大语言模型的回答质量。其核心流程包括'
        '查询理解、向量检索、上下文融合和答案生成四个阶段。\n'
        '2) 主流向量检索方法对比：包括基于倒排索引的传统方法（如 BM25）、'
        '基于稠密向量的 ANN 检索方法（如 HNSW、IVF）、以及混合检索策略。'
        '需要对比它们在召回率、延迟、内存占用等维度的表现。\n'
        '3) 2024 年 RAG 最新进展：包括多模态 RAG、自适应检索、'
        '基于强化学习的检索优化、以及 RAG 在企业级应用中的落地经验。\n'
        '请针对上述三个子问题分别进行深入研究，并给出综合分析报告。'
    ),
}

MOCK_MEMORIES: dict[str, str] = {
    "mem_042": (
        '我们之前对向量数据库进行了系统调研，结论如下：\n'
        '1) ChromaDB 适合嵌入式场景，支持元数据过滤，部署简单，'
        '适合本项目的轻量级需求。\n'
        '2) Milvus 适合大规模生产环境，支持分布式部署和多种索引类型，'
        '但部署和运维成本较高。\n'
        '3) FAISS 是 Facebook 开源的向量检索库，性能优秀但缺乏完整的数据库功能。\n'
        '综合来看，本项目推荐使用 ChromaDB 作为向量存储方案。'
    ),
    "mem_055": (
        '关于多智能体通信协议的研究笔记：\n'
        '结构化通信协议相比纯文本方式可以显著减少 token 开销。'
        '通过引用机制（上下文引用、记忆引用）和紧凑序列化（msgpack），'
        '可以将通信 token 减少 80% 以上。'
    ),
}


def mock_context_resolver(ref: str) -> str:
    return MOCK_CONTEXTS.get(ref, f"[未知上下文 {ref}]")


def mock_memory_resolver(mem_id: str) -> str:
    return MOCK_MEMORIES.get(mem_id, f"[未知记忆 {mem_id}]")


def mock_state_resolver(handle: dict) -> str:
    shape = handle.get("shape", "unknown")
    dtype = handle.get("dtype", "unknown")
    return (
        f"共享内存中的嵌入向量矩阵，shape={shape}，dtype={dtype}，"
        f"包含 5 条文档的 512 维嵌入向量，用于语义检索和相似度计算。"
    )


# ----------------------------------------------------------------
# 构造测试消息
# ----------------------------------------------------------------

def make_rich_message() -> Message:
    """构造一条带长上下文 + 记忆引用 + 状态句柄的 Message（对标标准样例）。"""
    return Message(
        version=1,
        src="planner",
        dst="retriever",
        msg_type="request",
        action="search",
        params={
            "queries": [
                "RAG基本原理",
                "向量检索方法对比",
                "RAG 2024最新进展",
            ],
            "top_k": 5,
        },
        context_ref="ctx_001",
        memory_refs=["mem_042", "mem_055"],
        state_handle={
            "shm_name": "embed_shm_001",
            "shape": [5, 512],
            "dtype": "float32",
        },
    )


def make_simple_message() -> Message:
    """构造一条简单消息（无引用）。"""
    return Message(
        version=1,
        src="executor",
        dst="summarizer",
        msg_type="request",
        action="summarize",
        params={"task": "总结分析结果"},
    )


# ----------------------------------------------------------------
# 测试 StructCodec
# ----------------------------------------------------------------

class TestStructCodec:
    """结构化编解码器测试。"""

    def setup_method(self) -> None:
        self.codec = StructCodec()

    def test_encode_decode_roundtrip(self) -> None:
        """编码→解码应能还原核心字段。"""
        msg = make_rich_message()
        raw = self.codec.encode(msg)
        decoded = self.codec.decode(raw)

        assert decoded.src == msg.src
        assert decoded.dst == msg.dst
        assert decoded.action == msg.action
        assert decoded.context_ref == msg.context_ref
        assert decoded.memory_refs == msg.memory_refs
        assert decoded.state_handle == msg.state_handle
        assert decoded.params == msg.params

    def test_encode_is_compact(self) -> None:
        """msgpack 编码应相当紧凑。"""
        msg = make_rich_message()
        raw = self.codec.encode(msg)
        # msgpack 二进制应比 JSON 等价物更小
        import json
        json_size = len(json.dumps(
            {"v": 1, "src": msg.src, "dst": msg.dst, "act": msg.action,
             "p": msg.params, "ctx": msg.context_ref, "mem": msg.memory_refs,
             "sh": msg.state_handle},
            ensure_ascii=False,
        ).encode("utf-8"))
        assert len(raw) < json_size

    def test_measure_returns_bytes_and_tokens(self) -> None:
        """measure() 应返回 (字节数, token数) 且都大于 0。"""
        msg = make_rich_message()
        byte_count, token_count = self.codec.measure(msg)
        assert byte_count > 0
        assert token_count > 0

    def test_refs_kept_as_ids(self) -> None:
        """结构化编码保持引用为 ID，不展开。"""
        msg = make_rich_message()
        raw = self.codec.encode(msg)
        raw_text = raw.decode("latin-1")  # msgpack 是二进制
        # 确保上下文全文不在编码中
        assert "检索增强生成是一种将信息检索与文本生成相结合" not in raw_text
        # 确保记忆全文不在编码中
        assert "ChromaDB 适合嵌入式场景" not in raw_text


# ----------------------------------------------------------------
# 测试 TextCodec
# ----------------------------------------------------------------

class TestTextCodec:
    """文本编解码器测试。"""

    def setup_method(self) -> None:
        self.codec = TextCodec(
            context_resolver=mock_context_resolver,
            memory_resolver=mock_memory_resolver,
            state_resolver=mock_state_resolver,
        )

    def test_encode_expands_context(self) -> None:
        """文本编码必须展开 context_ref 为完整文本。"""
        msg = make_rich_message()
        raw = self.codec.encode(msg)
        text = raw.decode("utf-8")
        # 上下文全文应该出现在文本中
        assert "检索增强生成是一种将信息检索与文本生成相结合" in text

    def test_encode_expands_memory(self) -> None:
        """文本编码必须展开 memory_refs 为完整文本。"""
        msg = make_rich_message()
        raw = self.codec.encode(msg)
        text = raw.decode("utf-8")
        # 记忆全文应该出现
        assert "ChromaDB 适合嵌入式场景" in text

    def test_encode_expands_state_handle(self) -> None:
        """文本编码必须展开 state_handle 为描述文本。"""
        msg = make_rich_message()
        raw = self.codec.encode(msg)
        text = raw.decode("utf-8")
        # 状态描述应该出现
        assert "嵌入向量矩阵" in text or "shape=" in text

    def test_measure_returns_bytes_and_tokens(self) -> None:
        """measure() 应返回 (字节数, token数) 且都大于 0。"""
        msg = make_rich_message()
        byte_count, token_count = self.codec.measure(msg)
        assert byte_count > 0
        assert token_count > 0

    def test_simple_message_no_refs(self) -> None:
        """无引用的简单消息也能正常编码。"""
        msg = make_simple_message()
        raw = self.codec.encode(msg)
        text = raw.decode("utf-8")
        assert "summarize" in text

    def test_encode_decode_produces_message(self) -> None:
        """encode→decode 应返回 Message 对象。"""
        msg = make_rich_message()
        raw = self.codec.encode(msg)
        decoded = self.codec.decode(raw)
        assert isinstance(decoded, Message)


# ----------------------------------------------------------------
# 核心测试：TextCodec vs StructCodec token 对比
# ----------------------------------------------------------------

class TestTokenComparison:
    """核心验证：同消息下 struct 的 token 显著小于 text。

    这是整个项目 25 分通信效率评分的关键证据。
    """

    def test_struct_tokens_significantly_less(self) -> None:
        """同一条消息，结构化 token 数应显著小于文本 token 数。"""
        msg = make_rich_message()

        text_codec = TextCodec(
            context_resolver=mock_context_resolver,
            memory_resolver=mock_memory_resolver,
            state_resolver=mock_state_resolver,
        )
        struct_codec = StructCodec()

        text_bytes, text_tokens = text_codec.measure(msg)
        struct_bytes, struct_tokens = struct_codec.measure(msg)

        # 结构化 token 应显著更少
        assert struct_tokens < text_tokens
        # 节省率应 > 50%（标准样例预期 ~94%）
        saving = (1 - struct_tokens / text_tokens) * 100
        assert saving > 50, f"token 节省率 {saving:.1f}% 不足 50%"

    def test_struct_bytes_significantly_less(self) -> None:
        """同一条消息，结构化字节数应显著小于文本书面字节数。"""
        msg = make_rich_message()

        text_codec = TextCodec(
            context_resolver=mock_context_resolver,
            memory_resolver=mock_memory_resolver,
            state_resolver=mock_state_resolver,
        )
        struct_codec = StructCodec()

        text_bytes, text_tokens = text_codec.measure(msg)
        struct_bytes, struct_tokens = struct_codec.measure(msg)

        assert struct_bytes < text_bytes
        saving = (1 - struct_bytes / text_bytes) * 100
        assert saving > 50, f"字节节省率 {saving:.1f}% 不足 50%"

    def test_comparison_with_multiple_memories(self) -> None:
        """多条记忆引用时，节省效果更明显。"""
        msg = Message(
            version=1,
            src="planner",
            dst="retriever",
            msg_type="request",
            action="search",
            params={"queries": ["多智能体协作", "通信协议优化"], "top_k": 3},
            context_ref="ctx_001",
            memory_refs=["mem_042", "mem_055"],
            state_handle={"shm_name": "s1", "shape": [10, 512], "dtype": "float32"},
        )

        text_codec = TextCodec(
            context_resolver=mock_context_resolver,
            memory_resolver=mock_memory_resolver,
            state_resolver=mock_state_resolver,
        )
        struct_codec = StructCodec()

        _, text_tokens = text_codec.measure(msg)
        _, struct_tokens = struct_codec.measure(msg)

        # 引用越多，结构化优势越大
        saving = (1 - struct_tokens / text_tokens) * 100
        assert saving > 60, f"多引用场景 token 节省率 {saving:.1f}% 不足 60%"

    def test_both_codecs_use_same_tiktoken(self) -> None:
        """两个 codec 的 measure 应使用相同的 tiktoken 编码。"""
        # 简单消息（无引用），两种方式 token 应接近
        msg = make_simple_message()

        text_codec = TextCodec()
        struct_codec = StructCodec()

        _, text_tokens = text_codec.measure(msg)
        _, struct_tokens = struct_codec.measure(msg)

        # 简单消息无引用展开差异，两者应在同一量级
        ratio = struct_tokens / text_tokens if text_tokens > 0 else 1.0
        assert 0.1 < ratio < 10, f"简单消息 token 比率 {ratio} 异常"

    def test_print_comparison_table(self, capsys: pytest.CaptureFixture) -> None:
        """打印对比表（方便查看和验证）。"""
        msg = make_rich_message()

        text_codec = TextCodec(
            context_resolver=mock_context_resolver,
            memory_resolver=mock_memory_resolver,
            state_resolver=mock_state_resolver,
        )
        struct_codec = StructCodec()

        text_bytes, text_tokens = text_codec.measure(msg)
        struct_bytes, struct_tokens = struct_codec.measure(msg)

        token_saving = (1 - struct_tokens / text_tokens) * 100
        byte_saving = (1 - struct_bytes / text_bytes) * 100

        print("\n" + "=" * 60)
        print("M1 通信效率对比（标准样例场景）")
        print("=" * 60)
        print(f"{'模式':<12} {'token':>8} {'字节':>8}")
        print("-" * 30)
        print(f"{'文本模式':<10} {text_tokens:>8} {text_bytes:>8}")
        print(f"{'结构化模式':<8} {struct_tokens:>8} {struct_bytes:>8}")
        print(f"{'节省':<10} {token_saving:>7.1f}% {byte_saving:>7.1f}%")
        print("=" * 60)
