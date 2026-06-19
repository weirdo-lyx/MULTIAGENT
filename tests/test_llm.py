"""M0 LLM 客户端测试：覆盖 api 和 mock 两种模式。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import reset_config_cache
from src.llm.client import LLMClient


@pytest.fixture(autouse=True)
def _reset_config():
    """每个测试前重置配置缓存。"""
    reset_config_cache()
    yield
    reset_config_cache()


# ----------------------------------------------------------------
# Mock 模式测试
# ----------------------------------------------------------------

class TestMockMode:
    """Mock 模式：不联网，确定性返回。"""

    def setup_method(self) -> None:
        self.client = LLMClient(mode="mock")

    def test_mock_mode_flag(self) -> None:
        """is_mock() 应返回 True。"""
        assert self.client.is_mock() is True

    def test_mock_plan_action(self) -> None:
        """action='plan' 应返回策划占位文本。"""
        messages = [{"role": "user", "content": "请分析 RAG 技术的核心原理"}]
        result = self.client.chat(messages, action="plan")
        assert "[Mock-Plan-" in result
        assert "planner" in result or "子任务" in result

    def test_mock_search_action(self) -> None:
        """action='search' 应返回检索占位文本。"""
        messages = [{"role": "user", "content": "RAG 技术"}]
        result = self.client.chat(messages, action="search")
        assert "[Mock-Search-" in result
        assert "检索结果" in result or "文档" in result

    def test_mock_execute_action(self) -> None:
        """action='execute' 应返回执行占位文本。"""
        messages = [{"role": "user", "content": "计算斐波那契数列"}]
        result = self.client.chat(messages, action="execute")
        assert "[Mock-Execute-" in result
        assert "执行" in result or "result" in result

    def test_mock_summarize_action(self) -> None:
        """action='summarize' 应返回摘要占位文本。"""
        messages = [{"role": "user", "content": "总结上述分析"}]
        result = self.client.chat(messages, action="summarize")
        assert "[Mock-Summary-" in result
        assert "摘要" in result or "结论" in result

    def test_mock_keyword_detection(self) -> None:
        """即使不传 action，消息内容包含关键字也能路由到正确分支。"""
        messages = [{"role": "user", "content": "请检索相关文档"}]
        result = self.client.chat(messages, action="")
        assert "[Mock-Search-" in result

    def test_mock_deterministic(self) -> None:
        """相同输入应返回完全相同的结果（确定性）。"""
        messages = [{"role": "user", "content": "RAG 技术原理"}]
        r1 = self.client.chat(messages, action="plan")
        r2 = self.client.chat(messages, action="plan")
        assert r1 == r2

    def test_mock_different_inputs_different_outputs(self) -> None:
        """不同输入应产生不同输出（哈希不同）。"""
        m1 = [{"role": "user", "content": "RAG 技术"}]
        m2 = [{"role": "user", "content": "多智能体协作"}]
        r1 = self.client.chat(m1, action="search")
        r2 = self.client.chat(m2, action="search")
        # 哈希部分不同
        assert r1 != r2

    def test_mock_default_fallback(self) -> None:
        """无匹配关键字时走默认回复。"""
        messages = [{"role": "user", "content": "你好"}]
        result = self.client.chat(messages, action="unknown_action")
        assert "[Mock-Response-" in result

    def test_mock_extracts_last_user_message(self) -> None:
        """Mock 应从最后一条 user 消息提取内容。"""
        messages = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "第一个问题"},
            {"role": "assistant", "content": "回复"},
            {"role": "user", "content": "检索 RAG 相关资料"},
        ]
        result = self.client.chat(messages, action="search")
        assert "RAG 相关资料" in result


# ----------------------------------------------------------------
# API 模式测试（使用 mock 替换网络调用）
# ----------------------------------------------------------------

class TestApiMode:
    """API 模式：验证 OpenAI 兼容接口的调用逻辑。"""

    def test_api_mode_flag(self) -> None:
        """api 模式 is_mock() 应返回 False。"""
        client = LLMClient(mode="api", base_url="http://fake", api_key="test", model="test")
        assert client.is_mock() is False

    def test_api_mode_reads_config(self) -> None:
        """未指定参数时应从 config 读取。"""
        client = LLMClient(mode="api")
        # config.yaml 里的值
        assert client.base_url == "https://api.deepseek.com/v1"
        assert client.model == "deepseek-chat"

    def test_api_chat_calls_openai(self) -> None:
        """api 模式 chat 应调用 OpenAI 客户端并返回回复。"""
        client = LLMClient(mode="api", base_url="http://fake", api_key="test", model="test-model")

        # 构造 mock 响应
        mock_choice = MagicMock()
        mock_choice.message.content = "这是 API 回复"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        # 替换 OpenAI client
        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response
        client._client = mock_openai_client

        messages = [{"role": "user", "content": "测试"}]
        result = client.chat(messages, action="plan")

        assert result == "这是 API 回复"
        mock_openai_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_openai_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "test-model"

    def test_api_returns_empty_on_none(self) -> None:
        """API 返回 None 时应返回空字符串。"""
        client = LLMClient(mode="api", base_url="http://fake", api_key="test", model="m")

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response
        client._client = mock_openai_client

        result = client.chat([{"role": "user", "content": "hi"}])
        assert result == ""

    def test_invalid_mode_raises(self) -> None:
        """未知模式应抛出 ValueError。"""
        client = LLMClient(mode="mock")
        client.mode = "invalid"
        with pytest.raises(ValueError, match="未知 LLM 模式"):
            client.chat([{"role": "user", "content": "test"}])


# ----------------------------------------------------------------
# 配置加载测试
# ----------------------------------------------------------------

class TestConfigIntegration:
    """验证 LLMClient 正确从 config 读取参数。"""

    def test_default_mode_from_config(self) -> None:
        """不传 mode 时应从 config.yaml 读取（当前配置为 mock）。"""
        client = LLMClient()
        assert client.mode == "mock"
        assert client.is_mock() is True
