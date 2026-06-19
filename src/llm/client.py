"""M0 LLM 客户端：支持 api（OpenAI 兼容）和 mock（离线确定性返回）两种模式。

配置从 config/config.yaml 读取，禁止硬编码。
mock 模式不联网，根据消息中的 action/关键字返回确定性占位文本，用于免费可复现实验。
"""
from __future__ import annotations

import hashlib
from typing import Any

from src.config import load_config


class LLMClient:
    """LLM 客户端，双模式支持。

    Args:
        mode: "api" 或 "mock"。若为 None 则从 config 读取。
        base_url: API 地址（仅 api 模式使用）。若为 None 则从 config 读取。
        api_key: API 密钥（仅 api 模式使用）。若为 None 则从 config 读取。
        model: 模型名称（仅 api 模式使用）。若为 None 则从 config 读取。
    """

    def __init__(
        self,
        mode: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        cfg = load_config()
        llm_cfg = cfg.get("llm", {})

        self.mode = mode or llm_cfg.get("mode", "mock")
        self.base_url = base_url or llm_cfg.get("base_url", "")
        self.api_key = api_key or llm_cfg.get("api_key", "")
        self.model = model or llm_cfg.get("model", "")

        self._client: Any = None
        if self.mode == "api":
            self._init_api_client()

    def _init_api_client(self) -> None:
        """初始化 OpenAI 兼容客户端（懒加载，仅 api 模式需要）。"""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        except ImportError:
            raise ImportError(
                "api 模式需要 openai 包，请运行: pip install openai"
            )

    def chat(
        self,
        messages: list[dict[str, str]],
        action: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """发送对话请求，返回文本回复。

        Args:
            messages: OpenAI 格式的消息列表 [{"role": "user", "content": "..."}]
            action: 当前任务动作（plan/search/execute/summarize），mock 模式用
            temperature: 生成温度
            max_tokens: 最大 token 数

        Returns:
            LLM 回复文本
        """
        if self.mode == "mock":
            return self._mock_chat(messages, action)
        elif self.mode == "api":
            return self._api_chat(messages, temperature, max_tokens)
        else:
            raise ValueError(f"未知 LLM 模式: {self.mode}，支持 api/mock")

    def _api_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """通过 OpenAI 兼容接口调用 LLM。"""
        if self._client is None:
            self._init_api_client()

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def _mock_chat(
        self,
        messages: list[dict[str, str]],
        action: str,
    ) -> str:
        """Mock 模式：根据 action 和消息内容返回确定性占位文本。

        不联网，完全确定性，用于离线可复现实验。
        通过 action 关键字和消息内容哈希生成一致的回复。
        """
        # 从最后一条 user 消息提取关键字
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # 根据 action 返回对应的确定性占位文本
        action_lower = action.lower()
        content_lower = user_content.lower()

        if action_lower == "plan" or "plan" in content_lower:
            return self._mock_plan(user_content)
        elif action_lower == "search" or "search" in content_lower or "检索" in user_content:
            return self._mock_search(user_content)
        elif action_lower == "execute" or "execute" in content_lower or "执行" in user_content:
            return self._mock_execute(user_content)
        elif action_lower == "summarize" or "summarize" in content_lower or "总结" in user_content:
            return self._mock_summarize(user_content)
        else:
            return self._mock_default(user_content)

    def _mock_plan(self, content: str) -> str:
        """Mock 策划回复。"""
        # 用内容哈希保证确定性
        h = hashlib.md5(content.encode()).hexdigest()[:8]
        return (
            f"[Mock-Plan-{h}] 任务分析与规划结果：\n"
            f"1. 子任务1：分析「{content[:20]}」的核心要点\n"
            f"2. 子任务2：检索相关证据和资料\n"
            f"3. 子任务3：综合总结并生成报告\n"
            f"分配：planner→retriever→summarizer"
        )

    def _mock_search(self, content: str) -> str:
        """Mock 检索回复。"""
        h = hashlib.md5(content.encode()).hexdigest()[:8]
        return (
            f"[Mock-Search-{h}] 检索结果：\n"
            f"文档1：关于「{content[:15]}」的基础概述，相关度0.92\n"
            f"文档2：「{content[:15]}」的技术细节分析，相关度0.87\n"
            f"文档3：「{content[:15]}」的最新进展综述，相关度0.81"
        )

    def _mock_execute(self, content: str) -> str:
        """Mock 执行回复（生成确定性 Python 代码）。"""
        h = hashlib.md5(content.encode()).hexdigest()[:8]
        return (
            f"[Mock-Execute-{h}] 代码执行结果：\n"
            f"```python\n"
            f"result = 42  # 确定性计算结果\n"
            f"print(f'计算完成: {{result}}')\n"
            f"```\n"
            f"stdout: 计算完成: 42\n"
            f"执行状态: ok"
        )

    def _mock_summarize(self, content: str) -> str:
        """Mock 摘要回复。"""
        h = hashlib.md5(content.encode()).hexdigest()[:8]
        return (
            f"[Mock-Summary-{h}] 综合分析摘要：\n"
            f"根据检索和分析结果，关于「{content[:15]}」的核心结论如下：\n"
            f"- 要点1：该领域的基本框架和方法论已较为成熟\n"
            f"- 要点2：主要挑战在于实际应用中的效率优化\n"
            f"- 要点3：未来发展方向集中在规模化和智能化"
        )

    def _mock_default(self, content: str) -> str:
        """Mock 默认回复。"""
        h = hashlib.md5(content.encode()).hexdigest()[:8]
        return (
            f"[Mock-Response-{h}] 已收到请求，处理完成。\n"
            f"针对「{content[:20]}」的回复：任务已执行，结果正常。"
        )

    def is_mock(self) -> bool:
        """是否为 mock 模式。"""
        return self.mode == "mock"
