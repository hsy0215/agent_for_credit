from __future__ import annotations

import os


def configure_provider(provider: str) -> None:
    """Validate provider secrets and map DeepSeek credentials in-process."""
    if provider != "deepseek":
        return
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "DeepSeek Provider 缺少 API Key，请先设置 DEEPSEEK_API_KEY 环境变量"
        )
    # mcp-agent's AnthropicSettings and the Anthropic SDK consume this name.
    # The value remains process-local and is never written to a project file.
    os.environ["ANTHROPIC_API_KEY"] = api_key

