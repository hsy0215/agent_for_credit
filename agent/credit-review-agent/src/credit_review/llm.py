from __future__ import annotations

from pathlib import Path
from time import perf_counter

from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.workflows.llm.augmented_llm_lm_studio import LMStudioAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

from .models import ApplicationFacts, EvidenceChunk
from .monitoring import MetricsRecorder

EXTRACTION_INSTRUCTION = """
你是企业授信材料抽取员。只抽取输入证据中明确出现的事实。
禁止使用常识补全；缺失字段保持 null，并加入 missing_fields。
每个非空字段必须填写 evidence_ids 和简短 source_quote。
金额单位统一为万元。不要恢复或猜测被方括号 token 替代的隐私信息。
financials 必须按“企业主体 + 报表口径 + 报告期间”分别输出，禁止把母公司、子公司、
关联企业或合并口径混在同一个 FinancialSnapshot。period_label 保留原文期间（例如
“2022年1-9月”“2021年9月”“2021年”），period_type 填 annual 或 interim；
每个快照都必须填写 entity_name、scope、evidence_ids 和 source_quotes。
同一年不同期间必须生成不同快照；同一字段冲突时加入 conflicts，不要擅自选择。
""".strip()


class MCPAgentExtractor:
    """Semantic extractor using mcp-agent and a selectable model provider."""

    def __init__(self, provider: str, config_path: Path, max_chars: int = 30_000, metrics: MetricsRecorder | None = None):
        if provider not in {"lm_studio", "openai", "deepseek"}:
            raise ValueError(f"不支持的 LLM provider: {provider}")
        self.provider = provider
        self.config_path = config_path
        self.max_chars = max_chars
        self.metrics = metrics

    async def extract(self, chunks: list[EvidenceChunk], source_file: Path) -> ApplicationFacts:
        evidence = "\n".join(
            f"[{chunk.evidence_id}] {chunk.masked_text or chunk.text}" for chunk in chunks
        )
        if len(evidence) > self.max_chars:
            raise ValueError("单批证据超过限制，请使用 MapReduceExtractor 分批")
        message = f"文件名：{source_file.name}\n\n证据：\n{evidence}"
        started = perf_counter()
        try:
            app = MCPApp(name="credit-review-extractor", settings=str(self.config_path))
            async with app.run():
                agent = Agent(name="credit_facts_extractor", instruction=EXTRACTION_INSTRUCTION)
                async with agent:
                    factory = {
                        "lm_studio": LMStudioAugmentedLLM,
                        "openai": OpenAIAugmentedLLM,
                        "deepseek": AnthropicAugmentedLLM,
                    }[self.provider]
                    llm = await agent.attach_llm(factory)
                    result = await llm.generate_structured(message=message, response_model=ApplicationFacts)
            if self.metrics:
                self.metrics.model_call(self.provider, "extractor", len(message), len(result.model_dump_json()), (perf_counter() - started) * 1000, True)
            return result
        except Exception:
            if self.metrics:
                self.metrics.model_call(self.provider, "extractor", len(message), 0, (perf_counter() - started) * 1000, False)
            raise
