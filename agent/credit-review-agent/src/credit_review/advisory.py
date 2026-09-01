from __future__ import annotations

import json
import re
from pathlib import Path
from time import perf_counter
from typing import Protocol

from mcp_agent.agents.agent import Agent
from mcp_agent.app import MCPApp
from mcp_agent.workflows.llm.augmented_llm_lm_studio import LMStudioAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm_anthropic import AnthropicAugmentedLLM

from .models import AdvisoryNarrative, CaseState, ReviewFinding, ReviewResult
from .monitoring import MetricsRecorder


class AdvisoryAgents(Protocol):
    async def write(self, state: CaseState) -> AdvisoryNarrative: ...
    async def review(self, state: CaseState, report: str) -> ReviewResult: ...


class DeterministicAdvisoryAgents:
    """Offline reviewer/writer used by CI and when no model endpoint is available."""

    async def write(self, state: CaseState) -> AdvisoryNarrative:
        assert state.facts and state.risk and state.amount and state.recommendation
        company = state.facts.company_name.value or "材料未提供企业"
        risks = "；".join(item.description for item in state.risk.items) or "当前规则未命中明确风险项"
        evidence = sorted({evidence for item in state.risk.items for evidence in item.evidence_ids})
        return AdvisoryNarrative(
            business_summary=f"{company}的评审材料已完成结构化核验。",
            risk_rationale=f"风险等级为{state.risk.risk_level}，风险得分{state.risk.total_score}；{risks}。",
            amount_rationale=f"规则引擎按{state.amount.rule_version}计算建议额度{state.amount.suggested_amount}万元。",
            recommendation_rationale=f"基于材料完整性、规则风险和额度结果，系统输出{state.recommendation.value}。",
            cited_evidence_ids=evidence,
        )

    async def review(self, state: CaseState, report: str) -> ReviewResult:
        assert state.amount and state.facts
        findings: list[ReviewFinding] = []
        if f"{state.amount.suggested_amount}" not in report:
            findings.append(ReviewFinding(code="AMOUNT_MISMATCH", severity="critical", message="报告未包含规则引擎建议额度"))
        known = {chunk.evidence_id for chunk in state.chunks}
        if state.narrative:
            invalid = set(state.narrative.cited_evidence_ids) - known
            if invalid:
                findings.append(ReviewFinding(
                    code="INVALID_EVIDENCE", severity="critical",
                    message=f"报告叙述引用不存在的证据: {sorted(invalid)}",
                ))
        pii_patterns = [r"(?<!\d)1[3-9]\d{9}(?!\d)", r"(?<!\d)\d{17}[\dXx](?!\d)"]
        if any(re.search(pattern, report) for pattern in pii_patterns):
            findings.append(ReviewFinding(code="PII_LEAK", severity="critical", message="报告包含未脱敏个人信息"))
        return ReviewResult(passed=not findings, findings=findings)


class MCPAdvisoryAgents:
    """Independent LLM writer and reviewer. Numeric decisions are immutable inputs."""

    def __init__(self, provider: str, config_path: Path, metrics: MetricsRecorder | None = None):
        self.provider = provider
        self.config_path = config_path
        self.metrics = metrics

    async def _structured(self, name: str, instruction: str, payload: dict, model):
        message = json.dumps(payload, ensure_ascii=False)
        started = perf_counter()
        try:
            app = MCPApp(name=f"credit-review-{name}", settings=str(self.config_path))
            async with app.run():
                agent = Agent(name=name, instruction=instruction)
                async with agent:
                    factory = {
                        "lm_studio": LMStudioAugmentedLLM,
                        "openai": OpenAIAugmentedLLM,
                        "deepseek": AnthropicAugmentedLLM,
                    }[self.provider]
                    llm = await agent.attach_llm(factory)
                    result = await llm.generate_structured(message=message, response_model=model)
            if self.metrics:
                self.metrics.model_call(self.provider, name, len(message), len(result.model_dump_json()), (perf_counter() - started) * 1000, True)
            return result
        except Exception:
            if self.metrics:
                self.metrics.model_call(self.provider, name, len(message), 0, (perf_counter() - started) * 1000, False)
            raise

    async def write(self, state: CaseState) -> AdvisoryNarrative:
        return await self._structured(
            "risk_analyst_writer",
            """你是授信风险分析和报告撰写 Agent。只能解释输入 JSON 中已有事实、风险项和额度步骤。
不得新增数字、公司事实或风险，不得修改 total_score、suggested_amount、recommendation。
不确定内容写材料未提供。每个事实性结论必须列出输入中存在的 evidence_id。文字简洁、专业。""",
            {
                "facts": state.facts.model_dump(mode="json"),
                "risk": state.risk.model_dump(mode="json"),
                "amount": state.amount.model_dump(mode="json"),
                "recommendation": state.recommendation,
            },
            AdvisoryNarrative,
        )

    async def review(self, state: CaseState, report: str) -> ReviewResult:
        deterministic = await DeterministicAdvisoryAgents().review(state, report)
        try:
            llm_result = await self._structured(
                "independent_reviewer",
                """你是独立授信报告审核 Agent。核对报告是否忠于输入事实、风险分、额度和建议；
检查无证据陈述、数字矛盾和隐私泄漏。不得重写报告。发现任何额度/结论矛盾时 passed=false。""",
                {
                    "facts": state.facts.model_dump(mode="json"),
                    "risk": state.risk.model_dump(mode="json"),
                    "amount": state.amount.model_dump(mode="json"),
                    "recommendation": state.recommendation,
                    "known_evidence_ids": [chunk.evidence_id for chunk in state.chunks],
                    "report": report,
                },
                ReviewResult,
            )
        except Exception as error:
            return ReviewResult(
                passed=False,
                findings=[
                    ReviewFinding(
                        code="MODEL_REVIEWER_UNAVAILABLE",
                        severity="high",
                        message=(
                            "独立模型审核未返回可解析的结构化结果，已完成确定性审核；"
                            f"在人工复核前不得自动通过。错误类型：{type(error).__name__}"
                        ),
                    ),
                    *deterministic.findings,
                ],
            )
        return ReviewResult(
            passed=llm_result.passed and deterministic.passed,
            findings=llm_result.findings + deterministic.findings,
        )
