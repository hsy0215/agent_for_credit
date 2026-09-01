import asyncio
from pathlib import Path

from credit_review.extraction import HeuristicExtractor
from credit_review.advisory import DeterministicAdvisoryAgents
from credit_review.models import AdvisoryNarrative, ReviewResult, Stage
from credit_review.reporting import ReportRenderer
from credit_review.rules import AmountEngine, RiskEngine
from credit_review.storage import FileCaseRepository
from credit_review.workflow import CreditReviewWorkflow

ROOT = Path(__file__).resolve().parents[1]


class InvalidCitationAdvisory:
    async def write(self, state):
        return AdvisoryNarrative(
            business_summary="测试企业概况",
            risk_rationale="测试风险说明",
            amount_rationale="测试额度说明",
            recommendation_rationale="测试建议",
            cited_evidence_ids=["EV-NOT-EXIST"],
        )

    async def review(self, state, report):
        return ReviewResult(passed=True)


class InvalidInlineCitationAdvisory(InvalidCitationAdvisory):
    async def write(self, state):
        return AdvisoryNarrative(
            business_summary="测试企业概况",
            risk_rationale="该结论来自 EV-FAKE-IN-PROSE",
            amount_rationale="测试额度说明",
            recommendation_rationale="测试建议",
            cited_evidence_ids=[],
        )


def test_end_to_end_offline(tmp_path):
    source = tmp_path / "深圳测试科技有限公司评审报告.txt"
    source.write_text("公司成立于2019年。申请授信额度300万元。联系电话13812345678。", encoding="utf-8")
    workflow = CreditReviewWorkflow(
        FileCaseRepository(tmp_path / "runtime"),
        HeuristicExtractor(),
        RiskEngine(ROOT / "config/risk_rules.yaml"),
        AmountEngine(ROOT / "config/amount_rules.yaml"),
        ReportRenderer(ROOT / "templates"),
        DeterministicAdvisoryAgents(),
    )
    state = asyncio.run(workflow.run(source))
    assert state.stage == Stage.PENDING_APPROVAL
    assert state.facts.requested_amount.value == 300
    assert "13812345678" not in state.chunks[0].masked_text
    assert Path(state.report_file).exists()
    approved = workflow.approve(state.case_id, "approve", "tester")
    assert approved.stage == Stage.APPROVED


def test_invalid_narrative_evidence_falls_back_without_aborting(tmp_path):
    source = tmp_path / "深圳测试科技有限公司评审报告.txt"
    source.write_text("申请授信额度300万元。", encoding="utf-8")
    workflow = CreditReviewWorkflow(
        FileCaseRepository(tmp_path / "runtime"),
        HeuristicExtractor(),
        RiskEngine(ROOT / "config/risk_rules.yaml"),
        AmountEngine(ROOT / "config/amount_rules.yaml"),
        ReportRenderer(ROOT / "templates"),
        InvalidCitationAdvisory(),
    )

    state = asyncio.run(workflow.run(source))

    assert state.stage == Stage.PENDING_APPROVAL
    assert state.narrative.cited_evidence_ids == []
    assert any(event.action == "narrative_evidence_rejected" for event in state.audit)


def test_invalid_inline_evidence_id_also_falls_back(tmp_path):
    source = tmp_path / "深圳测试科技有限公司评审报告.txt"
    source.write_text("申请授信额度300万元。", encoding="utf-8")
    workflow = CreditReviewWorkflow(
        FileCaseRepository(tmp_path / "runtime"),
        HeuristicExtractor(),
        RiskEngine(ROOT / "config/risk_rules.yaml"),
        AmountEngine(ROOT / "config/amount_rules.yaml"),
        ReportRenderer(ROOT / "templates"),
        InvalidInlineCitationAdvisory(),
    )

    state = asyncio.run(workflow.run(source))

    assert "EV-FAKE-IN-PROSE" not in Path(state.report_file).read_text(encoding="utf-8")
    assert any(event.action == "narrative_evidence_rejected" for event in state.audit)
