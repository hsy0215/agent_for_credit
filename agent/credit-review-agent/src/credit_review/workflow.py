from __future__ import annotations

import hashlib
import re
from pathlib import Path
from time import perf_counter

from .advisory import AdvisoryAgents, DeterministicAdvisoryAgents
from .document import DocumentParser
from .extraction import FactsExtractor
from .models import (
    ApprovalEvent, AuditEvent, CaseState, Recommendation, Stage,
)
from .monitoring import MetricsRecorder
from .privacy import PrivacyMasker
from .reporting import ReportRenderer
from .rules import AmountEngine, RiskEngine
from .storage import FileCaseRepository


class CreditReviewWorkflow:
    """Explicit workflow: LLM proposes facts; code owns decisions and transitions."""

    def __init__(
        self,
        repository: FileCaseRepository,
        extractor: FactsExtractor,
        risk_engine: RiskEngine,
        amount_engine: AmountEngine,
        renderer: ReportRenderer,
        advisory_agents: AdvisoryAgents,
        metrics: MetricsRecorder | None = None,
    ):
        self.repository = repository
        self.extractor = extractor
        self.risk_engine = risk_engine
        self.amount_engine = amount_engine
        self.renderer = renderer
        self.advisory_agents = advisory_agents
        self.metrics = metrics
        self.parser = DocumentParser()
        self.masker = PrivacyMasker()

    async def run(self, source_file: Path, case_id: str | None = None) -> CaseState:
        source_file = source_file.resolve()
        case_id = case_id or self._case_id(source_file)
        started = perf_counter()
        document_id, chunks = self.parser.parse(source_file)
        state = CaseState(case_id=case_id, source_file=str(source_file), stage=Stage.INGESTED, chunks=chunks)
        self._audit(state, "document_parsed", {"document_id": document_id, "chunks": len(chunks)})
        self.repository.save(state)
        self._stage_metric(case_id, "ingest", started)

        started = perf_counter()
        masked, mapping = self.masker.mask(chunks)
        mapping_file = self.repository.save_secret_mapping(case_id, mapping)
        state.chunks = masked
        state.mask_mapping_file = str(mapping_file)
        state.stage = Stage.MASKED
        self._audit(state, "pii_masked", {"masked_values": len(mapping)})
        self.repository.save(state)
        self._stage_metric(case_id, "privacy_mask", started)
        if self.metrics:
            masked_corpus = "\n".join(chunk.masked_text or "" for chunk in masked)
            leaked = sum(value in masked_corpus for value in mapping.values())
            self.metrics.record(
                "privacy", case_id=case_id, masked_values=len(mapping), leaked_values=leaked,
                leakage_rate=leaked / (len(mapping) or 1),
            )

        started = perf_counter()
        state.facts = await self.extractor.extract(masked, source_file)
        state.stage = Stage.EXTRACTED
        self._audit(state, "facts_extracted", {"missing_fields": state.facts.missing_fields})
        self._stage_metric(case_id, "extract", started)

        started = perf_counter()
        self._validate_evidence(state)
        state.stage = Stage.VALIDATED
        self._audit(state, "facts_validated")
        self._stage_metric(case_id, "validate", started)

        started = perf_counter()
        state.risk = self.risk_engine.assess(state.facts)
        state.stage = Stage.RISK_ASSESSED
        self._audit(state, "risk_assessed", {"score": state.risk.total_score, "level": state.risk.risk_level})
        self._stage_metric(case_id, "risk", started)

        started = perf_counter()
        state.amount = self.amount_engine.calculate(state.facts, state.risk)
        state.stage = Stage.AMOUNT_CALCULATED
        self._audit(state, "amount_calculated", {"suggested_amount": state.amount.suggested_amount})
        self._stage_metric(case_id, "amount", started)

        started = perf_counter()
        state.recommendation = self._recommend(state)
        try:
            state.narrative = await self.advisory_agents.write(state)
        except Exception as error:
            self._audit(state, "writer_model_unavailable", {"error_type": type(error).__name__})
            state.narrative = await DeterministicAdvisoryAgents().write(state)
        try:
            self._validate_narrative_evidence(state)
        except ValueError as error:
            # Never weaken the evidence gate. If a narrative model invents an
            # evidence ID, retain an audit event and fall back to deterministic
            # prose derived only from the validated state.
            self._audit(state, "narrative_evidence_rejected", {"error": str(error)})
            state.narrative = await DeterministicAdvisoryAgents().write(state)
            self._validate_narrative_evidence(state)
        report = self.repository.case_dir(case_id) / "credit_report.md"
        state.report_file = str(report)
        state.stage = Stage.REPORT_GENERATED
        self.renderer.render(state, report)
        state.review = await self.advisory_agents.review(state, report.read_text(encoding="utf-8"))
        if not state.review.passed:
            state.recommendation = Recommendation.SUPPLEMENT
            state.narrative = state.narrative.model_copy(update={
                "recommendation_rationale": "独立审核未通过，须补充或修正材料后重新评审。",
            })
            self.renderer.render(state, report)
        state.stage = Stage.PENDING_APPROVAL
        self._audit(state, "report_generated", {
            "recommendation": state.recommendation.value,
            "review_passed": state.review.passed,
            "review_findings": len(state.review.findings),
        })
        self.repository.save(state)
        self._stage_metric(case_id, "report_and_review", started)
        return state

    def approve(self, case_id: str, action: str, operator: str, comment: str = "") -> CaseState:
        state = self.repository.load(case_id)
        if state.stage != Stage.PENDING_APPROVAL:
            raise ValueError(f"案件当前状态 {state.stage} 不允许审批")
        targets = {
            "approve": Stage.APPROVED,
            "reject": Stage.REJECTED,
            "supplement": Stage.SUPPLEMENT_REQUIRED,
        }
        if action not in targets:
            raise ValueError("action 必须是 approve、reject 或 supplement")
        state.approvals.append(ApprovalEvent(action=action, operator=operator, comment=comment))
        state.stage = targets[action]
        self._audit(state, "human_approval", {"action": action, "operator": operator})
        self.repository.save(state)
        if self.metrics:
            expected = {
                Recommendation.PASS: "approve",
                Recommendation.REJECT: "reject",
                Recommendation.SUPPLEMENT: "supplement",
            }.get(state.recommendation)
            self.metrics.record(
                "human_approval", case_id=case_id, action=action,
                recommendation=state.recommendation, modified=action != expected,
            )
        return state

    def _stage_metric(self, case_id: str, stage: str, started: float) -> None:
        if self.metrics:
            self.metrics.record(
                "stage", case_id=case_id, stage=stage,
                latency_ms=round((perf_counter() - started) * 1000, 2), success=True,
            )

    @staticmethod
    def _case_id(source_file: Path) -> str:
        digest = hashlib.sha256(source_file.read_bytes()).hexdigest()[:10]
        return f"CR-{digest}"

    @staticmethod
    def _audit(state: CaseState, action: str, details: dict | None = None) -> None:
        state.audit.append(AuditEvent(stage=state.stage, action=action, details=details or {}))

    @staticmethod
    def _validate_evidence(state: CaseState) -> None:
        assert state.facts is not None
        known = {chunk.evidence_id for chunk in state.chunks}
        for name in (
            "company_name", "established_year", "registered_capital", "requested_amount",
            "industry", "litigation_count", "guarantee_amount",
        ):
            field = getattr(state.facts, name)
            invalid = set(field.evidence_ids) - known
            if invalid:
                raise ValueError(f"字段 {name} 引用了不存在的证据: {sorted(invalid)}")
        for index, snapshot in enumerate(state.facts.financials):
            invalid = set(snapshot.evidence_ids) - known
            if invalid:
                raise ValueError(f"财务快照 {index} 引用了不存在的证据: {sorted(invalid)}")

    @staticmethod
    def _validate_narrative_evidence(state: CaseState) -> None:
        assert state.narrative
        known = {chunk.evidence_id for chunk in state.chunks}
        narrative_ids = set(state.narrative.cited_evidence_ids)
        for field in (
            state.narrative.business_summary, state.narrative.risk_rationale,
            state.narrative.amount_rationale, state.narrative.recommendation_rationale,
        ):
            narrative_ids.update(re.findall(r"EV-[A-Za-z0-9_-]+", field))
        invalid = narrative_ids - known
        if invalid:
            raise ValueError(f"风险归纳引用了不存在的证据: {sorted(invalid)}")

    @staticmethod
    def _recommend(state: CaseState) -> Recommendation:
        assert state.facts and state.risk and state.amount
        if state.facts.missing_fields or state.facts.conflicts:
            return Recommendation.SUPPLEMENT
        if (
            state.facts.requested_amount.value is not None
            and state.facts.guarantee_amount.value is not None
            and state.facts.requested_amount.value != state.facts.guarantee_amount.value
        ):
            return Recommendation.SUPPLEMENT
        if state.risk.risk_level == "critical" or state.amount.suggested_amount <= 0:
            return Recommendation.REJECT
        return Recommendation.PASS
