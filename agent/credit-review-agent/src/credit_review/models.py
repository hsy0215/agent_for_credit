from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Stage(StrEnum):
    INGESTED = "ingested"
    MASKED = "masked"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    RISK_ASSESSED = "risk_assessed"
    AMOUNT_CALCULATED = "amount_calculated"
    REPORT_GENERATED = "report_generated"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPPLEMENT_REQUIRED = "supplement_required"


class Recommendation(StrEnum):
    PASS = "建议通过"
    REJECT = "建议拒绝"
    SUPPLEMENT = "建议补件"


class EvidenceChunk(BaseModel):
    evidence_id: str
    document_id: str
    file_name: str
    page: int | None = None
    section: str | None = None
    paragraph: int | None = None
    table_id: str | None = None
    table_row: int | None = None
    extraction_method: str = "native"
    text: str
    masked_text: str | None = None


class EvidenceValue(BaseModel, Generic[T]):
    value: T | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_quote: str | None = None


class FinancialSnapshot(BaseModel):
    year: int
    entity_name: str | None = None
    scope: str | None = None
    period_label: str | None = None
    period_type: str | None = None
    revenue: float | None = None
    net_profit: float | None = None
    total_assets: float | None = None
    total_liabilities: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    accounts_receivable: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    source_quotes: list[str] = Field(default_factory=list)


class ApplicationFacts(BaseModel):
    company_name: EvidenceValue[str] = Field(default_factory=EvidenceValue[str])
    established_year: EvidenceValue[int] = Field(default_factory=EvidenceValue[int])
    registered_capital: EvidenceValue[float] = Field(default_factory=EvidenceValue[float])
    requested_amount: EvidenceValue[float] = Field(default_factory=EvidenceValue[float])
    industry: EvidenceValue[str] = Field(default_factory=EvidenceValue[str])
    financials: list[FinancialSnapshot] = Field(default_factory=list)
    litigation_count: EvidenceValue[int] = Field(default_factory=EvidenceValue[int])
    guarantee_amount: EvidenceValue[float] = Field(default_factory=EvidenceValue[float])
    missing_fields: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)


class FinancialMetrics(BaseModel):
    debt_ratio: float | None = None
    current_ratio: float | None = None
    net_margin: float | None = None
    revenue_growth: float | None = None
    receivables_ratio: float | None = None


class RiskItem(BaseModel):
    risk_code: str
    category: str
    description: str
    severity: str
    score: int = Field(ge=0, le=100)
    evidence_ids: list[str] = Field(default_factory=list)
    rule_id: str
    mitigations: list[str] = Field(default_factory=list)
    metric: str | None = None
    metric_value: float | None = None
    operator: str | None = None
    threshold: float | None = None
    calculation: str | None = None
    source_quotes: list[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    metrics: FinancialMetrics
    items: list[RiskItem] = Field(default_factory=list)
    total_score: int = Field(ge=0, le=100)
    risk_level: str


class AdvisoryNarrative(BaseModel):
    business_summary: str
    risk_rationale: str
    amount_rationale: str
    recommendation_rationale: str
    cited_evidence_ids: list[str] = Field(default_factory=list)


class ReviewFinding(BaseModel):
    code: str
    severity: str
    message: str
    evidence_ids: list[str] = Field(default_factory=list)


class ReviewResult(BaseModel):
    passed: bool
    findings: list[ReviewFinding] = Field(default_factory=list)


class CalculationStep(BaseModel):
    name: str
    formula: str
    value: float


class AmountDecision(BaseModel):
    currency_unit: str = "万元"
    base_amount: float
    suggested_amount: float
    risk_factor: float
    data_quality_factor: float
    rule_version: str
    steps: list[CalculationStep]


class ApprovalEvent(BaseModel):
    action: str
    operator: str
    comment: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuditEvent(BaseModel):
    stage: Stage
    action: str
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CaseState(BaseModel):
    case_id: str
    source_file: str
    stage: Stage
    chunks: list[EvidenceChunk] = Field(default_factory=list)
    mask_mapping_file: str | None = None
    facts: ApplicationFacts | None = None
    risk: RiskAssessment | None = None
    amount: AmountDecision | None = None
    recommendation: Recommendation | None = None
    narrative: AdvisoryNarrative | None = None
    review: ReviewResult | None = None
    report_file: str | None = None
    approvals: list[ApprovalEvent] = Field(default_factory=list)
    audit: list[AuditEvent] = Field(default_factory=list)
