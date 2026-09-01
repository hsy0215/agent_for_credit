from pathlib import Path

from credit_review.models import ApplicationFacts, EvidenceValue, FinancialSnapshot
from credit_review.rules import AmountEngine, RiskEngine

ROOT = Path(__file__).resolve().parents[1]


def test_risk_and_amount_are_deterministic():
    facts = ApplicationFacts(financials=[
        FinancialSnapshot(year=2023, revenue=1000, net_profit=100, total_assets=800, total_liabilities=400),
        FinancialSnapshot(
            year=2024, revenue=600, net_profit=-20, total_assets=900, total_liabilities=700,
            current_assets=200, current_liabilities=300, accounts_receivable=400,
        ),
    ])
    risk = RiskEngine(ROOT / "config/risk_rules.yaml").assess(facts)
    amount = AmountEngine(ROOT / "config/amount_rules.yaml").calculate(facts, risk)
    assert risk.risk_level == "critical"
    assert risk.total_score == 90
    assert amount.suggested_amount == 0


def test_missing_financials_never_produce_credit_amount():
    facts = ApplicationFacts(missing_fields=["financials"])
    risk = RiskEngine(ROOT / "config/risk_rules.yaml").assess(facts)
    amount = AmountEngine(ROOT / "config/amount_rules.yaml").calculate(facts, risk)
    assert amount.base_amount == 0
    assert amount.suggested_amount == 0


def test_suggested_amount_never_exceeds_requested_amount():
    facts = ApplicationFacts(
        requested_amount=EvidenceValue(value=400),
        financials=[FinancialSnapshot(
            year=2024,
            revenue=10_000,
            net_profit=1_000,
            total_assets=8_000,
            total_liabilities=2_000,
        )],
    )
    risk = RiskEngine(ROOT / "config/risk_rules.yaml").assess(facts)
    amount = AmountEngine(ROOT / "config/amount_rules.yaml").calculate(facts, risk)

    assert amount.base_amount == 400
    assert amount.suggested_amount == 400
    assert any(step.name == "申请额度上限" for step in amount.steps)


def test_growth_uses_same_interim_period_instead_of_full_year():
    facts = ApplicationFacts(
        company_name=EvidenceValue(value="深圳之光传媒科技有限公司"),
        financials=[
            FinancialSnapshot(
                year=2021, entity_name="之光传媒", period_label="2021年",
                period_type="annual", revenue=11_456, evidence_ids=["EV-ANNUAL"],
            ),
            FinancialSnapshot(
                year=2021, entity_name="之光传媒", period_label="2021年1-9月",
                period_type="interim", revenue=7_909, evidence_ids=["EV-2021-09"],
                source_quotes=["销售额 | 2021年1-9月 | 7909"],
            ),
            FinancialSnapshot(
                year=2022, entity_name="之光传媒", period_label="2022年1-9月",
                period_type="interim", revenue=7_779, evidence_ids=["EV-2022-09"],
                source_quotes=["销售额 | 2022年1-9月 | 7779"],
            ),
        ],
    )

    risk = RiskEngine(ROOT / "config/risk_rules.yaml").assess(facts)

    assert risk.metrics.revenue_growth == (7_779 - 7_909) / 7_909
    assert not any(item.rule_id == "FIN-GROWTH-001" for item in risk.items)


def test_triggered_risk_contains_rule_math_and_source_evidence():
    facts = ApplicationFacts(
        company_name=EvidenceValue(value="深圳之光传媒科技有限公司"),
        financials=[
            FinancialSnapshot(
                year=2021, entity_name="之光传媒", period_label="2021年9月",
                period_type="interim", revenue=8_000, evidence_ids=["EV-OLD"],
                source_quotes=["2021年1-9月收入8000万元"],
            ),
            FinancialSnapshot(
                year=2022, entity_name="之光传媒", period_label="2022年9月",
                period_type="interim", revenue=5_000, evidence_ids=["EV-NEW"],
                source_quotes=["2022年1-9月收入5000万元"],
            ),
        ],
    )

    risk = RiskEngine(ROOT / "config/risk_rules.yaml").assess(facts)
    item = next(item for item in risk.items if item.rule_id == "FIN-GROWTH-001")

    assert item.metric_value == -0.375
    assert item.threshold == -0.30
    assert item.calculation == "(5000 - 8000) / 8000"
    assert item.evidence_ids == ["EV-NEW", "EV-OLD"]
    assert item.source_quotes == ["2022年1-9月收入5000万元", "2021年1-9月收入8000万元"]
