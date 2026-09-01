from __future__ import annotations

import operator
import re
from pathlib import Path

import yaml

from .models import (
    AmountDecision, ApplicationFacts, CalculationStep, FinancialMetrics,
    RiskAssessment, RiskItem,
)


def _divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def _normalise_company(value: str | None) -> str:
    if not value:
        return ""
    result = re.sub(r"[（）()\s]", "", value)
    for token in ("深圳市", "深圳", "有限公司", "股份", "科技", "传媒"):
        result = result.replace(token, "")
    return result


def _period(snapshot) -> tuple[int, int | None, bool]:
    label = snapshot.period_label or str(snapshot.year)
    months = [int(value) for value in re.findall(r"(?<!\d)(1[0-2]|[1-9])月", label)]
    month = months[-1] if months else None
    interim = snapshot.period_type == "interim" or month is not None or "季度" in label or "1-" in label
    return snapshot.year, month, not interim


def _target_financials(facts: ApplicationFacts):
    company = _normalise_company(facts.company_name.value)
    matching = [
        item for item in facts.financials
        if company and _normalise_company(item.entity_name)
        and (
            company in _normalise_company(item.entity_name)
            or _normalise_company(item.entity_name) in company
        )
    ]
    return matching or facts.financials


def _latest_and_comparable_previous(facts: ApplicationFacts):
    ordered = sorted(
        _target_financials(facts),
        key=lambda value: (value.year, _period(value)[1] or (12 if _period(value)[2] else 0)),
    )
    if not ordered:
        return None, None
    latest = ordered[-1]
    latest_year, latest_month, latest_annual = _period(latest)
    comparable = []
    for candidate in ordered[:-1]:
        year, month, annual = _period(candidate)
        if year != latest_year - 1:
            continue
        if latest_annual and annual:
            comparable.append(candidate)
        elif not latest_annual and not annual and month == latest_month:
            comparable.append(candidate)
    return latest, comparable[-1] if comparable else None


def _financial_evidence(*snapshots) -> tuple[list[str], list[str]]:
    evidence_ids: list[str] = []
    quotes: list[str] = []
    for snapshot in snapshots:
        if snapshot is None:
            continue
        evidence_ids.extend(snapshot.evidence_ids)
        quotes.extend(snapshot.source_quotes)
    return list(dict.fromkeys(evidence_ids)), list(dict.fromkeys(quotes))


class RiskEngine:
    OPERATORS = {"gt": operator.gt, "gte": operator.ge, "lt": operator.lt, "lte": operator.le}

    def __init__(self, config_path: Path):
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def assess(self, facts: ApplicationFacts) -> RiskAssessment:
        latest, previous = _latest_and_comparable_previous(facts)
        metrics = FinancialMetrics(
            debt_ratio=_divide(latest.total_liabilities, latest.total_assets) if latest else None,
            current_ratio=_divide(latest.current_assets, latest.current_liabilities) if latest else None,
            net_margin=_divide(latest.net_profit, latest.revenue) if latest else None,
            revenue_growth=_divide(latest.revenue - previous.revenue, previous.revenue)
            if latest and previous and latest.revenue is not None and previous.revenue is not None else None,
            receivables_ratio=_divide(latest.accounts_receivable, latest.revenue) if latest else None,
        )
        items: list[RiskItem] = []
        for rule in self.config["rules"]:
            value = getattr(metrics, rule["metric"])
            if value is not None and self.OPERATORS[rule["operator"]](value, rule["threshold"]):
                source_snapshots = (latest, previous) if rule["metric"] == "revenue_growth" else (latest,)
                evidence_ids, source_quotes = _financial_evidence(*source_snapshots)
                calculation = None
                if rule["metric"] == "revenue_growth" and latest and previous:
                    calculation = (
                        f"({latest.revenue:g} - {previous.revenue:g}) / {previous.revenue:g}"
                        if latest.revenue is not None and previous.revenue is not None else None
                    )
                items.append(RiskItem(
                    risk_code=rule["id"], category=rule["category"],
                    description=rule["description"], severity=rule["severity"],
                    score=rule["score"], rule_id=rule["id"], metric=rule["metric"],
                    metric_value=value, operator=rule["operator"], threshold=rule["threshold"],
                    calculation=calculation, evidence_ids=evidence_ids, source_quotes=source_quotes,
                ))
        score = min(100, sum(item.score for item in items))
        level = "critical" if score >= 70 else "high" if score >= 45 else "medium" if score >= 20 else "low"
        return RiskAssessment(metrics=metrics, items=items, total_score=score, risk_level=level)


class AmountEngine:
    def __init__(self, config_path: Path):
        self.config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    def calculate(self, facts: ApplicationFacts, risk: RiskAssessment) -> AmountDecision:
        latest, _ = _latest_and_comparable_previous(facts)
        caps = [float(self.config["maximum_amount"])]
        steps = [CalculationStep(name="产品上限", formula="maximum_amount", value=caps[0])]
        if facts.requested_amount.value is not None:
            requested_cap = float(facts.requested_amount.value)
            caps.append(requested_cap)
            steps.append(CalculationStep(
                name="申请额度上限",
                formula="requested_amount",
                value=requested_cap,
            ))
        if latest and latest.revenue is not None:
            revenue_cap = latest.revenue * self.config["revenue_factor"]
            caps.append(revenue_cap)
            steps.append(CalculationStep(name="收入上限", formula="revenue × revenue_factor", value=revenue_cap))
        if latest and latest.total_assets is not None and latest.total_liabilities is not None:
            net_asset_cap = (latest.total_assets - latest.total_liabilities) * self.config["net_asset_factor"]
            caps.append(max(0, net_asset_cap))
            steps.append(CalculationStep(name="净资产上限", formula="net_assets × net_asset_factor", value=max(0, net_asset_cap)))
        # An application amount or a product ceiling alone is not a credit basis.
        # Without revenue or net assets, keep the computed amount at zero and let
        # the workflow request supplementary material.
        has_financial_cap = any(step.name in {"收入上限", "净资产上限"} for step in steps)
        base = max(0, min(caps)) if has_financial_cap else 0.0
        risk_factor = float(self.config["risk_factors"][risk.risk_level])
        completeness = 1 - min(0.5, len(facts.missing_fields) * 0.1)
        suggested = min(float(self.config["maximum_amount"]), base * risk_factor * completeness)
        suggested = round(suggested / 10) * 10
        steps.append(CalculationStep(name="建议额度", formula="base × risk_factor × data_quality_factor", value=suggested))
        return AmountDecision(
            base_amount=base, suggested_amount=suggested, risk_factor=risk_factor,
            data_quality_factor=completeness, rule_version=self.config["version"], steps=steps,
        )
