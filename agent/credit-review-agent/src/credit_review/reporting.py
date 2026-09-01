import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .models import CaseState
from .rules import _latest_and_comparable_previous, _target_financials


class ReportRenderer:
    def __init__(self, template_dir: Path):
        self.environment = Environment(
            loader=FileSystemLoader(template_dir), undefined=StrictUndefined,
            autoescape=False, trim_blocks=False, lstrip_blocks=False,
        )

    def render(self, state: CaseState, output: Path) -> Path:
        template = self.environment.get_template("credit_report.md.j2")
        context = self._context(state)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(template.render(case=state, **context), encoding="utf-8")
        return output

    @staticmethod
    def _context(state: CaseState) -> dict:
        known = {chunk.evidence_id: chunk for chunk in state.chunks}
        evidence_ids: list[str] = []

        def include(values: list[str]) -> None:
            evidence_ids.extend(value for value in values if value in known)

        if state.facts:
            for name in (
                "company_name", "established_year", "registered_capital",
                "requested_amount", "industry", "litigation_count", "guarantee_amount",
            ):
                include(getattr(state.facts, name).evidence_ids)
            for snapshot in state.facts.financials:
                include(snapshot.evidence_ids)
            for conflict in state.facts.conflicts:
                include(re.findall(r"EV-[A-Za-z0-9_-]+", conflict))
        if state.risk:
            for item in state.risk.items:
                include(item.evidence_ids)
        if state.review:
            for finding in state.review.findings:
                include(finding.evidence_ids)

        excerpts = []
        for evidence_id in dict.fromkeys(evidence_ids):
            chunk = known[evidence_id]
            text = " ".join((chunk.masked_text or chunk.text).split())
            excerpts.append({
                "evidence_id": evidence_id,
                "location": ReportRenderer._location(chunk),
                "text": text[:500] + ("…" if len(text) > 500 else ""),
            })

        risk_rows = []
        if state.risk:
            for item in state.risk.items:
                actual = ReportRenderer._metric_value(item.metric, item.metric_value)
                threshold = ReportRenderer._metric_value(item.metric, item.threshold)
                operator_name = {"gt": ">", "gte": "≥", "lt": "<", "lte": "≤"}.get(
                    item.operator or "", item.operator or ""
                )
                risk_rows.append({
                    "item": item,
                    "actual": actual,
                    "threshold": threshold,
                    "condition": (
                        f"{actual} {operator_name} {threshold}"
                        if item.metric_value is not None and item.threshold is not None else "无法复核"
                    ),
                })

        financial_rows = []
        if state.facts:
            seen = set()
            for snapshot in sorted(
                _target_financials(state.facts),
                key=lambda value: (value.year, value.period_label or ""),
            ):
                key = (
                    snapshot.entity_name, snapshot.scope, snapshot.period_label,
                    snapshot.revenue, snapshot.net_profit, snapshot.total_assets,
                    snapshot.total_liabilities,
                )
                if key in seen:
                    continue
                seen.add(key)
                financial_rows.append(snapshot)
        latest, previous = _latest_and_comparable_previous(state.facts) if state.facts else (None, None)

        return {
            "risk_rows": risk_rows,
            "supplements": ReportRenderer._supplements(state),
            "evidence_excerpts": excerpts,
            "financial_rows": financial_rows,
            "latest_financial": latest,
            "previous_financial": previous,
        }

    @staticmethod
    def _location(chunk) -> str:
        values = [chunk.file_name]
        if chunk.page is not None:
            values.append(f"第{chunk.page}页")
        if chunk.section:
            values.append(chunk.section)
        if chunk.table_id:
            values.append(f"表{chunk.table_id}")
            if chunk.table_row is not None:
                values.append(f"第{chunk.table_row}行")
        elif chunk.paragraph is not None:
            values.append(f"第{chunk.paragraph}段")
        return " / ".join(values)

    @staticmethod
    def _metric_value(metric: str | None, value: float | None) -> str:
        if value is None:
            return "材料不足"
        if metric == "current_ratio":
            return f"{value:.2f}"
        return f"{value * 100:.2f}%"

    @staticmethod
    def _supplements(state: CaseState) -> list[dict[str, str]]:
        if not state.facts:
            return []
        items: list[dict[str, str]] = []

        def add(issue: str, material: str, purpose: str, owner: str = "客户补件") -> None:
            key = (issue, material, purpose, owner)
            if key not in {(i["issue"], i["material"], i["purpose"], i["owner"]) for i in items}:
                items.append({
                    "issue": issue, "material": material, "purpose": purpose, "owner": owner,
                })

        missing_map = {
            "requested_amount": (
                "加盖公章的授信/担保申请书（明确金额、期限、用途和还款来源）",
                "确认本次申请边界并作为额度上限依据",
            ),
            "financials": (
                "申请主体最近三年审计报告及最新一期财务报表，含资产负债表、利润表、现金流量表和附注",
                "补足偿债、盈利、现金流及额度测算依据",
            ),
            "industry": (
                "营业执照、主营业务说明及收入构成明细",
                "确认行业分类和业务实质",
            ),
        }
        for field in state.facts.missing_fields:
            material, purpose = missing_map.get(
                field,
                (f"能够证明 {field} 的盖章原件或权威查询结果", f"补足字段 {field}"),
            )
            add(f"缺少字段：{field}", material, purpose)

        financial_conflicts = [value for value in state.facts.conflicts if "financials." in value or "年" in value]
        if financial_conflicts:
            add(
                "财务数据存在多主体、合并/单体或期间口径冲突",
                "申请主体及相关子公司的对应年度审计报告（含附注）、纳税申报表、合并范围清单与合并抵销表，并由经办人出具逐项差异调节表",
                "逐项确认收入、净利润、总资产和应收账款采用的主体、期间及合并口径；冲突明细见本报告第三节",
            )

        requested = state.facts.requested_amount
        guarantee = state.facts.guarantee_amount
        if (
            requested.value is not None and guarantee.value is not None
            and requested.value != guarantee.value
        ):
            add(
                f"申请金额 {requested.value:g} 万元与征信担保交易余额 {guarantee.value:g} 万元不一致",
                "加盖公章的本次申请书、对应担保合同/批复、存量在保台账及余额证明，并提交金额差异说明",
                "确认两笔金额分别属于新增、续贷、存量在保还是不同申请主体，锁定本次额度计算基准",
            )

        if state.risk:
            for item in state.risk.items:
                if not item.evidence_ids:
                    add(
                        f"风险 {item.rule_id} 只有派生结果，缺少可定位原始证据",
                        "同一主体、同一报表口径、可比期间的盖章财务报表及相关科目明细",
                        f"复核指标 {item.metric or item.description}、计算过程及规则阈值，确认该风险是否真实触发",
                    )

        if state.review:
            for finding in state.review.findings:
                if finding.code in {"EVIDENCE_MISMATCH", "INVALID_EVIDENCE"}:
                    add(
                        f"报告存在无效证据引用：{finding.message}",
                        "删除无效引用，重新绑定现有 Evidence ID 后重新生成并复核报告",
                        "该项由系统修正，不要求客户为模型生成错误补材料",
                        owner="系统修正",
                    )
                elif finding.code == "EVIDENCE_EMPTY":
                    add(
                        f"审核发现派生风险缺少原始凭证：{finding.message}",
                        "触发该风险所用的同主体、同口径、同期原始报表及科目明细",
                        "使每个风险指标都能回溯到原文和具体位置",
                    )
                elif finding.code == "MODEL_REVIEWER_UNAVAILABLE":
                    add(
                        "独立模型 Reviewer 未返回可解析结果",
                        "重新执行独立模型审核；若仍失败，由有权限的人工审核员对照 state.json、规则计算链和 Evidence ID 完成复核并签署意见",
                        "补齐第二道独立审核，不把模型服务故障误当作客户材料缺失",
                        owner="系统/人工复核",
                    )
        return items
