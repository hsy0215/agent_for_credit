from __future__ import annotations

import re
from pathlib import Path
from collections import defaultdict
from typing import Protocol

from .models import ApplicationFacts, EvidenceChunk, EvidenceValue, FinancialSnapshot


class FactsExtractor(Protocol):
    async def extract(self, chunks: list[EvidenceChunk], source_file: Path) -> ApplicationFacts: ...


class MapReduceExtractor:
    """Extract bounded chunk batches, then deterministically merge candidates."""

    def __init__(self, delegate: FactsExtractor, max_chars: int = 20_000):
        self.delegate = delegate
        self.max_chars = max_chars

    async def extract(self, chunks: list[EvidenceChunk], source_file: Path) -> ApplicationFacts:
        batches: list[list[EvidenceChunk]] = []
        current: list[EvidenceChunk] = []
        size = 0
        for chunk in self._split_oversized_chunks(chunks):
            # Must match MCPAgentExtractor's serialized evidence format:
            # "[evidence_id] text\n". Keeping this budget below the delegate's
            # 30k hard limit also leaves room for filename and prompt framing.
            chunk_size = len(chunk.evidence_id) + len(chunk.masked_text or chunk.text) + 4
            if current and size + chunk_size > self.max_chars:
                batches.append(current)
                current, size = [], 0
            current.append(chunk)
            size += chunk_size
        if current:
            batches.append(current)
        partials = [await self.delegate.extract(batch, source_file) for batch in batches]
        for partial in partials:
            # Ambiguous financial rows are more dangerous than missing values:
            # a table without subject/scope/period provenance can silently mix
            # the applicant, guarantor and subsidiaries. Keep it out of rules;
            # the deterministic table parser below may recover it safely.
            partial.financials = [
                snapshot for snapshot in partial.financials
                if snapshot.entity_name
                and snapshot.scope
                and snapshot.scope.lower() not in {"未注明", "未知", "unknown", "n/a"}
                and snapshot.period_label
                and snapshot.evidence_ids
            ]
        semantic = merge_facts(partials)
        # LLM extraction is probabilistic and can occasionally miss an explicit
        # scalar even when its evidence is present. Recover only high-precision
        # fields with deterministic patterns; semantic candidates still win on
        # confidence and disagreements remain visible as conflicts.
        deterministic = await HeuristicExtractor().extract(chunks, source_file)
        return merge_facts([semantic, deterministic])

    def _split_oversized_chunks(self, chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
        result: list[EvidenceChunk] = []
        # Reserve formatting overhead even for unusually long evidence IDs.
        payload_limit = max(1_000, self.max_chars - 256)
        for chunk in chunks:
            text = chunk.masked_text or chunk.text
            if len(text) <= payload_limit:
                result.append(chunk)
                continue
            for index, start in enumerate(range(0, len(text), payload_limit), start=1):
                part = text[start:start + payload_limit]
                result.append(chunk.model_copy(update={
                    "text": part if chunk.masked_text is None else chunk.text,
                    "masked_text": part if chunk.masked_text is not None else None,
                    "section": f"{chunk.section or '长段落'}（分片{index}）",
                }))
        return result


def merge_facts(partials: list[ApplicationFacts]) -> ApplicationFacts:
    result = ApplicationFacts()
    scalar_names = (
        "company_name", "established_year", "registered_capital", "requested_amount",
        "industry", "litigation_count", "guarantee_amount",
    )
    for name in scalar_names:
        candidates = [getattr(partial, name) for partial in partials if getattr(partial, name).value is not None]
        if not candidates:
            continue
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        winner = candidates[0]
        distinct = {str(item.value) for item in candidates}
        setattr(result, name, winner)
        if len(distinct) > 1:
            result.conflicts.append(f"{name} 出现多个候选值: {sorted(distinct)}")
    by_period: dict[tuple[str, str, str], list] = defaultdict(list)
    for partial in partials:
        for snapshot in partial.financials:
            entity = (snapshot.entity_name or "").strip()
            entity_key = re.sub(r"深圳市?|有限公司|股份|科技|传媒|\s", "", entity)
            scope_key = (snapshot.scope or "").replace("报表", "").strip()
            key = (
                entity_key,
                scope_key,
                (snapshot.period_label or str(snapshot.year)).strip(),
            )
            by_period[key].append(snapshot)
    for key, candidates in sorted(by_period.items(), key=lambda item: (item[1][0].year, item[0])):
        merged = candidates[0].model_copy(deep=True)
        for candidate in candidates[1:]:
            for field in merged.model_fields:
                if field in {"year", "entity_name", "scope", "period_label", "period_type"}:
                    continue
                old, new = getattr(merged, field), getattr(candidate, field)
                if field in {"evidence_ids", "source_quotes"}:
                    setattr(merged, field, list(dict.fromkeys([*old, *new])))
                    continue
                if old is None and new is not None:
                    setattr(merged, field, new)
                elif old is not None and new is not None and old != new:
                    evidence = list(dict.fromkeys([
                        *merged.evidence_ids, *candidate.evidence_ids,
                    ]))
                    label = "/".join(part for part in key if part) or str(merged.year)
                    result.conflicts.append(
                        f"financials.{label}.{field}: {old} != {new}"
                        + (f"（证据：{', '.join(evidence)}）" if evidence else "")
                    )
        result.financials.append(merged)
    result.conflicts.extend(conflict for partial in partials for conflict in partial.conflicts)
    required = {
        "requested_amount": result.requested_amount.value,
        "financials": result.financials,
        "industry": result.industry.value,
    }
    result.missing_fields = [name for name, value in required.items() if not value]
    return result


class HeuristicExtractor:
    """Offline baseline. Replace with MCPAgentExtractor for semantic extraction."""

    MONEY = re.compile(r"(?:申请|授信|建议)[^。；\n]{0,20}?(\d+(?:\.\d+)?)\s*万")
    YEAR = re.compile(r"(?:成立于?|成立时间[^\d]{0,8})(20\d{2}|19\d{2})")

    async def extract(self, chunks: list[EvidenceChunk], source_file: Path) -> ApplicationFacts:
        facts = ApplicationFacts()
        company = re.sub(r"(?:评审报告.*|报告.*)$", "", source_file.stem).strip("-—_ ")
        facts.company_name = EvidenceValue(value=company, confidence=0.7)
        for chunk in chunks:
            text = chunk.masked_text or chunk.text
            cells = [cell.strip() for cell in text.split("|")]
            if cells and cells[0] == "借款企业名称" and len(cells) > 1 and cells[1]:
                facts.company_name = EvidenceValue(
                    value=cells[1], evidence_ids=[chunk.evidence_id],
                    confidence=0.98, source_quote=text,
                )
            if facts.requested_amount.value is None and (match := self.MONEY.search(text)):
                facts.requested_amount = EvidenceValue(
                    value=float(match.group(1)), evidence_ids=[chunk.evidence_id],
                    confidence=0.65, source_quote=match.group(0),
                )
            if facts.established_year.value is None and (match := self.YEAR.search(text)):
                facts.established_year = EvidenceValue(
                    value=int(match.group(1)), evidence_ids=[chunk.evidence_id],
                    confidence=0.75, source_quote=match.group(0),
                )
            if (
                facts.established_year.value is None
                and cells and cells[0] == "成立日期" and len(cells) > 1
                and (match := re.search(r"(20\d{2}|19\d{2})", cells[1]))
            ):
                facts.established_year = EvidenceValue(
                    value=int(match.group(1)), evidence_ids=[chunk.evidence_id],
                    confidence=0.98, source_quote=text,
                )
            if facts.registered_capital.value is None and cells and cells[0] == "注册资本":
                match = re.search(r"(\d+(?:\.\d+)?)\s*万", cells[1] if len(cells) > 1 else "")
                if match:
                    facts.registered_capital = EvidenceValue(
                        value=float(match.group(1)), evidence_ids=[chunk.evidence_id],
                        confidence=0.95, source_quote=text,
                    )
            if facts.industry.value is None and "广告营销行业" in text:
                facts.industry = EvidenceValue(
                    value="广告营销", evidence_ids=[chunk.evidence_id],
                    confidence=0.9, source_quote=text,
                )

        facts.financials = self._extract_financial_tables(chunks, facts.company_name.value)
        self._extract_guarantee_balance(chunks, facts)
        required = {
            "requested_amount": facts.requested_amount.value,
            "financials": facts.financials,
            "industry": facts.industry.value,
        }
        facts.missing_fields = [name for name, value in required.items() if not value]
        return facts

    @staticmethod
    def _extract_financial_tables(
        chunks: list[EvidenceChunk], company_name: str | None,
    ) -> list[FinancialSnapshot]:
        descriptors: list[tuple[str, str]] = []
        for chunk in chunks:
            text = (chunk.masked_text or chunk.text).strip()
            if text.startswith("合并财务简表"):
                descriptors.append((company_name or "申请企业", "合并"))
            elif match := re.match(r"(.+?)财务简表（", text):
                descriptors.append((match.group(1).strip(), "单体"))
            elif match := re.match(r"(.+?)财务报表（", text):
                descriptors.append((match.group(1).strip(), "母公司单体"))
            elif match := re.match(r"最新单体报表：(.+)", text):
                entity = re.sub(r"（.*", "", match.group(1)).strip()
                descriptors.append((entity, "单体"))

        table_order: list[str] = []
        tables: dict[str, list[EvidenceChunk]] = defaultdict(list)
        for chunk in chunks:
            if chunk.table_id:
                if chunk.table_id not in tables:
                    table_order.append(chunk.table_id)
                tables[chunk.table_id].append(chunk)

        financial_tables = []
        for table_id in table_order:
            rows = sorted(tables[table_id], key=lambda value: value.table_row or 0)
            if not rows:
                continue
            header = [cell.strip() for cell in (rows[0].masked_text or rows[0].text).split("|")]
            if not header or header[0].replace(" ", "") not in {"项目", "科目"}:
                continue
            if len(header) < 2 or not all(re.search(r"(?:19|20)\d{2}年", cell) for cell in header[1:]):
                continue
            financial_tables.append((header, rows))

        snapshots: list[FinancialSnapshot] = []
        metric_names = {
            "销售额": "revenue",
            "营业收入": "revenue",
            "收入": "revenue",
            "净利润": "net_profit",
            "总资产": "total_assets",
            "总负债": "total_liabilities",
            "流动资产": "current_assets",
            "流动负债": "current_liabilities",
            "应收账款": "accounts_receivable",
            "净资产": "_net_assets",
        }
        if descriptors:
            financial_tables = financial_tables[:len(descriptors)]
        for table_index, (header, rows) in enumerate(financial_tables):
            entity, scope = (
                descriptors[table_index]
                if table_index < len(descriptors)
                else (company_name or "申请企业", "未注明")
            )
            by_column: dict[int, dict] = {}
            for column, period_label in enumerate(header[1:], start=1):
                year_match = re.search(r"((?:19|20)\d{2})年", period_label)
                if not year_match:
                    continue
                interim = bool(re.search(r"月|季度|1-", period_label))
                by_column[column] = {
                    "year": int(year_match.group(1)),
                    "entity_name": entity,
                    "scope": scope,
                    "period_label": period_label,
                    "period_type": "interim" if interim else "annual",
                    "evidence_ids": [rows[0].evidence_id],
                    "source_quotes": [rows[0].masked_text or rows[0].text],
                }

            net_assets: dict[int, float] = {}
            for row in rows[1:]:
                cells = [cell.strip() for cell in (row.masked_text or row.text).split("|")]
                if not cells:
                    continue
                metric = metric_names.get(cells[0].replace(" ", ""))
                if not metric:
                    continue
                for column, target in by_column.items():
                    if column >= len(cells):
                        continue
                    try:
                        value = float(cells[column].replace(",", ""))
                    except ValueError:
                        continue
                    if metric == "_net_assets":
                        net_assets[column] = value
                    else:
                        target[metric] = value
                    target["evidence_ids"].append(row.evidence_id)
                    target["source_quotes"].append(row.masked_text or row.text)

            for column, values in by_column.items():
                if (
                    values.get("total_liabilities") is None
                    and values.get("total_assets") is not None
                    and column in net_assets
                ):
                    values["total_liabilities"] = values["total_assets"] - net_assets[column]
                values["evidence_ids"] = list(dict.fromkeys(values["evidence_ids"]))
                values["source_quotes"] = list(dict.fromkeys(values["source_quotes"]))
                if any(values.get(field) is not None for field in (
                    "revenue", "net_profit", "total_assets", "total_liabilities",
                    "current_assets", "current_liabilities", "accounts_receivable",
                )):
                    snapshots.append(FinancialSnapshot(**values))
        return snapshots

    @staticmethod
    def _extract_guarantee_balance(chunks: list[EvidenceChunk], facts: ApplicationFacts) -> None:
        by_table: dict[str, list[EvidenceChunk]] = defaultdict(list)
        for chunk in chunks:
            if chunk.table_id:
                by_table[chunk.table_id].append(chunk)
        for rows in by_table.values():
            rows = sorted(rows, key=lambda value: value.table_row or 0)
            if not rows:
                continue
            header = [cell.strip() for cell in (rows[0].masked_text or rows[0].text).split("|")]
            if "担保交易余额" not in header:
                continue
            amount_index = header.index("担保交易余额")
            for row in rows[1:]:
                cells = [cell.strip() for cell in (row.masked_text or row.text).split("|")]
                if "申请主体" not in cells or amount_index >= len(cells):
                    continue
                try:
                    value = float(cells[amount_index].replace(",", ""))
                except ValueError:
                    continue
                facts.guarantee_amount = EvidenceValue(
                    value=value, evidence_ids=[row.evidence_id],
                    confidence=0.98, source_quote=row.masked_text or row.text,
                )
                return
