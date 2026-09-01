import asyncio
from pathlib import Path

from credit_review.extraction import HeuristicExtractor, MapReduceExtractor
from credit_review.models import ApplicationFacts, EvidenceChunk


class RecordingExtractor:
    def __init__(self):
        self.serialized_sizes = []

    async def extract(self, chunks, source_file):
        self.serialized_sizes.append(sum(
            len(chunk.evidence_id) + len(chunk.masked_text or chunk.text) + 4
            for chunk in chunks
        ))
        return ApplicationFacts()


def test_map_reduce_counts_evidence_formatting_overhead():
    delegate = RecordingExtractor()
    extractor = MapReduceExtractor(delegate, max_chars=5000)
    chunks = [
        EvidenceChunk(
            evidence_id=f"EV-{index:012d}", document_id="D", file_name="x",
            text="字" * 10, masked_text="字" * 10,
        )
        for index in range(500)
    ]
    asyncio.run(extractor.extract(chunks, Path("x.docx")))
    assert len(delegate.serialized_sizes) > 1
    assert max(delegate.serialized_sizes) <= 5000


def test_map_reduce_splits_one_oversized_chunk():
    delegate = RecordingExtractor()
    extractor = MapReduceExtractor(delegate, max_chars=5000)
    chunks = [EvidenceChunk(
        evidence_id="EV-LONG", document_id="D", file_name="x",
        text="字" * 12_000, masked_text="字" * 12_000,
    )]
    asyncio.run(extractor.extract(chunks, Path("x.docx")))
    assert len(delegate.serialized_sizes) == 3
    assert max(delegate.serialized_sizes) <= 5000


def test_map_reduce_recovers_explicit_amount_missed_by_delegate():
    delegate = RecordingExtractor()
    extractor = MapReduceExtractor(delegate, max_chars=5000)
    chunks = [EvidenceChunk(
        evidence_id="EV-AMOUNT", document_id="D", file_name="x",
        text="申请担保金额：400万元，申请期限12个月",
        masked_text="申请担保金额：400万元，申请期限12个月",
    )]

    facts = asyncio.run(extractor.extract(chunks, Path("x.docx")))

    assert facts.requested_amount.value == 400
    assert facts.requested_amount.evidence_ids == ["EV-AMOUNT"]
    assert "requested_amount" not in facts.missing_fields


def test_heuristic_recovers_financial_table_with_period_and_evidence():
    chunks = [
        EvidenceChunk(
            evidence_id="EV-DESC", document_id="D", file_name="x.docx",
            text="之光传媒财务简表（借款企业，主要销售主体）",
        ),
        EvidenceChunk(
            evidence_id="EV-HEADER", document_id="D", file_name="x.docx",
            table_id="DOCX-T1", table_row=1,
            text="项  目 | 2022年9月 | 2021年9月",
        ),
        EvidenceChunk(
            evidence_id="EV-ASSETS", document_id="D", file_name="x.docx",
            table_id="DOCX-T1", table_row=2,
            text="总资产 | 20801 | 12371",
        ),
        EvidenceChunk(
            evidence_id="EV-NET-ASSETS", document_id="D", file_name="x.docx",
            table_id="DOCX-T1", table_row=3,
            text="净资产 | 9681 | 9126",
        ),
        EvidenceChunk(
            evidence_id="EV-REVENUE", document_id="D", file_name="x.docx",
            table_id="DOCX-T1", table_row=4,
            text="销售额 | 7779 | 7909",
        ),
        EvidenceChunk(
            evidence_id="EV-PROFIT", document_id="D", file_name="x.docx",
            table_id="DOCX-T1", table_row=5,
            text="净利润 | 20 | 648",
        ),
    ]

    facts = asyncio.run(HeuristicExtractor().extract(chunks, Path("深圳之光传媒科技有限公司评审报告.docx")))

    latest = next(item for item in facts.financials if item.period_label == "2022年9月")
    assert latest.entity_name == "之光传媒"
    assert latest.period_type == "interim"
    assert latest.revenue == 7779
    assert latest.total_liabilities == 11120
    assert "EV-REVENUE" in latest.evidence_ids
    assert "financials" not in facts.missing_fields
