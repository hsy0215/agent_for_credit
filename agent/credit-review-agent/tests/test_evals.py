import json
from pathlib import Path

from openpyxl import Workbook

from credit_review.evals import HistoricalDatasetBuilder, company_split, evaluate_jsonl


def test_dataset_alignment_and_company_split(tmp_path):
    documents = tmp_path / "docs"
    documents.mkdir()
    (documents / "深圳有芯电子有限公司评审报告.docx").write_bytes(b"placeholder")
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["序号", "公司名称", "授信金额（万元）"])
    sheet.append([1, "深圳有芯电子有芯公司", 400])
    amount_file = tmp_path / "amount.xlsx"
    workbook.save(amount_file)
    records = HistoricalDatasetBuilder().build(documents, amount_file)
    assert len(records) == 1
    assert records[0].approved_amount == 400
    assert records[0].split == company_split(records[0].company_name)


def test_field_and_amount_eval(tmp_path):
    gold = tmp_path / "gold.jsonl"
    prediction = tmp_path / "prediction.jsonl"
    gold.write_text(json.dumps({
        "record_id": "1", "approved_amount": 300,
        "gold": {"company_name": "测试有限公司", "industry": None},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    prediction.write_text(json.dumps({
        "record_id": "1", "suggested_amount": 280,
        "facts": {"company_name": {"value": "测试有限公司"}, "industry": {"value": "制造业"}},
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    result = evaluate_jsonl(gold, prediction)
    assert result.records == 1
    assert result.missing_predictions == 0
    assert result.field_exact_accuracy == 1
    assert result.unsupported_value_rate == 0.5
    assert result.amount_mae == 20
    assert result.amount_band_accuracy == 1
