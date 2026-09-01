from credit_review.models import EvidenceChunk
from credit_review.privacy import PrivacyMasker


def test_masks_pii_consistently():
    chunks = [
        EvidenceChunk(evidence_id="E1", document_id="D", file_name="x", text="电话13812345678"),
        EvidenceChunk(evidence_id="E2", document_id="D", file_name="x", text="再次联系13812345678"),
    ]
    masked, mapping = PrivacyMasker().mask(chunks)
    assert masked[0].masked_text == "电话[PHONE_001]"
    assert masked[1].masked_text == "再次联系[PHONE_001]"
    assert mapping["[PHONE_001]"] == "13812345678"


def test_masks_chinese_identity_and_business_identifiers():
    chunk = EvidenceChunk(
        evidence_id="E1", document_id="D", file_name="x",
        text="法定代表人：张三，注册地址：深圳市南山区科技园1号；统一代码91440300MA5ABCDE12",
    )
    masked, mapping = PrivacyMasker().mask([chunk])
    text = masked[0].masked_text
    assert "张三" not in text
    assert "深圳市南山区科技园1号" not in text
    assert "91440300MA5ABCDE12" not in text
    assert set(mapping.values()) == {"张三", "深圳市南山区科技园1号", "91440300MA5ABCDE12"}


def test_masks_unlabelled_recurrence_after_identity_discovery():
    chunks = [
        EvidenceChunk(evidence_id="E1", document_id="D", file_name="x", text="法定代表人：张三，"),
        EvidenceChunk(evidence_id="E2", document_id="D", file_name="x", text="张三负责企业经营。"),
    ]
    masked, _ = PrivacyMasker().mask(chunks)
    assert all("张三" not in chunk.masked_text for chunk in masked)
