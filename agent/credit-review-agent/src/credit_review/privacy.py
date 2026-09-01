from __future__ import annotations

import re
from collections import defaultdict

from .models import EvidenceChunk


PATTERNS = [
    ("IDCARD", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")),
    ("USCC", re.compile(r"(?<![0-9A-Z])[0-9A-HJ-NPQRTUWXY]{18}(?![0-9A-Z])")),
    ("PHONE", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("LANDLINE", re.compile(r"(?<!\d)(?:0\d{2,3}[-—]?)?\d{7,8}(?!\d)")),
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("BANKCARD", re.compile(r"(?<!\d)\d{16,19}(?!\d)")),
    ("PERSON", re.compile(
        r"(?:法定代表人|法人代表|实际控制人|联系人|负责人|股东姓名|姓名)"
        r"(?P<prefix>\s*[：:]?\s*)(?P<value>[\u3400-\u9fff·]{2,8})"
    )),
    ("ADDRESS", re.compile(
        r"(?:住所|住址|注册地址|联系地址|经营地址|家庭地址)"
        r"(?P<prefix>\s*[：:]?\s*)(?P<value>[\u3400-\u9fffA-Za-z0-9（）()\-]{6,80})"
    )),
]


class PrivacyMasker:
    """Deterministically mask high-confidence PII before an external LLM sees it."""

    def mask(self, chunks: list[EvidenceChunk]) -> tuple[list[EvidenceChunk], dict[str, str]]:
        counters: dict[str, int] = defaultdict(int)
        value_to_token: dict[tuple[str, str], str] = {}
        token_to_value: dict[str, str] = {}

        def replace(kind: str, match: re.Match[str]) -> str:
            value = match.groupdict().get("value") or match.group(0)
            key = (kind, value)
            if key not in value_to_token:
                counters[kind] += 1
                token = f"[{kind}_{counters[kind]:03d}]"
                value_to_token[key] = token
                token_to_value[token] = value
            if "value" in match.groupdict():
                start, end = match.span("value")
                relative_start = start - match.start()
                relative_end = end - match.start()
                original = match.group(0)
                return original[:relative_start] + value_to_token[key] + original[relative_end:]
            return value_to_token[key]

        masked: list[EvidenceChunk] = []
        for chunk in chunks:
            text = chunk.text
            for kind, pattern in PATTERNS:
                text = pattern.sub(lambda match, kind=kind: replace(kind, match), text)
            masked.append(chunk.model_copy(update={"masked_text": text}))
        # A name/address first discovered in a labelled field may recur later in
        # free text without the label. Mask every recurrence after discovery.
        replacements = sorted(token_to_value.items(), key=lambda item: len(item[1]), reverse=True)
        masked = [
            chunk.model_copy(update={
                "masked_text": self._replace_known(chunk.masked_text or "", replacements),
            })
            for chunk in masked
        ]
        return masked, token_to_value

    @staticmethod
    def _replace_known(text: str, replacements: list[tuple[str, str]]) -> str:
        for token, value in replacements:
            text = text.replace(value, token)
        return text
