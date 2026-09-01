from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Protocol

import pymupdf
from docx import Document
from PIL import Image
import pytesseract

from .models import EvidenceChunk


def _id(document_id: str, location: str, text: str) -> str:
    digest = hashlib.sha256(f"{document_id}:{location}:{text}".encode()).hexdigest()[:12]
    return f"EV-{digest}"


class OCRProvider(Protocol):
    def recognize(self, image: Image.Image, language: str = "chi_sim+eng") -> str: ...


class TesseractOCR:
    def recognize(self, image: Image.Image, language: str = "chi_sim+eng") -> str:
        try:
            return pytesseract.image_to_string(image, lang=language)
        except pytesseract.TesseractNotFoundError as error:
            raise RuntimeError("扫描 PDF 需要安装 tesseract-ocr 和中文语言包") from error


class DocumentParser:
    """Parse source files into stable, traceable evidence chunks."""

    def __init__(self, ocr: OCRProvider | None = None, min_native_chars: int = 30):
        self.ocr = ocr or TesseractOCR()
        self.min_native_chars = min_native_chars

    def parse(self, path: Path) -> tuple[str, list[EvidenceChunk]]:
        raw = path.read_bytes()
        document_id = hashlib.sha256(raw).hexdigest()[:16]
        suffix = path.suffix.lower()
        if suffix == ".docx":
            chunks = self._docx(path, document_id)
        elif suffix == ".pdf":
            chunks = self._pdf(path, document_id)
        elif suffix in {".txt", ".md"}:
            chunks = self._text(path, document_id)
        else:
            raise ValueError(f"暂不支持的文件类型: {suffix}")
        if not chunks:
            raise ValueError("文档未解析出文本；扫描 PDF 需要配置 OCR 服务")
        return document_id, chunks

    def _docx(self, path: Path, document_id: str) -> list[EvidenceChunk]:
        doc = Document(path)
        chunks: list[EvidenceChunk] = []
        section: str | None = None
        for index, paragraph in enumerate(doc.paragraphs, start=1):
            text = paragraph.text.strip()
            if not text:
                continue
            if paragraph.style and paragraph.style.name.startswith("Heading"):
                section = text
            chunks.append(EvidenceChunk(
                evidence_id=_id(document_id, f"p:{index}", text),
                document_id=document_id,
                file_name=path.name,
                section=section,
                paragraph=index,
                text=text,
            ))
        for table_index, table in enumerate(doc.tables, start=1):
            for row_index, row in enumerate(table.rows, start=1):
                text = " | ".join(cell.text.strip().replace("\n", " ") for cell in row.cells)
                if text.strip(" |"):
                    chunks.append(EvidenceChunk(
                        evidence_id=_id(document_id, f"t:{table_index}:r:{row_index}", text),
                        document_id=document_id,
                        file_name=path.name,
                        section=f"表格{table_index}",
                        paragraph=row_index,
                        table_id=f"DOCX-T{table_index}",
                        table_row=row_index,
                        text=text,
                    ))
        return chunks

    def _pdf(self, path: Path, document_id: str) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        previous_header: tuple[str, ...] | None = None
        logical_table = 0
        with pymupdf.open(path) as doc:
            for page_number, page in enumerate(doc, start=1):
                blocks = page.get_text("blocks")
                for block_number, block in enumerate(blocks, start=1):
                    text = " ".join(str(block[4]).split())
                    if text:
                        chunks.append(EvidenceChunk(
                            evidence_id=_id(document_id, f"page:{page_number}:b:{block_number}", text),
                            document_id=document_id,
                            file_name=path.name,
                            page=page_number,
                            paragraph=block_number,
                            text=text,
                        ))
                # PyMuPDF recovers ruled and many unruled tables. Matching the
                # normalized header continues the same logical table across pages.
                try:
                    tables = page.find_tables().tables
                except (AttributeError, ValueError):
                    tables = []
                for page_table_index, table in enumerate(tables, start=1):
                    rows = table.extract()
                    if not rows:
                        continue
                    header = tuple(" ".join(str(cell or "").split()) for cell in rows[0])
                    if header != previous_header:
                        logical_table += 1
                    previous_header = header
                    for row_index, row in enumerate(rows, start=1):
                        if page_number > 1 and row_index == 1 and header == previous_header:
                            # Header is retained as evidence but marked consistently;
                            # downstream aggregation can de-duplicate it by table_id.
                            pass
                        text = " | ".join(" ".join(str(cell or "").split()) for cell in row)
                        if text.strip(" |"):
                            table_id = f"PDF-T{logical_table}"
                            chunks.append(EvidenceChunk(
                                evidence_id=_id(document_id, f"page:{page_number}:{table_id}:r:{row_index}", text),
                                document_id=document_id, file_name=path.name, page=page_number,
                                section=f"表格{logical_table}", paragraph=row_index,
                                table_id=table_id, table_row=row_index, text=text,
                                extraction_method="table",
                            ))
                native_text = "".join(str(block[4]) for block in blocks).strip()
                if len(native_text) < self.min_native_chars:
                    pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
                    ocr_text = self.ocr.recognize(image)
                    for line_number, line in enumerate(ocr_text.splitlines(), start=1):
                        text = " ".join(line.split())
                        if text:
                            chunks.append(EvidenceChunk(
                                evidence_id=_id(document_id, f"page:{page_number}:ocr:{line_number}", text),
                                document_id=document_id, file_name=path.name, page=page_number,
                                paragraph=line_number, text=text, extraction_method="ocr",
                            ))
        return chunks

    def _text(self, path: Path, document_id: str) -> list[EvidenceChunk]:
        return [
            EvidenceChunk(
                evidence_id=_id(document_id, f"line:{index}", text),
                document_id=document_id,
                file_name=path.name,
                paragraph=index,
                text=text,
            )
            for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
            if (text := line.strip())
        ]
