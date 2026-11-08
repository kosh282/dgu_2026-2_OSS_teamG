"""교수자가 올린 강의 자료(PDF·PPTX)를 페이지 단위 텍스트로 읽는다.

자료는 메모리에서만 다루며 파일로 저장하지 않는다. `document_id`는 파일 내용의
해시에서 만든 UUID라서 같은 자료를 수업 중과 사후 검토에서 다시 올려도 같은 ID가 된다.
사용자 식별자가 아니라 자료 식별자다.
"""
from dataclasses import dataclass
import hashlib
import io
from pathlib import PurePath
from uuid import UUID, uuid5

# 자료 ID 전용 이름 공간. 값을 바꾸면 기존 세션의 자료 ID와 연결이 끊긴다.
_DOCUMENT_NAMESPACE = UUID("6f3c1f5e-2f7a-4c55-9a0b-6d1f0d4b7e21")
SUPPORTED_SUFFIXES = (".pdf", ".pptx")


class MaterialError(ValueError):
    """읽을 수 없거나 지원하지 않는 자료."""


@dataclass(frozen=True)
class MaterialPage:
    document_id: str
    document_name: str
    page: int  # 1부터 시작. PPTX는 슬라이드 번호
    text: str


@dataclass(frozen=True)
class Material:
    document_id: str
    name: str
    pages: tuple

    @property
    def empty_pages(self):
        """텍스트를 찾지 못한 페이지 수(스캔 이미지 PDF 등)."""
        return sum(1 for page in self.pages if not page.text.strip())


def document_id_for(data):
    return str(uuid5(_DOCUMENT_NAMESPACE, hashlib.sha256(data).hexdigest()))


def _pdf_pages(data):
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as error:
        raise MaterialError("PDF를 읽으려면 pypdf가 필요합니다. ai_agent/requirements.txt를 설치하세요") from error
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise MaterialError("암호가 걸린 PDF는 읽을 수 없습니다")
        return [page.extract_text() or "" for page in reader.pages]
    except PdfReadError as error:
        raise MaterialError(f"PDF를 읽을 수 없습니다: {error}") from error


def _pptx_pages(data):
    try:
        from pptx import Presentation
    except ImportError as error:
        raise MaterialError("PPTX를 읽으려면 python-pptx가 필요합니다. ai_agent/requirements.txt를 설치하세요") from error
    try:
        presentation = Presentation(io.BytesIO(data))
    except Exception as error:  # python-pptx는 손상 파일에 여러 예외를 낸다
        raise MaterialError(f"PPTX를 읽을 수 없습니다: {error}") from error
    pages = []
    for slide in presentation.slides:
        lines = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                lines.append(shape.text_frame.text)
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    lines.append(" ".join(cell.text for cell in row.cells))
        if slide.has_notes_slide:
            lines.append(slide.notes_slide.notes_text_frame.text)
        pages.append("\n".join(line for line in lines if line.strip()))
    return pages


def load_material(name, data):
    """파일 이름과 내용(bytes)으로 `Material`을 만든다."""
    suffix = PurePath(name).suffix.lower()
    if suffix == ".ppt":
        raise MaterialError("구형 PPT는 지원하지 않습니다. PPTX 또는 PDF로 저장해 올려 주세요")
    if suffix not in SUPPORTED_SUFFIXES:
        raise MaterialError(f"지원하지 않는 형식입니다: {suffix or '확장자 없음'}")
    if not data:
        raise MaterialError("빈 파일입니다")
    texts = _pdf_pages(data) if suffix == ".pdf" else _pptx_pages(data)
    document_id = document_id_for(data)
    pages = tuple(MaterialPage(document_id, name, number, text.strip())
                  for number, text in enumerate(texts, start=1))
    if not pages:
        raise MaterialError("페이지가 없는 자료입니다")
    return Material(document_id, name, pages)
