import fitz
from dataclasses import dataclass

@dataclass
class Page:
    page_number: int
    text : str
    char_count: int


def extract_pages(pdf_path: str) -> list[Page]:
    """
    Extracts text from a PDF, one Page record per page.
    Skips pages with no extractable text.
    """
    pages = []
    doc = fitz.open(pdf_path)

    for i, page in enumerate(doc):
        text = page.get_text()

        if not text or not text.strip():
            continue

        clean_text = text.strip()
        pages.append(
            Page(page_number=i + 1, text=clean_text, char_count=len(clean_text))
        )
    doc.close()

    return pages


