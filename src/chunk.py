from dataclasses import dataclass
from ingest import Page

@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    source: str  


def chunk_pages(pages: list[Page], source: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """
    Splits each page's text into overlapping fixed-size chunks.
    chunk_size = max characters per chunk.
    overlap = characters repeated between consecutive chunks, so ideas near
    a chunk boundary aren't lost entirely in either chunk.
    """
    chunks = []

    for page in pages:
        text = page.text
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + chunk_size
            piece = text[start:end].strip()

            if piece:
                chunk_id = f"{source}_p{page.page_number}_c{chunk_index}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=piece,
                        page_number=page.page_number,
                        source=source,
                    )
                )
                chunk_index += 1

            start += chunk_size - overlap  # step forward, but less than chunk_size = overlap

    return chunks


if __name__ == "__main__":
    from ingest import extract_pages

    pages = extract_pages("documents/Masked Autoencoders Are Scalable Vision Learners.pdf")
    chunks = chunk_pages(pages, source="vit_paper")
    print(f"Created {len(chunks)} chunks from {len(pages)} pages")
    print(chunks[0].text[-150:])
    print("---")
    print(chunks[1].text[:150])