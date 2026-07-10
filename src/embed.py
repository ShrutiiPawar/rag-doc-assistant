from sentence_transformers import SentenceTransformer
from chunk import Chunk


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """
    Converts chunk text into embedding vectors, one vector per chunk.
    Returns vectors in the same order as the input chunks list.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c.text for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings.tolist()

if __name__ == "__main__":
    from ingest import extract_pages
    from chunk import chunk_pages

    pages = extract_pages("documents/AN IMAGE IS WORTH 16X16 WORDS- TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE.pdf")
    chunks = chunk_pages(pages, source="vit_paper")
    vectors = embed_chunks(chunks)

    print(f"Created {len(vectors)} vectors for {len(chunks)} chunks")
    print(f"Vector length: {len(vectors[0])}")
