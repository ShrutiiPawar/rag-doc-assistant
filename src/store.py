from chunk import Chunk
import chromadb

client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection("documents")


def add_to_store(chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """
    Stores chunks and their embeddings in a persistent Chroma vector store.
    """
    client = chromadb.PersistentClient(path="./data/chroma_db")
    collection = client.get_or_create_collection("documents")

    collection.add(
        ids=[c.chunk_id for c in chunks],
        embeddings=vectors,
        documents=[c.text for c in chunks],
        metadatas=[{"page_number": c.page_number, "source": c.source} for c in chunks],
    )


def query_store(question: str, model, top_k: int = 4) -> list[dict]:
    """
    Embeds a question and returns the top_k most similar chunks from the store.
    """
    client = chromadb.PersistentClient(path="./data/chroma_db")
    collection = client.get_or_create_collection("documents")

    question_vector = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_vector,
        n_results=top_k,
    )

    matches = []
    for i in range(len(results["ids"][0])):
        matches.append({
            "text": results["documents"][0][i],
            "page_number": results["metadatas"][0][i]["page_number"],
            "distance": results["distances"][0][i],
        })

    return matches



if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from ingest import extract_pages
    from chunk import chunk_pages
    from embed import embed_chunks

    pages = extract_pages("documents/AN IMAGE IS WORTH 16X16 WORDS- TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE.pdf")
    chunks = chunk_pages(pages, source="vit_paper")
    vectors = embed_chunks(chunks)
    add_to_store(chunks, vectors)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    results = query_store("What is a Vision Transformer?", model)

    for r in results:
        print(f"[page {r['page_number']}] distance={r['distance']:.3f}")
        print(r["text"][:150], "\n")
