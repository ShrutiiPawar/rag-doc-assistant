
import ollama
import os

# Use host.docker.internal when running in Docker, localhost otherwise.
# This lets the same code work both on your machine directly and inside
# a container without needing two separate versions.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
client = ollama.Client(host=OLLAMA_HOST)

PROMPT_TEMPLATE = """You are a helpful assistant answering questions based only on the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer using only the information in the context above.
- If the context does not contain enough information to answer, say "I don't have enough information in the document to answer that."
- Provide a thorough, detailed answer using multiple sentences where relevant, rather than a single brief sentence.

Answer:"""

def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    Takes retrieved chunks and a question, returns a grounded answer plus
    the source pages it was based on.
    """
    context = "\n\n".join(f"[Page {c['page_number']}] {c['text']}" for c in chunks)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = ollama.generate(model="llama3.2:3b", prompt=prompt)

    sources = sorted(set(c["page_number"] for c in chunks))

    return {
        "answer": response["response"],
        "sources": sources,
    }

if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from ingest import extract_pages
    from chunk import chunk_pages
    from embed import embed_chunks
    from store import add_to_store, query_store

    pages = extract_pages("documents/Masked Autoencoders Are Scalable Vision Learners.pdf")
    chunks = chunk_pages(pages, source="vit_paper")
    vectors = embed_chunks(chunks)
    add_to_store(chunks, vectors)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    question= "What is a Vision Transformer?"
    retrieved = query_store(question, model)


    result = generate_answer(question, retrieved)
    print(f"Answer: {result['answer']}")
    print(f"Sources: pages {result['sources']}")