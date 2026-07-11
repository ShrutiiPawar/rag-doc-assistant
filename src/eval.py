"""
eval.py
Small evaluation harness for the RAG pipeline.

Measures two things separately (don't conflate them - see LEARNING_NOTES
Lesson 1, errors propagate silently through a pipeline):

1. Retrieval accuracy: did the system pull a chunk from the page that
   actually contains the answer?
2. Answer quality: does the generated answer actually reflect the correct
   information? (manual judgment here - "LLM-as-judge" is a fancier
   alternative but manual is fine and honest for a first pass)

NOTE: expected_pages below are based on the actual structure of "Masked
Autoencoders Are Scalable Vision Learners" (He et al., arXiv 2111.06377) -
title/abstract p.1, related work p.3, approach/masking/encoder/decoder
p.4, reconstruction target p.5, ImageNet experiments + Table 1 p.6,
Table 2/mask token p.7, Table 3 comparisons + COCO results p.8. Page
numbers were read directly from the uploaded PDF text, not estimated -
but PyMuPDF's page count for YOUR local copy of this PDF should be
double-checked against these before trusting results, since a different
PDF export/version could paginate slightly differently.
"""

from sentence_transformers import SentenceTransformer
from store import query_store
from generate import generate_answer

TEST_SET = [
    {
        "question": "What is a masked autoencoder and what are its two core design choices?",
        "expected_pages": [1, 2],
    },
    {
        "question": "What masking ratio works best, and how does it compare to BERT's masking ratio?",
        "expected_pages": [6],
    },
    {
        "question": "Why does the encoder skip mask tokens, and what benefit does that provide?",
        "expected_pages": [4, 7],
    },
    {
        "question": "What accuracy does the vanilla ViT-Huge model achieve on ImageNet-1K?",
        "expected_pages": [1, 8],
    },
    {
        "question": "How does MAE performance compare to MoCo v3 and BEiT?",
        "expected_pages": [8],
    },
    {
        "question": "What is the capital of France?",  # should be refused - out of scope
        "expected_pages": [],
    },
    {
        "question": "What reconstruction target does MAE use, pixels or tokens?",
        "expected_pages": [5, 6],
    },
    {
        "question": "How does MAE perform on object detection and segmentation on COCO?",
        "expected_pages": [8],
    },
]


def evaluate(pdf_already_indexed: bool = True):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    retrieval_hits = 0
    results = []

    for item in TEST_SET:
        question = item["question"]
        expected_pages = set(item["expected_pages"])

        retrieved = query_store(question, model, top_k=4)
        retrieved_pages = set(r["page_number"] for r in retrieved)

        # A "hit" = at least one retrieved page overlaps with an expected page.
        # For the out-of-scope question (expected_pages=[]), we don't score
        # retrieval at all - there's nothing correct to retrieve.
        if expected_pages:
            hit = bool(retrieved_pages & expected_pages)
            if hit:
                retrieval_hits += 1
        else:
            hit = None  # not applicable

        result = generate_answer(question, retrieved)

        results.append({
            "question": question,
            "expected_pages": sorted(expected_pages) if expected_pages else "N/A (out of scope)",
            "retrieved_pages": sorted(retrieved_pages),
            "retrieval_hit": hit,
            "answer": result["answer"],
        })

    # Only count questions that actually have expected pages for the accuracy score
    scored_questions = [q for q in TEST_SET if q["expected_pages"]]
    accuracy = retrieval_hits / len(scored_questions) if scored_questions else 0

    return results, accuracy


def print_report(results, accuracy):
    print("=" * 70)
    print("RAG PIPELINE EVALUATION REPORT")
    print("=" * 70)

    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"  Expected pages:  {r['expected_pages']}")
        print(f"  Retrieved pages: {r['retrieved_pages']}")
        hit_display = "✅ HIT" if r["retrieval_hit"] else ("N/A" if r["retrieval_hit"] is None else "❌ MISS")
        print(f"  Retrieval:       {hit_display}")
        print(f"  Answer:          {r['answer'][:200]}{'...' if len(r['answer']) > 200 else ''}")

    print("\n" + "=" * 70)
    print(f"RETRIEVAL ACCURACY: {accuracy:.1%} ({sum(1 for r in results if r['retrieval_hit'])}/{len([r for r in results if r['retrieval_hit'] is not None])} scored questions)")
    print("=" * 70)


if __name__ == "__main__":
    results, accuracy = evaluate()
    print_report(results, accuracy)