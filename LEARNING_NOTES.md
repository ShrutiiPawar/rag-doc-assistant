# RAG Project — Learning Notes

A running log of engineering lessons from building this project, not just a
build log. The goal is to remember *why*, not just *what*.

---

## Lesson 1 — Errors propagate silently through a pipeline

In a multi-stage pipeline (extract → chunk → embed → store → retrieve →
generate), a mistake early on doesn't crash anything. It just quietly makes
every downstream stage worse, and the further downstream you look for the
bug, the more wrong you'll be about where it actually lives.

**Concrete case:** if chunking cuts a sentence in half, no error is thrown —
retrieval will run fine, the LLM will generate fluent text, and the final
answer will just be subtly wrong. A great LLM makes bad context sound
*convincing* — "garbage in, eloquent garbage out." A confidently wrong
answer is harder to catch than an obviously broken one.

**Takeaway:** when debugging bad output, check upstream stages first
(especially chunking and retrieval) before assuming the LLM/prompt is at
fault.

---

## Lesson 2 — Dataclass vs. dictionary: the real reason to prefer dataclass

For structured records (e.g. `{page_number: 1, text: "..."}`), a dataclass
is often assumed to be "better for adding fields" — but dictionaries can
have keys added just as easily. That's not the real advantage.

**The real reason:** typo safety.
- Dictionary: `page["pagenumber"]` (typo, missing underscore) — Python
  accepts any string as a key, so this fails *silently until runtime*,
  and only on the exact line that executes it. Could hide in an
  untested code path for a long time.
- Dataclass: `page.pagenumbr` (typo) — fails immediately, caught by
  the IDE and/or Python, because the attribute simply doesn't exist.

**Takeaway:** dataclasses convert a category of bugs from "silent, caught
late" into "loud, caught immediately." This is the same theme as Lesson 1 —
prefer designs that fail loudly and early over ones that fail quietly and
late.

---

## Architecture reference

Two pipelines:
- **Indexing (build once):** PDF document → Chunking → Embeddings → Vector store
- **Querying (per question):** User question → Retrieval (reads vector store) → Generation → Answer

Build-yourself vs. use-a-library:
| Step | Build yourself | Use a library |
|---|---|---|
| Extraction | — | PDF parsing (pdfplumber) |
| Chunking | chunk size/overlap decisions | — |
| Embeddings | — | sentence-transformers |
| Vector store | — | Chroma |
| Retrieval | orchestration (how many chunks, fallback logic) | similarity search internals |
| Generation | prompt template, context structuring | Ollama client |

---

## Lesson 3 — A library can "succeed" while silently producing garbage

`pdfplumber.extract_text()` ran without any errors on a LaTeX-generated
academic PDF (arXiv-style, two-column layout) — but the output text had
missing spaces between words: `"PublishedasaconferencepaperatICLR2021"`.

**Why this happened:** a PDF doesn't store text as "words with spaces" like
a `.txt` file — it stores character/glyph *positions* on the page. The
extraction library has to *infer* word boundaries from the horizontal gaps
between characters. Some PDFs (especially LaTeX output with tight kerning
or unusual font embedding) confuse that inference, and the library quietly
guesses wrong — no exception, no warning, just bad text.

**What we tried first:** tuning `pdfplumber`'s tolerance parameters
(`x_tolerance`, `y_tolerance`) and switching to `extract_words()` — both are
reasonable first fixes for word-boundary issues, but barely changed the
output here. That result was itself useful diagnostic information: it told
us the problem wasn't a tuning issue, but something more fundamental about
how this particular PDF's fonts/encoding interact with `pdfplumber`.

**Fix:** switched extraction libraries entirely, from `pdfplumber` to
`PyMuPDF` (`fitz`), which uses a different underlying approach to reading
the text layer and handles this class of PDF more reliably.

**Takeaway (ties back to Lesson 1):** "the code ran without errors" is
never sufficient evidence that a pipeline stage worked correctly. Always
manually inspect a sample of real output before building the next stage on
top of it — especially right after ingesting real-world, messy input data
(vs. clean toy examples). When a first fix barely moves the result, that's
a signal to reconsider the root cause rather than keep tweaking parameters.

---

## Lesson 4 — Overlap isn't about "fixing" a cut, it's about not losing context at a boundary

Fixed-size chunking will inevitably cut some sentences/ideas mid-way,
regardless of chunk size. Overlap doesn't prevent this — it ensures that
even if an idea is cut in one chunk, it's very likely to appear *whole* in
the neighboring chunk, because each new chunk starts partway *into* the
previous one, not exactly where it ended.

Mechanism: `start += chunk_size - overlap` (not `+= chunk_size`). With
`chunk_size=800, overlap=150`, each chunk only advances 650 characters, so
the last 150 characters of one chunk reappear at the start of the next.

Verified by checking `chunks[0].text[-150:]` against `chunks[1].text[:150]`
— confirmed matching text, i.e. real overlap, not just adjacent slices.

---

## Progress checklist

- [x] Milestone 1 — PDF extraction (PyMuPDF, `Page` dataclass, char_count)
- [x] Milestone 2 — Chunking (fixed-size + overlap, `Chunk` dataclass with unique `chunk_id`)
- [ ] Milestone 3 — Embeddings (sentence-transformers)
- [ ] Milestone 4 — Vector store (Chroma) + retrieval
- [ ] Milestone 5 — Generation (Ollama + prompt design)
- [ ] Milestone 6 — Streamlit frontend
- [ ] Milestone 7 — Evaluation
- [ ] Milestone 8 — Deployment

---

## Lesson 5 — Batch operations vs. looping one item at a time

`model.encode([c.text for c in chunks])` (one call, whole list) is much
faster than calling `model.encode(chunk.text)` in a loop, 200 times.
Neural network-based operations are optimized for parallel/batch
processing — each individual call carries its own overhead, so batching
saves that overhead across the whole set.

**Rule of thumb:** whenever a library offers a batch/list-based version of
an operation instead of one-at-a-time, prefer the batch version.

**Sanity check pattern:** after embedding, `len(vectors) == len(chunks)`
should always hold. If it doesn't, something was silently skipped or
duplicated — cheap, useful check to keep in mind at every pipeline stage
where a list goes in and a same-length list should come out.

---

## Progress checklist

- [x] Milestone 1 — PDF extraction (PyMuPDF, `Page` dataclass, char_count)
- [x] Milestone 2 — Chunking (fixed-size + overlap, `Chunk` dataclass with unique `chunk_id`)
- [x] Milestone 3 — Embeddings (sentence-transformers, all-MiniLM-L6-v2, 384-dim vectors)
- [ ] Milestone 4 — Vector store (Chroma) + retrieval
- [ ] Milestone 5 — Generation (Ollama + prompt design)
- [ ] Milestone 6 — Streamlit frontend
- [ ] Milestone 7 — Evaluation
- [ ] Milestone 8 — Deployment

---

## Lesson 6 — Distance vs. similarity, and matching embedding models

**Distance and similarity move in opposite directions.** Chroma returns
*distance* (how far apart two vectors are) — lower distance means more
similar meaning, not higher. Verified end-to-end: querying "What is a
Vision Transformer?" against the ViT paper returned the Method section
(distance 0.824) ranked above the title/abstract page (distance 0.954-0.960)
— correct, since the abstract only loosely relates to that specific
question while the Method section is directly on-topic.

**The embedding model must be identical between indexing and querying.**
Chunks were embedded with `all-MiniLM-L6-v2`; the question must be embedded
with that *same* model instance/name, otherwise the vectors live in
different mathematical spaces and comparing distances between them would
be meaningless — not an error, just silently wrong results (same "silent
failure" theme as Lessons 1 and 3).

**Debugging note:** pasting Python code directly into a PowerShell prompt
produces PowerShell parser errors (`ParserError`, `UnexpectedToken`), not
Python errors — a useful tell for diagnosing "did this code actually run
as Python, or did it go to the wrong interpreter?"

---

## Progress checklist

- [x] Milestone 1 — PDF extraction (PyMuPDF, `Page` dataclass, char_count)
- [x] Milestone 2 — Chunking (fixed-size + overlap, `Chunk` dataclass with unique `chunk_id`)
- [x] Milestone 3 — Embeddings (sentence-transformers, all-MiniLM-L6-v2, 384-dim vectors)
- [x] Milestone 4 — Vector store + retrieval (Chroma, verified correct ranking on a real question)
- [ ] Milestone 5 — Generation (Ollama + prompt design)
- [ ] Milestone 6 — Streamlit frontend
- [ ] Milestone 7 — Evaluation
- [ ] Milestone 8 — Deployment

---

## Lesson 7 — Grounding the LLM: prompt design controls hallucination

Wiring retrieved chunks into an LLM call is mechanically simple (one
library call to `ollama.generate`), but the prompt's *instructions* are
what actually make it a RAG system instead of a generic chatbot with extra
text pasted in front of it.

**Key design choice:** explicitly instruct the model to (1) use only the
provided context, and (2) say it doesn't know if the context is
insufficient, rather than guessing. Verified both behaviors with real
tests, not just assumed:
- On-topic question ("What is a Vision Transformer?") → correct, grounded
  answer using paper-specific language (patches, attention), not generic
  textbook phrasing — evidence it was actually using retrieved context.
- Off-topic question ("What is the capital of France?") → correctly
  refused: "I don't have enough information in the document to answer
  that," instead of hallucinating.

**Why this matters for a portfolio project specifically:** anyone can wrap
an LLM API call. Explicitly constraining and *verifying* grounded behavior
is the signal that separates an engineered RAG system from a toy chatbot
wrapper — and it's the natural centerpiece of the evaluation section later.

**Debugging note:** `ollama._types.ResponseError: model not found (404)`
meant the model referenced in code (`llama3.2:3b`) wasn't actually pulled
on this machine yet — checked with `ollama list` to see what was actually
available locally, rather than guessing. Reminder: code referencing a
model name and the model actually being present locally are two separate
things that can silently drift apart across machines/sessions.

---

## Lesson 8 — Don't trust the LLM to self-report its own sources

Initial version only returned `response["response"]` (the answer text) and
discarded the retrieved chunks' metadata. That meant source pages were
only available if the LLM happened to mention them correctly inside its
free-text answer — unreliable, since nothing forces it to, and it could
misstate them.

**Better approach:** derive sources directly from the retrieved chunks'
metadata (`page_number`), which you already know for certain — not from
parsing or trusting the model's generated text. `generate_answer` now
returns `{"answer": ..., "sources": [...]}`, where `sources` is built with
`sorted(set(c["page_number"] for c in chunks))` (dedup + order).

**General principle:** whenever a piece of information is already known
deterministically upstream (here: which pages were retrieved), don't make
the LLM re-derive or re-state it — pull it from the source of truth
directly. Only ask the LLM to do the part that actually requires language
generation (the answer itself).

---

## Lesson 9 — Evaluation findings: test-set errors vs. real system findings

Ran the 8-question eval harness against the MAE paper. Initial retrieval
accuracy: 71.4% (5/7 scored questions, excluding the out-of-scope
control question). Two misses, investigated individually rather than
averaged away:

**Miss 1 (fixed - was a test-set bug, not a system bug):** expected page 6
for a masking-ratio question; system retrieved page 4. Manually checked
the source PDF - the "75% ratio, vs. BERT's 15%" text is actually on page
4 (two-column layout, sits right next to the "4. ImageNet Experiments"
header). The retrieval was correct; the hand-written ground truth was
wrong. Fixed `expected_pages` to `[4]`. **Takeaway:** evaluation is only
as good as its ground truth - verify the test set itself, not just the
system under test.

**Miss 2 (real finding, kept in the eval and flagged, not "fixed"):**
question about ViT-Huge's ImageNet accuracy. Retrieved pages [7, 12]
(mask-token ablation table, appendix config tables) - neither obviously
contains the answer. Yet the generated answer stated "87.8%," which is
*correct* and matches page 8's content exactly. Likely explanation:
`llama3.2:3b` may have seen this well-known paper during its own
pre-training and is recalling the number from memory rather than purely
from the retrieved context, despite the prompt's grounding instructions.
**Takeaway:** a correct-looking answer is not proof of correct grounding.
Small local models don't always follow "use only the provided context"
instructions perfectly - checking retrieved pages against the answer's
claims (not just checking if the answer sounds right) is what surfaced
this. This is a genuinely useful, portfolio-worthy finding, not a bug to
hide - it demonstrates real evaluation rigor rather than a shallow "it
works" claim.

---

## Lesson 10 — Docker debugging: three separate real-world issues

Getting the app running in Docker surfaced three distinct problems, each
worth understanding on its own (not just "it broke, then it worked"):

**1. Bloated `requirements.txt` from environment contamination.**
`pip freeze > requirements.txt` was run with an Anaconda `base` environment
active *alongside* venv (visible in the prompt: `(venv) (base)`), so it
captured 120+ packages from both environments, not just what the project
actually needs. **Fix:** rewrote `requirements.txt` by hand with only the
~7 top-level packages actually imported in the code, letting pip resolve
sub-dependencies fresh. **Takeaway:** always double-check which environment
is active before freezing dependencies - a contaminated environment
produces a requirements file that "works" locally (everything's already
installed) but silently balloons builds elsewhere.

**2. PyTorch defaults to a CUDA/GPU build on Linux.** Docker containers
run Linux even on a Windows host. Installing `torch` (a `sentence-
transformers` dependency) without specifying otherwise pulled multiple GB
of unnecessary NVIDIA/CUDA packages - a portfolio project doesn't need GPU
support. **Fix:** installed the CPU-only build explicitly first, from
PyTorch's own package index (`pip install torch --index-url
https://download.pytorch.org/whl/cpu`), *before* installing the rest of
requirements - so when `sentence-transformers` later asks for `torch`, a
compatible version is already satisfied and the huge CUDA build is never
downloaded.

**3. Slow/unstable network caused pip to time out mid-download**, even
after fixing (1) and (2) - a `ReadTimeoutError` on a large package
partway through. **Fix:** added `--default-timeout=120 --retries 10` to
the pip install command, making it tolerant of a stalled connection
instead of aborting the whole build on first hiccup.

**4. `localhost` inside a container isn't the same `localhost` as the
host machine.** The app worked until it needed to reach Ollama, which
runs on the host, not inside the container - `ollama.generate()` defaulted
to `localhost:11434`, which inside the container refers to the container
itself (nothing listening there), not the host machine. **Fix:** used
Docker's special DNS name `host.docker.internal` to reach the host from
inside a container, passed in via an `OLLAMA_HOST` environment variable
(defaulting to `localhost` when not set) - so the same code works
unmodified both run directly and inside Docker, just with a different
flag at launch time (`-e OLLAMA_HOST=http://host.docker.internal:11434`).

**Overall takeaway:** none of these were exotic problems - a
misconfigured environment, a platform-default that doesn't fit the use
case, network flakiness, and a container-networking basics gap. This is
genuinely representative of what real deployment work looks like:
mostly configuration and environment issues, not application logic bugs.

---

## Progress checklist

- [x] Milestone 1 — PDF extraction (PyMuPDF, `Page` dataclass, char_count)
- [x] Milestone 2 — Chunking (fixed-size + overlap, `Chunk` dataclass with unique `chunk_id`)
- [x] Milestone 3 — Embeddings (sentence-transformers, all-MiniLM-L6-v2, 384-dim vectors)
- [x] Milestone 4 — Vector store + retrieval (Chroma, verified correct ranking on a real question)
- [x] Milestone 5 — Generation (Ollama + llama3.2:3b, grounded answers + verified hallucination refusal + reliable source citation)
- [x] Milestone 6 — Streamlit frontend (upload → index once via session_state → chat interface, verified end-to-end in browser)
- [x] Milestone 7 — Evaluation (8-question harness, retrieval accuracy metric, caught both a test-set bug and a real possible-memorization finding)
- [x] Milestone 8 — Deployment (Dockerized: clean requirements, CPU-only torch, network-tolerant pip install, host.docker.internal networking - verified full end-to-end run in a container)

---

**Core RAG pipeline: complete.** All 8 milestones built, tested, and
documented from first principles, with real debugging at every stage
rather than a clean guided path. Next steps are optional polish: README
with architecture diagram + eval results, demo GIF, and optionally a live
hosted deployment using an API-based model instead of local Ollama.