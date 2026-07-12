# RAG Document Assistant

Upload a PDF, ask it questions, get answers grounded in the actual document, with page citations, and a straight "I don't know" when the answer isn't in there.

## What it does

- Upload a PDF (research paper, report, whatever)
- Ask questions about it in a chat interface
- Get answers pulled from the actual text, with the source page numbers shown
- If the doc doesn't have the answer, it says so instead of making something up

## Stack

- **PDF parsing:** PyMuPDF (switched from pdfplumber after it silently mangled spacing on LaTeX PDFs — see `LEARNING_NOTES.md`)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** Chroma
- **LLM:** Ollama, running `llama3.2:3b` locally
- **Frontend:** Streamlit
- **Containerized:** Docker

## Running it

You'll need [Ollama](https://ollama.com) installed and running locally — this app calls a local model, not a paid API.

```bash
ollama pull llama3.2:3b
```

Then, with Docker:

```bash
git clone https://github.com/ShrutiiPawar/rag-document-assistant.git
cd rag-document-assistant
docker build -t rag-assistant .
docker run -p 8501:8501 -e OLLAMA_HOST=http://host.docker.internal:11434 rag-assistant
```

Open `http://localhost:8501`, upload a PDF, start asking questions.

(No Docker? `pip install -r requirements.txt` and `streamlit run src/app.py` works too, as long as Ollama's running:)

## How it works

PDF → chunked into overlapping pieces → embedded → stored in Chroma → on each question, the most relevant chunks are retrieved and handed to the LLM along with strict instructions to only use what's there.

## Does it actually work?

Ran an 8-question eval against a real paper, checking whether retrieval actually pulled the right page for each answer: **85.7% retrieval accuracy**.

The more interesting result: one question got the *right* answer, but the retrieved chunks didn't obviously support it; turned out the model likely remembered that number from its own training data rather than pulling it from context. Worth knowing local models don't always stick to "only use the provided context," even when you tell them to. Full breakdown in `LEARNING_NOTES.md`.

## Known limitations

- One document at a time for now
- Runs on a small local model (`llama3.2:3b`), so answers are decent but not GPT-4 level
- Needs Ollama running locally 

## Worth reading

`LEARNING_NOTES.md` has the actual build log --- every bug, every wrong assumption, every fix, written down as I went. More honest picture of the project than this README.
