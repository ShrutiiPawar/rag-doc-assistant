import streamlit as st
import tempfile
import os

from ingest import extract_pages
from chunk import chunk_pages
from embed import embed_chunks
from store import add_to_store, query_store
from generate import generate_answer
from sentence_transformers import SentenceTransformer


st.set_page_config(page_title="RAG Document Assistant", layout="centered")
st.title("RAG Document Assistant")
st.caption("Upload a PDF and ask questions about it — answers are grounded in the document, with page citations.")


# --- Load the embedding model once, cache it across reruns ---
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()


# --- Session state: tracks whether a document has been indexed yet ---
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "messages" not in st.session_state:
    st.session_state.messages = []


# --- Upload + indexing ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and not st.session_state.indexed:
    with st.spinner("Indexing document... (extracting, chunking, embedding)"):
        # Save uploaded file to a temp path, since extract_pages expects a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        pages = extract_pages(tmp_path)
        chunks = chunk_pages(pages, source=uploaded_file.name)
        vectors = embed_chunks(chunks)
        add_to_store(chunks, vectors)

        os.remove(tmp_path)  # clean up temp file

    st.session_state.indexed = True
    st.success(f"Indexed {len(pages)} pages, {len(chunks)} chunks.")


# --- Chat interface ---
if st.session_state.indexed:
    # Show past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about the document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved = query_store(question, model)
                result = generate_answer(question, retrieved)

            st.write(result["answer"])
            st.caption(f"Sources: pages {result['sources']}")

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
else:
    st.info("Upload a PDF above to get started.")
