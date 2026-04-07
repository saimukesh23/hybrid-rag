"""
RAG App with Hybrid Search (FINAL WORKING VERSION)

- Embeddings : NVIDIA (direct API, no LiteLLM)
- Reranking  : NVIDIA rerank API
- LLM        : Groq llama3-70b-8192
- Database   : PostgreSQL (Neon) or DuckDB fallback
- RAGLite    : 1.0.0 (stable API)
"""

import os
import logging
import tempfile
import warnings
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """
You are a helpful assistant. Answer ONLY using the provided context.
Do not mention context explicitly.
""".strip()

# ─────────────────────────────────────────────────────────
# Secrets

def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


def get_db_url():
    url = get_secret("DATABASE_URL")
    if not url:
        return "duckdb:///raglite.duckdb"

    if url.startswith("postgresql") and "sslmode" not in url:
        url += "?sslmode=require"

    return url

# ─────────────────────────────────────────────────────────
# NVIDIA EMBEDDINGS (FIXED)

def nvidia_embed(texts: List[str], api_key: str):
    import requests

    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
            "input": texts,
            "input_type": "passage",
        },
        timeout=60,
    )

    resp.raise_for_status()
    return [x["embedding"] for x in resp.json()["data"]]

# ─────────────────────────────────────────────────────────
# NVIDIA RERANKER

class NvidiaReranker:
    def __init__(self, api_key):
        self.api_key = api_key

    def rank(self, query, docs, **kwargs):
        import requests

        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/ranking",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
                "query": {"role": "user", "content": query},
                "passages": [{"role": "user", "content": d} for d in docs],
            },
            timeout=60,
        )

        resp.raise_for_status()
        rankings = resp.json()["rankings"]

        return [
            r["index"]
            for r in sorted(rankings, key=lambda x: x["logit"], reverse=True)
        ]

# ─────────────────────────────────────────────────────────
# GROQ LLM

def groq_chat(system, user, key):
    from groq import Groq

    client = Groq(api_key=key)

    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        max_tokens=1024,
    )

    return resp.choices[0].message.content

# ─────────────────────────────────────────────────────────
# RAG CONFIG

def build_config(nvidia_key, groq_key, db_url):
    from raglite import RAGLiteConfig

    os.environ["GROQ_API_KEY"] = groq_key

    def embedder(texts):
        return nvidia_embed(texts, nvidia_key)

    return RAGLiteConfig(
        db_url=db_url,
        llm="groq/llama3-70b-8192",
        embedder=embedder,   # ✅ FIXED
        embedder_normalize=True,
        reranker=NvidiaReranker(nvidia_key),
    )

# ─────────────────────────────────────────────────────────
# DOCUMENT INGESTION

def process_document(file_path, config):
    try:
        from raglite import Document, insert_documents

        doc = Document.from_path(Path(file_path))
        insert_documents([doc], config=config)
        return True

    except Exception as e:
        logger.error(f"INSERT ERROR: {e}", exc_info=True)
        return False

# ─────────────────────────────────────────────────────────
# SEARCH PIPELINE

def search_and_rerank(query, config):
    try:
        from raglite import hybrid_search, rerank_chunks, retrieve_context

        chunk_ids, _ = hybrid_search(query, num_results=20, config=config)

        if not chunk_ids:
            return []

        ranked = rerank_chunks(query, chunk_ids, config=config)
        return retrieve_context(ranked[:5], config=config)

    except Exception as e:
        logger.error(f"SEARCH ERROR: {e}", exc_info=True)
        return []

# ─────────────────────────────────────────────────────────
# ANSWER GENERATION

def generate_answer(query, spans, history, groq_key):
    context_parts = []

    for span in spans:
        text = " ".join(
            getattr(c, "text", "") for c in (span if isinstance(span, list) else [span])
        )
        context_parts.append(text)

    context = "\n\n---\n\n".join(context_parts)

    system = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}"

    hist = ""
    for u, a in history[-4:]:
        hist += f"User: {u}\nAssistant: {a}\n"

    return groq_chat(system, hist + f"\nUser: {query}", groq_key)


def fallback_answer(query, key):
    return groq_chat("You are helpful.", query, key)

# ─────────────────────────────────────────────────────────
# STREAMLIT UI

def main():
    st.set_page_config(page_title="RAG App", layout="wide")

    if "config" not in st.session_state:
        st.session_state.config = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "ready" not in st.session_state:
        st.session_state.ready = False

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Setup")

        nvidia = get_secret("NVIDIA_API_KEY")
        groq   = get_secret("GROQ_API_KEY")
        db     = get_db_url()

        if st.button("Initialize"):
            st.session_state.config = build_config(nvidia, groq, db)
            st.session_state.ready = True
            st.success("Ready!")

    st.title("📄 RAG Chat")

    if not st.session_state.ready:
        st.info("Initialize from sidebar")
        return

    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if files:
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(f.read())
                path = tmp.name

            if process_document(path, st.session_state.config):
                st.success(f"{f.name} processed")
            else:
                st.error(f"{f.name} failed")

    query = st.chat_input("Ask something")

    if query:
        st.chat_message("user").write(query)

        chunks = search_and_rerank(query, st.session_state.config)

        if chunks:
            ans = generate_answer(query, chunks, st.session_state.history, get_secret("GROQ_API_KEY"))
        else:
            ans = fallback_answer(query, get_secret("GROQ_API_KEY"))

        st.chat_message("assistant").write(ans)
        st.session_state.history.append((query, ans))


if __name__ == "__main__":
    main()
