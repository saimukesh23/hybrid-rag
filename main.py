"""
RAG App with Hybrid Search
- Embeddings : NVIDIA llama-3.2-nv-embedqa-1b-v2
- Reranking  : NVIDIA llama-3.2-nv-rerankqa-1b-v2
- LLM        : Groq  llama3-70b-8192
- Database   : PostgreSQL (Neon) or DuckDB (local fallback)
"""

import os
import logging
import tempfile
import warnings
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # no-op on Streamlit Cloud, works locally
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── RAG system prompt ───────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Answer the user's question using only the context below.
When responding, you MUST NOT reference the existence of the context, directly or indirectly.
Instead, treat the context as if its contents are entirely part of your working memory.
""".strip()

# ─── Secret helpers ──────────────────────────────────────────────────────────

def get_secret(key: str, default: str = "") -> str:
    """st.secrets first (Streamlit Cloud), then env (local .env)."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)


def get_db_url() -> str:
    url = get_secret("DATABASE_URL")
    if not url:
        return "duckdb:///raglite.duckdb"
    if url.startswith("postgresql") and "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url

# ─── NVIDIA Embeddings ───────────────────────────────────────────────────────

def nvidia_embed(texts: List[str], nvidia_key: str, input_type: str = "passage") -> List[List[float]]:
    """Embed texts using NVIDIA NIM llama-3.2-nv-embedqa-1b-v2."""
    import requests
    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        headers={"Authorization": f"Bearer {nvidia_key}", "Content-Type": "application/json"},
        json={
            "model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
            "truncate": "END",
        },
        timeout=60,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]

# ─── NVIDIA Reranker ─────────────────────────────────────────────────────────

class NvidiaReranker:
    """RAGLite-compatible reranker using NVIDIA NIM llama-3.2-nv-rerankqa-1b-v2."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def rank(self, query: str, docs: List[str], **kwargs):
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
        rankings = resp.json().get("rankings", [])
        return sorted(rankings, key=lambda x: x.get("logit", 0), reverse=True)

# ─── Groq LLM ────────────────────────────────────────────────────────────────

def groq_chat(system_prompt: str, user_message: str, groq_key: str) -> str:
    from groq import Groq
    client = Groq(api_key=groq_key)
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=1024,
        temperature=0.7,
    )
    return completion.choices[0].message.content

# ─── RAGLite config ──────────────────────────────────────────────────────────

def build_config(nvidia_key: str, groq_key: str, db_url: str):
    from raglite import RAGLiteConfig

    os.environ["GROQ_API_KEY"]   = groq_key
    os.environ["NVIDIA_API_KEY"] = nvidia_key

    def _embedder(texts: List[str]) -> List[List[float]]:
        return nvidia_embed(texts, nvidia_key, input_type="passage")

    return RAGLiteConfig(
        db_url=db_url,
        llm="groq/llama3-70b-8192",
        embedder=_embedder,
        embedder_normalize=True,
        reranker=NvidiaReranker(api_key=nvidia_key),
    )

# ─── Document processing ─────────────────────────────────────────────────────

def process_document(file_path: str, config) -> bool:
   def process_document(file_path: str, config) -> bool:
    try:
        from raglite import ingest
        ingest(Path(file_path), config=config)
        return True
    except Exception as e:
        logger.error(f"Document insert error: {e}")
        return False

# ─── Search & rerank ─────────────────────────────────────────────────────────

def search_and_rerank(query: str, config) -> List:
    try:
        from raglite import hybrid_search, rerank_chunks
        chunk_ids, _ = hybrid_search(query, num_results=20, config=config)
        if not chunk_ids:
            return []
        return rerank_chunks(query, chunk_ids, config=config)
    except Exception as e:
        logger.error(f"Search/rerank error: {e}")
        return []

# ─── Answer generation ───────────────────────────────────────────────────────

def generate_answer(query: str, chunks: List, history: List, groq_key: str) -> str:
    context_parts = []
    for chunk in chunks[:5]:
        text = getattr(chunk, "text", None) or getattr(chunk, "content", str(chunk))
        if text:
            context_parts.append(text.strip())

    context = "\n\n---\n\n".join(context_parts)
    system  = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}"

    history_text = ""
    for u, a in history[-4:]:
        history_text += f"User: {u}\nAssistant: {a}\n\n"

    return groq_chat(
        system_prompt=system,
        user_message=f"{history_text}User: {query}",
        groq_key=groq_key,
    )


def fallback_answer(query: str, groq_key: str) -> str:
    system = (
        "You are a helpful AI assistant. Answer clearly and concisely. "
        "If you don't know something, say so honestly."
    )
    try:
        return groq_chat(system_prompt=system, user_message=query, groq_key=groq_key)
    except Exception as e:
        logger.error(f"Fallback error: {e}")
        return "I encountered an error processing your request. Please try again."

# ─── Streamlit app ───────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="RAG App – Hybrid Search", layout="wide")

    for key, default in [
        ("chat_history", []),
        ("documents_loaded", False),
        ("my_config", None),
        ("groq_key", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.caption("Keys are read from **st.secrets** (Streamlit Cloud) or `.env` (local).")

        nvidia_key = get_secret("NVIDIA_API_KEY")
        groq_key   = get_secret("GROQ_API_KEY")
        db_url     = get_db_url()

        with st.expander("🔑 Override keys (optional)"):
            nvidia_key = st.text_input("NVIDIA API Key", value=nvidia_key, type="password",
                                       placeholder="nvapi-...")
            groq_key   = st.text_input("Groq API Key",   value=groq_key,   type="password",
                                       placeholder="gsk_...")
            db_url     = st.text_input("Database URL",   value=db_url)

        if st.button("💾 Save & Initialise", use_container_width=True):
            if not all([nvidia_key, groq_key, db_url]):
                st.error("All three fields are required.")
            else:
                with st.spinner("Initialising…"):
                    try:
                        st.session_state.my_config = build_config(nvidia_key, groq_key, db_url)
                        st.session_state.groq_key  = groq_key
                        st.success("✅ Ready!")
                    except Exception as e:
                        st.error(f"Config error: {e}")
                        logger.exception("Config error")

        st.divider()
        st.markdown("**Models in use**")
        st.markdown("🔵 Embeddings — `llama-3.2-nv-embedqa-1b-v2`")
        st.markdown("🟣 Reranker   — `llama-3.2-nv-rerankqa-1b-v2`")
        st.markdown("🟢 LLM        — `llama3-70b-8192` via Groq")

    # ── Main ─────────────────────────────────────────────────────────────────
    st.title("👀 RAG App with Hybrid Search")

    if not st.session_state.my_config:
        st.info("👈 Enter your API keys in the sidebar and click **Save & Initialise**.")
        return

    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
    )

    if uploaded_files:
        for uf in uploaded_files:
            with st.spinner(f"Processing **{uf.name}**…"):
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.getvalue())
                        tmp_path = tmp.name
                    if process_document(tmp_path, st.session_state.my_config):
                        st.success(f"✅ {uf.name}")
                        st.session_state.documents_loaded = True
                    else:
                        st.error(f"❌ Failed: {uf.name}")
                except Exception as e:
                    st.error(f"❌ {uf.name}: {e}")
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)

        if st.session_state.documents_loaded:
            st.success("Documents indexed — ask away below 👇")

    if not st.session_state.documents_loaded:
        st.info("Upload at least one PDF document to enable the chat.")
        return

    st.divider()
    st.subheader("💬 Chat")

    for u, a in st.session_state.chat_history:
        with st.chat_message("user"):      st.write(u)
        with st.chat_message("assistant"): st.write(a)

    user_input = st.chat_input("Ask a question about your documents…")
    if not user_input:
        return

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner("Thinking…"):
            try:
                groq_key = st.session_state.groq_key or get_secret("GROQ_API_KEY")
                chunks   = search_and_rerank(user_input, st.session_state.my_config)

                if chunks:
                    response = generate_answer(
                        user_input, chunks, st.session_state.chat_history, groq_key
                    )
                else:
                    st.info("No relevant document context found — using general knowledge.")
                    response = fallback_answer(user_input, groq_key)

                placeholder.markdown(response)
                st.session_state.chat_history.append((user_input, response))
            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Chat error")


if __name__ == "__main__":
    main()
