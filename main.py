"""
RAG App with Hybrid Search
- Embeddings : NVIDIA llama-3.2-nv-embedqa-1b-v2  (via LiteLLM openai/ prefix)
- Reranking  : NVIDIA llama-3.2-nv-rerankqa-1b-v2 (direct HTTP, custom wrapper)
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

load_dotenv()
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

# ─── Secrets ─────────────────────────────────────────────────────────────────

def get_secret(key: str, default: str = "") -> str:
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

# ─── NVIDIA Reranker (direct HTTP — not via LiteLLM) ─────────────────────────

class NvidiaReranker:
    """
    Minimal reranker object that RAGLite accepts.
    RAGLite calls  reranker.rank(query, docs) -> ranked list of indices.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key

    def rank(self, query: str, docs: List[str], **_):
        import requests
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/ranking",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
                "query":    {"role": "user", "content": query},
                "passages": [{"role": "user", "content": d} for d in docs],
            },
            timeout=60,
        )
        resp.raise_for_status()
        rankings = resp.json().get("rankings", [])
        # Return indices sorted by descending logit (most relevant first)
        return [r["index"] for r in sorted(rankings, key=lambda x: x.get("logit", 0), reverse=True)]

# ─── Groq chat ───────────────────────────────────────────────────────────────

def groq_chat(system: str, user: str, groq_key: str) -> str:
    from groq import Groq
    client = Groq(api_key=groq_key)
    resp = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        max_tokens=1024,
        temperature=0.7,
    )
    return resp.choices[0].message.content

# ─── RAGLite config ──────────────────────────────────────────────────────────

def build_config(nvidia_key: str, groq_key: str, db_url: str):
    from raglite import RAGLiteConfig

    # LiteLLM uses OPENAI_API_KEY + api_base for openai-compatible endpoints.
    # Setting NVIDIA_API_KEY and pointing LiteLLM to the NVIDIA base URL works
    # via the "openai/" model prefix (LiteLLM's generic OpenAI-compat gateway).
    os.environ["GROQ_API_KEY"]   = groq_key
    os.environ["NVIDIA_API_KEY"] = nvidia_key
    # LiteLLM needs these to route to NVIDIA for embeddings
    os.environ["OPENAI_API_KEY"]  = nvidia_key   # used by litellm openai/ prefix
    os.environ["OPENAI_BASE_URL"] = "https://integrate.api.nvidia.com/v1"

    return RAGLiteConfig(
        db_url=db_url,
        # LLM: Groq via LiteLLM's groq/ prefix
        llm="groq/llama3-70b-8192",
        # Embedder: NVIDIA via LiteLLM's openai/ prefix pointing at NVIDIA base URL
        # The model string must be the exact model name as NVIDIA serves it.
        embedder="openai/nvidia/llama-3.2-nv-embedqa-1b-v2",
        embedder_normalize=True,
        reranker=NvidiaReranker(api_key=nvidia_key),
    )

# ─── Document processing ─────────────────────────────────────────────────────

def process_document(file_path: str, config) -> bool:
    try:
        from raglite import insert_document
        insert_document(Path(file_path), config=config)
        return True
    except Exception as e:
        logger.error(f"insert_document error: {e}", exc_info=True)
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
        logger.error(f"search_and_rerank error: {e}", exc_info=True)
        return []

# ─── Answer generation ───────────────────────────────────────────────────────

def generate_answer(query: str, chunks: List, history: List, groq_key: str) -> str:
    parts = []
    for chunk in chunks[:5]:
        text = getattr(chunk, "text", None) or getattr(chunk, "content", str(chunk))
        if text:
            parts.append(text.strip())
    context = "\n\n---\n\n".join(parts)
    system  = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}"
    hist    = "".join(f"User: {u}\nAssistant: {a}\n\n" for u, a in history[-4:])
    return groq_chat(system, f"{hist}User: {query}", groq_key)


def fallback_answer(query: str, groq_key: str) -> str:
    try:
        return groq_chat(
            "You are a helpful AI assistant. Be clear, concise, and honest.",
            query, groq_key,
        )
    except Exception as e:
        logger.error(f"fallback error: {e}")
        return "I encountered an error. Please try again."

# ─── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="RAG App – Hybrid Search", layout="wide")

    for k, v in [("chat_history", []), ("documents_loaded", False),
                 ("my_config", None), ("groq_key", "")]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.caption("Keys loaded from **st.secrets** (Cloud) or `.env` (local).")

        nvidia_key = get_secret("NVIDIA_API_KEY")
        groq_key   = get_secret("GROQ_API_KEY")
        db_url     = get_db_url()

        with st.expander("🔑 Override keys (optional)"):
            nvidia_key = st.text_input("NVIDIA API Key", value=nvidia_key,
                                       type="password", placeholder="nvapi-...")
            groq_key   = st.text_input("Groq API Key",   value=groq_key,
                                       type="password", placeholder="gsk_...")
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
        "Upload PDF documents", type=["pdf"],
        accept_multiple_files=True, key="pdf_uploader",
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
                        st.error(f"❌ Failed: {uf.name} — check logs for details")
                except Exception as e:
                    st.error(f"❌ {uf.name}: {e}")
                    logger.exception(f"Upload error: {uf.name}")
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
                    response = generate_answer(user_input, chunks,
                                               st.session_state.chat_history, groq_key)
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
