import os
import logging
import tempfile
import warnings
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# Load .env for local development
load_dotenv()

warnings.filterwarnings("ignore", message=".*torch.classes.*")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Lazy imports to avoid startup crashes ───────────────────────────────────

def _import_raglite():
    from raglite import RAGLiteConfig, insert_document, hybrid_search, retrieve_chunks, rerank_chunks, rag
    return RAGLiteConfig, insert_document, hybrid_search, retrieve_chunks, rerank_chunks, rag

def _import_reranker():
    from rerankers import Reranker
    return Reranker

# ─── Prompts ─────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Answer the user's question using only the context below.
When responding, you MUST NOT reference the existence of the context, directly or indirectly.
Instead, you MUST treat the context as if its contents are entirely part of your working memory.
""".strip()

# ─── Secret / env helpers ────────────────────────────────────────────────────

def get_secret(key: str) -> str:
    """Read from st.secrets (Streamlit Cloud) then env (local)."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, "")


def get_db_url() -> str:
    url = get_secret("DATABASE_URL")
    if not url:
        url = "sqlite:///raglite.sqlite"
    # Neon / other cloud Postgres needs SSL
    if url.startswith("postgresql") and "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url

# ─── NVIDIA Embeddings ───────────────────────────────────────────────────────

def nvidia_embed(texts: List[str], api_key: str) -> List[List[float]]:
    """Embed a list of texts using NVIDIA's nv-embed-v1 model."""
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "nv-embed-v1",
        "input": texts,
        "encoding_format": "float",
    }
    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        json=payload,
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return [item["embedding"] for item in data["data"]]

# ─── NVIDIA Reranker wrapper compatible with RAGLite ─────────────────────────

class NvidiaReranker:
    """Thin wrapper so RAGLite's rerank_chunks can call us like a Reranker."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def rank(self, query: str, docs: List[str]):
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "rerank-qa-mistral-4b",
            "query": query,
            "passages": docs,
        }
        resp = requests.post(
            "https://integrate.api.nvidia.com/v1/rerank",
            json=payload,
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        results = resp.json().get("rankings", [])
        # Return sorted indices by descending relevance score
        sorted_results = sorted(results, key=lambda x: x["logit"], reverse=True)
        return [r["index"] for r in sorted_results]

# ─── Groq LLM ────────────────────────────────────────────────────────────────

def groq_chat(system_prompt: str, user_message: str, api_key: str) -> str:
    """Call Groq's llama3-70b-8192 and return the assistant reply."""
    from groq import Groq

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1024,
        temperature=0.7,
    )
    return completion.choices[0].message.content

# ─── RAGLite config ──────────────────────────────────────────────────────────

def build_config(nvidia_key: str, groq_key: str, db_url: str):
    RAGLiteConfig, *_ = _import_raglite()

    # Store NVIDIA key so the custom embedder closure can use it
    os.environ["NVIDIA_API_KEY"] = nvidia_key
    os.environ["GROQ_API_KEY"] = groq_key

    reranker = NvidiaReranker(api_key=nvidia_key)

    # RAGLite supports an "embedder" callable; wrap our NVIDIA function
    def _embedder(texts: List[str]) -> List[List[float]]:
        return nvidia_embed(texts, nvidia_key)

    return RAGLiteConfig(
        db_url=db_url,
        llm="groq/llama3-70b-8192",          # uses groq via LiteLLM prefix
        embedder=_embedder,
        embedder_normalize=True,
        chunk_max_size=2000,
        embedder_sentence_window_size=2,
        reranker=reranker,
    )

# ─── Document processing ─────────────────────────────────────────────────────

def process_document(file_path: str, config) -> bool:
    try:
        _, insert_document, *_ = _import_raglite()
        insert_document(Path(file_path), config=config)
        return True
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        return False

# ─── Search & retrieval ──────────────────────────────────────────────────────

def perform_search(query: str, config) -> List:
    try:
        _, _, hybrid_search, retrieve_chunks, rerank_chunks, _ = _import_raglite()
        chunk_ids, _ = hybrid_search(query, num_results=10, config=config)
        if not chunk_ids:
            return []
        chunks = retrieve_chunks(chunk_ids, config=config)
        return rerank_chunks(query, chunks, config=config)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

# ─── RAG answer ──────────────────────────────────────────────────────────────

def rag_answer(query: str, context_chunks: List, history: List, config) -> str:
    """Build a context string from chunks and call Groq."""
    groq_key = get_secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")

    context_parts = []
    for chunk in context_chunks[:5]:
        text = getattr(chunk, "text", None) or getattr(chunk, "content", str(chunk))
        context_parts.append(text)

    context = "\n\n---\n\n".join(context_parts)
    system = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}"

    # Build conversation history prefix
    history_text = ""
    for user_msg, assistant_msg in history[-4:]:
        history_text += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"

    full_user_message = f"{history_text}User: {query}"

    return groq_chat(system_prompt=system, user_message=full_user_message, api_key=groq_key)

# ─── Fallback (no docs) ──────────────────────────────────────────────────────

def handle_fallback(query: str) -> str:
    groq_key = get_secret("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")
    system = (
        "You are a helpful AI assistant. When you don't know something, be honest about it. "
        "Provide clear, concise, and accurate responses."
    )
    try:
        return groq_chat(system_prompt=system, user_message=query, api_key=groq_key)
    except Exception as e:
        logger.error(f"Fallback error: {e}")
        return "I encountered an error while processing your request. Please try again."

# ─── Streamlit UI ────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="RAG App – Hybrid Search", layout="wide")

    # Session state defaults
    defaults = {
        "chat_history": [],
        "documents_loaded": False,
        "my_config": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("Keys are loaded from **st.secrets** (Streamlit Cloud) or `.env` (local).")

        nvidia_key = get_secret("NVIDIA_API_KEY")
        groq_key   = get_secret("GROQ_API_KEY")
        db_url     = get_db_url()

        # Allow manual override in the UI (useful for local runs without .env)
        with st.expander("Override keys (optional)"):
            nvidia_key = st.text_input("NVIDIA API Key", value=nvidia_key, type="password")
            groq_key   = st.text_input("Groq API Key",   value=groq_key,   type="password")
            db_url     = st.text_input("Database URL",   value=db_url)

        if st.button("💾 Save & Initialise", use_container_width=True):
            if not all([nvidia_key, groq_key, db_url]):
                st.error("NVIDIA key, Groq key and Database URL are all required.")
            else:
                with st.spinner("Building configuration…"):
                    try:
                        st.session_state.my_config = build_config(nvidia_key, groq_key, db_url)
                        st.success("✅ Configuration saved!")
                    except Exception as e:
                        st.error(f"Configuration error: {e}")

        st.divider()
        st.caption("Models in use")
        st.markdown("🔵 **Embeddings** — NVIDIA nv-embed-v1")
        st.markdown("🟣 **Reranker** — NVIDIA rerank-qa-mistral-4b")
        st.markdown("🟢 **LLM** — Groq llama3-70b-8192")

    # ── Main area ────────────────────────────────────────────────────────────
    st.title("👀 RAG App with Hybrid Search")

    if not st.session_state.my_config:
        st.info("👈 Configure your API keys in the sidebar to get started.")
        return

    # Document upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"],
        accept_multiple_files=True, key="pdf_uploader"
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}…"):
                try:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    if process_document(tmp_path, st.session_state.my_config):
                        st.success(f"✅ {uploaded_file.name}")
                        st.session_state.documents_loaded = True
                    else:
                        st.error(f"❌ Failed: {uploaded_file.name}")
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        if st.session_state.documents_loaded:
            st.success("Documents ready! Ask away 👇")

    # Chat
    if not st.session_state.documents_loaded:
        st.info("Upload at least one PDF document to enable the chat.")
        return

    st.divider()
    st.subheader("💬 Chat")

    for user_msg, assistant_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user_msg)
        with st.chat_message("assistant"):
            st.write(assistant_msg)

    user_input = st.chat_input("Ask a question about your documents…")
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking…"):
                try:
                    chunks = perform_search(user_input, st.session_state.my_config)
                    if chunks:
                        response = rag_answer(
                            user_input, chunks,
                            st.session_state.chat_history,
                            st.session_state.my_config
                        )
                    else:
                        st.info("No relevant document context found — using general knowledge.")
                        response = handle_fallback(user_input)

                    placeholder.markdown(response)
                    st.session_state.chat_history.append((user_input, response))
                except Exception as e:
                    st.error(f"Error: {e}")
                    logger.exception("Chat error")


if __name__ == "__main__":
    main()
