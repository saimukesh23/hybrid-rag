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

RAG_SYSTEM_PROMPT = """
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Answer the user's question using only the context below.
Do not mention the context explicitly.
""".strip()

# ─── Secrets ─────────────────────────────────────────

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

# ─── NVIDIA Reranker ─────────────────────────────────

class NvidiaReranker:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def rank(self, query: str, docs: List[str], **_):
        import requests
        try:
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
            return [r["index"] for r in sorted(rankings, key=lambda x: x.get("logit", 0), reverse=True)]
        except Exception as e:
            logger.error(f"Reranker error: {e}")
            return list(range(len(docs)))

# ─── Groq LLM ───────────────────────────────────────

def groq_chat(system: str, user: str, groq_key: str) -> str:
    from groq import Groq
    try:
        resp = Groq(api_key=groq_key).chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ FIXED
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=1024,
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return "Error generating response."

# ─── RAG Config ─────────────────────────────────────

def build_config(nvidia_key: str, groq_key: str, db_url: str):
    from raglite import RAGLiteConfig

    os.environ["OPENAI_API_KEY"] = nvidia_key
    os.environ["OPENAI_BASE_URL"] = "https://integrate.api.nvidia.com/v1"

    return RAGLiteConfig(
        db_url=db_url,
        llm="groq/llama-3.3-70b-versatile",  # ✅ FIXED
        embedder="nvidia/llama-3.2-nv-embedqa-1b-v2",  # ✅ FIXED
        embedder_normalize=True,
        reranker=NvidiaReranker(api_key=nvidia_key),
    )

# ─── Document Processing ─────────────────────────────

def process_document(file_path: str, config) -> bool:
    try:
        from raglite import Document, insert_documents
        doc = Document.from_path(Path(file_path))
        insert_documents([doc], config=config)
        return True
    except Exception as e:
        logger.error(f"Document error: {e}")
        return False

# ─── Search ─────────────────────────────────────────

def search_and_rerank(query: str, config):
    try:
        from raglite import hybrid_search, rerank_chunks, retrieve_context
        chunk_ids, _ = hybrid_search(query, num_results=20, config=config)
        if not chunk_ids:
            return []
        ranked = rerank_chunks(query, chunk_ids, config=config)
        return retrieve_context(ranked[:5], config=config)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

# ─── Answer ─────────────────────────────────────────

def generate_answer(query, spans, history, groq_key):
    context = "\n\n---\n\n".join(
        " ".join(getattr(c, "text", str(c)) for c in span)
        for span in spans
    )
    system = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}"
    hist = "".join(f"User:{u}\nAssistant:{a}\n" for u, a in history[-4:])
    return groq_chat(system, hist + query, groq_key)

def fallback_answer(query, groq_key):
    return groq_chat("You are a helpful assistant.", query, groq_key)

# ─── UI ─────────────────────────────────────────────

def main():
    st.set_page_config(page_title="RAG App", layout="wide")

    if "config" not in st.session_state:
        st.session_state.config = None
        st.session_state.chat = []
        st.session_state.docs = False

    st.sidebar.title("Settings")

    nvidia = st.text_input("NVIDIA Key", type="password")
    groq = st.text_input("Groq Key", type="password")
    db = st.text_input("DB URL", value="duckdb:///raglite.duckdb")

    if st.sidebar.button("Init"):
        if not (nvidia and groq):
            st.error("Keys required")
        else:
            st.session_state.config = build_config(nvidia, groq, db)
            st.session_state.groq_key = groq
            st.success("Ready")

    if not st.session_state.config:
        st.info("Configure first")
        return

    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if files:
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(f.read())
                process_document(tmp.name, st.session_state.config)
                os.remove(tmp.name)
        st.session_state.docs = True

    if not st.session_state.docs:
        st.info("Upload documents")
        return

    for u, a in st.session_state.chat:
        st.chat_message("user").write(u)
        st.chat_message("assistant").write(a)

    q = st.chat_input("Ask...")
    if q:
        st.chat_message("user").write(q)

        groq_key = st.session_state.get("groq_key", groq)

        spans = search_and_rerank(q, st.session_state.config)

        if spans:
            ans = generate_answer(q, spans, st.session_state.chat, groq_key)
        else:
            ans = fallback_answer(q, groq_key)

        st.chat_message("assistant").write(ans)
        st.session_state.chat.append((q, ans))


if __name__ == "__main__":
    main()
