import os
import logging
import tempfile
import warnings
from typing import List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAG_SYSTEM_PROMPT = """
You are a friendly and knowledgeable assistant.
Answer ONLY using the provided context.
Do not make up information.
""".strip()

# ─── Secrets ─────────────────────────

def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

def get_db_url():
    url = get_secret("DATABASE_URL")
    return url if url else "duckdb:///raglite.duckdb"

# ─── NVIDIA Reranker ─────────────────

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

# ─── Groq LLM ───────────────────────

def groq_chat(system: str, user: str, groq_key: str) -> str:
    from groq import Groq
    try:
        resp = Groq(api_key=groq_key).chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return "Error generating response."

# ─── RAG Config ─────────────────────

def build_config(nvidia_key, groq_key, db_url):
    from raglite import RAGLiteConfig

    os.environ["OPENAI_API_KEY"] = nvidia_key
    os.environ["OPENAI_BASE_URL"] = "https://integrate.api.nvidia.com/v1"

    return RAGLiteConfig(
        db_url=db_url,
        llm="groq/llama-3.3-70b-versatile",
        embedder="nvidia/llama-3.2-nv-embedqa-1b-v2",
        embedder_normalize=True,
        reranker=NvidiaReranker(nvidia_key),
    )

# ─── FIXED PDF PROCESSING ─────────────────

def process_document(file_path, config):
    try:
        from raglite import insert_documents, Document
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            logger.error("No text extracted")
            return False

        doc = Document(text=text)
        insert_documents([doc], config=config)

        return True

    except Exception as e:
        logger.error(f"Document error: {e}")
        return False

# ─── SEARCH ─────────────────────────

def search_and_rerank(query, config):
    try:
        from raglite import hybrid_search, rerank_chunks, retrieve_context
        chunk_ids, _ = hybrid_search(query, config=config)
        if not chunk_ids:
            return []
        ranked = rerank_chunks(query, chunk_ids, config=config)
        return retrieve_context(ranked[:5], config=config)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

# ─── ANSWER ─────────────────────────

def generate_answer(query, spans, history, groq_key):
    context = "\n".join(
        " ".join(getattr(c, "text", str(c)) for c in span)
        for span in spans
    )
    system = f"{RAG_SYSTEM_PROMPT}\n\nContext:\n{context}"
    return groq_chat(system, query, groq_key)

def fallback_answer(query, groq_key):
    return groq_chat("You are a helpful assistant.", query, groq_key)

# ─── UI ─────────────────────────────

def main():
    st.set_page_config(page_title="RAG App", layout="wide")

    if "config" not in st.session_state:
        st.session_state.config = None
        st.session_state.chat = []
        st.session_state.docs = False

    st.sidebar.title("Settings")

    nvidia = st.text_input("NVIDIA Key", type="password")
    groq = st.text_input("Groq Key", type="password")

    if st.sidebar.button("Init"):
        st.session_state.config = build_config(nvidia, groq, get_db_url())
        st.session_state.groq = groq
        st.success("Ready")

    if not st.session_state.config:
        st.info("Configure first")
        return

    files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)

    if files:
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(f.read())
                path = tmp.name

            success = process_document(path, st.session_state.config)

            if success:
                st.success(f"Indexed: {f.name}")
                st.session_state.docs = True
            else:
                st.error(f"Failed: {f.name}")

            os.remove(path)

    if not st.session_state.docs:
        st.info("Upload document")
        return

    for u, a in st.session_state.chat:
        st.chat_message("user").write(u)
        st.chat_message("assistant").write(a)

    q = st.chat_input("Ask question")

    if q:
        st.chat_message("user").write(q)

        spans = search_and_rerank(q, st.session_state.config)
        st.write("DEBUG spans:", spans)

        if spans:
            st.success("RAG Answer")
            ans = generate_answer(q, spans, st.session_state.chat, st.session_state.groq)
        else:
            st.warning("Fallback (Groq)")
            ans = fallback_answer(q, st.session_state.groq)

        st.chat_message("assistant").write(ans)
        st.session_state.chat.append((q, ans))

if __name__ == "__main__":
    main()
