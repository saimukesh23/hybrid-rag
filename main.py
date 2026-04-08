import os
import logging
import tempfile
from typing import List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── PROMPT ─────────────────────────

RAG_SYSTEM_PROMPT = """
Answer ONLY using the provided context.
Do not hallucinate.
""".strip()

# ─── SECRETS ────────────────────────

def get_secret(key):
    return st.secrets.get(key, "")

# ✅ FORCE DUCKDB (no Neon issues)
def get_db_url():
    return "duckdb:///raglite.duckdb"

# ─── NVIDIA RERANKER ───────────────

class NvidiaReranker:
    def __init__(self, api_key):
        self.api_key = api_key

    def rank(self, query: str, docs: List[str], **_):
        import requests
        try:
            r = requests.post(
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
            r.raise_for_status()
            rankings = r.json().get("rankings", [])
            return [x["index"] for x in sorted(rankings, key=lambda i: i.get("logit", 0), reverse=True)]
        except Exception as e:
            logger.error(f"Reranker error: {e}")
            return list(range(len(docs)))

# ─── GROQ LLM ──────────────────────

def groq_chat(system, user, key):
    from groq import Groq
    try:
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return "Error generating response."

# ─── CONFIG ────────────────────────

def build_config():
    from raglite import RAGLiteConfig

    embed_key = get_secret("NVIDIA_EMBED_API_KEY")
    rerank_key = get_secret("NVIDIA_RERANK_API_KEY")

    # NVIDIA as OpenAI-compatible
    os.environ["OPENAI_API_KEY"] = embed_key
    os.environ["OPENAI_BASE_URL"] = "https://integrate.api.nvidia.com/v1"

    return RAGLiteConfig(
        db_url=get_db_url(),
        llm="groq/llama-3.3-70b-versatile",
        embedder="openai/nvidia/llama-3.2-nv-embedqa-1b-v2",  # ✅ FIXED
        embedder_normalize=True,
        reranker=NvidiaReranker(rerank_key),
    )

# ─── PDF PROCESSING ────────────────

def process_document(path, config):
    try:
        from raglite import insert_documents, Document
        from pypdf import PdfReader

        reader = PdfReader(path)
        text = ""

        for page in reader.pages:
            try:
                text += page.extract_text() or ""
            except:
                continue

        if not text.strip():
            logger.error("No text extracted from PDF")
            return False

        doc = Document(content=text)
        insert_documents([doc], config=config)
        return True

    except Exception as e:
        logger.error(f"Doc error: {e}")
        return False

# ─── SEARCH ─────────────────────────

def search(query, config):
    try:
        from raglite import hybrid_search, rerank_chunks, retrieve_context

        ids, _ = hybrid_search(query, config=config)

        if not ids:
            return []

        ranked = rerank_chunks(query, ids, config=config)
        return retrieve_context(ranked[:5], config=config)

    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

# ─── UI ─────────────────────────────

def main():
    st.set_page_config(page_title="RAG App", layout="wide")
    st.title("👀 RAG App (Hybrid Search)")

    config = build_config()
    groq_key = get_secret("GROQ_API_KEY")

    # ── Upload ──
    files = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=True)

    if files:
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(f.read())
                path = tmp.name

            if process_document(path, config):
                st.success(f"✅ Indexed: {f.name}")
            else:
                st.error(f"❌ Failed: {f.name}")

            os.remove(path)

    # ── Chat ──
    q = st.chat_input("Ask about your document...")

    if q:
        spans = search(q, config)

        st.write("DEBUG:", spans)

        if spans:
            context = "\n\n".join(
                " ".join(getattr(c, "text", str(c)) for c in span)
                for span in spans
            )

            ans = groq_chat(RAG_SYSTEM_PROMPT + "\n\n" + context, q, groq_key)
            st.success("✅ Answer from document (RAG)")

        else:
            ans = groq_chat("You are a helpful assistant.", q, groq_key)
            st.warning("⚠️ No document match (Fallback)")

        st.write(ans)

# ─── RUN ───────────────────────────

if __name__ == "__main__":
    main()
