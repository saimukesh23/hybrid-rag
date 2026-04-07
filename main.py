import os
import logging
import streamlit as st
from raglite import RAGLiteConfig, insert_document, hybrid_search, retrieve_chunks, rerank_chunks, rag
from typing import List
from pathlib import Path
import time
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

RAG_SYSTEM_PROMPT = """
You are a friendly and knowledgeable assistant that provides complete and insightful answers.
Answer the user's question using only the context below.
When responding, you MUST NOT reference the existence of the context, directly or indirectly.
Instead, you MUST treat the context as if its contents are entirely part of your working memory.
""".strip()

# ✅ NEW: NVIDIA RERANKER (replaces Cohere)
class NvidiaReranker:
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
                "model": "rerank-qa-mistral-4b",
                "query": {"role": "user", "content": query},
                "passages": [{"role": "user", "content": d} for d in docs],
            },
            timeout=60,
        )
        resp.raise_for_status()
        rankings = resp.json().get("rankings", [])
        return [r["index"] for r in sorted(rankings, key=lambda x: x.get("logit", 0), reverse=True)]

# ✅ NEW: NVIDIA EMBEDDINGS (replaces OpenAI)
def nvidia_embed(texts: List[str], api_key: str) -> List[List[float]]:
    import requests
    resp = requests.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "nv-embed-v1",
            "input": texts,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return [item["embedding"] for item in resp.json()["data"]]

# ✅ NEW: GROQ (replaces Claude)
def groq_chat(prompt: str, system_prompt: str, groq_key: str) -> str:
    from groq import Groq
    client = Groq(api_key=groq_key)
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.7,
    )
    return completion.choices[0].message.content


def initialize_config(nvidia_embed_key: str, groq_key: str, nvidia_rerank_key: str, db_url: str) -> RAGLiteConfig:
    try:
        os.environ["NVIDIA_API_KEY"] = nvidia_embed_key
        os.environ["GROQ_API_KEY"] = groq_key

        def embedder_fn(texts: List[str]):
            return nvidia_embed(texts, nvidia_embed_key)

        return RAGLiteConfig(
            db_url=db_url,
            llm="groq/llama3-70b-8192",  # ✅ changed from Claude
            embedder=embedder_fn,        # ✅ changed from OpenAI
            embedder_normalize=True,
            chunk_max_size=2000,
            embedder_sentence_window_size=2,
            reranker=NvidiaReranker(api_key=nvidia_rerank_key)  # ✅ changed from Cohere
        )
    except Exception as e:
        raise ValueError(f"Configuration error: {e}")


def process_document(file_path: str) -> bool:
    try:
        if not st.session_state.get('my_config'):
            raise ValueError("Configuration not initialized")
        insert_document(Path(file_path), config=st.session_state.my_config)
        return True
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return False


def perform_search(query: str) -> List[dict]:
    try:
        chunk_ids, scores = hybrid_search(query, num_results=10, config=st.session_state.my_config)
        if not chunk_ids:
            return []
        chunks = retrieve_chunks(chunk_ids, config=st.session_state.my_config)
        return rerank_chunks(query, chunks, config=st.session_state.my_config)
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []


# ✅ UPDATED fallback (Claude → Groq)
def handle_fallback(query: str) -> str:
    try:
        return groq_chat(
            prompt=query,
            system_prompt="You are a helpful AI assistant. Give clear answers.",
            groq_key=st.session_state.user_env["GROQ_API_KEY"]
        )
    except Exception as e:
        logger.error(f"Fallback error: {str(e)}")
        return "Error occurred."


def main():
    st.set_page_config(page_title="LLM-Powered Hybrid Search-RAG Assistant", layout="wide")

    for state_var in ['chat_history', 'documents_loaded', 'my_config', 'user_env']:
        if state_var not in st.session_state:
            st.session_state[state_var] = [] if state_var == 'chat_history' else False if state_var == 'documents_loaded' else None if state_var == 'my_config' else {}

    with st.sidebar:
        st.title("Configuration")

        # ✅ UPDATED INPUTS
        nvidia_embed_key = st.text_input("NVIDIA Embed API Key", type="password")
        groq_key = st.text_input("Groq API Key", type="password")
        nvidia_rerank_key = st.text_input("NVIDIA Rerank API Key", type="password")
        db_url = st.text_input("Database URL", value="sqlite:///raglite.sqlite")

        if st.button("Save Configuration"):
            try:
                if not all([nvidia_embed_key, groq_key, nvidia_rerank_key, db_url]):
                    st.error("All fields are required!")
                    return

                st.session_state.my_config = initialize_config(
                    nvidia_embed_key,
                    groq_key,
                    nvidia_rerank_key,
                    db_url
                )

                st.session_state.user_env = {"GROQ_API_KEY": groq_key}
                st.success("Configuration saved successfully!")
            except Exception as e:
                st.error(f"Configuration error: {str(e)}")

    st.title("👀 RAG App with Hybrid Search")

    if st.session_state.my_config:
        uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

        if uploaded_files:
            success = False
            for uploaded_file in uploaded_files:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                if process_document(temp_path):
                    st.success(f"Successfully processed: {uploaded_file.name}")
                    success = True
                else:
                    st.error(f"Failed to process: {uploaded_file.name}")

                os.remove(temp_path)

            if success:
                st.session_state.documents_loaded = True

    if st.session_state.documents_loaded:
        user_input = st.chat_input("Ask a question...")
        if user_input:
            with st.chat_message("assistant"):
                reranked_chunks = perform_search(user_input)
                if not reranked_chunks:
                    response = handle_fallback(user_input)
                else:
                    response = rag(
                        prompt=user_input,
                        system_prompt=RAG_SYSTEM_PROMPT,
                        search=hybrid_search,
                        max_contexts=5,
                        config=st.session_state.my_config
                    )
                st.write(response)


if __name__ == "__main__":
    main()
