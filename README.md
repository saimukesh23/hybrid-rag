# 👀 RAG App with Hybrid Search

A production-ready document Q&A application using **Hybrid Search (RAG)**, powered by:

| Layer | Provider | Model |
|---|---|---|
| Embeddings | NVIDIA | `nv-embed-v1` |
| Reranking | NVIDIA | `rerank-qa-mistral-4b` |
| LLM | Groq | `llama3-70b-8192` |
| Vector DB | Neon PostgreSQL / SQLite | — |

Built with [RAGLite](https://github.com/superlinear-ai/raglite) and [Streamlit](https://streamlit.io).

---

## Features

- PDF document upload & processing (chunking, embedding, hybrid search)
- NVIDIA-powered dense embeddings + reranking
- Groq LLM for fast, high-quality answers
- Automatic fallback to general knowledge when no document context is found
- Works with Neon PostgreSQL (cloud) or SQLite (local)
- One-click deploy to Streamlit Community Cloud

---

## Prerequisites

### API Keys
| Key | Where to get it |
|---|---|
| `NVIDIA_API_KEY` | [build.nvidia.com](https://build.nvidia.com) → API Keys |
| `GROQ_API_KEY` | [console.groq.com](https://console.groq.com) → API Keys |
| `DATABASE_URL` | [neon.tech](https://neon.tech) → new project → connection string |

---

## Local Development

```bash
# 1. Clone
git clone https://github.com/<you>/hybrid-search-rag.git
cd hybrid-search-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install spaCy model (required by RAGLite)
pip install https://github.com/explosion/spacy-models/releases/download/xx_sent_ud_sm-3.7.0/xx_sent_ud_sm-3.7.0-py3-none-any.whl

# 4. Set environment variables
cp .env.example .env
# Fill in NVIDIA_API_KEY, GROQ_API_KEY, DATABASE_URL in .env

# 5. Run
streamlit run main.py
```

### .env.example

```
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
GROQ_API_KEY=gsk_xxxxxxxxxxxx
DATABASE_URL=sqlite:///raglite.sqlite
```

---

## Deploying to Streamlit Community Cloud

1. Push your repository to GitHub (make sure `.streamlit/secrets.toml` is in `.gitignore`).

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → select your repo.

3. Set **Main file path** to `main.py`.

4. Open **Advanced settings → Secrets** and paste:

```toml
NVIDIA_API_KEY = "nvapi-xxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY   = "gsk_xxxxxxxxxxxxxxxxxxxx"
DATABASE_URL   = "postgresql://user:pass@ep-xyz.region.aws.neon.tech/dbname"
```

5. Click **Deploy** — that's it!

> **Note**: Streamlit Cloud provisions a fresh environment on every deploy. API keys are read from `st.secrets`; no `.env` file is needed in production.

---

## Project Structure

```
.
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── secrets.toml         # Local secrets (git-ignored)
├── .env.example             # Template for local env vars
└── README.md
```

---

## How It Works

1. **Upload PDFs** — documents are chunked and embedded with NVIDIA `nv-embed-v1`.
2. **Hybrid Search** — BM25 keyword search + dense vector search are combined.
3. **Reranking** — NVIDIA `rerank-qa-mistral-4b` picks the top-5 most relevant chunks.
4. **Answer Generation** — Groq `llama3-70b-8192` synthesises the answer from context.
5. **Fallback** — If no relevant chunks are found, the question goes directly to the LLM.

---

## Database Options

| Backend | Connection string |
|---|---|
| Neon PostgreSQL (recommended) | `postgresql://user:pass@ep-xyz.region.aws.neon.tech/dbname` |
| Local PostgreSQL | `postgresql://user:pass@localhost:5432/dbname` |
| SQLite (dev only) | `sqlite:///raglite.sqlite` |

SSL is automatically appended for PostgreSQL connections that don't already include `sslmode`.

---

## Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.
