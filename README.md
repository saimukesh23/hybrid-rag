## 📖 Usage

1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
2. **Wait for Processing**: The app will chunk and index your documents
3. **Ask Questions**: Type your question in the chat input
4. **View Answers**: Get streaming responses based on your documents
5. **Check Sources**: Expand "View relevant chunks" to see source passages

## 🏗️ Architecture

```
app.py              # Main Streamlit application
pdf_utils.py        # PDF loading and chunking (400 tokens, 50 overlap)
embeddings.py       # Embedding generation and indexing (FAISS + Qdrant)
retrieval.py        # Hybrid search (FAISS + BM25 + RRF)
qa.py               # Groq API integration with streaming
requirements.txt    # Python dependencies
```

### How It Works

1. **PDF Processing**: PDFs are loaded and split into 400-token chunks with 50-token overlap using tiktoken
2. **Embedding Generation**: Each chunk is embedded using `all-MiniLM-L6-v2`
3. **Indexing**: 
   - FAISS: In-memory vector index for semantic search
   - BM25: Keyword-based index for exact matching
   - Qdrant (optional): Persistent vector storage
4. **Query Processing**:
   - Semantic search via FAISS (cosine similarity)
   - Keyword search via BM25
   - Results combined using Reciprocal Rank Fusion
5. **Answer Generation**:
   - Top 5 chunks passed to Groq LLM
   - Streaming response with conversation history
   - Fallback for low-confidence results

## 🎛️ Configuration

You can customize these parameters in the code:

- **Chunk size**: 400 tokens (in `pdf_utils.py`)
- **Chunk overlap**: 50 tokens (in `pdf_utils.py`)
- **Top-K results**: 5 (in `app.py`)
- **Similarity threshold**: 0.3 (in `qa.py`)
- **Groq model**: llama-3.3-70b-versatile (in `qa.py`)
- **Temperature**: 0.3 (in `qa.py`)
- **Max tokens**: 1024 (in `qa.py`)
- **Conversation history**: Last 10 messages (in `app.py`)

## 🐛 Troubleshooting

### "GROQ_API_KEY not found"
- Ensure `.streamlit/secrets.toml` exists and contains your API key
- For Streamlit Cloud, check App Settings → Secrets

### "No text extracted from PDFs"
- Check if PDF is password-protected
- Try a different PDF file
- Ensure PDF contains extractable text (not scanned images)

### "Failed to connect to Qdrant"
- Verify QDRANT_API_KEY and QDRANT_ENDPOINT are correct
- Check Qdrant cluster is running
- App will automatically fallback to FAISS if Qdrant fails

### Memory issues with large PDFs
- Try processing fewer PDFs at once
- Reduce chunk size in `pdf_utils.py`
- Use Qdrant for persistence instead of in-memory FAISS

## 📝 License

MIT License - Feel free to use and modify!

## 🙏 Acknowledgments

- [Groq](https://groq.com) for blazing-fast LLM inference
- [SentenceTransformers](https://www.sbert.net/) for embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Qdrant](https://qdrant.tech/) for vector database
- [Streamlit](https://streamlit.io) for the amazing framework

---

**Ready to deploy!** This app is production-ready and optimized for Streamlit Cloud. 🚀
