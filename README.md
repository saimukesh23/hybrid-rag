
Action: file_editor view /app/README.md
Observation: [Showing lines 101-186 of 186 total] /app/README.md:
101|## 📖 Usage
102|
103|1. **Upload PDFs**: Use the sidebar to upload one or more PDF files
104|2. **Wait for Processing**: The app will chunk and index your documents
105|3. **Ask Questions**: Type your question in the chat input
106|4. **View Answers**: Get streaming responses based on your documents
107|5. **Check Sources**: Expand "View relevant chunks" to see source passages
108|
109|## 🏗️ Architecture
110|
111|```
112|├── app.py              # Main Streamlit application
113|├── pdf_utils.py        # PDF loading and chunking (400 tokens, 50 overlap)
114|├── embeddings.py       # Embedding generation and indexing (FAISS + Qdrant)
115|├── retrieval.py        # Hybrid search (FAISS + BM25 + RRF)
116|├── qa.py               # Groq API integration with streaming
117|└── requirements.txt    # Python dependencies
118|```
119|
120|### How It Works
121|
122|1. **PDF Processing**: PDFs are loaded and split into 400-token chunks with 50-token overlap using tiktoken
123|2. **Embedding Generation**: Each chunk is embedded using `all-MiniLM-L6-v2`
124|3. **Indexing**: 
125|   - FAISS: In-memory vector index for semantic search
126|   - BM25: Keyword-based index for exact matching
127|   - Qdrant (optional): Persistent vector storage
128|4. **Query Processing**:
129|   - Semantic search via FAISS (cosine similarity)
130|   - Keyword search via BM25
131|   - Results combined using Reciprocal Rank Fusion
132|5. **Answer Generation**:
133|   - Top 5 chunks passed to Groq LLM
134|   - Streaming response with conversation history
135|   - Fallback for low-confidence results
136|
137|## 🎛️ Configuration
138|
139|You can customize these parameters in the code:
140|
141|- **Chunk size**: 400 tokens (in `pdf_utils.py`)
142|- **Chunk overlap**: 50 tokens (in `pdf_utils.py`)
143|- **Top-K results**: 5 (in `app.py`)
144|- **Similarity threshold**: 0.3 (in `qa.py`)
145|- **Groq model**: llama-3.3-70b-versatile (in `qa.py`)
146|- **Temperature**: 0.3 (in `qa.py`)
147|- **Max tokens**: 1024 (in `qa.py`)
148|- **Conversation history**: Last 10 messages (in `app.py`)
149|
150|## 🐛 Troubleshooting
151|
152|### "GROQ_API_KEY not found"
153|- Ensure `.streamlit/secrets.toml` exists and contains your API key
154|- For Streamlit Cloud, check App Settings → Secrets
155|
156|### "No text extracted from PDFs"
157|- Check if PDF is password-protected
158|- Try a different PDF file
159|- Ensure PDF contains extractable text (not scanned images)
160|
161|### "Failed to connect to Qdrant"
162|- Verify QDRANT_API_KEY and QDRANT_ENDPOINT are correct
163|- Check Qdrant cluster is running
164|- App will automatically fallback to FAISS if Qdrant fails
165|
166|### Memory issues with large PDFs
167|- Try processing fewer PDFs at once
168|- Reduce chunk size in `pdf_utils.py`
169|- Use Qdrant for persistence instead of in-memory FAISS
170|
171|## 📝 License
172|
173|MIT License - Feel free to use and modify!
174|
175|## 🙏 Acknowledgments
176|
177|- [Groq](https://groq.com) for blazing-fast LLM inference
178|- [SentenceTransformers](https://www.sbert.net/) for embeddings
179|- [FAISS](https://github.com/facebookresearch/faiss) for vector search
180|- [Qdrant](https://qdrant.tech/) for vector database
181|- [Streamlit](https://streamlit.io) for the amazing framework
182|
183|---
184|
185|**Ready to deploy!** This app is production-ready and optimized for Streamlit Cloud. 🚀
186|
[End of file]
