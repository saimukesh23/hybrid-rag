I'll provide you with all the files and their complete code. Copy each one exactly as shown:

---

## **File 1: app.py**
Action: file_editor view /app/app.py
Observation: [Showing lines 1-218 of 218 total] /app/app.py:
1|import streamlit as st
2|import os
3|from pdf_utils import load_and_chunk_pdfs
4|from embeddings import EmbeddingManager
5|from retrieval import HybridRetriever
6|from qa import QASystem
7|import tempfile
8|
9|# Page configuration
10|st.set_page_config(
11|    page_title="Hybrid PDF Q&A System",
12|    page_icon="📚",
13|    layout="wide"
14|)
15|
16|# Title and description
17|st.title("📚 Hybrid PDF Q&A System")
18|st.markdown("Upload PDFs and ask questions using advanced hybrid search (FAISS + BM25 + optional Qdrant)")
19|
20|# Initialize session state
21|if 'documents' not in st.session_state:
22|    st.session_state.documents = None
23|if 'embedding_manager' not in st.session_state:
24|    st.session_state.embedding_manager = None
25|if 'retriever' not in st.session_state:
26|    st.session_state.retriever = None
27|if 'qa_system' not in st.session_state:
28|    st.session_state.qa_system = None
29|if 'chat_history' not in st.session_state:
30|    st.session_state.chat_history = []
31|if 'processed_files' not in st.session_state:
32|    st.session_state.processed_files = []
33|
34|# Check for Groq API key
35|try:
36|    groq_api_key = st.secrets["GROQ_API_KEY"]
37|except (KeyError, FileNotFoundError):
38|    st.error("⚠️ GROQ_API_KEY not found in Streamlit secrets. Please add it to continue.")
39|    st.info("Go to Streamlit Cloud dashboard → App settings → Secrets → Add GROQ_API_KEY")
40|    st.stop()
41|
42|# Check for optional Qdrant keys
43|try:
44|    qdrant_api_key = st.secrets.get("QDRANT_API_KEY", None)
45|    qdrant_endpoint = st.secrets.get("QDRANT_ENDPOINT", None)
46|    use_qdrant = bool(qdrant_api_key and qdrant_endpoint)
47|except:
48|    use_qdrant = False
49|
50|if use_qdrant:
51|    st.sidebar.success("✅ Qdrant integration enabled")
52|else:
53|    st.sidebar.info("ℹ️ Using FAISS in-memory (add QDRANT_API_KEY & QDRANT_ENDPOINT for persistence)")
54|
55|# Sidebar - File upload
56|st.sidebar.header("📁 Upload PDFs")
57|uploaded_files = st.sidebar.file_uploader(
58|    "Choose PDF files",
59|    type="pdf",
60|    accept_multiple_files=True,
61|    help="Upload one or more PDF files to ask questions about"
62|)
63|
64|# Process uploaded files
65|if uploaded_files:
66|    current_files = [f.name for f in uploaded_files]
67|    
68|    # Check if files have changed
69|    if current_files != st.session_state.processed_files:
70|        with st.spinner("🔄 Processing PDFs..."):
71|            try:
72|                # Save uploaded files temporarily
73|                temp_paths = []
74|                for uploaded_file in uploaded_files:
75|                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
76|                        tmp_file.write(uploaded_file.read())
77|                        temp_paths.append(tmp_file.name)
78|                
79|                # Load and chunk PDFs
80|                documents = load_and_chunk_pdfs(temp_paths, chunk_size=400, overlap=50)
81|                
82|                # Clean up temp files
83|                for path in temp_paths:
84|                    try:
85|                        os.unlink(path)
86|                    except:
87|                        pass
88|                
89|                if not documents:
90|                    st.error("❌ No text extracted from PDFs. Please check your files.")
91|                    st.stop()
92|                
93|                st.session_state.documents = documents
94|                st.sidebar.success(f"✅ Processed {len(uploaded_files)} PDFs into {len(documents)} chunks")
95|                
96|                # Initialize embedding manager
97|                embedding_manager = EmbeddingManager(
98|                    use_qdrant=use_qdrant,
99|                    qdrant_api_key=qdrant_api_key if use_qdrant else None,
100|                    qdrant_endpoint=qdrant_endpoint if use_qdrant else None
101|                )
102|                
103|                # Build or load index
104|                embedding_manager.build_index(documents)
105|                st.session_state.embedding_manager = embedding_manager
106|                
107|                # Initialize retriever
108|                retriever = HybridRetriever(
109|                    embedding_manager=embedding_manager,
110|                    documents=documents,
111|                    top_k=5
112|                )
113|                st.session_state.retriever = retriever
114|                
115|                # Initialize QA system
116|                qa_system = QASystem(
117|                    groq_api_key=groq_api_key,
118|                    retriever=retriever
119|                )
120|                st.session_state.qa_system = qa_system
121|                
122|                # Reset chat history for new documents
123|                st.session_state.chat_history = []
124|                st.session_state.processed_files = current_files
125|                
126|            except Exception as e:
127|                st.error(f"❌ Error processing PDFs: {str(e)}")
128|                st.stop()
129|
130|# Main content area
131|if st.session_state.documents:
132|    st.sidebar.markdown("---")
133|    st.sidebar.markdown(f"**📊 Statistics:**")
134|    st.sidebar.markdown(f"- Total chunks: {len(st.session_state.documents)}")
135|    st.sidebar.markdown(f"- Files: {len(st.session_state.processed_files)}")
136|    
137|    # Display chat history
138|    st.markdown("### 💬 Conversation")
139|    for message in st.session_state.chat_history:
140|        with st.chat_message(message["role"]):
141|            st.markdown(message["content"])
142|    
143|    # Question input
144|    question = st.chat_input("Ask a question about your PDFs...")
145|    
146|    if question:
147|        # Display user message
148|        with st.chat_message("user"):
149|            st.markdown(question)
150|        
151|        # Add to chat history
152|        st.session_state.chat_history.append({"role": "user", "content": question})
153|        
154|        # Generate answer with streaming
155|        with st.chat_message("assistant"):
156|            message_placeholder = st.empty()
157|            full_response = ""
158|            
159|            try:
160|                # Stream the response
161|                for chunk in st.session_state.qa_system.answer_question(
162|                    question,
163|                    chat_history=st.session_state.chat_history[-11:]  # Last 5-6 Q&A pairs
164|                ):
165|                    full_response += chunk
166|                    message_placeholder.markdown(full_response + "▌")
167|                
168|                message_placeholder.markdown(full_response)
169|                
170|                # Add to chat history
171|                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
172|                
173|                # Show relevant chunks in expander (for debugging)
174|                with st.expander("🔍 View relevant chunks"):
175|                    results = st.session_state.retriever.hybrid_search(question)
176|                    for i, (doc, score) in enumerate(results[:3], 1):
177|                        st.markdown(f"**Chunk {i}** (Score: {score:.3f})")
178|                        st.text(doc[:300] + "..." if len(doc) > 300 else doc)
179|                        st.markdown("---")
180|            
181|            except Exception as e:
182|                st.error(f"❌ Error generating answer: {str(e)}")
183|    
184|    # Clear chat button
185|    if st.sidebar.button("🗑️ Clear Chat History"):
186|        st.session_state.chat_history = []
187|        st.rerun()
188|
189|else:
190|    # Welcome message
191|    st.info("👈 Please upload PDF files from the sidebar to get started")
192|    
193|    st.markdown("### 🚀 Features")
194|    col1, col2 = st.columns(2)
195|    
196|    with col1:
197|        st.markdown("""
198|        - 📄 **Multi-PDF Support**: Upload multiple PDFs
199|        - 🔍 **Hybrid Search**: FAISS + BM25 + RRF
200|        - 🧠 **Semantic Understanding**: MiniLM embeddings
201|        """)
202|    
203|    with col2:
204|        st.markdown("""
205|        - 💾 **Optional Persistence**: Qdrant integration
206|        - 💬 **Multi-turn Conversations**: Context-aware responses
207|        - ⚡ **Streaming Responses**: Real-time answer generation
208|        """)
209|
210|# Footer
211|st.sidebar.markdown("---")
212|st.sidebar.markdown("**Powered by:**")
213|st.sidebar.markdown("- Groq API")
214|st.sidebar.markdown("- SentenceTransformers")
215|st.sidebar.markdown("- FAISS + BM25")
216|if use_qdrant:
217|    st.sidebar.markdown("- Qdrant Vector DB")
218|
[End of file]
