import streamlit as st
import os
from pdf_utils import load_and_chunk_pdfs
from embeddings import EmbeddingManager
from retrieval import HybridRetriever
from qa import QASystem
import tempfile

# Page configuration
st.set_page_config(
    page_title="Hybrid PDF Q&A System",
    page_icon="📚",
    layout="wide"
)

# Title and description
st.title("📚 Hybrid PDF Q&A System")
st.markdown("Upload PDFs and ask questions using advanced hybrid search (FAISS + BM25 + optional Qdrant)")

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'embedding_manager' not in st.session_state:
    st.session_state.embedding_manager = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Check for Groq API key
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("⚠️ GROQ_API_KEY not found in Streamlit secrets. Please add it to continue.")
    st.info("Go to Streamlit Cloud dashboard → App settings → Secrets → Add GROQ_API_KEY")
    st.stop()

# Check for optional Qdrant keys
try:
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY", None)
    qdrant_endpoint = st.secrets.get("QDRANT_ENDPOINT", None)
    use_qdrant = bool(qdrant_api_key and qdrant_endpoint)
except:
    use_qdrant = False

if use_qdrant:
    st.sidebar.success("✅ Qdrant integration enabled")
else:
    st.sidebar.info("ℹ️ Using FAISS in-memory (add QDRANT_API_KEY & QDRANT_ENDPOINT for persistence)")

# Sidebar - File upload
st.sidebar.header("📁 Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type="pdf",
    accept_multiple_files=True,
    help="Upload one or more PDF files to ask questions about"
)

# Process uploaded files
if uploaded_files:
    current_files = [f.name for f in uploaded_files]
    
    # Check if files have changed
    if current_files != st.session_state.processed_files:
        with st.spinner("🔄 Processing PDFs..."):
            try:
                # Save uploaded files temporarily
                temp_paths = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_paths.append(tmp_file.name)
                
                # Load and chunk PDFs
                documents = load_and_chunk_pdfs(temp_paths, chunk_size=400, overlap=50)
                
                # Clean up temp files
                for path in temp_paths:
                    try:
                        os.unlink(path)
                    except:
                        pass
                
                if not documents:
                    st.error("❌ No text extracted from PDFs. Please check your files.")
                    st.stop()
                
                st.session_state.documents = documents
                st.sidebar.success(f"✅ Processed {len(uploaded_files)} PDFs into {len(documents)} chunks")
                
                # Initialize embedding manager
                embedding_manager = EmbeddingManager(
                    use_qdrant=use_qdrant,
                    qdrant_api_key=qdrant_api_key if use_qdrant else None,
                    qdrant_endpoint=qdrant_endpoint if use_qdrant else None
                )
                
                # Build or load index
                embedding_manager.build_index(documents)
                st.session_state.embedding_manager = embedding_manager
                
                # Initialize retriever
                retriever = HybridRetriever(
                    embedding_manager=embedding_manager,
                    documents=documents,
                    top_k=5
                )
                st.session_state.retriever = retriever
                
                # Initialize QA system
                qa_system = QASystem(
                    groq_api_key=groq_api_key,
                    retriever=retriever
                )
                st.session_state.qa_system = qa_system
                
                # Reset chat history for new documents
                st.session_state.chat_history = []
                st.session_state.processed_files = current_files
                
            except Exception as e:
                st.error(f"❌ Error processing PDFs: {str(e)}")
                st.stop()

# Main content area
if st.session_state.documents:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**📊 Statistics:**")
    st.sidebar.markdown(f"- Total chunks: {len(st.session_state.documents)}")
    st.sidebar.markdown(f"- Files: {len(st.session_state.processed_files)}")
    
    # Display chat history
    st.markdown("### 💬 Conversation")
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Question input
    question = st.chat_input("Ask a question about your PDFs...")
    
    if question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Generate answer with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Stream the response
                for chunk in st.session_state.qa_system.answer_question(
                    question,
                    chat_history=st.session_state.chat_history[-11:]  # Last 5-6 Q&A pairs
                ):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                
                # Show relevant chunks in expander (for debugging)
                with st.expander("🔍 View relevant chunks"):
                    results = st.session_state.retriever.hybrid_search(question)
                    for i, (doc, score) in enumerate(results[:3], 1):
                        st.markdown(f"**Chunk {i}** (Score: {score:.3f})")
                        st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                        st.markdown("---")
            
            except Exception as e:
                st.error(f"❌ Error generating answer: {str(e)}")
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

else:
    # Welcome message
    st.info("👈 Please upload PDF files from the sidebar to get started")
    
    st.markdown("### 🚀 Features")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - 📄 **Multi-PDF Support**: Upload multiple PDFs
        - 🔍 **Hybrid Search**: FAISS + BM25 + RRF
        - 🧠 **Semantic Understanding**: MiniLM embeddings
        """)
    
    with col2:
        st.markdown("""
        - 💾 **Optional Persistence**: Qdrant integration
        - 💬 **Multi-turn Conversations**: Context-aware responses
        - ⚡ **Streaming Responses**: Real-time answer generation
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by:**")
st.sidebar.markdown("- Groq API")
st.sidebar.markdown("- SentenceTransformers")
st.sidebar.markdown("- FAISS + BM25")
if use_qdrant:
    st.sidebar.markdown("- Qdrant Vector DB")
