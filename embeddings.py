"""Embedding generation and vector index management."""
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

class EmbeddingManager:
    """
    Manages embeddings using SentenceTransformers and vector indexes using FAISS and optionally Qdrant.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_qdrant: bool = False,
        qdrant_api_key: Optional[str] = None,
        qdrant_endpoint: Optional[str] = None,
        collection_name: str = "pdf_chunks"
    ):
        """
        Initialize embedding manager.
        
        Args:
            model_name: SentenceTransformer model to use
            use_qdrant: Whether to use Qdrant for persistence
            qdrant_api_key: Qdrant API key (required if use_qdrant=True)
            qdrant_endpoint: Qdrant endpoint URL (required if use_qdrant=True)
            collection_name: Name of the Qdrant collection
        """
        # Load embedding model
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # FAISS index (always available as fallback)
        self.faiss_index = None
        self.documents = []
        
        # Qdrant setup
        self.use_qdrant = use_qdrant and QDRANT_AVAILABLE
        self.qdrant_client = None
        self.collection_name = collection_name
        
        if self.use_qdrant:
            if not qdrant_api_key or not qdrant_endpoint:
                print("Warning: Qdrant credentials not provided. Falling back to FAISS.")
                self.use_qdrant = False
            else:
                try:
                    self.qdrant_client = QdrantClient(
                        url=qdrant_endpoint,
                        api_key=qdrant_api_key
                    )
                    self._initialize_qdrant_collection()
                except Exception as e:
                    print(f"Warning: Failed to connect to Qdrant: {str(e)}. Falling back to FAISS.")
                    self.use_qdrant = False
    
    def _initialize_qdrant_collection(self):
        """Initialize or load Qdrant collection."""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(c.name == self.collection_name for c in collections)
            
            if not collection_exists:
                # Create new collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created new Qdrant collection: {self.collection_name}")
            else:
                print(f"Using existing Qdrant collection: {self.collection_name}")
        
        except Exception as e:
            raise Exception(f"Failed to initialize Qdrant collection: {str(e)}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index from embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
        """
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.faiss_index.add(embeddings)
    
    def upload_to_qdrant(self, embeddings: np.ndarray, documents: List[Dict[str, str]]):
        """
        Upload embeddings and documents to Qdrant.
        
        Args:
            embeddings: Numpy array of embeddings
            documents: List of document dictionaries
        """
        if not self.use_qdrant or not self.qdrant_client:
            return
        
        try:
            # Prepare points for upload
            points = []
            for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
                point = PointStruct(
                    id=i,
                    vector=embedding.tolist(),
                    payload={
                        'text': doc['text'],
                        'source': doc['metadata'].get('source', 'unknown'),
                        'chunk_id': doc['metadata'].get('chunk_id', 0)
                    }
                )
                points.append(point)
            
            # Upload in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            print(f"Uploaded {len(points)} points to Qdrant")
        
        except Exception as e:
            print(f"Error uploading to Qdrant: {str(e)}")
    
    def build_index(self, documents: List[Dict[str, str]]):
        """
        Build vector index from documents.
        
        Args:
            documents: List of document dictionaries with 'text' and 'metadata'
        """
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.generate_embeddings(texts)
        
        # Build FAISS index (always)
        print("Building FAISS index...")
        self.build_faiss_index(embeddings)
        
        # Upload to Qdrant if enabled
        if self.use_qdrant:
            print("Uploading to Qdrant...")
            self.upload_to_qdrant(embeddings, documents)
        
        print("Index built successfully!")
    
    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Search for similar documents using semantic search.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of tuples (document_text, similarity_score)
        """
        # Generate query embedding
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx]['text'], float(score)))
        
        return results
