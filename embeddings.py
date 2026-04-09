---

## **File 3: embeddings.py**
Action: file_editor view /app/embeddings.py
Observation: [Showing lines 1-205 of 205 total] /app/embeddings.py:
1|"""Embedding generation and vector index management."""
2|from typing import List, Dict, Optional
3|import numpy as np
4|from sentence_transformers import SentenceTransformer
5|import faiss
6|
7|try:
8|    from qdrant_client import QdrantClient
9|    from qdrant_client.models import Distance, VectorParams, PointStruct
10|    QDRANT_AVAILABLE = True
11|except ImportError:
12|    QDRANT_AVAILABLE = False
13|
14|class EmbeddingManager:
15|    """
16|    Manages embeddings using SentenceTransformers and vector indexes using FAISS and optionally Qdrant.
17|    """
18|    
19|    def __init__(
20|        self,
21|        model_name: str = "all-MiniLM-L6-v2",
22|        use_qdrant: bool = False,
23|        qdrant_api_key: Optional[str] = None,
24|        qdrant_endpoint: Optional[str] = None,
25|        collection_name: str = "pdf_chunks"
26|    ):
27|        """
28|        Initialize embedding manager.
29|        
30|        Args:
31|            model_name: SentenceTransformer model to use
32|            use_qdrant: Whether to use Qdrant for persistence
33|            qdrant_api_key: Qdrant API key (required if use_qdrant=True)
34|            qdrant_endpoint: Qdrant endpoint URL (required if use_qdrant=True)
35|            collection_name: Name of the Qdrant collection
36|        """
37|        # Load embedding model
38|        self.model = SentenceTransformer(model_name)
39|        self.embedding_dim = self.model.get_sentence_embedding_dimension()
40|        
41|        # FAISS index (always available as fallback)
42|        self.faiss_index = None
43|        self.documents = []
44|        
45|        # Qdrant setup
46|        self.use_qdrant = use_qdrant and QDRANT_AVAILABLE
47|        self.qdrant_client = None
48|        self.collection_name = collection_name
49|        
50|        if self.use_qdrant:
51|            if not qdrant_api_key or not qdrant_endpoint:
52|                print("Warning: Qdrant credentials not provided. Falling back to FAISS.")
53|                self.use_qdrant = False
54|            else:
55|                try:
56|                    self.qdrant_client = QdrantClient(
57|                        url=qdrant_endpoint,
58|                        api_key=qdrant_api_key
59|                    )
60|                    self._initialize_qdrant_collection()
61|                except Exception as e:
62|                    print(f"Warning: Failed to connect to Qdrant: {str(e)}. Falling back to FAISS.")
63|                    self.use_qdrant = False
64|    
65|    def _initialize_qdrant_collection(self):
66|        """Initialize or load Qdrant collection."""
67|        try:
68|            # Check if collection exists
69|            collections = self.qdrant_client.get_collections().collections
70|            collection_exists = any(c.name == self.collection_name for c in collections)
71|            
72|            if not collection_exists:
73|                # Create new collection
74|                self.qdrant_client.create_collection(
75|                    collection_name=self.collection_name,
76|                    vectors_config=VectorParams(
77|                        size=self.embedding_dim,
78|                        distance=Distance.COSINE
79|                    )
80|                )
81|                print(f"Created new Qdrant collection: {self.collection_name}")
82|            else:
83|                print(f"Using existing Qdrant collection: {self.collection_name}")
84|        
85|        except Exception as e:
86|            raise Exception(f"Failed to initialize Qdrant collection: {str(e)}")
87|    
88|    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
89|        """
90|        Generate embeddings for a list of texts.
91|        
92|        Args:
93|            texts: List of text strings
94|            
95|        Returns:
96|            Numpy array of embeddings
97|        """
98|        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
99|        return embeddings
100|    
101|    def build_faiss_index(self, embeddings: np.ndarray):
102|        """
103|        Build FAISS index from embeddings.
104|        
105|        Args:
106|            embeddings: Numpy array of embeddings
107|        """
108|        # Normalize embeddings for cosine similarity
109|        faiss.normalize_L2(embeddings)
110|        
111|        # Create FAISS index
112|        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
113|        self.faiss_index.add(embeddings)
114|    
115|    def upload_to_qdrant(self, embeddings: np.ndarray, documents: List[Dict[str, str]]):
116|        """
117|        Upload embeddings and documents to Qdrant.
118|        
119|        Args:
120|            embeddings: Numpy array of embeddings
121|            documents: List of document dictionaries
122|        """
123|        if not self.use_qdrant or not self.qdrant_client:
124|            return
125|        
126|        try:
127|            # Prepare points for upload
128|            points = []
129|            for i, (embedding, doc) in enumerate(zip(embeddings, documents)):
130|                point = PointStruct(
131|                    id=i,
132|                    vector=embedding.tolist(),
133|                    payload={
134|                        'text': doc['text'],
135|                        'source': doc['metadata'].get('source', 'unknown'),
136|                        'chunk_id': doc['metadata'].get('chunk_id', 0)
137|                    }
138|                )
139|                points.append(point)
140|            
141|            # Upload in batches
142|            batch_size = 100
143|            for i in range(0, len(points), batch_size):
144|                batch = points[i:i + batch_size]
145|                self.qdrant_client.upsert(
146|                    collection_name=self.collection_name,
147|                    points=batch
148|                )
149|            
150|            print(f"Uploaded {len(points)} points to Qdrant")
151|        
152|        except Exception as e:
153|            print(f"Error uploading to Qdrant: {str(e)}")
154|    
155|    def build_index(self, documents: List[Dict[str, str]]):
156|        """
157|        Build vector index from documents.
158|        
159|        Args:
160|            documents: List of document dictionaries with 'text' and 'metadata'
161|        """
162|        self.documents = documents
163|        texts = [doc['text'] for doc in documents]
164|        
165|        # Generate embeddings
166|        print(f"Generating embeddings for {len(texts)} chunks...")
167|        embeddings = self.generate_embeddings(texts)
168|        
169|        # Build FAISS index (always)
170|        print("Building FAISS index...")
171|        self.build_faiss_index(embeddings)
172|        
173|        # Upload to Qdrant if enabled
174|        if self.use_qdrant:
175|            print("Uploading to Qdrant...")
176|            self.upload_to_qdrant(embeddings, documents)
177|        
178|        print("Index built successfully!")
179|    
180|    def search(self, query: str, top_k: int = 5) -> List[tuple]:
181|        """
182|        Search for similar documents using semantic search.
183|        
184|        Args:
185|            query: Query text
186|            top_k: Number of results to return
187|            
188|        Returns:
189|            List of tuples (document_text, similarity_score)
190|        """
191|        # Generate query embedding
192|        query_embedding = self.model.encode([query], convert_to_numpy=True)
193|        faiss.normalize_L2(query_embedding)
194|        
195|        # Search FAISS index
196|        scores, indices = self.faiss_index.search(query_embedding, top_k)
197|        
198|        # Return results
199|        results = []
200|        for score, idx in zip(scores[0], indices[0]):
201|            if idx < len(self.documents):
202|                results.append((self.documents[idx]['text'], float(score)))
203|        
204|        return results
205|
[End of file]
