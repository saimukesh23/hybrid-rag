---

## **File 4: retrieval.py**
Action: file_editor view /app/retrieval.py
Observation: [Showing lines 1-135 of 135 total] /app/retrieval.py:
1|"""Hybrid retrieval combining semantic search (FAISS) and keyword search (BM25)."""
2|from typing import List, Dict, Tuple
3|from rank_bm25 import BM25Okapi
4|import numpy as np
5|
6|class HybridRetriever:
7|    """
8|    Combines semantic search (FAISS) and keyword search (BM25) using Reciprocal Rank Fusion (RRF).
9|    """
10|    
11|    def __init__(
12|        self,
13|        embedding_manager,
14|        documents: List[Dict[str, str]],
15|        top_k: int = 5,
16|        rrf_k: int = 60
17|    ):
18|        """
19|        Initialize hybrid retriever.
20|        
21|        Args:
22|            embedding_manager: EmbeddingManager instance
23|            documents: List of document dictionaries
24|            top_k: Number of results to return
25|            rrf_k: RRF constant (default 60)
26|        """
27|        self.embedding_manager = embedding_manager
28|        self.documents = documents
29|        self.top_k = top_k
30|        self.rrf_k = rrf_k
31|        
32|        # Build BM25 index
33|        self._build_bm25_index()
34|    
35|    def _build_bm25_index(self):
36|        """Build BM25 index for keyword search."""
37|        # Tokenize documents (simple whitespace tokenization)
38|        tokenized_docs = [doc['text'].lower().split() for doc in self.documents]
39|        self.bm25 = BM25Okapi(tokenized_docs)
40|    
41|    def semantic_search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
42|        """
43|        Perform semantic search using FAISS.
44|        
45|        Args:
46|            query: Query text
47|            top_k: Number of results (uses self.top_k if None)
48|            
49|        Returns:
50|            List of tuples (document_text, score)
51|        """
52|        k = top_k if top_k else self.top_k
53|        return self.embedding_manager.search(query, top_k=k)
54|    
55|    def keyword_search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
56|        """
57|        Perform keyword search using BM25.
58|        
59|        Args:
60|            query: Query text
61|            top_k: Number of results (uses self.top_k if None)
62|            
63|        Returns:
64|            List of tuples (document_text, score)
65|        """
66|        k = top_k if top_k else self.top_k
67|        
68|        # Tokenize query
69|        tokenized_query = query.lower().split()
70|        
71|        # Get BM25 scores
72|        scores = self.bm25.get_scores(tokenized_query)
73|        
74|        # Get top-k indices
75|        top_indices = np.argsort(scores)[::-1][:k]
76|        
77|        # Return results
78|        results = []
79|        for idx in top_indices:
80|            if idx < len(self.documents):
81|                results.append((self.documents[idx]['text'], float(scores[idx])))
82|        
83|        return results
84|    
85|    def reciprocal_rank_fusion(
86|        self,
87|        semantic_results: List[Tuple[str, float]],
88|        keyword_results: List[Tuple[str, float]]
89|    ) -> List[Tuple[str, float]]:
90|        """
91|        Combine results using Reciprocal Rank Fusion.
92|        
93|        Args:
94|            semantic_results: Results from semantic search
95|            keyword_results: Results from keyword search
96|            
97|        Returns:
98|            Fused and re-ranked results
99|        """
100|        # Create document to score mapping
101|        doc_scores = {}
102|        
103|        # Add semantic search scores
104|        for rank, (doc, score) in enumerate(semantic_results, 1):
105|            doc_scores[doc] = doc_scores.get(doc, 0) + 1 / (self.rrf_k + rank)
106|        
107|        # Add keyword search scores
108|        for rank, (doc, score) in enumerate(keyword_results, 1):
109|            doc_scores[doc] = doc_scores.get(doc, 0) + 1 / (self.rrf_k + rank)
110|        
111|        # Sort by fused score
112|        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
113|        
114|        # Return top-k results
115|        return sorted_docs[:self.top_k]
116|    
117|    def hybrid_search(self, query: str) -> List[Tuple[str, float]]:
118|        """
119|        Perform hybrid search combining semantic and keyword search.
120|        
121|        Args:
122|            query: Query text
123|            
124|        Returns:
125|            List of tuples (document_text, fused_score)
126|        """
127|        # Perform both searches
128|        semantic_results = self.semantic_search(query, top_k=self.top_k * 2)
129|        keyword_results = self.keyword_search(query, top_k=self.top_k * 2)
130|        
131|        # Fuse results
132|        fused_results = self.reciprocal_rank_fusion(semantic_results, keyword_results)
133|        
134|        return fused_results
135|
[End of file]
