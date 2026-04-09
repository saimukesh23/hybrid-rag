"""Hybrid retrieval combining semantic search (FAISS) and keyword search (BM25)."""
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    """
    Combines semantic search (FAISS) and keyword search (BM25) using Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(
        self,
        embedding_manager,
        documents: List[Dict[str, str]],
        top_k: int = 5,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            embedding_manager: EmbeddingManager instance
            documents: List of document dictionaries
            top_k: Number of results to return
            rrf_k: RRF constant (default 60)
        """
        self.embedding_manager = embedding_manager
        self.documents = documents
        self.top_k = top_k
        self.rrf_k = rrf_k
        
        # Build BM25 index
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index for keyword search."""
        # Tokenize documents (simple whitespace tokenization)
        tokenized_docs = [doc['text'].lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def semantic_search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Perform semantic search using FAISS.
        
        Args:
            query: Query text
            top_k: Number of results (uses self.top_k if None)
            
        Returns:
            List of tuples (document_text, score)
        """
        k = top_k if top_k else self.top_k
        return self.embedding_manager.search(query, top_k=k)
    
    def keyword_search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Perform keyword search using BM25.
        
        Args:
            query: Query text
            top_k: Number of results (uses self.top_k if None)
            
        Returns:
            List of tuples (document_text, score)
        """
        k = top_k if top_k else self.top_k
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Return results
        results = []
        for idx in top_indices:
            if idx < len(self.documents) and scores[idx] > 0:
                results.append((self.documents[idx]['text'], float(scores[idx])))
        
        return results
    
    def reciprocal_rank_fusion(
        self,
        semantic_results: List[Tuple[str, float]],
        keyword_results: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            
        Returns:
            Fused and re-ranked results
        """
        # Create document to score mapping
        doc_scores = {}
        
        # Add semantic search scores
        for rank, (doc, score) in enumerate(semantic_results, 1):
            doc_scores[doc] = doc_scores.get(doc, 0) + 1 / (self.rrf_k + rank)
        
        # Add keyword search scores
        for rank, (doc, score) in enumerate(keyword_results, 1):
            doc_scores[doc] = doc_scores.get(doc, 0) + 1 / (self.rrf_k + rank)
        
        # Sort by fused score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        return sorted_docs[:self.top_k]
    
    def hybrid_search(self, query: str) -> List[Tuple[str, float]]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Query text
            
        Returns:
            List of tuples (document_text, fused_score)
        """
        # Perform both searches
        semantic_results = self.semantic_search(query, top_k=self.top_k * 2)
        keyword_results = self.keyword_search(query, top_k=self.top_k * 2)
        
        # Fuse results
        fused_results = self.reciprocal_rank_fusion(semantic_results, keyword_results)
        
        return fused_results
