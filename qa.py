"""Q&A system using Groq API with streaming support."""
from typing import List, Dict, Generator
from groq import Groq

class QASystem:
    """
    Question-answering system using Groq API with streaming and fallback logic.
    """
    
    def __init__(
        self,
        groq_api_key: str,
        retriever,
        model: str = "llama-3.3-70b-versatile",
        similarity_threshold: float = 0.3
    ):
        """
        Initialize QA system.
        
        Args:
            groq_api_key: Groq API key
            retriever: HybridRetriever instance
            model: Groq model to use
            similarity_threshold: Minimum similarity score for relevant results
        """
        self.client = Groq(api_key=groq_api_key)
        self.retriever = retriever
        self.model = model
        self.similarity_threshold = similarity_threshold
    
    def build_context(self, query: str) -> tuple:
        """
        Retrieve relevant context for the query.
        
        Args:
            query: User question
            
        Returns:
            Tuple of (context_text, max_score)
        """
        # Perform hybrid search
        results = self.retriever.hybrid_search(query)
        
        if not results:
            return "", 0.0
        
        # Get max score
        max_score = results[0][1] if results else 0.0
        
        # Build context from top results
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[Chunk {i}]\n{doc}\n")
        
        context = "\n".join(context_parts)
        return context, max_score
    
    def build_messages(
        self,
        query: str,
        context: str,
        chat_history: List[Dict[str, str]] = None
    ) -> List[Dict[str, str]]:
        """
        Build message history for Groq API.
        
        Args:
            query: Current user question
            context: Retrieved context
            chat_history: Previous conversation (last 5-6 Q&A pairs)
            
        Returns:
            List of message dictionaries
        """
        messages = []
        
        # System message
        system_message = """You are a helpful AI assistant that answers questions based on provided PDF documents.

Instructions:
- Answer questions accurately using ONLY the provided context
- If the context doesn't contain relevant information, say "I don't have enough information to answer that question based on the provided documents."
- Be concise and clear in your responses
- Cite relevant information from the context when answering
- If asked about something not in the context, politely decline"""
        
        messages.append({"role": "system", "content": system_message})
        
        # Add recent chat history (last 5-6 Q&A pairs = 10-12 messages)
        if chat_history:
            for msg in chat_history[-10:]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append(msg)
        
        # Add current query with context
        user_message = f"""Context from PDF documents:
{context}

Question: {query}

Please answer the question based on the context provided above."""
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def answer_question(
        self,
        query: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Generator[str, None, None]:
        """
        Answer a question with streaming response.
        
        Args:
            query: User question
            chat_history: Previous conversation history
            
        Yields:
            Chunks of the response text
        """
        try:
            # Retrieve context
            context, max_score = self.build_context(query)
            
            # Check if results are relevant
            if not context or max_score < self.similarity_threshold:
                yield "I don't have enough relevant information in the provided documents to answer that question. Please try rephrasing your question or ask about topics covered in the uploaded PDFs."
                return
            
            # Build messages
            messages = self.build_messages(query, context, chat_history)
            
            # Stream response from Groq
            stream = self.client.chat.completions.create(
                messages=messages,
                model=self.model,
                max_tokens=1024,
                temperature=0.3,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            yield f"Error generating answer: {str(e)}"
