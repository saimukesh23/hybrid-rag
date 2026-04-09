---

## **File 5: qa.py**
Action: file_editor view /app/qa.py
Observation: [Showing lines 1-149 of 149 total] /app/qa.py:
1|"""Q&A system using Groq API with streaming support."""
2|from typing import List, Dict, Generator
3|from groq import Groq
4|
5|class QASystem:
6|    """
7|    Question-answering system using Groq API with streaming and fallback logic.
8|    """
9|    
10|    def __init__(
11|        self,
12|        groq_api_key: str,
13|        retriever,
14|        model: str = "llama-3.3-70b-versatile",
15|        similarity_threshold: float = 0.3
16|    ):
17|        """
18|        Initialize QA system.
19|        
20|        Args:
21|            groq_api_key: Groq API key
22|            retriever: HybridRetriever instance
23|            model: Groq model to use
24|            similarity_threshold: Minimum similarity score for relevant results
25|        """
26|        self.client = Groq(api_key=groq_api_key)
27|        self.retriever = retriever
28|        self.model = model
29|        self.similarity_threshold = similarity_threshold
30|    
31|    def build_context(self, query: str) -> tuple:
32|        """
33|        Retrieve relevant context for the query.
34|        
35|        Args:
36|            query: User question
37|            
38|        Returns:
39|            Tuple of (context_text, max_score)
40|        """
41|        # Perform hybrid search
42|        results = self.retriever.hybrid_search(query)
43|        
44|        if not results:
45|            return "", 0.0
46|        
47|        # Get max score
48|        max_score = results[0][1] if results else 0.0
49|        
50|        # Build context from top results
51|        context_parts = []
52|        for i, (doc, score) in enumerate(results, 1):
53|            context_parts.append(f"[Chunk {i}]\n{doc}\n")
54|        
55|        context = "\n".join(context_parts)
56|        return context, max_score
57|    
58|    def build_messages(
59|        self,
60|        query: str,
61|        context: str,
62|        chat_history: List[Dict[str, str]] = None
63|    ) -> List[Dict[str, str]]:
64|        """
65|        Build message history for Groq API.
66|        
67|        Args:
68|            query: Current user question
69|            context: Retrieved context
70|            chat_history: Previous conversation (last 5-6 Q&A pairs)
71|            
72|        Returns:
73|            List of message dictionaries
74|        """
75|        messages = []
76|        
77|        # System message
78|        system_message = """You are a helpful AI assistant that answers questions based on provided PDF documents.
79|
80|Instructions:
81|- Answer questions accurately using ONLY the provided context
82|- If the context doesn't contain relevant information, say "I don't have enough information to answer that question based on the provided documents."
83|- Be concise and clear in your responses
84|- Cite relevant information from the context when answering
85|- If asked about something not in the context, politely decline"""
86|        
87|        messages.append({"role": "system", "content": system_message})
88|        
89|        # Add recent chat history (last 5-6 Q&A pairs = 10-12 messages)
90|        if chat_history:
91|            for msg in chat_history[-10:]:
92|                if msg["role"] in ["user", "assistant"]:
93|                    messages.append(msg)
94|        
95|        # Add current query with context
96|        user_message = f"""Context from PDF documents:
97|{context}
98|
99|Question: {query}
100|
101|Please answer the question based on the context provided above."""
102|        
103|        messages.append({"role": "user", "content": user_message})
104|        
105|        return messages
106|    
107|    def answer_question(
108|        self,
109|        query: str,
110|        chat_history: List[Dict[str, str]] = None
111|    ) -> Generator[str, None, None]:
112|        """
113|        Answer a question with streaming response.
114|        
115|        Args:
116|            query: User question
117|            chat_history: Previous conversation history
118|            
119|        Yields:
120|            Chunks of the response text
121|        """
122|        try:
123|            # Retrieve context
124|            context, max_score = self.build_context(query)
125|            
126|            # Check if results are relevant
127|            if not context or max_score < self.similarity_threshold:
128|                yield "I don't have enough relevant information in the provided documents to answer that question. Please try rephrasing your question or ask about topics covered in the uploaded PDFs."
129|                return
130|            
131|            # Build messages
132|            messages = self.build_messages(query, context, chat_history)
133|            
134|            # Stream response from Groq
135|            stream = self.client.chat.completions.create(
136|                messages=messages,
137|                model=self.model,
138|                max_tokens=1024,
139|                temperature=0.3,
140|                stream=True
141|            )
142|            
143|            for chunk in stream:
144|                if chunk.choices[0].delta.content:
145|                    yield chunk.choices[0].delta.content
146|        
147|        except Exception as e:
148|            yield f"Error generating answer: {str(e)}"
149|
[End of file]
