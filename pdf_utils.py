---

## **File 2: pdf_utils.py**
Action: file_editor view /app/pdf_utils.py
Observation: [Showing lines 1-177 of 177 total] /app/pdf_utils.py:
1|"""PDF loading and chunking utilities."""
2|import re
3|from typing import List, Dict
4|from PyPDF2 import PdfReader
5|import tiktoken
6|
7|def load_pdf(pdf_path: str) -> str:
8|    """
9|    Load text from a PDF file.
10|    
11|    Args:
12|        pdf_path: Path to the PDF file
13|        
14|    Returns:
15|        Extracted text from all pages
16|        
17|    Raises:
18|        Exception: If PDF cannot be read
19|    """
20|    try:
21|        reader = PdfReader(pdf_path)
22|        text = ""
23|        
24|        for page in reader.pages:
25|            page_text = page.extract_text()
26|            if page_text:
27|                text += page_text + "\n"
28|        
29|        return text.strip()
30|    
31|    except Exception as e:
32|        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")
33|
34|def clean_text(text: str) -> str:
35|    """
36|    Clean extracted text by removing extra whitespace and special characters.
37|    
38|    Args:
39|        text: Raw text to clean
40|        
41|    Returns:
42|        Cleaned text
43|    """
44|    # Remove multiple newlines
45|    text = re.sub(r'\n+', '\n', text)
46|    
47|    # Remove multiple spaces
48|    text = re.sub(r'\s+', ' ', text)
49|    
50|    # Remove leading/trailing whitespace
51|    text = text.strip()
52|    
53|    return text
54|
55|def chunk_text_by_tokens(
56|    text: str,
57|    chunk_size: int = 400,
58|    overlap: int = 50,
59|    encoding_name: str = "cl100k_base"
60|) -> List[str]:
61|    """
62|    Chunk text into segments based on token count with overlap.
63|    
64|    Args:
65|        text: Text to chunk
66|        chunk_size: Maximum tokens per chunk
67|        overlap: Number of overlapping tokens between chunks
68|        encoding_name: Tokenizer encoding to use
69|        
70|    Returns:
71|        List of text chunks
72|    """
73|    try:
74|        encoding = tiktoken.get_encoding(encoding_name)
75|    except:
76|        # Fallback to simple word-based chunking if tiktoken fails
77|        return chunk_text_by_words(text, chunk_size * 2, overlap * 2)
78|    
79|    # Tokenize the entire text
80|    tokens = encoding.encode(text)
81|    
82|    chunks = []
83|    start = 0
84|    
85|    while start < len(tokens):
86|        # Get chunk of tokens
87|        end = start + chunk_size
88|        chunk_tokens = tokens[start:end]
89|        
90|        # Decode back to text
91|        chunk_text = encoding.decode(chunk_tokens)
92|        chunks.append(chunk_text.strip())
93|        
94|        # Move start position with overlap
95|        start += chunk_size - overlap
96|    
97|    return chunks
98|
99|def chunk_text_by_words(
100|    text: str,
101|    chunk_size: int = 800,
102|    overlap: int = 100
103|) -> List[str]:
104|    """
105|    Fallback chunking method using word count.
106|    
107|    Args:
108|        text: Text to chunk
109|        chunk_size: Maximum words per chunk
110|        overlap: Number of overlapping words
111|        
112|    Returns:
113|        List of text chunks
114|    """
115|    words = text.split()
116|    chunks = []
117|    start = 0
118|    
119|    while start < len(words):
120|        end = start + chunk_size
121|        chunk = ' '.join(words[start:end])
122|        chunks.append(chunk)
123|        start += chunk_size - overlap
124|    
125|    return chunks
126|
127|def load_and_chunk_pdfs(
128|    pdf_paths: List[str],
129|    chunk_size: int = 400,
130|    overlap: int = 50
131|) -> List[Dict[str, str]]:
132|    """
133|    Load multiple PDFs and chunk them into segments.
134|    
135|    Args:
136|        pdf_paths: List of paths to PDF files
137|        chunk_size: Maximum tokens per chunk
138|        overlap: Number of overlapping tokens
139|        
140|    Returns:
141|        List of document dictionaries with 'text' and 'metadata'
142|    """
143|    documents = []
144|    
145|    for pdf_path in pdf_paths:
146|        try:
147|            # Extract text from PDF
148|            raw_text = load_pdf(pdf_path)
149|            
150|            if not raw_text:
151|                print(f"Warning: No text extracted from {pdf_path}")
152|                continue
153|            
154|            # Clean the text
155|            cleaned_text = clean_text(raw_text)
156|            
157|            # Chunk the text
158|            chunks = chunk_text_by_tokens(cleaned_text, chunk_size, overlap)
159|            
160|            # Create document objects
161|            for i, chunk in enumerate(chunks):
162|                if chunk.strip():  # Only add non-empty chunks
163|                    documents.append({
164|                        'text': chunk,
165|                        'metadata': {
166|                            'source': pdf_path,
167|                            'chunk_id': i,
168|                            'total_chunks': len(chunks)
169|                        }
170|                    })
171|        
172|        except Exception as e:
173|            print(f"Error processing {pdf_path}: {str(e)}")
174|            continue
175|    
176|    return documents
177|
[End of file]
