"""PDF loading and chunking utilities."""
import re
from typing import List, Dict
from PyPDF2 import PdfReader
import tiktoken

def load_pdf(pdf_path: str) -> str:
    """
    Load text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from all pages
        
    Raises:
        Exception: If PDF cannot be read
    """
    try:
        reader = PdfReader(pdf_path)
        text = ""
        
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        return text.strip()
    
    except Exception as e:
        raise Exception(f"Error reading PDF {pdf_path}: {str(e)}")

def clean_text(text: str) -> str:
    """
    Clean extracted text by removing extra whitespace and special characters.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def chunk_text_by_tokens(
    text: str,
    chunk_size: int = 400,
    overlap: int = 50,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Chunk text into segments based on token count with overlap.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum tokens per chunk
        overlap: Number of overlapping tokens between chunks
        encoding_name: Tokenizer encoding to use
        
    Returns:
        List of text chunks
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except:
        # Fallback to simple word-based chunking if tiktoken fails
        return chunk_text_by_words(text, chunk_size * 2, overlap * 2)
    
    # Tokenize the entire text
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    
    while start < len(tokens):
        # Get chunk of tokens
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        
        # Decode back to text
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        
        # Move start position with overlap
        start += chunk_size - overlap
    
    return chunks

def chunk_text_by_words(
    text: str,
    chunk_size: int = 800,
    overlap: int = 100
) -> List[str]:
    """
    Fallback chunking method using word count.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum words per chunk
        overlap: Number of overlapping words
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks

def load_and_chunk_pdfs(
    pdf_paths: List[str],
    chunk_size: int = 400,
    overlap: int = 50
) -> List[Dict[str, str]]:
    """
    Load multiple PDFs and chunk them into segments.
    
    Args:
        pdf_paths: List of paths to PDF files
        chunk_size: Maximum tokens per chunk
        overlap: Number of overlapping tokens
        
    Returns:
        List of document dictionaries with 'text' and 'metadata'
    """
    documents = []
    
    for pdf_path in pdf_paths:
        try:
            # Extract text from PDF
            raw_text = load_pdf(pdf_path)
            
            if not raw_text:
                print(f"Warning: No text extracted from {pdf_path}")
                continue
            
            # Clean the text
            cleaned_text = clean_text(raw_text)
            
            # Chunk the text
            chunks = chunk_text_by_tokens(cleaned_text, chunk_size, overlap)
            
            # Create document objects
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # Only add non-empty chunks
                    documents.append({
                        'text': chunk,
                        'metadata': {
                            'source': pdf_path,
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        }
                    })
        
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            continue
    
    return documents
