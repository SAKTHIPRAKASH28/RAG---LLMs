import PyPDF2
from collections import Counter

import io

def read_pdf(file_bytes):
    """
    Reads PDF content from a byte stream (in-memory).
    """
    # Create a binary stream from the bytes
    pdf_stream = io.BytesIO(file_bytes)
    
    # Use PyPDF2 to read the PDF from the binary stream
    reader = PyPDF2.PdfReader(pdf_stream)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return text
def preprocess_text(text):
    chunks = text.split('\n')
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def create_index(chunks):
    word_freq = Counter()
    for chunk in chunks:
        word_freq.update(chunk.lower().split())
    return word_freq, chunks

def retrieve_context(query, index, chunks, k=3):
    query_words = set(query.lower().split())
    chunk_scores = [(i, len(set(chunk.lower().split()) & query_words)) for i, chunk in enumerate(chunks)]
    top_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:k]
    return [chunks[i] for i, _ in top_chunks]