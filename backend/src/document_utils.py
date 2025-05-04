import os
import tempfile

import urllib
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from typing import Optional, Tuple, List, Any
from src.logger_setup import setup_logger
from src.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, embeddings

logger = setup_logger('RAG_eval - document_utils')

def download_sample_document(url: Optional[str] = None) -> Tuple[str, str]:
    """Download a sample document from a URL or use a default document"""
    temp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(temp_dir, "sample-document.pdf")
    
    # Get paper url
    if url is None:
        url = "https://arxiv.org/pdf/2503.18968.pdf"
    
    # Download the document
    urllib.request.urlretrieve(url, pdf_path)
    logger.info(f"Downloaded document to: {pdf_path}")
    return pdf_path, temp_dir

def load_and_split_document(file_path: str, 
    chunk_size: int = DEFAULT_CHUNK_SIZE, 
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> List[Document]:
    """Load a document and split it into chunks"""
    # Determine loader based on file extension
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    # Load the document
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} document pages/segments")
    
    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    
    return chunks

def create_vectorstore(chunks: List[Document], 
    embedding_model: Any = embeddings
) -> VectorStore:
    """Create a vector store from document chunks"""
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    logger.info(f"Created vector store with {len(chunks)} documents")
    return vectorstore