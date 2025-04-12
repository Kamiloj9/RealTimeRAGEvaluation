from src.document_utils import download_sample_document, load_and_split_document, create_vectorstore
from src.rag import create_rag_chain

def build_complete_rag_system(prompt_template, document_url=None):
    """Build a complete RAG system from a document URL"""
    # Download the document
    file_path, temp_dir = download_sample_document(document_url)
    
    # Load and split the document
    chunks = load_and_split_document(file_path)
    
    # Create the vector store
    vectorstore = create_vectorstore(chunks)
    
    # Create the RAG chain
    rag_chain = create_rag_chain(vectorstore, prompt_template)
    
    return {
        "file_path": file_path,
        "temp_dir": temp_dir,
        "chunks": chunks,
        "vectorstore": vectorstore,
        "rag_chain": rag_chain
    }