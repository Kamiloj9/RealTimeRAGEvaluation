from typing import Any, Dict
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable
from src.config import llm, DEFAULT_RETRIEVER_K

def create_rag_chain(
    vectorstore: VectorStore,
    prompt_template: PromptTemplate,
    model: BaseLanguageModel = llm,
    k: int = DEFAULT_RETRIEVER_K
) -> Runnable:
    """Create a RAG chain from a vector store"""
    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    RAG_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create the RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": RAG_PROMPT},
        return_source_documents=True
    )
    
    return rag_chain

def query_rag_system(
    rag_chain: Runnable,
    query: str
) -> Dict[str, Any]:
    """Query the RAG system with a question"""
    result = rag_chain.invoke(query)
    
    return {
        "query": query,
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }