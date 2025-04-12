from src.transformations import rewrite_query, generate_step_back_query, decompose_query
from src.templates import summarization_prompt
from langchain.chains import LLMChain
from src.config import llm

def query_with_rewriting(rag_chain, query):
    rewritten = rewrite_query(query)
    result = rag_chain.invoke(rewritten)
    return {
        "original_query": query,
        "rewritten_query": rewritten,
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }

def query_with_step_back(rag_chain, query):
    step_back = generate_step_back_query(query)
    result = rag_chain.invoke(step_back)
    return {
        "original_query": query,
        "step_back_query": step_back,
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }

def query_with_subqueries(rag_chain, query):
    sub_queries = decompose_query(query)
    answers = []
    sources = []
    for sub in sub_queries:
        result = rag_chain.invoke(sub)
        answers.append(f"Q: {sub}\nA: {result['result']}")
        sources.extend(result["source_documents"])
    join_sub_answers = "\n".join(answers)
    summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)
    return summarization_chain.run(query=query, answers=join_sub_answers)