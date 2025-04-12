from src.config import llm

def evaluate_rag_response(query, answer, reference_answer=None):
    """
    Evaluate a RAG response based on various criteria
    """
    if reference_answer:
        evaluation_prompt = f"""
        Question: {query}
        
        Generated Answer: {answer}
        
        Reference Answer: {reference_answer}
        
        Evaluate this answer on a scale of 1-10 for:
        1. Relevance: How well the answer addresses the question
        2. Accuracy: How factually correct the information is compared to the reference
        3. Completeness: How thoroughly the answer covers the topic
        4. Conciseness: How focused and to-the-point the answer is
        
        Provide a score for each criterion and a brief explanation.
        """
    else:
        evaluation_prompt = f"""
        Question: {query}
        
        Generated Answer: {answer}
        
        Evaluate this answer on a scale of 1-10 for:
        1. Relevance: How well the answer addresses the question
        2. Accuracy: How factually correct the information appears to be
        3. Completeness: How thoroughly the answer covers the topic
        4. Conciseness: How focused and to-the-point the answer is
        
        Provide a score for each criterion and a brief explanation.
        """
    
    evaluation = llm.invoke(evaluation_prompt).content
    return evaluation