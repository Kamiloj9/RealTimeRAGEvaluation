from langchain.prompts import PromptTemplate

query_rewrite_prompt = PromptTemplate(
    input_variables=["original_query"],
    template="""You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.\nGiven the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.\n\nOriginal query: {original_query}\n\nRewritten query:"""
)

step_back_prompt = PromptTemplate(
    input_variables=["original_query"],
    template="""You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.\nGiven the original query, generate a step-back query that is more general and can help retrieve relevant background information.\n\nOriginal query: {original_query}\n\nStep-back query:"""
)

subquery_decomposition_prompt = PromptTemplate(
    input_variables=["original_query"],
    template="""You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.\nGiven the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.\n\nOriginal query: {original_query}\n\nexample: What are the impacts of climate change on the environment?\n\nSub-queries:\n1. What are the impacts of climate change on biodiversity?\n2. How does climate change affect the oceans?\n3. What are the effects of climate change on agriculture?\n4. What are the impacts of climate change on human health?"""
)

summarization_prompt = PromptTemplate(
        input_variables=["query", "answers"],
        template="""You are an expert assistant. Summarize the answer to the following original question based on the sub-answers below. The asswer must be maximum 3 sentences long.

Question: {query}

Sub-answers:
{answers}

Final Answer:"""
    )

negative_example_prompt = PromptTemplate(
    input_variables=["topic"],
    template="""Provide a brief explanation to the question or topic {topic}.
Do NOT include any of the following in your explanation:
- Technical jargon or complex terminology
- Historical background or dates
- Comparisons to other related topics
Your explanation should be simple, direct, and focus only on the core concept."""
)

exclusion_prompt = PromptTemplate(
    input_variables=["topic", "exclude"],
    template="""Write a short paragraph about this question or topic {topic}.
Important: Do not mention or reference anything related to {exclude}."""
)

constraint_prompt = PromptTemplate(
    input_variables=["topic", "style", "excluded_words"],
    template="""Write a {style} description of this question or topic {topic}.
Constraints:
1. Do not use any of these words: {excluded_words}
2. Keep the description under 100 words
3. Do not use analogies or metaphors
4. Focus only on factual information
5. Your answer must not exceed three sentences."""
)

rag_prompt_template_simple = """
    You are an AI assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.
    
    Question: {question}
    
    Context: {context}
    
    Answer:
    """