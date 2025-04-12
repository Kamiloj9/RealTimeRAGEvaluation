from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

from src.templates import query_rewrite_prompt, step_back_prompt, subquery_decomposition_prompt, negative_example_prompt, exclusion_prompt, constraint_prompt
from src.config import BASE_MODEL

re_write_llm = ChatOpenAI(temperature=0, model_name=BASE_MODEL, max_tokens=4000)
step_back_llm = ChatOpenAI(temperature=0, model_name=BASE_MODEL, max_tokens=4000)
sub_query_llm = ChatOpenAI(temperature=0, model_name=BASE_MODEL, max_tokens=4000)
constraint_llm = ChatOpenAI(temperature=0, model_name=BASE_MODEL, max_tokens=4000)

query_rewriter = query_rewrite_prompt | re_write_llm
step_back_chain = step_back_prompt | step_back_llm
subquery_decomposer_chain = subquery_decomposition_prompt | sub_query_llm

def rewrite_query(original_query):
    return query_rewriter.invoke(original_query).content

def generate_step_back_query(original_query):
    return step_back_chain.invoke(original_query).content

def decompose_query(original_query):
    response = subquery_decomposer_chain.invoke(original_query).content
    return [q.strip() for q in response.split('\n') if q.strip() and not q.lower().startswith("sub-queries")]