from src.evaluation import evaluate_rag_response
from src.utils import load_config
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.pipeline import build_complete_rag_system
from src.rag import query_rag_system
from src.runners import (
    query_with_rewriting,
    query_with_step_back,
    query_with_subqueries
)
from src.templates import rag_prompt_template_simple, negative_example_prompt, exclusion_prompt, constraint_prompt
from src.config import llm
import json
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas import evaluate
from flask import Flask, Response, stream_with_context
from src.logger_setup import setup_logger

logger = setup_logger()

# used for testing
if __name__ == '__main2__':
    # Build the RAG system
    rag_system = build_complete_rag_system(rag_prompt_template_simple)
    rag_chain = rag_system["rag_chain"]

    # Test queries
    test_queries = [
        "What is the MedAgent-Pro?",
        "Explain Knowledge-based Task-level Reasoning like I am five."
    ]

    # Store all responses
    all_results = []

    for query in test_queries:
        new_queries = {
            "negative_prompt": negative_example_prompt.format(topic=query),
            "exclusion_prompt": exclusion_prompt.format(topic=query, exclude="AI"),
            "constraint_prompt": constraint_prompt.format(
                topic=query,
                style="technical",
                excluded_words="robot, human-like, science fiction"
            )
        }

        for method, prompt in new_queries.items():
            result_entry = {
                "query": query,
                "method": method,
                "rewrite": query_with_rewriting(rag_chain, prompt)['answer'],
                "step_back": query_with_step_back(rag_chain, prompt)['answer'],
                "sub_queries": query_with_subqueries(rag_chain, prompt)
            }
            all_results.append(result_entry)

    # Save to JSON
    with open("rag_outputs.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
    
    print("✅ Results saved to rag_outputs.json")

rag_system = build_complete_rag_system(rag_prompt_template_simple)
rag_chain = rag_system["rag_chain"]
        
app = Flask(__name__)

@app.route("/stream-evaluations")
def stream_evaluations():
    def generate():
        with open("qa_pairs.json", "r", encoding="utf-8") as f:
            qa_pairs = json.load(f)

        query_methods = {
            "rewriting": query_with_rewriting,
            "step_back": query_with_step_back,
            "sub_queries": query_with_subqueries,
        }

        for pair in qa_pairs:
            question = pair["question"]
            reference_answer = pair["answer"]
            logger.info(f"Processing question: {question}")

            new_queries = {
                "negative_prompt": negative_example_prompt.format(topic=question),
                "exclusion_prompt": exclusion_prompt.format(topic=question, exclude="AI"),
                "constraint_prompt": constraint_prompt.format(
                    topic=question,
                    style="technical",
                    excluded_words="robot, human-like, science fiction"
                )
            }

            for prompt_type, prompt in new_queries.items():
                for method_name, method_func in query_methods.items():
                    logger.debug(f"Method: {method_name}, Prompt: {prompt_type}")
                    result = method_func(rag_chain, prompt)

                    # Handle string vs dict
                    if isinstance(result, dict):
                        answer = result.get('answer', '')
                        context = result.get('context', '')
                    else:
                        answer = result
                        context = ''

                    # LLM Evaluation
                    llm_eval = evaluate_rag_response(question, answer, reference_answer)

                    # RAGAs input
                    ragas_sample = {
                        "question": question,
                        "answer": answer,
                        "contexts": [context],
                        "ground_truth": reference_answer,
                    }

                    ragas_dataset = Dataset.from_list([ragas_sample])
                    ragas_result = evaluate(ragas_dataset, metrics=[
                        faithfulness,
                        answer_relevancy,
                        context_precision,
                    ],
                    raise_exceptions=False)
                    ragas_scores = ragas_result.to_pandas().to_dict(orient="records")[0]

                    logger.debug(f"Generated answer (truncated): {answer[:100]}...")
                    logger.debug(f"LLM evaluation result: {llm_eval}")
                    logger.debug(f"RAGAS scores: {ragas_scores}")

                    response = {
                        "question": question,
                        "reference_answer": reference_answer,
                        "method": method_name,
                        "prompt_type": prompt_type,
                        "generated_answer": answer,
                        "llm_evaluation": llm_eval,
                        "ragas_scores": ragas_scores,
                    }

                    # Convert any ndarray values in ragas_scores to native Python types
                    for k, v in ragas_scores.items():
                        if hasattr(v, "tolist"):
                            ragas_scores[k] = v.tolist()
                    # SSE format
                    yield f"data: {json.dumps(response, ensure_ascii=False)}\n\n"

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    return Response(stream_with_context(generate()), headers=headers)

# main entry
if __name__ == "__main__":
    logger.info("Starting Flask app")
    app.run(debug=True)