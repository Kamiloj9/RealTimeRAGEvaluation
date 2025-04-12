# Query Transformations & Negative Prompting in RAG Systems

This project implements and evaluates multiple **query transformation techniques** and **negative prompting strategies** to improve the performance of **Retrieval-Augmented Generation (RAG)** systems.

It features both a **backend** (Flask) for streaming evaluations and a **frontend** (Streamlit) for live visualizations.

---

## Repository Structure

Backend is an Flask app serving RAG evaluations over SSE.
Frontend is using Streamlit displaying dashboard for real-time visualization.
In qa_pairs.json is a set of QA pairs used for evaluation.


---

## Overview

### Query Transformations for Improved Retrieval in RAG Systems

This module implements three query transformation techniques to enhance the relevance and comprehensiveness of information retrieved by RAG pipelines:

- **Query Rewriting**  
- **Step-back Prompting**  
- **Sub-query Decomposition**

Each method modifies the original query in a unique way to improve downstream retrieval and generation quality.

---

## Motivation

RAG systems often struggle with **ambiguous** or **complex** queries, leading to irrelevant or incomplete retrievals. These transformation techniques help bridge that gap by **rephrasing** or **expanding** the input query to better align with indexed content.

---

## Query Transformation Methods

### 1. Query Rewriting  
**Goal:** Make queries more specific and focused.  
**How:** Reformulates user input using a custom GPT-4 prompt.

### 2. Step-back Prompting  
**Goal:** Generalize the query to retrieve broader context.  
**How:** Applies a GPT-4 prompt that steps back from the original query to a more general one.

### 3. Sub-query Decomposition  
**Goal:** Break down complex questions into simpler components.  
**How:** Generates 2–4 smaller sub-questions using GPT-4.

---

## Negative Prompting & Avoiding Undesired Outputs

This component explores techniques for **controlling and constraining** model outputs by specifying what **not** to generate.

### Key Concepts

- **Negative Examples:** Show what responses are *not* acceptable.  
- **Exclusion Constraints:** Explicitly ban terms, styles, or tones in the prompt.  
- **Constraint Prompting:** Use LangChain to create precise prompts with structured exclusions.  
- **Iterative Refinement:** Evaluate and tweak prompts based on outputs.

---

## Evaluation Pipeline

The system uses two layers of evaluation:

- **LLM-Based Evaluation** – Assesses relevance, completeness, and clarity of generated answers using GPT-based heuristics.  
- **RAGAs Metrics** – Faithfulness, Answer Relevancy, and Context Precision calculated with [RAGAS](https://github.com/explodinggradients/ragas).

Evaluations are streamed via SSE from the backend and visualized live in the frontend with bar charts and answer summaries.

---

## Frontend (Streamlit)

- Live updates via Server-Sent Events  
- Interactive dashboards  
- LLM evaluation + RAGAs metrics display  
- Grouped bar charts for visual performance comparison

---

## Backend (Flask)

- Loads QA pairs from `qa_pairs.json`  
- Applies 3 query transformation methods to each input  
- Uses 3 prompt variations for negative prompting  
- Streams results as they are generated and evaluated

---

## Getting Started

### 1. Install Dependencies

**Backend:**
```bash
cd backend
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
pip install -r requirements.txt
```

### 2. How to run

**Backend:**
```bash
cd backend
python main.py
```

**Frontend:**
```bash
cd frontend
streamlit run app.py
```