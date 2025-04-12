import streamlit as st
import requests
import sseclient
import json
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Live RAG Evaluations", layout="wide")

st.title("📡 Live RAG Evaluation Stream")

st.markdown("Streaming real-time results from `/stream-evaluations`...")

placeholder = st.empty()
chart_placeholder = st.empty()

results = []

def render_results():
    with placeholder.container():
        for item in reversed(results[-10:]):
            st.markdown(f"### 🧠 {item['question']}")
            st.markdown(f"- **Method**: `{item['method']}`")
            st.markdown(f"- **Prompt Type**: `{item['prompt_type']}`")
            st.markdown("#### 📝 Generated Answer")
            st.write(item['generated_answer'])

            st.markdown("#### 🤖 LLM Evaluation")
            st.markdown(item['llm_evaluation'])

            st.markdown("#### 📊 RAGAs Scores")
            st.markdown(f"""
- **Faithfulness**: `{item['ragas_scores']['faithfulness']:.2f}`
- **Relevancy**: `{item['ragas_scores']['answer_relevancy']:.2f}`
- **Context Precision**: `{item['ragas_scores']['context_precision']:.2f}`
---
""")



def render_charts():
    if not results:
        return

    # Build DataFrame for the last 20 evaluations
    df = pd.DataFrame([{
        "Method": r["method"],
        "Prompt Type": r["prompt_type"],
        "Faithfulness": r["ragas_scores"]["faithfulness"],
        "Relevancy": r["ragas_scores"]["answer_relevancy"],
        "Context Precision": r["ragas_scores"]["context_precision"],
        "Label": f"{r['method']} | {r['prompt_type']}"
    } for r in results])

    df = df.tail(20)  # Only show last 20

    # Melt dataframe for grouped bar chart
    df_melted = df.melt(id_vars=["Label"], value_vars=["Faithfulness", "Relevancy", "Context Precision"],
                        var_name="Metric", value_name="Score")

    with chart_placeholder.container():
        st.subheader("📊 RAGAs Metrics (Grouped by Evaluation Block)")

        fig = px.bar(
            df_melted,
            x="Label",
            y="Score",
            color="Metric",
            barmode="group",
            title="RAGAs Scores per Evaluation",
            labels={"Label": "Method | Prompt Type"},
            height=400
        )
        fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(fig, use_container_width=True)


def stream_results():
    try:
        response = requests.get(
            "http://localhost:5000/stream-evaluations",
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=None,
        )
        response.raise_for_status()
        st.success("✅ Connected to backend!")

        client = sseclient.SSEClient(response)  # DO NOT use iter_lines()

        for event in client.events():
            if not event.data.strip():
                continue

            try:
                data = json.loads(event.data)
                results.append(data)

                render_results()
                render_charts()

            except json.JSONDecodeError as json_err:
                st.error(f"❌ JSON decode error: {json_err}")
                st.text(f"Raw event:\n{event.data}")

    except requests.exceptions.RequestException as e:
        st.error("🚫 Unable to connect to the backend. Please make sure the Flask server is running.")
        st.exception(e)
        st.stop()
    except Exception as e:
        st.error("❌ An unexpected error occurred.")
        st.exception(e)
        st.stop()


# Launch stream
stream_results()
