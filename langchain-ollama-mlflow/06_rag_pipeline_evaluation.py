"""
============================================================
 06_rag_pipeline_evaluation.py
 ─────────────────────────────────────────────────────────────
 Tutorial Part 6 – RAG Pipeline Evaluation with MLflow
 ─────────────────────────────────────────────────────────────
 What you will learn
 ───────────────────
   • How to build a Retrieval-Augmented Generation (RAG)
     pipeline using LangChain + Ollama (no external DB needed)
   • How MLflow traces the retrieval step AND the generation step
   • How to score faithfulness: does the answer stay inside
     the retrieved context?
   • How to log retrieval quality metrics

 Pipeline architecture
 ─────────────────────
   User Query
       │
       ▼
   SimpleRetriever  (keyword search over an in-memory corpus)
       │
       ▼
   Retrieved Context
       │
       ▼
   OllamaLLM  ──►  Answer
       │
       ▼
   MLflow Trace  +  Evaluation Scores

 Run
 ───
   ~/python3-env/bin/python3 06_rag_pipeline_evaluation.py
============================================================
"""

import os
import mlflow
from mlflow.genai import scorer

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── 1. Setup ──────────────────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "dummy")

mlflow.set_tracking_uri("http://localhost:6001")
mlflow.set_experiment("langchain-ollama-rag-eval")
mlflow.langchain.autolog()

# ── 2. Tiny in-memory knowledge base ─────────────────────────
CORPUS = [
    {
        "id": "mlflow-1",
        "content": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. "
                   "It includes tools for tracking experiments, packaging code, and deploying models.",
    },
    {
        "id": "mlflow-2",
        "content": "MLflow Tracking lets you log parameters, metrics, and artifacts from your ML training runs. "
                   "You can compare runs across experiments in the MLflow UI.",
    },
    {
        "id": "langchain-1",
        "content": "LangChain is a framework for developing applications powered by language models. "
                   "It provides abstractions for chains, agents, memory, and tools.",
    },
    {
        "id": "langchain-2",
        "content": "LangChain agents use an LLM as a reasoning engine to decide which tools to call "
                   "and in which order, enabling dynamic multi-step workflows.",
    },
    {
        "id": "ollama-1",
        "content": "Ollama allows you to run large language models locally on your own hardware. "
                   "It exposes a REST API compatible with the OpenAI API format.",
    },
    {
        "id": "rag-1",
        "content": "Retrieval-Augmented Generation (RAG) combines a retrieval system with a language model. "
                   "The retriever finds relevant documents; the LLM generates an answer grounded in those documents.",
    },
    {
        "id": "mlops-1",
        "content": "MLOps is the practice of applying DevOps principles to machine learning. "
                   "It encompasses experiment tracking, model versioning, CI/CD, and production monitoring.",
    },
]

# ── 3. Simple keyword retriever ───────────────────────────────
def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """Return the top_k corpus entries with the most keyword overlaps."""
    query_words = set(query.lower().split())
    scored = []
    for doc in CORPUS:
        doc_words = set(doc["content"].lower().split())
        overlap = len(query_words & doc_words)
        scored.append((overlap, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ── 4. RAG chain ──────────────────────────────────────────────
llm = OllamaLLM(
    model="qwen3.5:0.8b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

rag_prompt = PromptTemplate.from_template(
    "Use the following context to answer the question. "
    "If the context does not contain the answer, say 'I don't know based on the context.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

rag_chain = rag_prompt | llm

def rag_predict_fn(question: str) -> str:
    """Full RAG pipeline: retrieve → generate."""
    docs = retrieve(question)
    context = "\n\n".join(d["content"] for d in docs)
    return rag_chain.invoke({"question": question, "context": context})


# ── 5. RAG-specific scorers ───────────────────────────────────

@scorer
def is_not_empty(outputs: str) -> bool:
    """Pass if the model returned a non-empty answer."""
    return bool(outputs and outputs.strip())


@scorer
def not_hallucinating(outputs: str) -> bool:
    """
    Heuristic: pass if the model does NOT admit it has no context.
    (A proper faithfulness scorer would compare against retrieved docs.)
    """
    refusal_phrases = ["i don't know based on", "not in the context", "cannot find"]
    return not any(p in outputs.lower() for p in refusal_phrases)


@scorer
def keyword_present(inputs: dict, outputs: str, expectations: dict) -> bool:
    """Pass if the expected keyword appears in the output."""
    kw = expectations.get("keyword", "")
    return kw.lower() in outputs.lower() if kw else False


@scorer
def response_length_ok(outputs: str) -> bool:
    """Pass if the response is between 5 and 100 words."""
    wc = len(outputs.split())
    return 5 <= wc <= 100


# ── 6. Evaluation dataset ─────────────────────────────────────
eval_dataset = [
    {
        "inputs":       {"question": "What is MLflow used for?"},
        "expectations": {"keyword": "machine learning"},
    },
    {
        "inputs":       {"question": "What does MLflow Tracking log?"},
        "expectations": {"keyword": "metrics"},
    },
    {
        "inputs":       {"question": "What is LangChain?"},
        "expectations": {"keyword": "language models"},
    },
    {
        "inputs":       {"question": "How does Ollama expose its API?"},
        "expectations": {"keyword": "REST"},
    },
    {
        "inputs":       {"question": "What is RAG?"},
        "expectations": {"keyword": "retrieval"},
    },
]

# ── 7. Run evaluation ─────────────────────────────────────────
print("Running RAG pipeline evaluation …\n")

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=rag_predict_fn,
    scorers=[is_not_empty, not_hallucinating, keyword_present, response_length_ok],
)

print("\n" + "="*60)
print("RAG EVALUATION RESULTS")
print("="*60)
print("\nAggregate metrics:")
for k, v in results.metrics.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
print("\nPer-sample results:")
scorer_cols = [c for c in results.result_df.columns if c.endswith("/value")]
print(results.result_df[["request", "response"] + scorer_cols].to_string(index=False))
print("\n✅  Done – open http://localhost:6001 to view per-sample scores.")
