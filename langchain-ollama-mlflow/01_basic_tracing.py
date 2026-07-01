"""
============================================================
 01_basic_tracing.py
 ─────────────────────────────────────────────────────────────
 Tutorial Part 1 – Basic LangChain + Ollama Tracing with MLflow
 ─────────────────────────────────────────────────────────────
 What you will learn
 ───────────────────
   • How to connect MLflow to a local tracking server
   • How to enable automatic tracing for LangChain calls
   • How to run a simple LLM chain and inspect the trace on
     the MLflow UI

 Pre-requisites
 ──────────────
   • Ollama running locally  (ollama serve)
   • qwen3.5:0.8b model pulled (ollama pull qwen3.5:0.8b)
   • MLflow UI running on port 6001
       mlflow server --host 127.0.0.1 --port 6001

 Run
 ───
   ~/python3-env/bin/python3 01_basic_tracing.py
============================================================
"""

import mlflow
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── 1. Point MLflow at your local server ──────────────────────
mlflow.set_tracking_uri("http://localhost:6001")
mlflow.set_experiment("langchain-ollama-basic")

# ── 2. Enable auto-tracing for every LangChain invocation ─────
#       This single line instruments all LangChain / LangGraph
#       runnables automatically – no extra code needed.
mlflow.langchain.autolog()

# ── 3. Build a minimal LangChain chain ────────────────────────
#       OllamaLLM speaks directly to the local Ollama REST API.
llm = OllamaLLM(
    model="qwen3.5:0.8b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

prompt = PromptTemplate.from_template(
    "Answer the following question in one short sentence.\n\nQuestion: {question}"
)

chain = prompt | llm

# ── 4. Run inside an MLflow run so results are persisted ──────
with mlflow.start_run(run_name="basic-chain-demo"):
    questions = [
        "What is the capital of France?",
        "Who invented the telephone?",
        "What is 15 multiplied by 7?",
    ]

    for q in questions:
        answer = chain.invoke({"question": q})
        print(f"\nQ: {q}")
        print(f"A: {answer}")

print("\n✅  Run complete – open http://localhost:6001 to view traces.")
