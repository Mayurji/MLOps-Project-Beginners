"""
============================================================
 05_monitoring_dashboard.py
 ─────────────────────────────────────────────────────────────
 Tutorial Part 5 – Production-Style Agent Monitoring
 ─────────────────────────────────────────────────────────────
 What you will learn
 ───────────────────
   • How to log custom metrics (latency, token counts) per call
   • How to tag runs for easy filtering in the MLflow UI
   • How to simulate a "continuous monitoring" loop with
     periodic metric logging
   • How to log agent errors as run metadata

 Metrics logged per request
 ──────────────────────────
   latency_ms     Wall-clock time for one chain invocation
   response_len   Character count of the response
   word_count     Word count of the response
   error_count    Cumulative errors across all requests

 Run
 ───
   ~/python3-env/bin/python3 05_monitoring_dashboard.py
============================================================
"""

import os
import time
import mlflow
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── 1. Setup ──────────────────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "dummy")

mlflow.set_tracking_uri("http://localhost:6001")
mlflow.set_experiment("langchain-ollama-monitoring")
mlflow.langchain.autolog()

# ── 2. Chain ──────────────────────────────────────────────────
llm = OllamaLLM(
    model="qwen3.5:0.8b",
    base_url="http://localhost:11434",
    temperature=0.3,
)

prompt = PromptTemplate.from_template(
    "Answer the following question concisely and helpfully.\n\nQuestion: {question}"
)
chain = prompt | llm

# ── 3. Monitoring wrapper ─────────────────────────────────────
def monitored_invoke(question: str, run, step: int) -> dict:
    """
    Invoke the chain and log per-request monitoring metrics to MLflow.
    Returns a dict with the answer and timing info.
    """
    start = time.perf_counter()
    error = None
    answer = ""

    try:
        answer = chain.invoke({"question": question})
    except Exception as e:
        error = str(e)
        answer = f"ERROR: {e}"

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Log per-step metrics (step = request number)
    mlflow.log_metric("latency_ms",    elapsed_ms,         step=step)
    mlflow.log_metric("response_len",  len(answer),        step=step)
    mlflow.log_metric("word_count",    len(answer.split()), step=step)

    if error:
        mlflow.log_metric("error_count", 1, step=step)
        mlflow.set_tag(f"error_step_{step}", error[:200])

    return {
        "question":    question,
        "answer":      answer,
        "latency_ms":  elapsed_ms,
        "word_count":  len(answer.split()),
        "error":       error,
    }

# ── 4. Simulate a monitoring session ─────────────────────────
#       In production you would call monitored_invoke() from
#       your API handler / message queue consumer.
monitoring_questions = [
    "What is machine learning?",
    "Explain DevOps in one sentence.",
    # "What is the difference between Docker and Kubernetes?",
    # "What is MLflow used for?",
    # "Name three benefits of using LangChain.",
    # "What is a large language model?",
    # "Describe the concept of MLOps.",
    # "What is Ollama?",
]

with mlflow.start_run(run_name="monitoring-session-II") as run:
    # Tag the run for easy filtering
    mlflow.set_tags({
        "environment":   "local",
        "llm_model":     "qwen3.5:0.8b",
        "provider":      "ollama",
        "session_type":  "monitoring",
    })

    total_latency = 0.0
    error_count = 0
    results = []

    print(f"{'─'*65}")
    print(f"{'MONITORING SESSION':^65}")
    print(f"{'Run ID: ' + run.info.run_id:^65}")
    print(f"{'─'*65}\n")

    for step, question in enumerate(monitoring_questions, start=1):
        print(f"[{step}/{len(monitoring_questions)}] {question}")
        result = monitored_invoke(question, run, step)
        results.append(result)

        total_latency += result["latency_ms"]
        if result["error"]:
            error_count += 1

        status = "✅" if not result["error"] else "❌"
        print(f"  {status} {result['latency_ms']:.0f} ms | {result['word_count']} words")
        print(f"     {result['answer'][:100]}{'…' if len(result['answer']) > 100 else ''}\n")

    # ── 5. Log summary / aggregate metrics ────────────────────
    avg_latency = total_latency / len(monitoring_questions)
    avg_words   = sum(r["word_count"] for r in results) / len(results)

    mlflow.log_metrics({
        "avg_latency_ms":   avg_latency,
        "avg_word_count":   avg_words,
        "total_requests":   len(monitoring_questions),
        "total_errors":     error_count,
        "success_rate":     (len(monitoring_questions) - error_count) / len(monitoring_questions),
    })

    print(f"{'─'*65}")
    print(f"SESSION SUMMARY")
    print(f"  Requests   : {len(monitoring_questions)}")
    print(f"  Avg latency: {avg_latency:.0f} ms")
    print(f"  Avg words  : {avg_words:.1f}")
    print(f"  Errors     : {error_count}")
    print(f"  Success    : {(len(monitoring_questions)-error_count)/len(monitoring_questions)*100:.0f}%")
    print(f"{'─'*65}")

print("\n✅  Monitoring session done – open http://localhost:6001 to view metrics.")
