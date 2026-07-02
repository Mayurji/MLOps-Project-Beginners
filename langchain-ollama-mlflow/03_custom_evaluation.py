"""
============================================================
 03_custom_evaluation.py
 ─────────────────────────────────────────────────────────────
 Tutorial Part 3 – Evaluation with Custom Scorers
 ─────────────────────────────────────────────────────────────
 What you will learn
 ───────────────────
   • How to define a predict_fn that wraps your LangChain agent
   • How to write pure-Python custom scorers (no external LLM
     judge needed – perfect for local/offline environments)
   • How to use mlflow.genai.evaluate() to score a Q&A dataset
   • How to read evaluation metrics from the MLflow UI

 Scorers implemented here
 ─────────────────────────
   ┌──────────────────┬────────────────────────────────────────┐
   │ Scorer           │ What it checks                         │
   ├──────────────────┼────────────────────────────────────────┤
   │ is_concise       │ ≤ 30 words                             │
   │ is_not_empty     │ non-blank response                     │
   │ keyword_match    │ expected keyword appears in output     │
   │ no_apology       │ model does not say "I don't know" etc. │
   └──────────────────┴────────────────────────────────────────┘

 Run
 ───
   ~/python3-env/bin/python3 03_custom_evaluation.py
============================================================
"""

import os
import mlflow
from mlflow.genai import scorer

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# ── 1. Env / MLflow setup ─────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "dummy")   # silence any key warnings

mlflow.set_tracking_uri("http://localhost:6001")
mlflow.set_experiment("langchain-ollama-evaluation")
mlflow.langchain.autolog()

# ── 2. Build the LangChain chain ──────────────────────────────
llm = OllamaLLM(
    model="qwen3.5:0.8b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Answer concisely.\n\nQuestion: {question}"
)

chain = prompt | llm

# ── 3. predict_fn expected by mlflow.genai.evaluate() ─────────
def qa_predict_fn(question: str) -> str:
    """Invoke the LangChain chain for a single question."""
    return chain.invoke({"question": question})

# ── 4. Evaluation dataset ─────────────────────────────────────
eval_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"keyword": "Paris"},
    },
    {
        "inputs": {"question": "Who invented the telephone?"},
        "expectations": {"keyword": "Bell"},
    },
    {
        "inputs": {"question": "What programming language was Python named after?"},
        "expectations": {"keyword": "Monty"},
    },
    {
        "inputs": {"question": "What does MLflow help you manage?"},
        "expectations": {"keyword": "machine learning"},
    },
    {
        "inputs": {"question": "In which year did World War II end?"},
        "expectations": {"keyword": "1945"},
    },
]

# ── 5. Custom scorers ─────────────────────────────────────────

@scorer
def is_concise(outputs: str) -> bool:
    """Pass if the response is 30 words or fewer."""
    return len(outputs.split()) <= 30


@scorer
def is_not_empty(outputs: str) -> bool:
    """Pass if the response is not blank."""
    return bool(outputs and outputs.strip())


@scorer
def keyword_match(inputs: dict, outputs: str, expectations: dict) -> bool:
    """
    Pass if the expected keyword (case-insensitive) appears in the output.
    Requires 'keyword' in expectations.
    """
    keyword = expectations.get("keyword", "")
    if not keyword:
        return False
    return keyword.lower() in outputs.lower()


@scorer
def no_apology(outputs: str) -> bool:
    """Fail if the model produces a typical refusal / apology phrase."""
    apology_phrases = [
        "i don't know",
        "i do not know",
        "i'm not sure",
        "i am not sure",
        "cannot answer",
        "unable to answer",
    ]
    text = outputs.lower()
    return not any(phrase in text for phrase in apology_phrases)


# ── 6. Run evaluation ─────────────────────────────────────────
print("Starting evaluation – this may take a minute …")

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=qa_predict_fn,
    scorers=[is_concise, is_not_empty, keyword_match, no_apology],
)

# ── 7. Print summary ──────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print("\nAggregate metrics:")
for k, v in results.metrics.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
print("\nPer-sample results:")
# Select readable columns: request, response, and all scorer /value columns
scorer_cols = [c for c in results.result_df.columns if c.endswith("/value")]
print(results.result_df[["request", "response"] + scorer_cols].to_string(index=False))
print("\n✅  Evaluation complete – open http://localhost:6001 to view results.")
