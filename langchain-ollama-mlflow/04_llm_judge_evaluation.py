"""
============================================================
 04_llm_judge_evaluation.py
 ─────────────────────────────────────────────────────────────
 Tutorial Part 4 – LLM-as-a-Judge Evaluation via Ollama
 ─────────────────────────────────────────────────────────────
 What you will learn
 ───────────────────
   • How to use a LOCAL Ollama model as the evaluation judge
     (no OpenAI key required)
   • How to write a custom @scorer that calls the Ollama API
     directly to assess response quality
   • How to combine LLM-judge scorers with rule-based scorers

 Judge scorers implemented here
 ──────────────────────────────
   ┌───────────────────┬──────────────────────────────────────┐
   │ Scorer            │ What the judge checks                │
   ├───────────────────┼──────────────────────────────────────┤
   │ llm_correctness   │ Is the answer factually correct?     │
   │ llm_helpfulness   │ Is the answer genuinely helpful?     │
   │ llm_language      │ Is the response written in English?  │
   └───────────────────┴──────────────────────────────────────┘

 Run
 ───
   ~/python3-env/bin/python3 04_llm_judge_evaluation.py
============================================================
"""

import os
import mlflow
from mlflow.genai import scorer
from mlflow.entities import Feedback

from langchain_ollama import OllamaLLM, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# ── 1. Setup ──────────────────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", "dummy")

mlflow.set_tracking_uri("http://localhost:6001")
mlflow.set_experiment("langchain-ollama-llm-judge")
mlflow.langchain.autolog()

# ── 2. The model under test ───────────────────────────────────
llm = OllamaLLM(
    model="qwen3.5:0.8b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Answer concisely.\n\nQuestion: {question}"
)
chain = prompt | llm

def qa_predict_fn(question: str) -> str:
    return chain.invoke({"question": question})


# ── 3. Judge model (same Ollama endpoint, can be a diff model) ─
judge_llm = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",
    temperature=0.0,   # deterministic judgements
)

def _judge(system_prompt: str, user_prompt: str) -> str:
    """Call the local Ollama judge and return its raw text response."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = judge_llm.invoke(messages)
    return response.content.strip()


# ── 4. LLM-judge custom scorers ───────────────────────────────

@scorer
def llm_correctness(inputs: dict, outputs: str, expectations: dict) -> Feedback:
    """
    Ask the judge: is the answer factually correct given the expected answer?
    Returns a score between 0 and 1 and a brief rationale.
    """
    expected = expectations.get("expected_response", "")
    system = (
        "You are an evaluation judge. "
        "Respond with ONLY a JSON object: {\"score\": <0.0-1.0>, \"reason\": \"<one sentence>\"}"
    )
    user = (
        f"Question: {inputs.get('question', '')}\n"
        f"Expected answer: {expected}\n"
        f"Actual answer: {outputs}\n\n"
        "Score 1.0 if the actual answer is fully correct, 0.5 if partially correct, 0.0 if wrong."
    )
    raw = _judge(system, user)
    import json, re
    try:
        # Extract JSON even if surrounded by markdown fences
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        return Feedback(name="llm_correctness", value=float(data.get("score", 0.0)), rationale=data.get("reason", raw))
    except Exception as e:
        return Feedback(name="llm_correctness", value=0.0, rationale=raw, error=str(e))


@scorer
def llm_helpfulness(inputs: dict, outputs: str) -> Feedback:
    """
    Ask the judge: is the response genuinely helpful to the user?
    Returns a score between 0 and 1.
    """
    system = (
        "You are an evaluation judge. "
        "Respond with ONLY a JSON object: {\"score\": <0.0-1.0>, \"reason\": \"<one sentence>\"}"
    )
    user = (
        f"Question: {inputs.get('question', '')}\n"
        f"Response: {outputs}\n\n"
        "Score 1.0 if fully helpful, 0.5 if somewhat helpful, 0.0 if unhelpful or evasive."
    )
    raw = _judge(system, user)
    import json, re
    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        data = json.loads(match.group()) if match else {}
        return Feedback(name="llm_helpfulness", value=float(data.get("score", 0.0)), rationale=data.get("reason", raw))
    except Exception as e:
        return Feedback(name="llm_helpfulness", value=0.0, rationale=raw, error=str(e))


@scorer
def llm_language_check(outputs: str) -> bool:
    """Ask the judge: is the response written in English?"""
    system = "You are a language detector. Reply with only YES or NO."
    user = f"Is the following text written in English?\n\n{outputs}"
    raw = _judge(system, user)
    return "yes" in raw.lower()


# ── 5. Dataset ────────────────────────────────────────────────
eval_dataset = [
    {
        "inputs":  {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"},
    },
    {
        "inputs":  {"question": "Who invented the telephone?"},
        "expectations": {"expected_response": "Alexander Graham Bell"},
    },
    {
        "inputs":  {"question": "What is the boiling point of water in Celsius?"},
        "expectations": {"expected_response": "100 degrees Celsius"},
    },
]

# ── 6. Run evaluation ─────────────────────────────────────────
print("Starting LLM-judge evaluation (local Ollama judge) …")
print("Note: each scorer calls the judge LLM – this takes ~1-2 min\n")

results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=qa_predict_fn,
    scorers=[llm_correctness, llm_helpfulness, llm_language_check],
)

print("\n" + "="*60)
print("LLM-JUDGE EVALUATION RESULTS")
print("="*60)
print("\nAggregate metrics:")
for k, v in results.metrics.items():
    print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
print("\nPer-sample results:")
scorer_cols = [c for c in results.result_df.columns if c.endswith("/value")]
print(results.result_df[["request", "response"] + scorer_cols].to_string(index=False))
print("\n✅  Done – open http://localhost:6001 to inspect trace assessments.")
