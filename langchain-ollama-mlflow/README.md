# AI Agent Monitoring & Evaluation with MLflow + LangChain + Ollama

A **complete, self-contained tutorial** on how to monitor, trace, and evaluate
AI agents running entirely on your local machine — no cloud API keys required.

```
LangChain Agents
      │
      │  auto-traced by
      ▼
   MLflow  ────► Tracking UI (localhost:6001)
      │
      ▼
  Evaluation
  Scorers ─────► Per-sample scores + aggregate metrics
```

---

## Prerequisites

| Requirement | Check |
|---|---|
| Python 3.10+ | `python3 --version` |
| Ollama installed & running | `ollama serve` |
| `qwen3.5:0.8b` model pulled | `ollama pull qwen3.5:0.8b` |
| Virtual env at `~/python3-env` | see setup below |

### Install dependencies

```bash
~/python3-env/bin/pip install -r requirements.txt
```

### Start the MLflow tracking server

Open a **separate terminal** and run:

```bash
~/python3-env/bin/mlflow server --host 127.0.0.1 --port 6001
```

Then open **http://localhost:6001** in your browser — the dashboard will show
experiments and runs as you execute the tutorial scripts.

---

## Tutorial Structure

```
langchain-ollama-mlflow/
├── 01_basic_tracing.py          Part 1 – Basic chain + auto-tracing
├── 02_agent_tracing.py          Part 2 – ReAct agent with tools
├── 03_custom_evaluation.py      Part 3 – Rule-based custom scorers
├── 04_llm_judge_evaluation.py   Part 4 – LLM-as-a-Judge (local Ollama)
├── 05_monitoring_dashboard.py   Part 5 – Production-style monitoring loop
├── 06_rag_pipeline_evaluation.py Part 6 – RAG pipeline evaluation
└── requirements.txt
```

---

## Part 1 — Basic Tracing (`01_basic_tracing.py`)

**Concepts:** `mlflow.langchain.autolog()`, MLflow experiments, basic chains.

```bash
~/python3-env/bin/python3 01_basic_tracing.py
```

`mlflow.langchain.autolog()` is the single line that instruments **every**
LangChain runnable automatically. Each `chain.invoke()` call creates a trace
in the MLflow UI, showing:

- The prompt template and rendered prompt
- The LLM response
- Wall-clock latency

```python
mlflow.langchain.autolog()          # ← one line, full instrumentation

chain = prompt | llm                # standard LangChain LCEL chain

with mlflow.start_run():
    answer = chain.invoke({"question": "What is the capital of France?"})
```

**What you'll see in the UI:**
- Experiment `langchain-ollama-basic`
- Run `basic-chain-demo` with 3 traces, one per question

---

## Part 2 — Agent Tracing (`02_agent_tracing.py`)

**Concepts:** ReAct agents, tool tracing, run tags.

```bash
~/python3-env/bin/python3 02_agent_tracing.py
```

This script builds a ReAct-style agent with three tools:

| Tool | Description |
|---|---|
| `calculator` | Evaluates arithmetic expressions |
| `get_current_date` | Returns today's date |
| `wiki_stub` | Returns facts about Python / MLflow / LangChain / Ollama |

MLflow traces the **entire agent loop** — every LLM reasoning step, every
tool call, and every observation — in a single hierarchical trace.

```
Trace root
  ├── LLM call (Thought → Action)
  ├── Tool: calculator
  ├── LLM call (Observation → Thought)
  └── LLM call (Final Answer)
```

**Key code:**

```python
# Tags make runs easy to filter in the UI
mlflow.set_tags({
    "agent_type": "ReAct",
    "llm_model": "qwen3.5:0.8b",
})

result = agent_executor.invoke({"input": "What is 128 / 4 * 3?"})
```

---

## Part 3 — Custom Evaluation (`03_custom_evaluation.py`)

**Concepts:** `mlflow.genai.evaluate()`, `@scorer`, rule-based scorers, eval datasets.

```bash
~/python3-env/bin/python3 03_custom_evaluation.py
```

### Evaluation dataset format

```python
eval_dataset = [
    {
        "inputs":       {"question": "What is the capital of France?"},
        "expectations": {"keyword": "Paris"},
    },
    ...
]
```

### Scorer anatomy

```python
from mlflow.genai import scorer

@scorer
def keyword_match(inputs: dict, outputs: str, expectations: dict) -> bool:
    """Pass if the expected keyword appears in the output."""
    keyword = expectations.get("keyword", "")
    return keyword.lower() in outputs.lower()
```

A scorer receives any combination of these parameters:
- `inputs` — the dataset row's `inputs` dict
- `outputs` — the `predict_fn` return value
- `expectations` — the dataset row's `expectations` dict

Return `bool` for pass/fail, `float` for a numeric score, or `dict` for
score + rationale.

### Scorers in this part

| Scorer | Logic |
|---|---|
| `is_concise` | `len(outputs.split()) <= 30` |
| `is_not_empty` | `bool(outputs.strip())` |
| `keyword_match` | `expected_keyword in outputs` |
| `no_apology` | detects refusal phrases |

---

## Part 4 — LLM-as-a-Judge (`04_llm_judge_evaluation.py`)

**Concepts:** LLM judges, structured judge prompts, combining judge + rule scorers.

```bash
~/python3-env/bin/python3 04_llm_judge_evaluation.py
```

Instead of calling OpenAI, the **judge is the same local Ollama model**.
The judge receives a structured prompt and returns a JSON score + rationale:

```python
judge_llm = ChatOllama(model="qwen3.5:0.8b", temperature=0.0)

@scorer
def llm_correctness(inputs, outputs, expectations) -> dict:
    system = "Respond with ONLY: {\"score\": <0-1>, \"reason\": \"...\"}"
    user   = f"Question: {inputs['question']}\n" \
             f"Expected: {expectations['expected_response']}\n" \
             f"Actual: {outputs}"
    raw = judge_llm.invoke([SystemMessage(system), HumanMessage(user)])
    data = json.loads(raw.content)
    return {"score": data["score"], "rationale": data["reason"]}
```

> **Tip:** Use `temperature=0.0` for the judge to get deterministic scores.

---

## Part 5 — Production Monitoring (`05_monitoring_dashboard.py`)

**Concepts:** per-step metrics, latency tracking, success rate, session tags.

```bash
~/python3-env/bin/python3 05_monitoring_dashboard.py
```

This script simulates a **monitoring session** — a loop of real requests with
per-request metrics logged to MLflow using `step` so you can plot time-series
charts in the UI.

```python
mlflow.log_metric("latency_ms",  elapsed_ms,  step=step)
mlflow.log_metric("word_count",  word_count,  step=step)
```

At the end, summary / aggregate metrics are logged:

```python
mlflow.log_metrics({
    "avg_latency_ms": avg_latency,
    "success_rate":   (total - errors) / total,
})
```

**What you'll see in the UI:**
- Metric charts with one point per request (step-based)
- Tags: `environment`, `llm_model`, `provider`, `session_type`
- Quick comparison between monitoring sessions

---

## Part 6 — RAG Pipeline Evaluation (`06_rag_pipeline_evaluation.py`)

**Concepts:** RAG chains, retrieval + generation tracing, faithfulness scoring.

```bash
~/python3-env/bin/python3 06_rag_pipeline_evaluation.py
```

A full RAG pipeline (no external vector DB required):

```
User Query → Keyword Retriever → Top-K Docs → LLM → Answer
                                                │
                                           MLflow Trace
```

The retriever and the LLM generation step are both captured in a single
hierarchical trace, giving you end-to-end visibility.

### RAG scorers

| Scorer | Checks |
|---|---|
| `is_not_empty` | model returned something |
| `not_hallucinating` | model didn't refuse due to missing context |
| `keyword_present` | expected term appears in the answer |
| `response_length_ok` | 5–100 words (not too short, not too long) |

---

## MLflow UI Walkthrough

After running any script, open **http://localhost:6001**:

1. **Experiments tab** — lists all experiments created by the tutorial
2. **Runs tab** — each script execution creates one or more runs
3. **Traces** — click a run → "Traces" to see the full call tree
4. **Metrics** — charts for latency, word count, success rate (Part 5)
5. **Evaluation** — per-sample scorer results (Parts 3, 4, 6)

---

## Key APIs Reference

```python
# Auto-instrument all LangChain calls
mlflow.langchain.autolog()

# Create / reuse an experiment
mlflow.set_experiment("my-experiment")

# Wrap work in a named run
with mlflow.start_run(run_name="my-run"):
    ...

# Log scalar metrics (optional step for time-series)
mlflow.log_metric("latency_ms", 123.4, step=1)

# Log multiple metrics at once
mlflow.log_metrics({"avg_latency": 200, "success_rate": 0.95})

# Tag a run for filtering
mlflow.set_tags({"model": "qwen3.5:0.8b", "env": "local"})

# Define a custom scorer
from mlflow.genai import scorer

@scorer
def my_scorer(outputs: str) -> bool:
    return len(outputs) > 0

# Run structured evaluation
mlflow.genai.evaluate(
    data=eval_dataset,         # list of {inputs, expectations}
    predict_fn=my_predict_fn,  # fn(question: str) -> str
    scorers=[my_scorer],
)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection refused` on port 11434 | Run `ollama serve` in a terminal |
| `Cannot set a deleted experiment` | Change the experiment name or restore it in the UI |
| `OPENAI_API_KEY must be set` | Add `os.environ["OPENAI_API_KEY"] = "dummy"` before imports |
| Scorers fail 3/3 | The scorer judge needs `OPENAI_BASE_URL` and `OPENAI_API_KEY` set |
| Very slow evaluation (10+ min) | Use `qwen3.5:0.8b` (fastest) or reduce dataset size |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Your Machine                        │
│                                                         │
│  ┌───────────┐   LangChain    ┌──────────────────────┐  │
│  │  Python   │ ─────────────► │  Ollama REST API     │  │
│  │  Script   │                │  localhost:11434     │  │
│  └─────┬─────┘                │  model: qwen3.5:0.8b │  │
│        │                      └──────────────────────┘  │
│        │ MLflow SDK                                      │
│        ▼                                                 │
│  ┌───────────┐                                           │
│  │  MLflow   │                                           │
│  │  Tracking │  localhost:6001                           │
│  │  Server   │                                           │
│  └─────┬─────┘                                           │
│        │                                                 │
│        ▼                                                 │
│  ┌───────────┐                                           │
│  │  mlruns/  │  (local filesystem store)                 │
│  │  database │                                           │
│  └───────────┘                                           │
└─────────────────────────────────────────────────────────┘
```
