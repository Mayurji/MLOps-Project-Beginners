"""
============================================================
 02_agent_tracing.py
 ─────────────────────────────────────────────────────────────
 Tutorial Part 2 – Tracing a LangGraph ReAct Agent
 ─────────────────────────────────────────────────────────────
 What you will learn
 ───────────────────
   • How to build a ReAct-style tool-using agent with LangGraph
     (the modern successor to LangChain AgentExecutor)
   • How MLflow traces every tool call AND the LLM reasoning
     step automatically via mlflow.langchain.autolog()
   • How to tag runs for easy filtering in the MLflow UI

 Architecture of the agent
 ─────────────────────────
     User Question
          │
          ▼
     ReAct Agent  ◄──── ChatOllama (qwen3.5:0.8b)
          │
    ┌─────┴──────┐
    │  Tools     │
    │  ─────     │
    │ calculator │
    │ date_tool  │
    │ wiki_stub  │
    └────────────┘
          │
          ▼
     Final Answer → MLflow Trace

 Run
 ───
   ~/python3-env/bin/python3 02_agent_tracing.py
============================================================
"""

import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*create_react_agent.*")

import mlflow
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# ── 1. MLflow setup ───────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:6001")
mlflow.set_experiment("langchain-ollama-agents")
mlflow.langchain.autolog()

# ── 2. Define custom tools ────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a basic math expression and return the numeric result.
    Input: a Python-evaluable arithmetic string, e.g. '15 * 7 + 3'.
    """
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {e}"


@tool
def get_current_date(query: str = "") -> str:
    """Return today's date in YYYY-MM-DD format."""
    return datetime.date.today().isoformat()


@tool
def wiki_stub(topic: str) -> str:
    """
    Return a short factual summary about a well-known topic.
    Supported topics: python, mlflow, langchain, ollama, langgraph.
    """
    facts = {
        "python":     "Python is a high-level, interpreted language created by Guido van Rossum in 1991.",
        "mlflow":     "MLflow is an open-source platform for managing the ML lifecycle: tracking, packaging, and deployment.",
        "langchain":  "LangChain is a framework for building LLM-powered apps with chains, agents, and memory.",
        "ollama":     "Ollama lets you run large language models locally via a simple REST API.",
        "langgraph":  "LangGraph is a library for building stateful, multi-step agents and workflows using LangChain components.",
    }
    key = topic.lower().strip()
    return facts.get(key, f"No stub for '{topic}'. Try: python, mlflow, langchain, ollama, langgraph.")


# ── 3. Create the LangGraph ReAct agent ──────────────────────
llm = ChatOllama(
    model="qwen3.5:0.8b",
    base_url="http://localhost:11434",
    temperature=0.1,
)

tools = [calculator, get_current_date, wiki_stub]

# create_react_agent from LangGraph is the modern replacement
# for LangChain's AgentExecutor + create_react_agent pattern
agent = create_react_agent(
    model=llm,
    tools=tools,
)

# ── 4. Run with MLflow tracking ───────────────────────────────
test_questions = [
    "What is 128 divided by 4, then multiplied by 3?",
    "What is today's date?",
    "Tell me about MLflow.",
    "What is LangGraph?",
]

with mlflow.start_run(run_name="react-agent-demo"):
    mlflow.set_tags({
        "agent_type":  "ReAct",
        "llm_model":   "qwen3.5:0.8b",
        "provider":    "Ollama",
        "framework":   "LangGraph",
    })

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*60}")
        print(f"Question {i}: {question}")
        print("="*60)
        try:
            result = agent.invoke({"messages": [("human", question)]})
            # The final answer is in the last message
            final_answer = result["messages"][-1].content
            print(f"Final Answer: {final_answer}")
        except Exception as e:
            print(f"Agent error: {e}")

print("\n✅  Agent run complete – open http://localhost:6001 to view traces.")
