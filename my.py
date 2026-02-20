from fastapi import FastAPI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()


class State(TypedDict):
    ml_output: str
    summary: str



llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


def load_ml(state: State):
    with open("ml_report.txt", "r", encoding="utf-8") as f:
        data = f.read()

    return {"ml_output": data}


def summarize(state:State):
    text = state["ml_output"]

    prompt = f"""
You are a business reporting assistant.

Read the ML sales forecast and format it clearly.

Rules:
- Keep product order exactly as in ML report
- Do NOT change numbers
- Separate growing products into a different section
- Add only 2 short summary lines at the end
- No extra explanation

Format:

DAILY SALES REPORT
------------------

FORECAST RESULTS:
(list products in order exactly as given)

GROWING STOCK:
(only products with positive forecast)

SUMMARY:
Line 1
Line 2

ML REPORT:
{text}

Return plain text only.
"""

    result = llm.invoke(prompt).content
    return {"summary": result}

    with open("manager_summary.txt", "w", encoding="utf-8") as f:
        f.write(result)

    return {"summary": result}


graph = StateGraph(State)

graph.add_node("load", load_ml)
graph.add_node("summarize", summarize)

graph.add_edge(START, "load")
graph.add_edge("load", "summarize")
graph.add_edge("summarize", END)

workflow = graph.compile()



app = FastAPI()

@app.get("/report")
def get_report():
    result = workflow.invoke({})
    return {
        "manager_summary": result["summary"]
    }