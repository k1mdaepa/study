from typing import Literal, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class State(TypedDict):
    amount: int
    status: str


def check_amount(state: State) -> Command[Literal["auto_approve", "manual_review"]]:
    if state["amount"] < 100_000:
        return Command(update={"status": "소액"}, goto="auto_approve")
    return Command(update={"status": "고액"}, goto="manual_review")

def auto_approve(state: State) -> dict:
    return {"status": state["status"] + " -> 자동 승인"}


def manual_review(state: State) -> dict:
    return {"status": state["status"] + " -> 담당자 검토 필요"}


builder = StateGraph(State)
builder.add_node("check_amount", check_amount)
builder.add_node("auto_approve", auto_approve)
builder.add_node("manual_review", manual_review)

builder.add_edge(START, "check_amount")
# check_amount는 Command로 스스로 이동하므로 조건부 엣지를 따로 안 붙여도 됨
builder.add_edge("auto_approve", END)
builder.add_edge("manual_review", END)
graph = builder.compile()


if __name__ == "__main__":
    print(graph.invoke({"amount": 50_000, "status": ""}))
    print(graph.invoke({"amount": 500_000, "status": ""}))