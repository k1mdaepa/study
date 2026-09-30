from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# State Schema
class State(TypedDict):
    text: str
    count: int

# Node
def uppercase(state: State) -> dict:
    return {"text": state["text"].upper()}

def add_exclaim(state: State) -> dict:
    return {
        "text": state["text"] + "!!!",
        "count": state["count"] + 1,
    }

# Graph 조립
builder = StateGraph(State)

builder.add_node("uppercase", uppercase)
builder.add_node("add_exclaim", add_exclaim)

builder.add_edge(START, "uppercase")
builder.add_edge("uppercase", "add_exclaim")
builder.add_edge("add_exclaim", END)

graph = builder.compile()

# 실행
if __name__ == "__main__":
    result = graph.invoke(
        {
            "text": "hello langgraph",
            "count": 2,
        }
    )
    print(result)

    print(graph.get_graph().draw_mermaid())