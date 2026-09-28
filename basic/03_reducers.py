import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    # 리스트 + operator.add 리듀서 -> 새 리스트가 뒤에 이어붙음(append)
    logs: Annotated[list[str], operator.add]
    # 메시지 전용 리듀서: append + 같은 id면 교체 + dict를 Message 객체로 자동 변환
    messages: Annotated[list[BaseMessage], add_messages]
    # 덮어쓰기
    step: int

def node_a(state: State) -> dict:
    return {
        "logs": ["A 실행"],
        "messages": [AIMessage(content = "A에서 인사드려요")],
        "step": 1,
    }

def node_b(state: State) -> dict:
    return {
        "logs": ["B 실행"],
        "messages": [{"role": "ai", "content": "B에서도 인사드려요"}],
        "step": 2,
    }

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)
graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke(
        {
            "logs": ["시작"],
            "messages": [HumanMessage(content="안녕?")],
            "step": 0
        }
    )
    print("logs :", result["logs"])
    print("step :", result["step"])
    for m in result["messages"]:
        print(f"  [{m.type}] {m.content}")