from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from common import config


@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 알려준다."""
    fake_db = {"서울": "맑음, 24도", "부산": "흐림, 21도", "제주": "비, 19도"}
    return fake_db.get(city, f"{city}의 날씨 정보가 없어요")


tools = [get_weather]

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


llm_with_tools = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=config.openai_api_key
).bind_tools(tools)

def call_model(state: AgentState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


builder = StateGraph(AgentState)

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

# CheckPointer
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer)

def chat(thread_id: str, text: str) -> str:
    configs = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=text)]}, configs)
    print(f"chat result: {result}")
    print(f"chat result content: {result["messages"][-1].content}")
    return result["messages"][-1].content



if __name__ == "__main__":
    # 스레드 A: 이름을 알려줌
    print("A:", chat("thread-A", "내 이름은 민수야. 기억해줘."))
    print("A:", chat("thread-A", "제주 날씨 알려줘"))
    print("A:", chat("thread-A", "내 이름이 뭐라고 했지?"))   # 기억함

    # 스레드 B: 완전히 별개의 대화
    print("B:", chat("thread-B", "내 이름이 뭐라고 했지?"))   # 모름

    # ---------- 상태 들여다보기 ----------
    config_a = {"configurable": {"thread_id": "thread-A"}}
    snapshot = graph.get_state(config_a)
    print("\n[thread-A 메시지 수]", len(snapshot.values["messages"]))
    print("[다음 실행될 노드]", snapshot.next)  # 끝난 상태면 빈 튜플 ()

    # 체크포인트 히스토리 (최신 -> 과거 순)
    history = list(graph.get_state_history(config_a))
    print("[체크포인트 개수]", len(history))
    print("[가장 오래된 체크포인트의 메시지 수]", len(history[-1].values["messages"]))