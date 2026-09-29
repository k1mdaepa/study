from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver


# ---------- 도구 ----------
@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 조회한다."""
    fake = {"제주": "비, 18도", "부산": "맑음, 23도", "서울": "흐림, 20도"}
    return f"{city}: {fake.get(city, '정보 없음')}"


tools = [get_weather]
CITIES = ["제주", "부산", "서울"]


# ---------- 가짜 LLM agent ----------
def agent(state: MessagesState):
    last = state["messages"][-1]

    # 사용자 질문이 들어오면 → 도구 호출 요청 (LLM이 tool_calls 만드는 흉내)
    if isinstance(last, HumanMessage):
        city = next((c for c in CITIES if c in last.content), "서울")
        return {"messages": [AIMessage(
            content="",
            tool_calls=[{"name": "get_weather", "args": {"city": city}, "id": "call_1"}],
        )]}

    # 도구 결과가 들어오면 → 최종 답변
    if isinstance(last, ToolMessage):
        return {"messages": [AIMessage(content=f"조회 결과: {last.content}")]}


# ---------- 그래프 ----------
builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=InMemorySaver(), interrupt_before=["tools"])

# 멈춘 상태 확인
def show_pending(config):
    snap = graph.get_state(config)
    print("  next:", snap.next)
    print("  실행 예정 도구:", snap.values["messages"][-1].tool_calls[0]["args"])


# ---------- 케이스 1: 그대로 재개 ----------
print("=== 케이스 1: 그대로 재개 ===")
config1 = {"configurable": {"thread_id": "debug-1"}}

graph.invoke({"messages": [("user", "제주 날씨?")]}, config1)
print("⏸ 멈춤")
show_pending(config1)

result = graph.invoke(None, config1)
print("▶ 재개 후 답변:", result["messages"][-1].content)


# ---------- 케이스 2: 도구 인자 고쳐서 재개 ----------
print("\n=== 케이스 2: 제주 → 부산으로 바꿔서 재개 ===")
config2 = {"configurable": {"thread_id": "debug-2"}}

graph.invoke({"messages": [("user", "제주 날씨?")]}, config2)
print("⏸ 멈춤 (수정 전)")
show_pending(config2)

# 직접 업데이트
last = graph.get_state(config2).values["messages"][-1]
last.tool_calls[0]["args"] = {"city": "부산"}
graph.update_state(config2, {"messages": [last]})   # 같은 id라 교체됨

print("✏️ 수정 후")
show_pending(config2)

result = graph.invoke(None, config2)
print("▶ 재개 후 답변:", result["messages"][-1].content)


# ---------- 메시지 흐름 ----------
print("\n=== 케이스 2 메시지 전체 ===")
for m in result["messages"]:
    m.pretty_print()