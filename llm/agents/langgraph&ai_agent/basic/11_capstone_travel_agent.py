# 실습
# 날씨를 조회하고 호텔을 검색하는건 자유, 실제 예약은 사람 승인을 받는 에이전트
# 라우터가 도구 이름을 보고 결정

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from common import config

# 도구
@tool
def get_weather(city: str) -> str:
    """도시의 이번 주말 날씨 예보를 알려준다."""
    fake_db = {
        "서울": "맑음, 24도",
        "부산": "흐림, 21도",
        "제주": "비, 19도",
        "강릉": "맑음, 22도"
    }
    return fake_db.get(city, f"{city}의 예보가 없어요.")

@tool
def search_hotels(city: str, max_price: int) -> str:
    """도시와 1박 최대 가격(원)으로 호텔을 검색한다."""
    fake_hotels = {
        "강릉": [("바다뷰 호텔", 120000), ("솔향기 리조트", 95000), ("경포 게스트하우스", 45000)],
        "제주": [("한라 리조트", 150000), ("올레 게스트하우스", 50000)],
        "부산": [("해운대 그랜드", 180000), ("광안리 비치텔", 80000)]
    }
    candidates = [f"{name} ({price:,}원)" for name, price in fake_hotels.get(city, []) if price <= max_price]
    return ", ".join(candidates) if candidates else f"{city}에서 {max_price:,}원 이하 호텔이 없어요."

@tool
def book_hotel(hotel_name: str, nights: int) -> str:
    """호텔을 실제로 예약한다. 되돌릴 수 없으므로 신중히 호출한 것"""
    return f"✅ 예약 완료: {hotel_name}, {nights}박. 예약번호 BK-{abs(hash(hotel_name)) % 100000:05d}"

tools = [get_weather, search_hotels, book_hotel]
SENSITIVE_TOOLS = {"book_hotel"}

# state & llm
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm_with_tools = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature = 0,
    api_key=config.openai_api_key
).bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content=(
        "너는 국내 여행 비서다. 날씨와 호텔을 조회해서 사용자에게 추천하고, "
        "사용자가 명시적으로 예약을 원할 때만 book_hotel을 호출해라. "
        "예약이 거부되면 사과하고 대안을 제시해라. 답변은 한국어로."
    )
)

# node
def agent(state: AgentState) -> dict:
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def human_review(state: AgentState) -> Command[Literal["tools", "agent"]]:
    last_ai = state["messages"][-1]
    pending = [c for c in last_ai.tool_calls if c["name"] in SENSITIVE_TOOLS]

    decision = interrupt(
        {
            "question": "다음 작업을 실행할까요? (yes / no)",
            "tool_calls": [{"name": c["name"], "args": c["args"]} for c in pending]
        }
    )

    if decision == "yes":
        return Command(goto="tools")

    # 거부: 모든 tool_call에 대해 ToolMessage를 남겨야 LLM 히스토리가 유효하게 유지
    denied = [
        ToolMessage(
            content="사용자가 이 작업을 거부했습니다. 실행되지 않았습니다.",
            tool_call_id=c["id"],
        )
        for c in last_ai.tool_calls
    ]
    return Command(goto="agent", update={"messages": denied})


# router
def route_after_agent(state: AgentState) -> Literal["tools", "human_review", "__end__"]:
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        return END
    if any(c["name"] in SENSITIVE_TOOLS for c in last.tool_calls):
        return "human_review"
    return "tools"

# graph
builder = StateGraph(AgentState)
builder.add_node("agent", agent)
builder.add_node("tools", tool_node)
builder.add_node("human_review", human_review)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "agent")

graph = builder.compile(checkpointer=InMemorySaver())

# conversion loop
def print_new_messages(messages: list[BaseMessage], start_index: int) -> None:
    for m in messages[start_index:]:
        if m.type == "ai" and m.tool_calls:
            calls = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
            print(f"  🤖 도구 호출 요청: {calls}")
        elif m.type == "ai":
            print(f"  🤖 {m.content}")
        elif m.type == "tool":
            print(f"  🔧 {m.content}")


def run_turn(user_text: str, configs: dict) -> None:
    before = len(graph.get_state(configs).values.get("messages", []))
    result = graph.invoke({"messages": [HumanMessage(content=user_text)]}, configs)

    # interrupt로 멈췄으면 사람에게 물어보고 재개 (승인 후 또 다른 승인이 필요할 수도 있으니 while)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print(f"\n  ✋ {payload['question']}")
        for call in payload["tool_calls"]:
            print(f"     - {call['name']} {call['args']}")
        answer = input("  👉 입력 (yes/no): ").strip().lower() or "no"
        result = graph.invoke(Command(resume=answer), configs)

    print_new_messages(result["messages"], before)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "trip-2026-autumn"}}

    print("=== 여행 비서 (종료: quit) ===")
    print("예시: '이번 주말 강릉 날씨 어때?' -> '10만원 이하 호텔 찾아줘' -> '솔향기 리조트 2박 예약해줘'")
    while True:
        text = input("\n👤 ").strip()
        if text.lower() in {"quit", "exit", "종료"}:
            break
        if text:
            run_turn(text, config)