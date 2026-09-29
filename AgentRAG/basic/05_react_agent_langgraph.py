from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from common import config


# ---------- 1) 도구: @tool 데코레이터가 docstring과 타입힌트로 스키마를 자동 생성 ----------
@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 알려준다. city는 도시 이름 (예: 서울)."""
    fake_db = {"서울": "맑음, 24도", "부산": "흐림, 21도", "제주": "비, 19도"}
    return fake_db.get(city, f"{city}의 날씨 정보가 없어요")


@tool
def calculator(expression: str) -> str:
    """사칙연산 수식을 계산한다. 예: '24 - 21'"""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"계산 실패: {e}"


tools = [get_weather, calculator]

# state
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# llm tool binding
llm = ChatOpenAI(
    model='gpt-4o-mini',
    temperature=0,
    api_key=config.openai_api_key,
)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = SystemMessage(
    content="너는 도구를 활용해 질문에 답하는 비서다. 필요하면 도구를 호출하고, "
            "결과를 확인한 뒤 최종 답을 한국어로 정리해라."
)

# node
def call_model(state: AgentState) -> dict:
    response=llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)  # "tools" 또는 END 반환
builder.add_edge("tools", "agent")                       # 도구 실행 후 다시 agent로 (루프)

graph = builder.compile()

if __name__ == "__main__":
    question = "서울이랑 부산 날씨 알려주고, 두 도시 기온 차이도 계산해줘"

    # stream_mode="updates": 각 노드가 반환한 업데이트만 순서대로 받음
    for chunk in graph.stream(
        {"messages": [HumanMessage(content=question)]},
        stream_mode="updates",
    ):
        for node_name, update in chunk.items():
            for m in update["messages"]:
                if m.type == "ai" and m.tool_calls:
                    calls = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
                    print(f"[{node_name}] Action     : {calls}")
                elif m.type == "tool":
                    print(f"[{node_name}] Observation: {m.content}")
                else:
                    print(f"[{node_name}] Answer     : {m.content}")