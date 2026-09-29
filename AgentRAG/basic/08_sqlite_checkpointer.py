from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    counter: int


def increment(state: State) -> dict:
    return {"counter": state["counter"] + 1}


builder = StateGraph(State)
builder.add_node("increment", increment)
builder.add_edge(START, "increment")
builder.add_edge("increment", END)

with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph=builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "counter-thread"}}

    # 첫 실행: 입력 필요
    print(graph.invoke({"counter": 0}, config))   # {'counter': 1}

    # 같은 스레드로 다시 실행: 마지막 체크포인트 위에서 이어감
    print(graph.invoke({"counter": 10}, config))  # {'counter': 11}

    print(graph.get_state(config).values)

# Postgres는 패키지만 다르고 사용법은 동일:
# pip install -U langgraph-checkpoint-postgres
# from langgraph.checkpoint.postgres import PostgresSaver
# with PostgresSaver.from_conn_string("postgresql://user:pw@host:5432/db") as cp:
#     cp.setup()  # 최초 1회 테이블 생성
#     graph = builder.compile(checkpointer=cp)



# 장기 기억은 Store 사용(InMemoryStore, PostgresStore)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store, get_config

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def remember(state: MessagesState):
    store=get_store()
    user_id=get_config()["configurable"]["user_id"]
    ns=(user_id, "preferences")
    last=state["messages"][-1].content

    if "반말" in last:
        store.put(ns, "tone", {"value": "반말 선호"})

    prefs=store.search(ns)
    tone=prefs[0].value["value"] if prefs else "기본"
    reply=llm.invoke([SystemMessage(f"말투: {tone}")] + state["messages"])
    return {"messages": [reply]}

builder = StateGraph(MessagesState)
builder.add_node("remember", remember)
builder.add_edge(START, "remember")

graph = builder.compile(
    checkpointer=InMemorySaver(),  # thread 안 기억
    store=InMemoryStore(),         # thread 넘는 기억
)