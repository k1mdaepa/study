from typing import Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt

class State(TypedDict):
    topic: str
    draft: str
    status: str

def write_draft(state: State) -> dict:
    # 실무에선 LLM 호출. 여기선 흐름에 집중하려고 문자열 조립만
    return {"draft": f"[공지] {state['topic']}에 대해 안내드립니다. 자세한 내용은 추후 공유하겠습니다."}

def human_review(state: State) -> Command[Literal["send", "revise"]]:
    # interrupt()에 넘긴 값이 그대로 호출자에게 전달됨 (UI에 띄울 정보)
    decision = interrupt(
        {
            "question": "이 초안을 발송할까요? (approve / revise)",
            "draft": state["draft"]
        }
    )
    # 재개 시 Command(resume=값)의 '값'이 여기로 들어옴
    if decision == "approve":
        return Command(goto="send", update={"status": "승인됨"})
    return Command(goto='revise', update={"status": "수정 요청"})

def send(state: State) -> dict:
    return {"status": state["status"] + " -> 발송 완료 ✅"}


def revise(state: State) -> dict:
    return {"draft": state["draft"] + " (담당자 검토 후 수정 예정)", "status": state["status"] + " -> 재작성"}


builder = StateGraph(State)
builder.add_node("write_draft", write_draft)
builder.add_node("human_review", human_review)
builder.add_node("send", send)
builder.add_node("revise", revise)

builder.add_edge(START, "write_draft")
builder.add_edge("write_draft", "human_review")
builder.add_edge("send", END)
builder.add_edge("revise", END)

# interrupt는 checkpointer가 필수
graph = builder.compile(checkpointer=InMemorySaver())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "notice-001"}}

    result = graph.invoke({"topic": "서버 점검", "draft": "", "status": ""}, config)
    print(f"first graph result: {result}")

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("⏸️  중단됨:", payload["question"])
        print("    초안:", payload["draft"])

    print("다음 노드:", graph.get_state(config).next)  # ('human_review',)

    # ---------- 2차 실행: 사람의 답을 들고 재개 ----------
    user_decision = input("\n결정을 입력하세요 (approve / revise): ").strip() or "approve"
    result = graph.invoke(Command(resume=user_decision), config)

    print("\n최종 상태:", result["status"])
    print("최종 초안:", result["draft"])

# 재개할 시, interrupt()가 있는 노드의 처음부터 다시 실행, interrupt() 앞에 있는 코드는 멱등성을 가져야 함.