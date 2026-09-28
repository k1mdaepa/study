from typing import Literal, TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    text: str
    sentiment: str
    reply: str

def classify(state: State) -> dict:
    """아주 단순한 규칙 기반 감정 분류 (실무에선 LLM이나 분류 모델로 교체)."""
    negative_words = ["짜증", "힘들", "싫", "우울"]
    is_negative = any(word in state["text"] for word in negative_words)
    return {"sentiment": "negative" if is_negative else "positive"}

def cheer(state: State) -> dict:
    return {"reply": "좋은 기분 그대로 쭉 가요! 🎉"}

def comfort(state: State) -> dict:
    return {"reply": "오늘 많이 힘드셨군요. 잠깐 쉬어가도 괜찮아요. ☕"}

# router: State를 읽고 "다음 노드 이름"을 반환
# Literal 타입 힌트는 시각화용
def route_by_sentiment(state: State) -> Literal["cheer", "comfort"]:
    if state["sentiment"] == "negative":
        return "comfort"
    return "cheer"

builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("cheer", cheer)
builder.add_node("comfort", comfort)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_by_sentiment)
builder.add_edge("cheer", END)
builder.add_edge("comfort", END)

graph = builder.compile()


if __name__ == "__main__":
    for text in ["오늘 코드가 한 번에 돌아갔어!", "배포 실패해서 너무 짜증나"]:
        result = graph.invoke({"text": text, "sentiment": "", "reply": ""})
        print(f"{text!r} -> [{result['sentiment']}] {result['reply']}")