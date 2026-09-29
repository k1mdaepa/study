import json
from openai import OpenAI
from common import config

client = OpenAI(
    api_key=config.openai_api_key
)

# 도구 정의
# ---------- 1) 도구(Tools) 정의: 그냥 평범한 파이썬 함수 ----------
def get_weather(city: str) -> str:
    """도시 이름을 받아 현재 날씨를 문자열로 돌려준다 (실습용 가짜 데이터)."""
    fake_db = {
        "서울": "맑음, 24도",
        "부산": "흐림, 21도",
        "제주": "비, 19도",
    }
    return fake_db.get(city, f"{city}의 날씨 정보가 없어요")


def calculator(expression: str) -> str:
    """사칙연산 수식을 계산한다. 예: '24 - 21'"""
    try:
        # 보안상 builtins를 비워서 eval 범위를 산술식으로 제한
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"계산 실패: {e}"

# 이름 -> 실제 함수 매핑
TOOLS = {
    "get_weather": get_weather,
    "calculator": calculator,
}

# LLM에게 보여줄 도구 명세 (JSON Schema)
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "도시의 현재 날씨를 알려준다",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "도시 이름 (예: 서울)"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "사칙연산 수식을 계산한다",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "계산할 수식 (예: 24 - 21)"}
                },
                "required": ["expression"],
            },
        },
    },
]

def run_agent(question: str, max_steps: int= 6) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "너는 도구를 활용해 질문에 답하는 비서다."
                "필요하면 도구를 호출하고, 결과를 확인한 뒤 최종 답을 한국어로 정리해라"
            )
        },
        {"role": "user", "content": question},
    ]

    for step in range(1, max_steps+1):
        # Thought + Action 결정
        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = messages,
            tools = TOOL_SPECS,
        )
        print(f"response: {response}")

        msg = response.choices[0].message
        print(f"msg: {msg}")

        if not msg.tool_calls:
            return msg.content

        # 메세지 저장
        messages.append(msg)

        # Action 실행 -> Observation 수집
        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"[Step {step}] Action {name}: {args}")

            result = TOOLS[name](**args)
            print(f"[Step {step}] Observation: {result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                }
            )

    return "최대 단계를 초과했어요. 질문을 더 구체적으로 해주세요."

if __name__ == "__main__":
    answer = run_agent("서울이랑 부산 날씨 알려주고, 두 도시 기온 차이도 계산해줘")
    print("최종 답변:", answer)
