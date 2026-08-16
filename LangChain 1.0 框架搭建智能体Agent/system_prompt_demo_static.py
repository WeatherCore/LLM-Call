from langchain.agents import create_agent
from langchain.tools import tool

# 1. 天气工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    weather_data = {
        "北京": "晴朗，气温25°C",
        "上海": "多云，气温28°C",
        "广州": "小雨，气温30°C"
    }
    return f"{city}的天气是: {weather_data.get(city, '未知')}"

# 2. 静态固定system_prompt
agent_static = create_agent(
    model="openai:gpt-4o-mini",
    tools=[get_weather],
    system_prompt=(
        "你是一个天气助手，回答不超过20字。\n"
        "调用工具时，严格按照以下格式：\n"
        "1. 使用 `get_weather(city: str)` 获取天气；\n"
        "2. 仅返回天气结果，不解释过程。"
    )
)

print("=== 静态 System Prompt ===")
response1 = agent_static.invoke({
    "messages": [{"role": "user", "content": "北京天气"}]
})
print(f"AI: {response1['messages'][-1].content}")