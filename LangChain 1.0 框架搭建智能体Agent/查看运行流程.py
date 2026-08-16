import getpass
import operator
from typing import Annotated, List, Union
import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# 引入可视化UI库
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from dotenv import load_dotenv

# 2. 加载.env环境变量（解决Key问题）
load_dotenv()  # 自动读取当前目录下的.env文件
# 初始化控制台
console = Console()

# ---------------------- 定义工具 ----------------------
# 天气查询工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。"""
    weather_data = {
        "北京": "晴朗，气温25°C",
        "上海": "多云，气温28°C",
        "广州": "小雨，气温30°C"
    }
    return f"{city}的天气是：{weather_data.get(city, '未知')}"

# 加法计算工具
@tool
def add(a: float, b: float) -> float:
    """计算两个数的和"""
    return a + b

tools = [get_weather, add]

# ---------------------- 初始化模型 & 构建Agent图 ----------------------
model = init_chat_model(
    model="qwen3.7-max",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
graph = create_agent(model, tools=tools)

# ---------------------- 可视化流式运行函数 ----------------------
def run_demo_with_visualization(user_input: str):
    print("\n" + "=" * 50)
    console.print(f"[bold yellow]开始任务：[/bold yellow] {user_input}")
    messages = [HumanMessage(content=user_input)]
    step_count = 1

    for event in graph.stream({"messages": messages}, stream_mode="values"):
        current_message = event["messages"][-1]

        # 跳过用户原始消息
        if isinstance(current_message, HumanMessage):
            continue

        # AI 决策 / 最终回复
        if isinstance(current_message, AIMessage):
            if current_message.tool_calls:
                for tool_call in current_message.tool_calls:
                    console.print(Panel(
                        Text(
                            f"🤔 AI 思考决定：需要调用外部工具\n"
                            f"🔧 工具名称: {tool_call['name']}\n"
                            f"📥 输入参数: {tool_call['args']}",
                            style="bold cyan"
                        ),
                        title=f"Step {step_count}: 决策 (Decision)",
                        border_style="cyan"
                    ))
            else:
                console.print(Panel(
                    Markdown(current_message.content),
                    title=f"Step {step_count}: 最终回复 (Final Answer)",
                    border_style="green"
                ))
            step_count += 1

        # 工具执行结果
        if isinstance(current_message, ToolMessage):
            console.print(Panel(
                Text(f"👀 工具返回结果 (Observation):\n{current_message.content}"),
                title=f"Step {step_count}: 执行与观察",
                border_style="magenta"
            ))
            step_count += 1

# ---------------------- 程序入口 ----------------------
if __name__ == "__main__":
    test_query = "查询一下北京和上海气温，并且计算一下北京的温度比上海低多少度"
    run_demo_with_visualization(test_query)