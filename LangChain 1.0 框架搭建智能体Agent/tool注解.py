from langchain_core.tools import tool
from langchain.agents import create_agent
from model_loader import load_chat_model
import os
from dotenv import load_dotenv

# 1. 加载环境变量（关键修复）
load_dotenv()  # 读取.env文件中的DEEPSEEK_API_KEY

# 1.自定义乘法工具
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b. 接收两个整数，返回相乘结果"""
    return a * b

# 2.加载模型（二选一，OpenAI/Kimi均可）
model = load_chat_model(model="deepseek-chat",provider="deepseek")

# 3.创建Agent，挂载自定义工具
agent = create_agent(model=model, tools=[multiply])

# 4.发起调用
response = agent.invoke({
    "messages": [{"role": "user", "content": "帮我计算12乘以6等于多少？"}]
})

# 查看全链路消息（包含思考、工具调用、结果）
print(response["messages"])
# 提取最终回答
print("最终答案：", response["messages"][-1].content)