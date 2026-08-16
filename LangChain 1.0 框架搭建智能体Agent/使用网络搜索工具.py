import os
from langchain.agents import create_agent
from langchain_tavily import TavilySearch  # 新版导入
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 2. 加载.env环境变量（解决Key问题）
load_dotenv()  # 自动读取当前目录下的.env文件

# 3.实例化搜索工具
web_search = TavilySearch(
    max_results=5,
    api_key=os.getenv("TAVILY_API_KEY"),
)  # max_results：单次搜索返回5条网页摘要

# 4.加载大模型
model = init_chat_model(
    model="qwen3.7-max",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 5.创建智能体
agent = create_agent(
    model=model,
    tools=[web_search],
    system_prompt="你是一名多才多艺的智能助手，可以调用工具帮助用户解决问题。"
)

# 6.发起请求
result = agent.invoke({
    "messages": [{"role":"user","content":"请帮我查询2026年菲尔兹得主是谁？"}]
})

# 7.提取最终回答
print(result['messages'][-1].content)


# 末尾消息=最终AI输出，一定是AIMessage
last_msg = result["messages"][-1]
blocks = last_msg.content_blocks
print("标准化内容块：",blocks)
