import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = init_chat_model(
    model="qwen3.7-max",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

res = model.invoke([
    SystemMessage("你是乐于助人的AI助手"),
    HumanMessage("你是哪个AI")
])

print(res)
print(" ----------------------")
print(res.content)
print(" ----------------------")
print(res.content_blocks)

# 标准流式写法
# for chunk in model.stream("用很多句话描述计算机行业"):
#     print(chunk.content, end="", flush=True)