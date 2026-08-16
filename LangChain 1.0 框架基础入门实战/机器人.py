import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage,AIMessage

load_dotenv()

# 1 初始化模型 (LangChain 1.0 接口)
# model = init_chat_model(
#     model="qwen3.7-max",
#     model_provider="openai",
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

model = init_chat_model(
    model="deepseek-ai/DeepSeek-V3",
    model_provider="openai",
    api_key=os.getenv("SILICON_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

# 2 初始化系统提示 (System Prompt)
system_message = SystemMessage(
    content="你叫 Weather，是一名乐于助人的智能助手。请在对话中保持温和、有耐心的语气。"
)

# 3 初始化消息历史
messages = [system_message]
print(" • 输入 exit 退出对话\n")

# 4 主循环（支持多轮对话 + 流式输出）
while True:
    user_input = input("👤 你: ")
    if user_input.lower() in {"exit", "quit"}:
        print("✨ 对话结束，期待下次又一次见面！")
        break

    # 追加用户消息
    messages.append(HumanMessage(content=user_input))

    # 实时输出模型生成内容
    print("🤖 Weather: ", end="", flush=True)
    full_reply = ""

    # LangChain 1.0 标准流式写法
    for chunk in model.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_reply += chunk.content

    print("\n" + "-" * 40)
    # 把AI完整回复存入消息历史
    messages.append(AIMessage(content=full_reply))
    # 滑动窗口：只保留最近50轮消息，控制token开销
    messages = messages[-50:]