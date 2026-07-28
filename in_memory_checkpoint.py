import os
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()  # 自动读取当前目录下的.env文件

# ========== 初始化 LLM ==========
# 请自行配置环境变量与模型，这里使用你笔记内的写法
model = init_chat_model(
    model="qwen3.7-max",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# ========== 定义工具函数 ==========
@tool
def get_user_info(name: str) -> str:
    """查询用户信息，返回姓名、年龄、爱好"""
    user_db = {
        "陈明": {"age": 28, "hobby": "旅游、滑雪、喝茶"},
        "张三": {"age": 32, "hobby": "编程、阅读、电影"}
    }
    info = user_db.get(name, {"age": "未知", "hobby": "未知"})
    return f"姓名: {name}, 年龄: {info['age']}岁, 爱好: {info['hobby']}"


# ========== 1. 基础短期记忆：InMemorySaver ==========
def demo_inmemory_memory():
    print("=" * 60)
    print("场景 1: 内存记忆（开发环境）")
    print("=" * 60)

    # 创建内存检查点
    memory = InMemorySaver()

    # 创建 Agent（自动继承对话记忆能力）
    agent = create_agent(
        model=model,
        tools=[get_user_info],
        checkpointer=memory  # 启用短期记忆
    )

    # 配置: thread_id 作为会话 ID
    config = {"configurable": {"thread_id": "user_123"}}

    # 第一轮对话: 自我介绍
    response1 = agent.invoke(
        {"messages": [{"role": "user", "content": "你好，我叫陈明，好久不见！"}]},
        config=config
    )
    print(f"用户: 你好，我叫陈明，好久不见！")
    print(f"AI: {response1['messages'][-1].content}")
    print("-" * 40)

    # 第二轮对话: 测试记忆
    response2 = agent.invoke(
        {"messages": [{"role": "user", "content": "请问你还记得我叫什么名字吗?"}]},
        config=config
    )
    print(f"用户: 请问你还记得我叫什么名字吗?")
    print(f"AI: {response2['messages'][-1].content}")
    print("-" * 40)

    # 验证记忆状态
    state = agent.get_state(config)
    print(f"当前记忆轮次: {len(state.values['messages'])} 条消息")
    print("-" * 40)

    # 新开一个会话（不同 thread_id）
    config2 = {"configurable": {"thread_id": "user_456"}}
    response3 = agent.invoke(
        {"messages": [{"role": "user", "content": "我们之前聊过吗?"}]},
        config=config2
    )
    print(f"新会话 AI: {response3['messages'][-1].content}")  # 应无记忆


if __name__ == "__main__":
    demo_inmemory_memory()