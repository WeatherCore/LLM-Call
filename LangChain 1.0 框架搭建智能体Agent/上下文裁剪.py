# ============ 核心模块导入 ============
import os
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver  # 短期记忆存储
from langchain_core.messages import trim_messages  # 消息裁剪工具
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from dotenv import load_dotenv

# 2. 加载.env环境变量（解决Key问题）
load_dotenv()  # 自动读取当前目录下的.env文件

# ============ 初始化 LLM 与工具 ============
model = init_chat_model(
    model="qwen3.7-max",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 1. 定义天气查询工具
@tool
def search_weather(city: str) -> str:
    """查询城市天气"""
    return f"{city}天气：晴，25°C"

# ============ 配置裁剪参数 ============
MAX_TOKENS = 100  # 根据模型上下文长度调整，上下文128K的话，一般设置为4000左右
TRIM_STRATEGY = "last"  # 保留最新消息
INCLUDE_SYSTEM = True  # 系统消息不参与裁剪

"""
tiktoken 是 OpenAI 官方 token 编码库，支持精确计数。
不同模型需使用不同编码：
- gpt-4o, gpt-4o-mini: o200k_base
- gpt-4-turbo: cl100k_base
- text-davinci-003: p50k_base
"""
import tiktoken

# 2. 定义 Token 编码器获取函数
def get_token_encoder(model_name: str = "gpt-4o-mini"):
    """获取 tiktoken 编码器实例"""
    try:
        # 自动映射模型到编码器
        encoding = tiktoken.encoding_for_model(model_name)
        print(f"✅ 已加载模型 '{model_name}' 的 tiktoken 编码器")
        return encoding
    except KeyError:
        # 如果模型不在官方列表，使用默认编码
        print(f"⚠️  模型 '{model_name}' 未在映射表中，使用默认编码 'o200k_base'")
        return tiktoken.get_encoding("o200k_base")

# 3. 初始化编码器（全局复用提升性能）
TOKEN_ENCODER = get_token_encoder("gpt-4o-mini")

# ============ 2. 精确 Token 计数函数 ============
def count_tokens_tiktoken(messages):
    """
    使用 tiktoken 精确计算消息列表的 token 总数

    参数:
        messages: List[BaseMessage] - 消息对象列表

    返回:
        int: 总 token 数

    说明:
        - 每个消息包含角色、内容和元数据，需完整编码
        - 不同消息格式（Human/AI/System）的 token 开销不同
        - 此函数模拟真实 API 调用的 token 计数逻辑
    """
    total_tokens = 0

    # 遍历每条消息，累加 token 数
    for message in messages:
        # 消息格式：role + content + 特殊标记
        # 典型格式：<|im_start|>role<|im_end|>content<|im_end|>

        # 角色 token（如 "user", "assistant", "system"）
        role_tokens = len(TOKEN_ENCODER.encode(message.type))

        # 内容 token（消息正文）
        content_tokens = len(TOKEN_ENCODER.encode(message.content))

        # 格式开销（OpenAI 消息边界标记）
        # 每条约 4 个特殊 token（开始、角色、结束、内容结束）
        format_overhead = 4

        total_tokens += role_tokens + content_tokens + format_overhead

    return total_tokens

# ============ 创建Agent ============
def create_trimmed_agent():
    """创建 Agent 并配置 InMemorySaver"""
    memory = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[search_weather],
        system_prompt="你是一个简洁的助手，记住用户提到的城市名称",
        checkpointer=memory  # 启用短期记忆
    )

    return agent

# ============ 手动裁剪并调用 Agent ============
def invoke_with_trim(agent, user_input: str, config: dict):
    """
    在调用 Agent 前手动裁剪上下文

    流程：
    1. 获取当前状态（所有历史消息）
    2. 使用 trim_messages 裁剪
    3. 构建新输入（裁剪后的消息 + 新消息）
    4. 调用 Agent
    """
    # 1. 获取当前记忆状态
    state = agent.get_state(config)
    existing_messages = state.values.get("messages", []) if state else []

    # 精确计算当前 token 数
    current_tokens = count_tokens_tiktoken(existing_messages)

    # 2. 如果有历史消息，先裁剪
    if existing_messages:
        print(f"裁剪前消息数: {len(existing_messages)}")

        # 核心：调用 trim_messages 进行裁剪
        trimmed_messages = trim_messages(
            existing_messages,                    # 待裁剪的消息列表
            max_tokens=MAX_TOKENS,                # 允许的最大 token 数，超过则触发裁剪
            token_counter=count_tokens_tiktoken,  # token 计数函数
            strategy=TRIM_STRATEGY,               # 裁剪策略，"last" 保留最新消息，"first" 保留最早消息
            include_system=INCLUDE_SYSTEM,        # 是否保留系统消息（通常必须保留）
            allow_partial=False,                  # False不允许部分消息，会尝试保留消息的完整性。如果无法在保持消息完整性的前提下将总token数裁剪到参数 max_tokens 以内，就会返回 空列表
            start_on="human"                      # 从 human 消息开始裁剪
        )

        # 计算裁剪后的 token 数
        new_tokens = count_tokens_tiktoken(trimmed_messages)

        print(f"裁剪后 token: {new_tokens}，裁剪后消息数: {len(trimmed_messages)}")
    else:
        trimmed_messages = []

    # 3. 添加新消息
    new_messages = trimmed_messages + [HumanMessage(content=user_input)]

    # 4. 调用 Agent（checkpointer 会自动保存新状态）
    response = agent.invoke(
        {"messages": new_messages},
        config=config
    )

    return response

# ============ 演示多轮对话与裁剪 ============
def demo_manual_trim():
    """演示手动裁剪上下文的多轮对话"""
    print("=" * 60)
    print("场景：手动 trim_messages + InMemorySaver")
    print("=" * 60)

    agent = create_trimmed_agent()
    config = {"configurable": {"thread_id": "trim_user_001"}}

    # 模拟多轮对话
    conversations = [
        "你好，我叫陈明",
        "查询北京天气",
        "上海呢？",
        "明天北京天气如何？",  # 此时会触发裁剪
        "我是谁？",  # 测试记忆是否保留
    ]

    for i, query in enumerate(conversations, 1):
        print(f"\n--- 第 {i} 轮 ---")
        print(f"用户: {query}")

        # 每次调用前自动裁剪
        response = invoke_with_trim(agent, query, config)

        print(f"AI: {response['messages'][-1].content}")
        print("-" * 40)

# ============ 运行演示 ============
if __name__ == "__main__":
    demo_manual_trim()