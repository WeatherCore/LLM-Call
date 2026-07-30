# ==================== SummarizationMiddleware 完整实现 ====================

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import ensure_config
from pydantic import BaseModel, Field
from typing import Optional
import logging
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# ==================== 1. 配置日志 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 2. 定义工具 ====================
@tool
def search_patent(query: str) -> str:
    """搜索专利数据库"""
    return f"专利搜索结果: 找到与 '{query}' 相关的 3 项专利..."

@tool
def analyze_technology(tech_desc: str) -> str:
    """分析技术可行性"""
    return f"技术分析: '{tech_desc}' 的实现可行性评估完成..."

tools = [search_patent, analyze_technology]

model = init_chat_model(
    model="qwen3.7-max",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# ==================== 3. 定义上下文 ====================
class UserContext(BaseModel):
    user_id: str = Field(..., description="用户唯一标识")
    department: str = Field(..., description="所属部门")
    max_history_tokens: Optional[int] = Field(default=1000, description="历史消息 token 阈值")

# ==================== 4. 配置中间件 ====================
summarization_middleware = SummarizationMiddleware(
    model=model,
    # max_tokens_before_summary=200,          # 历史消息 token 数量超过 200 时触发压缩
    messages_to_keep=5,                     # 保留最近 5 条消息
    summary_prompt="请将以下对话历史进行摘要，保留关键决策点和技术细节：\n\n{messages}\n\n摘要:"   # 摘要提示词
)

# ==================== 5. 创建 Agent ====================
agent = create_agent(
    model=model,
    tools=tools,
    middleware=[summarization_middleware],
    context_schema=UserContext,
    debug=True,
)

# ==================== 6. 执行测试 ====================
def run_summarization_test():
    logger.info("开始 SummarizationMiddleware 测试")

    # 创建长对话历史
    long_history = [HumanMessage(content=f"问题 {i+1}: 如何评估某项技术的专利风险？") for i in range(20)]
    logger.info(f"创建了 {len(long_history)} 条消息")

    # 执行
    result = agent.invoke(
        {"messages": long_history},
        context=UserContext(user_id="engineer_001", department="研发部"),
        config=ensure_config({"configurable": {"thread_id": "session_001"}})
    )

    result_messages = result.get("messages", [])
    logger.info(f"执行后消息数: {len(result_messages)}")

    if len(result_messages) < len(long_history):
        logger.info(f"中间件已触发！压缩了 {len(long_history) - len(result_messages)} 条消息")

    return result

# ==================== 7. 运行测试 ====================
result = run_summarization_test()
logger.info("测试完成")