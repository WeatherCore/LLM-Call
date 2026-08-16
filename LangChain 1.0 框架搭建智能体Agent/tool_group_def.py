from langchain.tools import tool
from langchain.agents import create_agent
import os
from dotenv import load_dotenv
from model_loader import load_chat_model

# 加载密钥环境变量
load_dotenv()

# 四类领域工具，严格遵循5.4统一工具规范
@tool
def search_web(query: str) -> str:
    """Web 搜索工具，用于查询网络公开信息，不适用于内部数据.参数: query 用户查询，如 OpenAI 最新模型"""
    return f"模拟搜索结果: 你搜索了 {query}"

@tool
def extract_pdf_text(path: str) -> str:
    """解析 PDF 文本文件。参数为文件的本地路径.参数: path 文件路径，如 /files/contract.pdf"""
    return f"模拟 PDF 内容: 从 {path} 中解析出的内容"

@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询，仅限内部业务数据库.参数: sql Sql语句，如 select * from users limit 5"""
    return f"模拟 SQL 执行: {sql}"

@tool
def calculate(expr: str) -> str:
    """计算数学表达式。适用于算式运算.参数: expr 数学表达式，如 (12+3)*(8-2)"""
    return str(eval(expr))

# 5.3 动态工具分组配置
TOOL_GROUPS = {
    "search": [search_web],
    "pdf": [extract_pdf_text],
    "database": [query_database],
    "math": [calculate],
}



# 2. 创建独立意图分类模型（5.2 工程级意图模型）
intent_llm = load_chat_model(model="deepseek-chat",provider="deepseek")

# 3. 意图分类系统提示词（5.1 Tool Router）
INTENT_SYSTEM_PROMPT = """
你是一个专业的意图分类器，请只返回以下类别之一:
- search
- pdf
- database
- math
- none

并严格只返回类别名，不要输出其它内容。
"""

# 4. 意图分类函数
def classify_intent(user_query: str) -> str:
    result = intent_llm.invoke(
        [
            ("system", INTENT_SYSTEM_PROMPT),
            ("user", user_query)
        ]
    )
    return result.content.strip()