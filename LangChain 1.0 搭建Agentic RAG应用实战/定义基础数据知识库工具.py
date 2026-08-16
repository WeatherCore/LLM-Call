from pydantic import BaseModel, Field
# 定义工具输入参数
from langchain_core.tools import StructuredTool

# 定义工具输入参数
class QAWithRetrievalArgs(BaseModel):
    query: str = Field(description="用户的问题")

def query_retrieval_knowledge(query: str) -> str:
    """
    一个基于LangChain知识库检索的问答工具。
    专门用于回答与 LangChain 相关的技术问题。

    ⚠️ 重要：此工具仅适用于 LangChain 相关问题！
    如果问题与 LangChain 无关，请使用网络搜索工具。
    """
    # 定义 LangChain 相关关键词
    langchain_keywords = [
        'langchain', 'langgraph', 'langsmith', 'lcel',
        'chain', 'agent', 'retriever', 'embedding', 'vector',
        'rag', 'prompt', 'llm', 'chatmodel', 'runnable',
        '链', '代理', '检索器', '向量', '提示词', '模型'
    ]

    # 检查查询是否包含 LangChain 相关关键词
    query_lower = query.lower()
    is_langchain_related = any(keyword in query_lower for keyword in langchain_keywords)

    # 如果查询与 LangChain 无关，返回提示
    if not is_langchain_related:
        return (
            "❌ 检测到此问题与 LangChain 知识库无关。\n"
            "建议：请使用网络搜索工具 (tavily_search_results_json) 来查找答案。\n"
            f"原始问题：{query}"
        )

    # 如果相关，则进行检索，返回检索文档内容
    retrieval_chain = ensemble_retriever | format_docs
    docs = retrieval_chain.invoke(query)

    # 检查检索结果质量
    if not docs or len(docs.strip()) < 50:
        return (
            f"⚠️ 知识库中未找到关于 '{query}' 的充分信息。\n"
            "建议：可以尝试使用网络搜索工具获取更多信息。"
        )

    return docs

# 定义工具StructuredTool
qa_tool = StructuredTool.from_function(
    func=query_retrieval_knowledge,        # 工具函数
    name="query_retrieval_knowledge",      # 工具名称
    description=(
        "🎯 专用于回答 LangChain 技术相关问题的知识库检索工具。\n"
        "适用范围：LangChain、LangGraph、LangSmith、LCEL、Agent、RAG、Retriever、Embedding、Prompt 等相关技术。\n"
        "⚠️ 限制：仅包含 LangChain 相关文档，不适用于其他领域问题（如烹饪、历史、科学等）。\n"
        "如果问题与 LangChain 无关，请使用网络搜索工具 tavily_search_results_json。"
    ),                                      # 工具描述
    args_schema=QAWithRetrievalArgs,        # 工具输入参数
    return_direct=False                     # 是否直接返回工具输出，而不是作为消息内容
)

result = qa_tool.invoke("LangChain这个框架是什么？")
print(result)