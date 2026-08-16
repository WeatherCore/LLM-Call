# 安装依赖
# !pip install langchain-tavily

# 导入联网搜索工具
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv
load_dotenv() 

# 实例化搜索工具
# max_results=2：限制最多返回2条搜索摘要
web_search = TavilySearch(
    max_results=5,
    api_key=os.getenv("TAVILY_API_KEY"),
)  # max_results：单次搜索返回5条网页摘要

# 手动调用测试
result = web_search.invoke("介绍一下LangChain这个框架")
# 打印查看
print("=====搜索结果=====")
print(result)