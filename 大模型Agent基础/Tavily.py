import math
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# 从环境变量中读取 Tavily API 密钥
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ====================== 模块1：天气查询工具函数 ======================
def get_weather(query: str) -> str:
    """
    使用 Tavily 搜索引擎获取天气信息的真实工具函数
    Args:
        query: 用户的天气查询请求（如"北京今天的天气"）
    Returns:
        JSON格式的天气查询结果，包含状态和结果文本
    """
    print(f"\n🌍 [Tool 执行中] 正在通过 Tavily 搜索: {query} ...")
    
    # API 配置
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": TAVILY_API_KEY,
        "query": f"{query} 今天天气怎么样？",
        "search_depth": "basic",  # 基础搜索模式，速度更快
        "include_answer": True,   # 优先返回 Tavily 直接生成的总结答案
        "max_results": 3          # 仅获取前3个最相关的网页结果
    }
    
    # 发送 POST 请求调用 Tavily API
    response = requests.post(url, json=payload, headers=headers)
    
    # 处理 API 响应
    if response.status_code == 200:
        data = response.json()
        # 优先使用 Tavily 直接返回的 answer（如果存在）
        result_text = data.get("answer", "")
        # 如果没有直接 answer，就组装搜索结果的 snippet
        if not result_text:
            snippets = [result["content"] for result in data.get("results", [])]
            result_text = "\n".join(snippets)
        # 返回标准化 JSON 结果（方便大模型解析）
        return json.dumps({"status": "success", "search_result": result_text}, ensure_ascii=False)
    else:
        # API 请求失败时返回错误信息
        return json.dumps({"status": "error", "message": f"Tavily API 请求失败: {response.status_code}"}, ensure_ascii=False)

# ====================== 模块2：安全数学计算工具函数 ======================
def calculate(expression: str) -> str:
    """
    安全的数学表达式计算工具，支持基础运算和常用数学函数
    Args:
        expression: 数学表达式字符串（如"1+2*3"、"sqrt(16)"）
    Returns:
        JSON格式的计算结果，包含表达式、结果/错误信息和状态
    """
    # 安全白名单：仅允许数学相关的函数/变量，防止代码注入攻击
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "pow": pow, "sum": sum,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
    }
    
    try:
        # 使用 eval 配合白名单执行计算，限制 __builtins__ 防止注入
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        # 返回成功结果（JSON格式，大模型易解析）
        return json.dumps({
            "expression": expression,
            "result": result,
            "status": "success"
        }, ensure_ascii=False)
    except Exception as e:
        # 捕获计算错误，返回标准化错误信息
        return json.dumps({
            "expression": expression,
            "error": str(e),
            "status": "failed"
        }, ensure_ascii=False)

# ====================== 模块3：工具函数测试代码 ======================
if __name__ == "__main__":
    # 测试天气查询工具
    print("天气查询测试: ", get_weather("搜索北京今天的天气"))
    # 测试数学计算工具
    print("计算测试: ", calculate("7654321 * 1234567"))