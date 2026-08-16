from Tavily import get_weather, calculate  # 导入你之前写的工具函数
import json

# 修复1：正确导入【函数】，而不是导入模块
from function_calling_pipeline import function_calling_pipeline

# ---------------------- 1. 工具注册表（第二张图的内容） ----------------------
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
}

# ---------------------- 2. 工具定义（第一张图的JSON Schema） ----------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "获取指定城市的当前天气信息，包括气温（摄氏度）、天气状况和湿度。"
                "当用户询问某个城市的天气、气温、是否需要带伞/穿外套等问题时，调用此工具。"
                "目前支持的城市：北京、上海、广州、深圳、杭州。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要查询天气的城市名称，例如：北京、上海、广州。"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "执行数学计算，支持加减乘除、幂运算、三角函数、对数等。"
                "当用户需要精确计算数学表达式时调用此工具。"
                "输入应为合法的 Python 数学表达式，例如：'2**10'、'sqrt(144)'、'7654321 * 1234567'。"
                "注意：不要用此工具回答不涉及计算的问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "要计算的数学表达式，使用 Python 语法，例如："
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# 测试 1: 需要天气工具的问题
print("=" * 60)
print("测试 1: 天气查询")
print("=" * 60)
answer = function_calling_pipeline(
    "上海今天天气怎么样？需要带伞吗？",
    tools=tools,
    tool_registry=TOOL_REGISTRY
)
print(f"\n☁️ {answer}\n")

# 测试 2: 需要计算工具的问题
print("=" * 60)
print("测试 2: 数学计算")
print("=" * 60)
answer = function_calling_pipeline(
    "请帮我计算 2 的 20 次方是多少？",
    tools=tools,
    tool_registry=TOOL_REGISTRY
)
print(f"\n🧮 {answer}\n")

# 测试 3: 不需要工具的问题
print("=" * 60)
print("测试 3: 纯知识问答（不需要工具）")
print("=" * 60)
answer = function_calling_pipeline(
    "请简要解释什么是 Function Calling？",
    tools=tools,
    tool_registry=TOOL_REGISTRY
)
print(f"\n📚 {answer}\n")

# 测试 4: 需要同时调用两个工具的问题
print("=" * 60)
print("测试 4: 多工具调用")
print("=" * 60)
answer = function_calling_pipeline(
    "北京今天多少度？另外帮我算一下 sqrt(144) + 3.14 * 2",
    tools=tools,
    tool_registry=TOOL_REGISTRY
)
print(f"\n🔧 {answer}")