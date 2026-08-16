import json
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv(override=True)

# 使用DeepSeek的API来调用大模型
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 1. 这是你本地真正能干活的函数（大模型并不知道它的具体实现代码）
def get_weather(location: str):
    print(f"🔧 [本地执行中] 正在查询 {location} 的天气...")
    # 这里可以是发HTTP请求、查数据库等真实操作
    if location == "北京":
        return '{"temp": 25, "condition": "晴"}'
    return '{"temp": 20, "condition": "未知"}'

# 2. 【关键抽象】建立“字符串名字”到内存里的真实函数的映射字典
available_functions = {
    "get_weather": get_weather,
    # 如果有别的工具: "search_database": search_database
}

# 3. 告诉大模型你有这个工具（只给说明书，不给代码）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    }
]


messages = [{"role": "user", "content": "北京今天热吗？"}]

# 大模型看到你的问题和工具说明书，它决定调用工具
response = client.chat.completions.create(
    model="qwen3.7-max",
    messages=messages,
    tools=tools  # 告诉大模型你有哪些工具
)

response_message = response.choices[0].message
# 查看大模型是否调用了工具
# print(response_message.tool_calls)

# 1. 检查大模型是不是发出了调用工具的请求
if response_message.tool_calls:
    # 记得把大模型的"请求调用"这条记录也放进历史对话里
    messages.append(response_message)

    # 遍历大模型想要调用的所有函数（有时候它会并行调用多个）
    for tool_call in response_message.tool_calls:
        # 1. 提取大模型建议的指令：把模型的指令拆解开，拿到后续需要的工具名和参数
        function_name = tool_call.function.name  # 比如提取到 "get_weather"
        function_args_json = tool_call.function.arguments  # 比如提取到 '{"location": "北京"}'

        # 2. 将大模型生成的 JSON 字符串解析为真正的 Python 字典
        function_args = json.loads(function_args_json)

        # 3. 【真正执行的魔法在此】通过大模型给的字符串名字，从你的映射字典里找到真正的 Python 函数内存地址
        function_to_call = available_functions.get(function_name)

        if function_to_call:
            # 4. 在你的本地机器上，真正执行这个函数，并传入解析好的参数！
            function_result = function_to_call(**function_args)
            print(f"✅ [本地执行完毕] 得到结果: {function_result}")
        else:
            function_result = "Error: 找不到该函数"

        # 5. 将执行得到的结果，打包成特定格式（role="tool"），准备发回给大模型
        messages.append({
            "tool_call_id": tool_call.id,  # 必须带上这个ID，告诉大模型这是对应它刚才哪个请求的结果
            "role": "tool",
            "name": function_name,
            "content": function_result  # 把真实结果（如 '{"temp": 25}'）塞进去
        })

# 打印最终的messages（调试用，看上下文是否正确）
# print(messages)

# 第四阶段：第二次请求大模型（带着工具结果）
second_response = client.chat.completions.create(
    model="qwen3.7-max",
    messages=messages,  # 这里的messages已经包含完整上下文了！
)

# 打印模型生成的最终回答
print("\n📌 最终回答: ", second_response.choices[0].message.content)