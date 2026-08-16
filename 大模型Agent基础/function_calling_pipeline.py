import os
from dotenv import load_dotenv
from openai import OpenAI
import json

# 从当前目录的.env文件加载环境变量
load_dotenv()

# 初始化OpenAI客户端（兼容DeepSeek API）
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

def function_calling_pipeline(
    user_message: str,
    tools: list,
    tool_registry: dict,
    system_prompt: str = "你是一个有用的助手。",
    model: str = "deepseek-chat",
    verbose: bool = True
) -> str:
    """
    完整的 Function Calling 管线（单轮工具调用）
    
    参数:
        user_message: 用户输入
        tools: 工具定义列表（JSON Schema）
        tool_registry: 工具名称到函数的映射字典
        system_prompt: 系统提示词
        model: 模型名称
        verbose: 是否打印中间过程
    
    返回:
        LLM 的最终回答文本
    """
    # 步骤 1: 构建消息历史
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    # 步骤 2-3: 发送请求，获取 LLM 决策
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.7,
        max_tokens=2048
    )
    assistant_message = response.choices[0].message

    # 如果 LLM 不需要调用工具，直接返回回答
    if not assistant_message.tool_calls:
        if verbose:
            print("💬 LLM 直接回答（未调用工具）")
        return assistant_message.content

    # 步骤 4: 执行工具调用
    if verbose:
        print(f"🔧 LLM 决定调用 {len(assistant_message.tool_calls)} 个工具")
    messages.append(assistant_message.model_dump())

    for tool_call in assistant_message.tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)

        if verbose:
            print(f" → {func_name}({func_args})")

        # 执行工具（通过映射字典调用）
        if func_name in tool_registry:
            result = tool_registry[func_name](**func_args)
        else:
            result = json.dumps({"error": f"未知工具: {func_name}"})

        if verbose:
            print(f" ← {result[:200]}")  # 截断过长的输出，避免日志刷屏

        # 步骤 5: 将结果追加到消息历史
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": func_name,
            "content": result
        })

    # 步骤 6: 再次调用 LLM 生成最终回答
    final_response = client.chat.completions.create(
        model=model, 
        messages=messages,
        tools=tools, 
        temperature=0.7, 
        max_tokens=2048
    )
    final_answer = final_response.choices[0].message.content
    if verbose:
        print(f"✅ 最终回答生成完成")
    return final_answer