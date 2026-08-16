import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("D:\\VS Code-wenjian\\大模型基础入门\\project_api\\.env")  # 加载.env文件

# 1. 平台配置字典（和你之前的代码一致）
PLATFORM_CONFIGS = {
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat"
    },
    "openrouter": {
        "api_key": os.getenv("OPENROUTER_API_KEY"),
        "base_url": "https://openrouter.ai/api/v1",
        "model": "nvidia/nemotron-3-super-120b-a12b:free"
    },
    "dashscope": {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo"
    },
    "zhipu": {
        "api_key": os.getenv("ZHIPUAI_API_KEY"),
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4"
    }
}

# 2. 通用调用函数（你图片里的代码）
def call_llm(platform_name, prompt, temperature=0.7, max_tokens=200):
    """
    通用的大模型调用函数
    Args:
        platform_name: 平台名称 (deepseek/openrouter/dashscope/zhipu)
        prompt: 用户输入
        temperature: 温度参数
        max_tokens: 最大输出 Token 数
    Returns:
        模型回复内容
    """
    # 获取对应平台的配置信息
    config = PLATFORM_CONFIGS[platform_name]

    # 初始化 OpenAI 客户端
    client = OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )

    # 调用大模型聊天接口
    response = client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )

    # 返回模型生成的文本内容
    return response.choices[0].message.content

# 3. 测试代码
if __name__ == "__main__":
    test_prompt = "用一句话解释什么是 AI"
    print("使用 openrouter:")
    print(call_llm("openrouter", test_prompt))