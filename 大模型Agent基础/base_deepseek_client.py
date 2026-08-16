import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 把路径改成你自己的！

# 初始化DeepSeek API客户端（兼容OpenAI SDK）
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),  # 从.env文件读取API Key
    base_url="https://api.deepseek.com"  # DeepSeek官方API端点
)

def chat(prompt: str, model: str = "deepseek-chat") -> str:
    """
    向DeepSeek发送单轮对话请求并返回回答
    :param prompt: 用户问题
    :param model: 模型名称（默认deepseek-chat）
    :return: 模型回答
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],  # 对话格式（和你之前学的一致）
        temperature=0.7,  # 控制回答随机性（0=确定，1=随机）
        max_tokens=1024   # 限制回答长度，避免过长
    )
    return response.choices[0].message.content

# 验证API连接（跑通这一步就说明环境没问题）
# if __name__ == "__main__":
#     result = chat("你好，请用一句话介绍你自己。")
#     print(f"API连接成功！模型返回：{result}")