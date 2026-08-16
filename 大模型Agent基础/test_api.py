import os
from dotenv import load_dotenv
from openai import OpenAI

# 从当前目录的.env文件加载环境变量
load_dotenv()

# 初始化OpenAI客户端（兼容DeepSeek API）
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 最小调用样例：测试对话接口
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好，你感觉今天怎么样？"}],
    max_tokens=50
)
print(f"✅ API 连接成功，模型响应: {response.choices[0].message.content}")