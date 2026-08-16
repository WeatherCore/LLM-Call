from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv("D:\\VS Code-wenjian\\大模型基础入门\\project_api\\.env")  # 加载.env文件

# 修正后的DeepSeek客户端配置（带/v1）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 调用模型（和示例完全一致）
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "你是谁？"}
    ]
)

# 提取并打印回复
answer = response.choices[0].message.content
print("模型回复:")
print(answer)