from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv("D:\\VS Code-wenjian\\大模型基础入门\\project_api\\.env")  # 加载.env文件

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY")  # 这里替换为你的 OpenRouter API Key
)

# 构造聊天对话请求（推理模式）
response = client.chat.completions.create(
  model="nvidia/nemotron-3-super-120b-a12b:free",
  messages=[
          {
            "role": "user",
            "content": "你好呀"
          }
        ],
  #extra_body={"reasoning": {"enabled": True}}  # 如果不需要推理模式，注销此行代码
)

# 输出模型回复
print(response.choices[0].message.content)



print(f"输入Token: {response.usage.prompt_tokens}")
print(f"输出Token: {response.usage.completion_tokens}")