from openai import OpenAI
import os
from dotenv import load_dotenv
import time

load_dotenv("D:\\VS Code-wenjian\\大模型基础入门\\project_api\\.env")  # 加载.env文件

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENROUTER_API_KEY")  # 这里替换为你的 OpenRouter API Key
)

# 构造聊天对话请求（推理模式）
stream = client.chat.completions.create(
  model="nvidia/nemotron-3-super-120b-a12b:free",
  messages=[
          {
            "role": "user",
            "content": "介绍一下艾尔登法环这款游戏"
          }
        ],
  stream=True  # ✅ 关键点1：启用流式模式
  #extra_body={"reasoning": {"enabled": True}}  # 如果不需要推理模式，注销此行代码
)

# 3. 循环迭代，处理每一段增量内容
for chunk in stream:  # ✅ 关键点2：stream是一个迭代器，需要循环获取内容
    # ✅ 关键点3：流式用 delta.content，不是 message.content
    delta_content = chunk.choices[0].delta.content

    if delta_content:  # 过滤掉空的内容块（比如开头的role信息）
        # ✅ 关键点4：实时打印，不换行+强制刷新缓冲区
        print(delta_content, end="", flush=True)
        time.sleep(0.1) # 可选：模拟打字机延迟，可删除

print("\n\n✅ 流式输出完成")