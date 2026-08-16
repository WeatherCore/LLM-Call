# 复用你的DeepSeek基础客户端，和前四个实验的导入逻辑一致
from base_deepseek_client import chat

# ---------------------- 实验五：与外部服务交互 ----------------------
print("【外部服务交互】")

# 补全图片中被截断的prompt，构造完整的HTTP请求任务
prompt = """请帮我向 https://httpbin.org/post 发送一个 POST 请求，body 为 {"test": "hello agent"}。"""

# 调用模型，让它帮你发送请求
response = chat(prompt)
print("模型回答：", response)

print("\n" + "="*60)

# 实验观察与结论
print("🔍 观察：LLM 会给出代码示例，但无法真正发送 HTTP 请求。")
print("🔧 Agent 解法：通过 http_request 工具直接发送请求并获取响应。")