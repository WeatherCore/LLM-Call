# 复用你的DeepSeek基础客户端，和实验一、二的导入逻辑一致
from base_deepseek_client import chat

# ---------------------- 实验三：文件系统操作 ----------------------
print("【文件系统操作】")

# 调用模型，让它帮你创建本地文件
prompt = "请帮我在桌面上创建一个名为 'agent_test.txt' 的文件，内容写入 'Hello Agent World'。"
response = chat(prompt)
print("模型回答：", response)

print("\n" + "="*60)

# 实验观察与结论
print("🔍 观察：LLM 只能告诉你'怎么做'，但无法真正执行文件操作。")
print("🔧 Agent 解法：通过 file_write 工具直接操作文件系统。")