# -------------------------- 关键：导入base_client里的chat函数 --------------------------
from base_deepseek_client import chat

# 1. 用LLM计算「近似数学结果」
print("【精确数学计算】")
prompt = "请计算 7654321 × 1234567 的结果。"
response = chat(prompt)
print("模型回答：", response)
print("\n" + "="*60)

# 用Python直接计算「真正的数学结果」（100%精确）
correct_result = 7654321 * 1234567
print(f"✅ Python计算的正确答案：{correct_result}")

# 实验观察与结论
print(f"🔍 观察：LLM的回答是否与正确答案一致？大概率不一致。")
print(f"🔧 Agent 解法：通过 calculator 工具或 code_executor 工具执行精确计算。")