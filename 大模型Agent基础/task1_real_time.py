# -------------------------- 关键：导入base_client里的chat函数 --------------------------
from base_deepseek_client import chat

# 实验一代码
print("【实时信息获取】")
response = chat("今天北京的天气怎么样？气温多少度？")
print(response)
print("\n" + "="*60)
print("🔍 观察：LLM 无法获取实时天气数据，只能给出模糊回答或声明自己无法访问实时信息。")
print("🔧 Agent 解法：通过 get_weather 工具调用天气 API 获取实时数据。")