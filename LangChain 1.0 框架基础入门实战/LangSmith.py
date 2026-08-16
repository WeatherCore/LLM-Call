import os
from dotenv import load_dotenv
load_dotenv()  # 加载.env文件中的环境变量
from langchain.chat_models import init_chat_model
# 2. 导入LangSmith客户端并初始化
from langsmith import Client

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))  # 用你的API Key连接LangSmith服务

# 3. 从官方拉取现成的RAG提示词模板
prompt = client.pull_prompt("weather/rag-cn-strict-no-hallucination-prompt", dangerously_pull_public_prompt=True)

# 4. 查看模板结构（验证和你自己写的模板是同一个东西）
print("=== 拉取的模板结构 ===")
print(prompt)
print("\n=== 模板需要的变量 ===")
print(prompt.input_variables)  # 输出: ['context', 'question']

# 初始化模型
model = init_chat_model(
    "qwen3.7-max",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 给模板传变量（和你之前用.format()完全一样）
# formatted_prompt = prompt.format(
#     context="LangChain 是一个构建LLM应用的框架，支持RAG、Agent等功能，能帮你快速连接模型和外部数据",
#     question="什么是LangChain？它有什么核心功能？"
# )

# 测试用例 1：精准匹配
# formatted_prompt = prompt.format(
#     context="DeepSeek-V3 是一个 MoE 模型，总参数量 671B，每 Token 激活 37B，训练成本 557 万美元。",
#     question="DeepSeek-V3 的总参数量和训练成本分别是多少？"
# )

# 测试用例 2：部分匹配（上下文有创始人，但无融资额）
formatted_prompt = prompt.format(
    context="深度求索公司成立于 2023 年，创始人梁文锋，专注 AGI 研究。",
    question="DeepSeek 的创始人是谁？最近一轮融资额是多少？"
)

# # 测试用例 3：完全无关（上下文讲天气，问题问 LangChain）
# formatted_prompt = prompt.format(
#     context="今天北京晴天，气温 25°C，适合出游。地铁 10 号线早高峰 7-9 点。",
#     question="LangChain 的 Agent 是如何调用工具的？"
# )

# # 测试用例 4：上下文有错误信息（矛盾事实）
# formatted_prompt = prompt.format(
#     context="Python 由微软公司于 1991 年首次发布，创始人是 Guido van Rossum。",
#     question="Python 是哪一年发布的？创始人是谁？"
# )

# # 测试用例 5：多跳推理（需整合上下文多个句子）
# formatted_prompt = prompt.format(
#     context="智能客服包含 ASR 和 NLG 模块。ASR 将语音转文本，NLG 生成回复。电商要求响应 <500ms。",
#     question="电商智能客服从语音到回复涉及哪两个模块？对延迟有何要求？"
# )

# 调用模型
response = model.invoke(formatted_prompt)
print("\n=== 模型回答 ===")
print(response.content)