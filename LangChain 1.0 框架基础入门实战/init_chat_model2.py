# 1. 导入统一模型入口
# 1. 导入依赖（加上dotenv，用来加载.env文件里的Key）
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# 2. 加载.env环境变量（解决Key问题）
load_dotenv()  # 自动读取当前目录下的.env文件

# 2. 初始化模型（统一入口写法）
model = init_chat_model(
    "qwen3.7-max",                # 指定模型名（必须是厂商支持的模型）
    model_provider="openai",      # 指定模型提供商，LangChain会自动匹配实现
    # 还可以加这些参数，和厂商专属类完全一致：
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    # temperature=0.0,
    # max_tokens=512
)

# 3. 业务代码（和模型无关，切换模型完全不用改）
question = "你好，请你介绍一下你自己。"
result = model.invoke(question)
print(result.content)