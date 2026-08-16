# 1. 导入依赖（加上dotenv，用来加载.env文件里的Key）
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek

# 2. 加载.env环境变量（解决Key问题）
load_dotenv()  # 自动读取当前目录下的.env文件

# 3. 初始化模型（修正base_url引号，加上api_key参数）
model = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从.env文件读取Key
    base_url="https://api.deepseek.com/v1",    # 修正：加上引号
    temperature=0.0,
    max_tokens=512,
    timeout=30
)

# 4. 定义问题
question = "你好，请你介绍一下你自己。"

# 5. 调用模型
result = model.invoke(question)

# 6. 输出结果
print(result.content)