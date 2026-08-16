import os
from dotenv import load_dotenv
load_dotenv()  # 加载.env文件中的环境变量
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent

# ========== 1. 定义结构化输出Pydantic模型 ==========
class WeatherForecast(BaseModel):
    """天气预报结构化输出"""
    city: str = Field(description="城市名称")
    temperature: int = Field(description="温度(摄氏度)")
    condition: Literal["晴", "雨", "多云", "雪"] = Field(description="天气状况")

# ========== 2. 加载裸大模型 ==========
model = init_chat_model(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),  # 从.env文件读取Key
    base_url="https://api.deepseek.com/v1",    # 修正：加上引号
)

# ========== 3. 创建Agent ==========
agent = create_agent(
    model=model,
    tools=[],  # 工具列表，为空代表不调用外部工具
    response_format=WeatherForecast  # 指定最终强制结构化输出模板
)

# ========== 4. 调用Agent（⚠️注意外层{"messages": ...}固定格式） ==========
result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "北京今天阳光明媚，温度10度"
        }
    ]
})

# ========== 5. 提取结构化结果 ==========
forecast = result["structured_response"]
print(f"{forecast.city}天气：{forecast.condition}, {forecast.temperature}°C")