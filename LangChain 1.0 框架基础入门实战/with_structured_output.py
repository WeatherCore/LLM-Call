import os
from dotenv import load_dotenv
load_dotenv()  # 加载.env文件中的环境变量
from langchain.chat_models import init_chat_model
from langchain_core.utils.pydantic import Field, BaseModel
from typing import List

# 1. 定义模板
class Person(BaseModel):
    name: str = Field(description="人的姓名")
    age: int = Field(description="人的年龄")
    high: int = Field(description="人的身高")
    hobbies: List[str] = Field(description="人的爱好列表")

model = init_chat_model(
    "qwen3.7-max",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
).with_structured_output(Person)

# 3. 调用模型
prompt = "提取名为约翰·多伊的人的信息。他30岁，喜欢阅读、远足和弹吉他"
result = model.invoke(prompt)

# ✅ result 直接就是 Person 类实例对象
print(type(result)) # <class '__main__.Person'>
print(result.name)
print(result.age)
print(result.hobbies)