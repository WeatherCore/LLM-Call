from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

# 1.Pydantic自定义参数结构：管控字段描述、数据约束
class DivideInput(BaseModel):
    """除法工具入参模型"""
    dividend: float = Field(description="被除数")
    divisor: float = Field(description="除数，不能为零")

# 2.业务逻辑函数
def divide(dividend: float, divisor: float) -> float:
    """执行除法运算，支持浮点数"""
    if divisor == 0:
        raise ValueError("除数不能为零")
    return dividend / divisor

# 3.实例化结构化工具（核心配置区）
division_tool = StructuredTool.from_function(
    func=divide,                     # 绑定同步业务函数
    name="DivisionTool",             # 工具唯一名称，LLM靠名称匹配工具调用
    description="安全执行除法运算，自动处理除零错误", # LLM决策是否调用工具的依据
    args_schema=DivideInput,         # 【关键】手动指定Pydantic入参，@tool无法自定义此项
    return_direct=False              # 结果流转控制
)

# 4.入参校验测试：参数名和schema不匹配，Pydantic自动抛错
try:
    division_tool.invoke({"a": 10, "b": 2})
except Exception as e:
    print(f"参数校验失败: {e}")

# 5.合法参数调用
result = division_tool.invoke({"dividend": 10, "divisor": 2})
print(f"除法结果: {result}")