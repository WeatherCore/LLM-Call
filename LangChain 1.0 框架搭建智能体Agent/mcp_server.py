from mcp.server.fastmcp import FastMCP

# 实例MCP服务，命名数学计算器服务
mcp = FastMCP("math_calc_server")

# 加法工具
@mcp.tool()
def add(a: float, b: float) -> float:
    """两个数字相加
    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a + b

# 乘法工具
@mcp.tool()
def multiply(a: float, b: float) -> float:
    """两个数字相乘
    Args:
        a: 第一个数字
        b: 第二个数字
    """
    return a * b

# 幂运算工具
@mcp.tool()
def power(base: float, exp: float) -> float:
    """幂运算 base 的 exp 次方
    Args:
        base: 底数
        exp: 指数
    """
    return base ** exp

# stdio 模式启动MCP服务，供LangChain客户端子进程调用
if __name__ == "__main__":
    mcp.run(transport="stdio")