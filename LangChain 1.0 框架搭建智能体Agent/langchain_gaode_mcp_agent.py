from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
import os
import asyncio
from dotenv import load_dotenv
from model_loader import load_chat_model

# 2. 加载.env环境变量（解决Key问题）
load_dotenv()  # 自动读取当前目录下的.env文件

# 1. 正确的 MCP 配置格式 (适用于 langchain_mcp_adapters)
# MultiServerMCPClient 需要的是扁平的字典结构，每个服务器是一个键值对
mcp_config = {
    # # 本地 Python MCP 服务
    # "math": {
    #     "transport": "stdio",
    #     "command": "python",
    #     "args": ["mcp_server.py"]
    # },
    # 高德地图 MCP 服务
    "amap-maps": {
        "transport": "stdio",
        "command": "npx",
        "args": ["-y", "@amap/amap-maps-mcp-server"],
        "env": {
            "AMAP_MAPS_API_KEY": os.getenv("AMAP_MAPS_API_KEY"),
        }
    }
}


# 全部await、async代码放入异步主函数
async def main():
    # 2. 创建 MCP 客户端
    client = MultiServerMCPClient(mcp_config)
    print("正在连接 MCP 服务器...")

    # 3. client.get_tools() 会自动:
    # 1. 调用所有服务器的 list_tools 接口
    # 2. 将 MCP Tool Schema 转换为 LangChain StructuredTool
    tools = await client.get_tools()
    print(f"成功加载 {len(tools)} 个工具: {[t.name for t in tools]}")

    # 4. 创建 Agent
    llm = load_chat_model(model="deepseek-chat",provider="deepseek")
    # 直接将转换好的 tools 传给 create_agent
    agent = create_agent(llm, tools, system_prompt="你是会调用工具进行天气查询、地图查询、")

    # 5. 运行 Agent
    print("\n--- 开始测试 Agent ---")

    # 6. 这里我们模拟一个请求 (具体 prompt 取决于你的工具功能)
    query = "请帮我搜索查询一下北京市今天的天气"
    inputs = {"messages": [HumanMessage(content=query)]}

    async for chunk in agent.astream(inputs, stream_mode="values"):
        last_msg = chunk["messages"][-1]
        print(f"\n{type(last_msg).__name__}:")
        print(last_msg.content)

        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            print(f">>> 调用工具详情: {last_msg.tool_calls}")


# 统一异步程序入口
if __name__ == "__main__":
    asyncio.run(main())




