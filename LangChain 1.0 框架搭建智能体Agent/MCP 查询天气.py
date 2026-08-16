from langchain_mcp_adapters.client import MultiServerMCPClient # MCP多服务客户端
import asyncio
import os
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


# 2. 加载.env环境变量（解决Key问题）
load_dotenv()  # 自动读取当前目录下的.env文件

async def main():
    # 获取MCP服务脚本绝对路径
    mcp_server_path = r"D:/IntelliJ-IDEA-wenjian/MCP-Servers/USA-weather/weather.py" 
    print(mcp_server_path)

    # 初始化MCP客户端配置
    mcp_client = MultiServerMCPClient({
        # 标识这个MCP服务别名：math
        "math": {
            "transport": "stdio", # 通信方式：标准输入输出管道（和Cline调用weather完全一致）
            "command": r"D:/python3.12.7/python.exe", # 启动命令
            "args": [mcp_server_path] # 启动参数：MCP服务脚本路径
        }
    })

    try:
        mcp_tools = await mcp_client.get_tools()
    except Exception as e:
        print(f"❌ 加载 MCP 工具失败: {e}")
        return

    # 加载兼容OpenAI格式大模型
    llm =  init_chat_model(
        model="qwen3.7-max",
        model_provider="openai",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # 创建ReAct智能体，具备工具选择、参数提取、循环调用能力
    agent = create_agent(
        model=llm,
        tools=mcp_tools,
        system_prompt="你是一个多功能的助手，可以查询天气。"
    )

    response = await agent.ainvoke({
        "messages": [{"role": "user", "content": "查询一下明天华盛顿气温"}]
    })

    # 打印Agent最终总结回答
    print(f"Agent: {response['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())



