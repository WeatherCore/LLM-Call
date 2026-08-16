from langchain.agents import create_agent
from model_loader import load_chat_model
# 依赖工具分组、意图函数
from tool_group_def import TOOL_GROUPS
from tool_group_def import classify_intent

# 5. 创建智能体函数
def create_agent_for_group(group: str):
    tools = TOOL_GROUPS.get(group, [])
    if not tools:
        return None

    model = load_chat_model(
        model="deepseek-chat",
        provider="deepseek",
    )
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你是一个 helpful assistant，可以使用工具回答问题。你必须严格根据工具描述选择工具！如果没有合适的工具，请回答“无合适工具”"
    )
    return agent

# 6. 路由智能体函数
def router_agent(user_query: str):
    # 1. 识别意图
    intent = classify_intent(user_query)
    print(f"[Router] 检测到意图: {intent}")

    # 2. 创建对应子 Agent
    sub_agent = create_agent_for_group(intent)
    if sub_agent is None:
        return "无法为该问题找到合适的工具或 Agent。"

    # 3. 调用子 Agent 执行任务
    result = sub_agent.invoke({
        "messages": [{"role": "user", "content": user_query}]
    })
    return result


# 7. 测试智能体
queries = [
    "请帮我搜索一下今年Google最新的大模型版本的发布会",
    "帮我解析一下这个PDF: /root/files/contract.pdf",
    "执行一个SQL: select * from products limit 5",
    "计算 (17+3)*(8-1)",
]

for q in queries:
    print("\n===== 用户问题 =====")
    print(q)
    print("====== Agent 回复 ======")
    print(router_agent(q)["messages"][2])