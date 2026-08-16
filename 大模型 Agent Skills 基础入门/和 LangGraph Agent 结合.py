def route_node(state):
    user_query = state["question"]
    executor = SkillExecutor()
    skill = executor.match_skill(user_query)
    if skill:
        # 把skill正文写入state，跳转skill执行节点
        return {"next_node": "run_skill", "skill_content": skill["body"]}
    else:
        # 跳转普通工具调用agent
        return {"next_node": "run_standard_agent"}

# graph.add_conditional_edges(起点, route_node, 分支映射)


# 程序启动 → SkillExecutor()实例化 → load_all_skills 扫描所有 SKILL.md
# 所有技能【name+description + 完整 body】载入内存（简化版本）
# 用户提问：帮我写单元测试
# 调用 match_skill() 检索，命中 unit_test 技能
# 拿到 target_skill
#    分支 A：找到 Skill → 将 skill 完整工作流文本追加进 LLM 系统提示词，进入 Skill 执行流程
#    分支 B：无匹配 Skill → Agent 直接使用普通 MCP/Tools，不启用任何 Skill