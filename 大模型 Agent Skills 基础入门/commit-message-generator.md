
---
name: commit-message-generator
description: 基于暂存区的更改生成符合规范的提交信息
---

# 提交信息生成器 (Commit Message Generator)

## Goal
生成清晰、符合最佳实践的规范化提交信息。

## Workflow
1. 使用 git diff 分析暂存区的代码更改
2. 识别提交的主要类型 (feat/fix/docs/refactor/test)
3. 从被修改的文件中提取作用范围 (scope)
4. 生成一条简明扼要的标题行 (最多 50 个字符)
5. 如果更改逻辑复杂，可以在之后添加详细的正文说明

## Constraints
- 遵循 Conventional Commits 格式规范: `type(scope): subject`
- 标题行必须使用祈使句语气 (例如使用 "add" 而不是 "added")
- 标题行的长度严格控制在 50 个字符以内
- 标题行和正文之间必须用一个空行分隔


