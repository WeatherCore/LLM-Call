import requests
import json
import os
from dotenv import load_dotenv  # 补上.env加载（原代码漏了）

# 1. 加载你的.env文件（适配你的项目结构）
load_dotenv("D:\\VS Code-wenjian\\大模型基础入门\\project_api\\.env")  # 加载.env文件
API_KEY = os.getenv("OPENROUTER_API_KEY")

# 2. 查余额请求
try:
    
    response = requests.get(
        "https://openrouter.ai/api/v1/auth/key",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    response.raise_for_status()  # 自动处理请求错误（比如Key错了、网络断了）

    # 3. 格式化打印返回的结果
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

    # 4. 只打印你关心的关键信息（不用看一长串JSON）
    data = response.json()
    print("=== OpenRouter 账户监控 ===")
    print(f"✅ 总消费金额: {data['data']['usage']:.4f} 美元")
    print(f"📅 今日消费: {data['data']['usage_daily']:.4f} 美元")
    print(f"🗓️ 本月消费: {data['data']['usage_monthly']:.4f} 美元")
    print(f"💰 剩余额度: {data['data']['limit_remaining'] if data['data']['limit_remaining'] is not None else '无限制/未设置'}")
    print(f"💳 账户类型: {'免费额度' if data['data']['is_free_tier'] else '付费账户'}")

except Exception as e:
    print(f"查询失败：{e}")
    print("请检查API Key是否正确、网络是否正常")