# model_loader.py
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

import os

# ==========全局统一限流配置（全项目共用一套限速规则）==========
rate_limiter = InMemoryRateLimiter(
    requests_per_second=5,    # 每秒最大并发请求5次
    check_every_n_seconds=1.0 # 1秒轮询校验调用频次
)

def load_chat_model(
    model: str,
    provider: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    base_url: str | None = None,
):
    """
    带接口限流的模型加载通用函数
    :param model: 模型名 gpt-4o-mini/deepseek-chat
    :param provider: 厂商 openai/deepseek
    :param temperature: 生成随机性
    :param max_tokens: 单次输出最大token
    :param base_url: 自定义反向代理地址
    :return: 封装限流后的LLM实例
    """

    # 👇 新增：自动从.env加载对应厂商的密钥
    api_key=os.getenv("DEEPSEEK_API_KEY")

    llm = init_chat_model(
        model=model,
        model_provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        rate_limiter=rate_limiter,
        api_key=api_key  # 👇 注入密钥
    )
    return llm
