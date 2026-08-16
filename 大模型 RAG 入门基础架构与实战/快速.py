import requests
import os
from dotenv import load_dotenv
load_dotenv(override=True)

resp = requests.post(
    "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings",
    headers={"Authorization": f"Bearer {os.getenv('DASHSCOPE_API_KEY')}"},
    json={"model":"qwen3.7-text-embedding","input":"测试"}
)
print(resp.status_code)
print(resp.text)