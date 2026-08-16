import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings.hunyuan import HunyuanEmbeddings
from dotenv import load_dotenv
import os

# 加载.env环境变量
load_dotenv(override=True)

embeddings = HunyuanEmbeddings(
    hunyuan_secret_id=os.environ["TENCENT_SECRET_ID"] ,
    hunyuan_secret_key=os.environ["TENCENT_SECRET_KEY"],
    region="ap-guangzhou" #必填，
)

print("✅ 模型加载成功")
print(embeddings.embed_query("Hello, world!")[:10])
print(embeddings.embed_documents(["文档1", "文档2"]))