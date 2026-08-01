import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings.hunyuan import HunyuanEmbeddings
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model

# 加载.env环境变量
load_dotenv(override=True)

embeddings = HunyuanEmbeddings(
    hunyuan_secret_id=os.environ["TENCENT_SECRET_ID"] ,
    hunyuan_secret_key=os.environ["TENCENT_SECRET_KEY"],
    region="ap-guangzhou" #必填，
)

# 读取基础数据文档
loader = TextLoader(r"D:\VS Code-wenjian\大模型\LangChain 1.0 搭建Agentic RAG应用实战\sample_document.txt", encoding="utf-8")
documents = loader.load()

# 定义文档切分器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,                # 切分文本块大小
    chunk_overlap=50,              # 文本块重叠大小
    separators=["\n\n", "\n", " ", ""] # 切分优先级符号
)

# 基础数据文档切分
texts = text_splitter.split_documents(documents)

# 根据切分好的文档块 + 嵌入模型，自动生成向量并构建索引
vector_store = FAISS.from_documents(texts, embeddings)
# 将向量索引持久化保存到本地文件夹 faiss_index
vector_store.save_local("faiss_index")

# 从磁盘加载向量库
vector_store = FAISS.load_local(
    "faiss_index",                # 文件夹名称
    embeddings,                   # 必须传入和入库完全一致的Embedding模型
    allow_dangerous_deserialization=True # ⚠️Windows必加，反序列化安全开关，否则抛异常
)

# 1. 创建BM25关键词检索器
bm25_retriever = BM25Retriever.from_documents(texts)
bm25_retriever.k = 3

# 2. 创建FAISS向量检索器
faiss_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# 3. 融合检索器（混合检索）
ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """你是一个专业的问答助手。请根据以下提供的上下文信息来回答用户的问题。
如果上下文中没有相关信息，请诚实地告诉用户你不知道，不要编造答案。

上下文信息：
{context}

问题: {question}

回答:"""

prompt = ChatPromptTemplate.from_template(template)

model = init_chat_model(
    model="qwen3.7-max",
    model_provider="openai",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# chain =  ensemble_retriever
# retrieval = chain.invoke("LangChain是什么？")
# print(f"检索到的内容：{retrieval}")

print("=" * 60)

retrieval_chain = (
    {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
content = retrieval_chain.invoke("LangChain是什么？")
print(f"大模型回复内容：{content}")