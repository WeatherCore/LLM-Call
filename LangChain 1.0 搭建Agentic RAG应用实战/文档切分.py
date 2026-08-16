from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, Docx2txtLoader

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

print(f"分割后的文本块数量: {len(texts)}")
print(texts[0].page_content)