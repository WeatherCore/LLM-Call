from langchain_community.document_loaders import TextLoader, Docx2txtLoader

# 读取基础数据文档
loader = TextLoader(r"D:\VS Code-wenjian\大模型\LangChain 1.0 搭建Agentic RAG应用实战\sample_document.txt", encoding="utf-8")
documents = loader.load()

print(documents[0].page_content)