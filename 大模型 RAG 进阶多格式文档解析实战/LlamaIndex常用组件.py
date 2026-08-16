from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse

# 如果文档结构复杂，优先使用 LlamaParse
# parser = LlamaParse(api_key="YOUR_LLAMA_CLOUD_API_KEY")
# documents = parser.load_data("sample.pdf")

# 或者使用简单读取器
documents = SimpleDirectoryReader(input_files=["D:/1/README.md"]).load_data()

print(documents[0].metadata)
print("===========================")
print(documents[0].text)
print("===========================")