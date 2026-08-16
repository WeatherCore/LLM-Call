from llama_index.readers.file.unstructured import UnstructuredReader
from pathlib import Path

reader = UnstructuredReader()
documents = reader.load_data(file=Path("D:/1/README.md"))

print("打印列表长度：" + str(len(documents)))
print("==================================")
print("打印解析的文本内容：" + documents[0].text[:100])
print("==================================")
print("打印元数据信息：" + str(documents[0].metadata))