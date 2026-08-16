# UnstructuredIO核心组件
from unstructured.partition.auto import partition
from unstructured.partition.md import partition_md
from typing import List
from unstructured.documents.elements import Element

# 使用partition函数自动检测文件类型并解析,默认strategy策略是auto，还会有fast策略，速度比image-to-text models的快100倍
# elements: List[Element] = partition(filename="D:/1/RAG评估.md", strategy="auto")
elements: List [Element] = partition_md (filename="D:/1/README.md", languages=["zho"],include_page_breaks=True)

# 元素的文本内容
print(elements[0].text)
print("===========================")

# 元素的类型
print(elements[0].category)
print("==================")

# 元素的元数据
print(elements[0].metadata.__dict__)
print("===========================")
