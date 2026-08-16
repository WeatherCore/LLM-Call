from unstructured.partition.auto import partition
from llama_index.core import Document

# ✅ 【重点】这里可以自由精细控制unstructured所有参数！
elements = partition(
    filename="D:/1/README.md",
    strategy="hi_res",          # 高清解析（识别表格首选！）
    split_pdf_page=True,
    infer_table_structure=True, # 开启表格结构化识别
    languages=["eng","chi_sim"] # 指定中英OCR
)

# ✅ 自己循环，把每一个element手动包装成Document
docs = [
    Document(
        text=e.text,
        metadata={
            "source":"甬兴证券-AI行业点评报告.pdf",
            "type": e.category  # ✅ 重点！保留元素类型：Title / Table / NarrativeText
        }
    )
    for e in elements
]