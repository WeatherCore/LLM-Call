import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader  # pypdf，PyPDF2的继任者
# 加载 .env 文件中的环境变量（override=True 确保 .env 优先级高于系统环境变量）
load_dotenv(override=True)

# — OpenAI Embedding —————————————————————————————————————————
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# — Qwen Embedding（兼容 OpenAI 协议，只需切换 base_url）——————
client_qwen = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
MODEL = "text-embedding-v3"

def get_embedding(text: str, client, model: str) -> list[float]:
    """单文本向量化：返回一个浮点数列表"""
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str], client, model: str) -> list[list[float]]:
    """批量向量化：一次 API 调用处理多个文本，节省延迟和成本"""
    response = client.embeddings.create(input=texts, model=model)
    # 注意：返回结果的顺序与 input 列表顺序严格一致
    return [item.embedding for item in response.data]


# 请确认模型名称为当前可用版本（查询日期：2026‑02）
# text‑embedding‑v3 仍有效；如需更强中文能力可升级为 text‑embedding‑v4
# openai的embedding模型为： text‑embedding‑3‑small

# client = client_qwen     # 或切换为 client_openai

# vec1 = get_embedding("如何申请年假", client, MODEL)
# vec2 = get_embedding("请假流程是什么", client, MODEL)
# vec3 = get_embedding("今天天气怎么样", client, MODEL)

# print(f"向量维度: {len(vec1)}")        # 应为 1024
# print(f"前 5 个数值: {vec1[:5]}")     # 浮点数数组，正负均有


def load_txt(file_path: str) -> dict:
    """加载 TXT / Markdown 文件, 统一编码为 utf‑8"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return {
            "content": f.read(),
            "metadata": {"source": file_path, "type": "txt"}
        }

def load_pdf(file_path: str) -> dict:
    """加载 PDF 文件: 逐页提取文本, 用换行符拼接各页内容"""
    reader = PdfReader(file_path)
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return {
        "content": text,
        "metadata": {"source": file_path, "type": "pdf", "pages": len(reader.pages)}
    }

def load_document(file_path: str) -> dict:
    """统一入口: 根据扩展名自动选择加载器【分发器模式】"""
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {".txt": load_txt, ".md": load_txt, ".pdf": load_pdf}
    loader = loaders.get(ext)
    if not loader:
        raise ValueError(f"不支持的文件格式: {ext} (当前支持: .txt .md .pdf)")
    return loader(file_path)

# # 调用，不管是txt/md/pdf写法完全一样
# doc_pdf = load_document("D:/VS Code-wenjian/大模型/大模型 RAG 入门基础架构与实战/data/company_leave_policy.pdf")
# print(doc_pdf["content"][:5000])       # 文档文本
# print(doc_pdf["metadata"])     # {'source':'./demo.pdf','type':'pdf','pages':12}

def split_text(text: str, chunk_size: int = 500, overlap: int = 100,
               separators: list[str] = None, metadata: dict = None) -> list[dict]:
    """
    递归字符切分器（保真版）

    设计约束：
    - chunk 的最终长度不超过 chunk_size
    - overlap 包含在 chunk_size 内
    - 不吞掉原文中的分隔符和连续空白
    """
    # 先做参数校验，避免出现无限切分或 overlap 挤占掉全部有效内容。
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # 默认按“段落 -> 换行 -> 句子 -> 词 -> 字符”的粒度逐级降级。
    if separators is None:
        separators = ["\n\n", "\n", "。", ".", " ", ""]
    metadata = metadata or {}

    # 预留 overlap 空间，确保最终 content 长度仍不超过 chunk_size。
    payload_size = chunk_size - overlap if overlap > 0 else chunk_size

    def _split_keep_separator(value: str, sep: str) -> list[str]:
        """
        根据指定分隔符切分文本，并确保分隔符保留在切分后的片段末尾。
        这样做可以避免在 chunk 边界处丢失原文的结构信息（如句号、换行符等）。
        """
        pieces = []
        start = 0
        while True:
            # 从当前起始位置查找分隔符
            index = value.find(sep, start)

            # 如果找不到分隔符，说明已到达文本末尾
            if index == -1:
                tail = value[start:]
                # 如果末尾还有剩余字符，则作为最后一个片段加入
                if tail:
                    pieces.append(tail)
                break
            
            # 计算包含分隔符在内的片段结束位置
            end = index + len(sep)

            # 截取从 start 到 end 的子串，确保分隔符被包含在内
            pieces.append(value[start:end])

            # 将下一次查找的起始位置更新为当前片段的结束位置
            start = end
            
        # 如果 pieces 为空（例如输入为空字符串），则返回包含原字符串的列表作为兜底
        return pieces or [value]

    def _split_recursive(value: str, seps: list[str]) -> list[str]:
        # 当前片段已经足够小，直接返回，不再继续降级切分。
        if len(value) <= payload_size:
            return [value]

        # 所有分隔符都用完后，才退化为按固定宽度硬切。
        if not seps:
            return [value[i:i + payload_size] for i in range(0, len(value), payload_size)]

        sep = seps[0]

        # 空字符串表示最后兜底：按字符窗口切分。
        if sep == "":
            return [value[i:i + payload_size] for i in range(0, len(value), payload_size)]

        result = []
        for piece in _split_keep_separator(value, sep):
            # 当前层切出来的片段放得下就收下，放不下才递归到更细粒度。
            if len(piece) <= payload_size:
                result.append(piece)
            else:
                result.extend(_split_recursive(piece, seps[1:]))
        return result

    def _merge_splits(splits: list[str]) -> list[str]:
        """
        将递归切分后的细小片段进行贪心合并。
        目标是在不超过 payload_size 的前提下，尽可能让每个 chunk 更完整，减少碎片化。
        """
        chunks = []
        current = ""
        for split in splits:
            # 如果当前缓存为空，直接放入第一个片段
            if not current:
                current = split
            # 如果当前缓存加上新片段的长度仍未超过预留的 payload_size，则进行合并
            elif len(current) + len(split) <= payload_size:
                current += split
            # 否则，说明当前缓存已达到合并上限，存入结果集，并开启新的缓存
            else:
                chunks.append(current)
                current = split
        
        # 处理最后一个残余的片段
        if current:
            chunks.append(current)
        return chunks

    # 先递归切到合法大小，再做一次贪心合并，减少碎片化。
    raw_chunks = _merge_splits(_split_recursive(text, separators))

    chunks = []
    current_start = 0
    for i, raw_chunk in enumerate(raw_chunks):
        # 计算当前 chunk 的实际内容：
        # overlap 不再依赖“前一个 raw_chunk 的尾巴”，而是直接根据当前片段在原文中的起点回溯。
        # 这样即使前一个 raw_chunk 很短，也能跨越多个 raw_chunk 从原文中拿满 overlap。
        # 由于 raw_chunk 的长度已在 merge 阶段限制在 payload_size (chunk_size - overlap) 之内，
        # 回溯 overlap 后的 content 总长度依然严格保证不超过原始定义的 chunk_size。
        if overlap == 0:
            # 无重叠情况：直接使用当前切分的片段内容
            content = raw_chunk
        else:
            # 有重叠情况：从原文起始位置回溯 overlap 长度，确保片段间存在上下文交叠
            start_with_overlap = max(0, current_start - overlap)
            current_end = current_start + len(raw_chunk)
            content = text[start_with_overlap:current_end]
        
        # 将文本内容与元数据封装为字典，方便后续向量化及检索溯源
        chunks.append({
            "content": content,
            "metadata": {
                **metadata,
                "chunk_index": i,
                "chunk_total": len(raw_chunks)
            }
        })

        # 更新当前片段在原文中的起始位点，供后续片段计算回溯使用
        current_start += len(raw_chunk)

    return chunks


# # 读取pdf文件
# doc = load_document("D:/VS Code-wenjian/大模型/大模型 RAG 入门基础架构与实战/data/company_leave_policy.pdf")

# # 分块大小为500，重叠部分为100
# chunks = split_text(doc["content"], chunk_size=500, overlap=100, metadata=doc["metadata"])

# lengths = [len(c["content"]) for c in chunks]
# print(f"总片段数：{len(chunks)}")
# print(f"长度范围：{min(lengths)} ~ {max(lengths)} 字符")
# print(f"平均长度：{sum(lengths) / len(lengths):.0f} 字符")
# print(f"\n片段 0 metadata：{chunks[0]['metadata']}")
# print("=" * 60)
# print(f"片段 0 前 100 字：{chunks[0]['content']}")
# print("=" * 60)
# print(f"片段 1 前 50 字（应包含片段 0 末尾内容）：{chunks[1]['content'][:50]}")


import numpy as np

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    计算两个向量之间的余弦相似度（NumPy 实现）

    Args:
        vec_a: 向量 A 的数值列表
        vec_b: 向量 B 的数值列表

    Returns:
        float: 余弦相似度得分，取值范围为 [-1, 1]，越接近 1 表示越相似
    """
    # 将输入列表转换为 NumPy 数组以便进行向量化计算
    a, b = np.array(vec_a), np.array(vec_b)

    # 计算向量点积 (Dot Product): A · B
    dot_product = np.dot(a, b)

    # 计算向量的 L2 范数（模长）: ||A|| 和 ||B||
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # 根据公式计算余弦相似度: cos(θ) = (A · B) / (||A|| * ||B||)
    # 注意: 实际应用中应考虑分母为 0 的情况
    return float(dot_product / (norm_a * norm_b))


# # 请确认模型名称为当前可用版本（查询日期：2026-02）
# MODEL = "text-embedding-v3"

# pairs = [
#     ("如何申请年假",     "请假流程是什么"),        # 中文同义，语义相近
#     ("如何申请年假",     "今天天气怎么样"),        # 语义无关
#     ("Python 列表排序",  "Python list sort"),      # 中英文同义
# ]

# for text_a, text_b in pairs:
#     # 调用 Embedding 接口获取文本 A 的向量表示
#     vec_a = get_embedding(text_a, client_qwen, MODEL)
#     # 调用 Embedding 接口获取文本 B 的向量表示
#     vec_b = get_embedding(text_b, client_qwen, MODEL)
    
#     # 计算两个向量之间的余弦相似度（Score 越接近 1 表示语义越相关）
#     score = cosine_similarity(vec_a, vec_b)
    
#     # 打印对比结果，展示不同文本组合下的语义匹配得分
#     print(f"  {text_a!r:20s} vs {text_b!r:20s}  →  {score:.4f}")

import faiss
import numpy as np
import json


def build_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatIP:
    """
    构建 FAISS 内积索引。
    通过对向量进行 L2 归一化，使得内积计算等价于余弦相似度。

    Args:
        embeddings: 嵌入向量列表，每个元素为一个浮点数列表。

    Returns:
        faiss.IndexFlatIP: 构建好的 FAISS 索引对象。
    """
    # 获取向量维度（特征长度）1024
    dim = len(embeddings[0])

    # 将输入列表转换为 float32 类型的 numpy 数组，这是 FAISS 指定的数值类型
    vectors = np.array(embeddings, dtype=np.float32)

    # 执行原地 L2 归一化：||v|| = 1。归一化后的向量点积即为余弦相似度
    faiss.normalize_L2(vectors)

    # 创建暴力搜索的内积索引 (IndexFlatIP)
    index = faiss.IndexFlatIP(dim)

    # 将处理后的向量添加到索引库中
    index.add(vectors)

    return index


def save_index(index: faiss.IndexFlatIP, chunks: list[dict],
               index_path: str, metadata_path: str) -> None:
    """
    持久化索引与元数据双文件方案。
    index_path 存储 FAISS 向量索引数据，metadata_path 存储对应的文本块及原始信息。

    Args:
        index: FAISS 索引对象。
        chunks: 包含 "content" 和 "metadata" 的原始文本块列表。
        index_path: 索引文件的保存路径（二进制格式）。
        metadata_path: 元数据文件的保存路径（JSON 格式）。
    """
    # 将向量索引二进制流写入磁盘
    faiss.write_index(index, index_path)

    # 将文本块及元数据序列化为 JSON，确保非 ASCII 字符（如中文）正常显示
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ 索引保存成功：{index_path}（包含 {index.ntotal} 条向量）")
    print(f"✅ 元数据保存成功：{metadata_path}")


def load_index(index_path: str, metadata_path: str) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    从磁盘恢复索引和关联的文本块数据。

    Args:
        index_path: 索引二进制文件路径。
        metadata_path: 元数据 JSON 文件路径。

    Returns:
        tuple: (index, chunks) 包含加载后的 FAISS 索引和对应的文本块列表。
    """
    # 读取二进制索引文件
    index = faiss.read_index(index_path)

    # 读取 JSON 格式的文本元数据
    with open(metadata_path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    return index, chunks

# chunks = [
#     {"content": "FAISS用于向量检索", "metadata": {"source":"xxx.pdf"}},
#     {"content": "RAG减少大模型幻觉", "metadata": {"source":"yyy.md"}}
# ]

# # 对所有 chunks 批量向量化（一次 API 调用，比循环逐个调用快得多）
# texts = [c["content"] for c in chunks]
# embeddings = get_embeddings_batch(texts, client_qwen, MODEL)

# # 构建索引并持久化
# index = build_faiss_index(embeddings)
# save_index(index, chunks, "faiss_data/rag.index", "faiss_data/rag_chunks.json")

# # 重新加载，验证一致性
# index2, chunks2 = load_index("faiss_data/rag.index", "faiss_data/rag_chunks.json")
# assert index2.ntotal == len(chunks2), "❌ 索引和 chunks 数量不匹配！"
# print(f"✅ 索引持久化验证通过：{index2.ntotal} 条向量 = {len(chunks2)} 个 chunks")


def search(query: str, client, model: str, index: faiss.IndexFlatIP,
           chunks: list[dict], top_k: int = 5,
           threshold: float = 0.5) -> list[dict]:
    """
    RAG 检索函数：向量化查询 → FAISS 检索 → 相似度过滤 → 排序
    参数：
      threshold: 余弦相似度阈值（0~1），低于此分数的结果被丢弃
    返回：
      按相似度降序排列的 chunk 列表，每项含 content / metadata / score
    """
    # Step 1: 查询向量化 + 归一化（必须与索引构建时用同一模型和同样的归一化）
    query_vec = np.array([get_embedding(query, client, model)], dtype=np.float32)
    faiss.normalize_L2(query_vec)

    # Step 2: FAISS 检索 Top-K
    # 注意：当 top_k > index.ntotal 时，多余的返回位置填充 -1（需过滤）
    scores, indices = index.search(query_vec, top_k)

    # Step 3: 过滤 + 组装输出
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:            # FAISS 填充的无效位置（top_k > 实际向量数时出现）
            continue
        if score < threshold:    # 相似度低于阈值，丢弃
            continue
        results.append({
            "content":  chunks[idx]["content"],
            "metadata": chunks[idx]["metadata"],
            "score":    float(score)
        })

    # Step 4: 按相似度降序排序（FAISS 已返回排序结果，此处为显式保证）
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# query = "年假可以申请几天？"

# results = search(
#     query,          # 查询文本
#     client_qwen,    # 嵌入模型客户端
#     MODEL,          # 嵌入模型名称
#     index,          # 向量索引
#     chunks,         # 原始文本块
#     top_k=5,        # 返回最相关的结果数量
#     threshold=0.5   # 相似度阈值
# )

# print(f"检索到 {len(results)} 条相关结果")
# for i, r in enumerate(results[:3]):
#     source = r['metadata'].get('source', '未知')
#     print(f"\n── Top {i+1}（得分 {r['score']:.4f}，来源：{source}）")
#     print(r["content"][:150] + "...")