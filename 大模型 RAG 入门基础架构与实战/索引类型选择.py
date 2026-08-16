# 此处已更新，以课件为主！
import faiss
import numpy as np

# 查询向量 A
A = np.array([1, 2, 3, 4, 5], dtype=np.float32)

# 4种关系：相同、反向、正交、部分相似
embeddings = np.array([
    [1, 2, 3, 4, 5],     # v0: 与A完全相同 -> cos = 1
    [-1, -2, -3, -4, -5],# v1: 与A方向相反 -> cos = -1
    [2, -1, 0, 0, 0],    # v2: 与A正交(点积=0) -> cos = 0
    [5, 4, 3, 2, 1],     # v3: 与A部分相似 -> cos = 0.636364
], dtype=np.float32)

dim = embeddings.shape[1]
print(f"dim={dim}, 向量数={len(embeddings)}")

# 归一化前模长
print("\n归一化前模长:")
for i, n in enumerate(np.linalg.norm(embeddings, axis=1)):
    print(f"v{i}: {n:.6f}")

# 关键：库向量和查询向量都要L2归一化
faiss.normalize_L2(embeddings)

query = A.reshape(1, -1).astype(np.float32)
faiss.normalize_L2(query)

# IndexFlatIP + 归一化 => 内积 = 余弦相似度
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

k = 4
scores, ids = index.search(query, k)

print("\nTop-k (score=cosine):")
for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), start=1):
    print(f"{rank}. id={idx}, score={score:.6f}")

# 归一化后，IndexFlatIP 的得分就是余弦相似度；1 表示同向，0 表示正交，-1 表示反向。