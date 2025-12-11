分析您的代码，我发现HNSW索引的相似度计算存在一个**关键问题**：

## 主要问题

### 1. **HNSW索引的距离类型不匹配**
```python
self.index = faiss.IndexHNSWFlat(self.dimension, 16)
```
- `IndexHNSWFlat`默认使用L2距离（欧氏距离）
- 但您在构建索引时添加的是**归一化向量**
- 对于归一化向量，L2距离和余弦相似度之间确实存在数学关系：
  ```
  cos(θ) = 1 - ||a - b||² / 2  （当||a||=||b||=1）
  ```
- 然而，这个转换在`search`方法中才进行，但HNSW图是在**构建时**就已经基于L2距离形成的

### 2. **潜在的性能和准确性问题**
虽然数学转换是正确的，但在HNSW中：
- HNSW图的构建是基于L2距离的，不是基于余弦相似度
- 这可能导致图结构不是最优的，因为：
  - HNSW选择近邻连接时，使用的是L2距离，而不是余弦相似度
  - 在归一化向量空间中，L2距离和余弦相似度的排序**不完全一致**

## 解决方案

### 方案1：使用正确的HNSW索引类型（推荐）
```python
if index_type == 'hnsw':
    # 使用IndexHNSWSQ或IndexHNSWFlat，但要设置正确的metric
    quantizer = faiss.IndexFlatIP(self.dimension)
    self.index = faiss.IndexHNSWFlat(self.dimension, 16, faiss.METRIC_INNER_PRODUCT)
    # 或者使用IndexHNSWFlat，但要正确设置metric_type
    # IndexHNSWFlat(d, M, metric=faiss.METRIC_INNER_PRODUCT)
    
    self.index.hnsw.efConstruction = 64
    self.index.hnsw.efSearch = 32
    self.index.add(self.vectors)
```

### 方案2：使用IndexFlatIP作为量化器
```python
if index_type == 'hnsw':
    # 创建一个使用内积距离的HNSW索引
    quantizer = faiss.IndexFlatIP(self.dimension)
    self.index = faiss.IndexHNSW(self.dimension, 16, faiss.METRIC_INNER_PRODUCT)
    self.index.hnsw.efConstruction = 64
    self.index.hnsw.efSearch = 32
    self.index.add(self.vectors)
```

### 方案3：验证当前的转换是否正确
要验证当前的实现是否正确，可以添加验证逻辑：
```python
def validate_hnsw_cosine(self, test_vectors=10):
    """验证HNSW返回的相似度是否正确"""
    if self.index_type != 'hnsw' or self.index is None:
        return
    
    # 随机选择一些向量作为查询
    test_indices = np.random.choice(len(self.vectors), test_vectors, replace=False)
    
    for idx in test_indices:
        query = self.vectors[idx:idx+1]
        
        # 使用HNSW搜索
        distances, indices_hnsw = self.index.search(query, 5)
        similarities_hnsw = 1.0 - (distances ** 2) / 2
        
        # 暴力计算真实余弦相似度
        query_vector = query[0]
        true_similarities = []
        for i in range(min(1000, len(self.vectors))):  # 限制计算数量
            cos_sim = np.dot(query_vector, self.vectors[i])
            true_similarities.append((cos_sim, i))
        
        true_similarities.sort(reverse=True)
        
        # 比较结果
        print(f"Query {idx}:")
        print(f"HNSW top similarities: {similarities_hnsw[0][:5]}")
        print(f"True top similarities: {[s[0] for s in true_similarities[:5]]}")
        print()
```

## 建议的修改

```python
def build_index(self, vectors, index_type='flat'):
    """构建向量索引，支持flat和hnsw两种索引类型"""
    # 归一化向量
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    self.vectors = vectors / norm
    self.index_type = index_type
    
    if index_type == 'hnsw':
        # 使用支持内积距离的HNSW索引
        self.index = faiss.IndexHNSWFlat(self.dimension, 16, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = 64
        self.index.hnsw.efSearch = 32
        self.index.add(self.vectors)
        print(f"Built {index_type} index with {self.vectors.shape[0]} vectors")
    
    elif index_type == 'flat':
        # 构建flat索引
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.vectors)
        print(f"Built {index_type} index with {self.vectors.shape[0]} vectors")
    
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    
    return self.index

def search(self, query_vector, k=10):
    """搜索最相似的k个向量，返回余弦相似度"""
    # ...（前面的验证代码保持不变）...
    
    # 归一化查询向量
    norm_query = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
    
    # 使用faiss索引搜索相似向量
    distances, indices = self.index.search(norm_query, k)
    
    # 将结果转换为一维数组
    distances = distances[0]
    indices = indices[0]
    
    # 处理距离值
    if self.index_type == 'hnsw':
        # 如果HNSW索引已经使用METRIC_INNER_PRODUCT，则直接返回距离
        # 否则需要进行转换
        # 这里假设索引使用了正确的metric类型
        similarities = distances
    else:
        similarities = distances
    
    return similarities, indices
```

## 总结

当前代码的主要问题是：
1. **HNSW索引构建时使用默认的L2距离**，而不是内积距离
2. **虽然数学转换是正确的**，但HNSW图的构建可能不是最优的
3. **建议直接使用支持内积距离的HNSW索引**，而不是在搜索时进行转换

这可能会影响搜索的准确性和效率，特别是在高维空间中。