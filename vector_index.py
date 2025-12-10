import numpy as np
import os
import faiss

class VectorIndex:
    def __init__(self, dimension=1024):
        self.dimension = dimension
        self.vectors = None
        self.index = None
        self.index_type = None
    
    def build_index(self, vectors, index_type='flat'):
        """构建向量索引，支持flat和hnsw两种索引类型"""
        self.vectors = vectors
        self.index_type = index_type
        
        if index_type == 'hnsw':
            # 构建HNSW索引
            # 使用IP（内积）距离，因为我们的向量已经归一化，内积等价于余弦相似度
            self.index = faiss.IndexHNSWFlat(self.dimension, 16)  # M=16是HNSW的连接数
            self.index.hnsw.efConstruction = 64  # 构建时的ef参数
            self.index.hnsw.efSearch = 32  # 搜索时的ef参数
            self.index.add(self.vectors)
            print(f"Built {index_type} index with {vectors.shape[0]} vectors")
        elif index_type == 'flat':
            # 构建flat索引
            self.index = faiss.IndexFlatIP(self.dimension)  # IP = Inner Product
            self.index.add(self.vectors)
            print(f"Built {index_type} index with {vectors.shape[0]} vectors")
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        return self.index
    
    def search(self, query_vector, k=10):
        """搜索最相似的k个向量"""
        try:
            # 验证查询向量
            if query_vector is None:
                raise ValueError("Query vector cannot be None")
            
            # 确保查询向量是numpy数组
            if not isinstance(query_vector, np.ndarray):
                query_vector = np.array(query_vector)
            
            # 确保查询向量维度正确
            if len(query_vector.shape) == 1:
                query_vector = np.reshape(query_vector, (1, -1))
            
            # 确保查询向量维度与索引维度匹配
            if query_vector.shape[1] != self.dimension:
                raise ValueError(f"Query vector dimension mismatch: expected {self.dimension}, got {query_vector.shape[1]}")
            
            # 确保索引已构建
            if self.index is None:
                raise ValueError("Index not built")
            
            # 使用faiss索引搜索相似向量
            # 对于IP索引，返回的是内积值，值越大越相似
            distances, indices = self.index.search(query_vector, k)
            
            # 将结果转换为一维数组
            distances = distances[0]
            indices = indices[0]
            
            return distances, indices
        except Exception as e:
            raise RuntimeError(f"Error during vector search: {str(e)}") from e
    
    def save_index(self, index_path):
        """保存索引到文件"""
        if self.vectors is None:
            raise ValueError("Vectors not loaded")
        if self.index is None:
            raise ValueError("Index not built")
        
        # 保存向量数据
        vectors_path = index_path.replace('.index', '_vectors.npy')
        np.save(vectors_path, self.vectors)
        
        # 保存faiss索引
        faiss.write_index(self.index, index_path)
        print(f"Index saved to {index_path}, vectors saved to {vectors_path}")
    
    def load_index(self, index_path, vectors=None):
        """从文件加载索引"""
        vectors_path = index_path.replace('.index', '_vectors.npy')
        
        # 检查faiss索引文件是否存在
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"faiss index file not found: {index_path}")
        
        # 加载向量数据
        if vectors is None:
            if os.path.exists(vectors_path):
                self.vectors = np.load(vectors_path)
            else:
                raise ValueError(f"Vectors file not found: {vectors_path}")
        else:
            self.vectors = vectors
        
        # 加载faiss索引
        self.index = faiss.read_index(index_path)
        # 确定索引类型
        if isinstance(self.index, faiss.IndexHNSWFlat):
            self.index_type = 'hnsw'
        elif isinstance(self.index, faiss.IndexFlatIP):
            self.index_type = 'flat'
        else:
            self.index_type = 'unknown'
        print(f"{self.index_type} index loaded from {index_path}")
        return self.index
    

