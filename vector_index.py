import numpy as np
import os

# # 尝试导入faiss，如不可用则使用纯numpy实现
# try:
#     import faiss
#     FAISS_AVAILABLE = True
# except ImportError:
#     FAISS_AVAILABLE = False
#     print("faiss library not found, using pure numpy implementation")
FAISS_AVAILABLE = False

class VectorIndex:
    def __init__(self, dimension=1024):
        self.dimension = dimension
        self.vectors = None
        self.index = None
        self.index_type = None
        self.use_faiss = FAISS_AVAILABLE
    
    def build_index(self, vectors, index_type='flat'):
        """构建向量索引，支持flat和hnsw两种索引类型"""
        self.vectors = vectors
        self.index_type = index_type
        
        # 如果faiss可用，使用faiss构建索引
        if self.use_faiss:
            try:
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
            except Exception as e:
                print(f"Failed to build {index_type} index with faiss, falling back to pure numpy implementation: {e}")
                self.use_faiss = False
        
        # 如果faiss不可用或构建失败，使用纯numpy实现
        if index_type == 'flat':
            # 纯numpy实现只支持flat索引
            print(f"Built {index_type} index with {vectors.shape[0]} vectors using pure numpy implementation")
            return self.vectors
        else:
            raise ValueError(f"Index type {index_type} not supported with pure numpy implementation, please use 'flat' instead")
    
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
            
            # 如果使用faiss索引
            if self.use_faiss and self.index is not None:
                # 搜索相似向量
                # 对于IP索引，返回的是内积值，值越大越相似
                distances, indices = self.index.search(query_vector, k)
                
                # 将结果转换为一维数组
                distances = distances[0]
                indices = indices[0]
            else:
                # 使用纯numpy实现搜索
                if self.vectors is None:
                    raise ValueError("Vectors not loaded")
                
                # 计算余弦相似度
                # 归一化向量
                norm_vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
                norm_query = query_vector / np.linalg.norm(query_vector)
                
                # 计算相似度
                similarities = np.dot(norm_vectors, norm_query.T)
                similarities = similarities.flatten()
                
                # 获取top k最相似的向量
                indices = np.argsort(similarities)[::-1][:k]  # 降序排列，取前k个
                distances = similarities[indices]
            
            return distances, indices
        except Exception as e:
            raise RuntimeError(f"Error during vector search: {str(e)}") from e
    
    def save_index(self, index_path):
        """保存索引到文件"""
        if self.vectors is None:
            raise ValueError("Vectors not loaded")
        
        # 保存向量数据
        vectors_path = index_path.replace('.index', '_vectors.npy')
        np.save(vectors_path, self.vectors)
        
        # 如果使用faiss索引，同时保存索引文件
        if self.use_faiss and self.index is not None:
            # 保存索引
            faiss.write_index(self.index, index_path)
            print(f"Index saved to {index_path}, vectors saved to {vectors_path}")
        else:
            # 使用纯numpy实现，只保存向量数据
            print(f"Vectors saved to {vectors_path}")
    
    def load_index(self, index_path, vectors=None):
        """从文件加载索引"""
        vectors_path = index_path.replace('.index', '_vectors.npy')
        
        # 检查faiss索引文件是否存在
        index_file_exists = os.path.exists(index_path)
        
        # 加载向量数据
        if vectors is None:
            if os.path.exists(vectors_path):
                self.vectors = np.load(vectors_path)
            else:
                raise ValueError(f"Vectors file not found: {vectors_path}")
        else:
            self.vectors = vectors
        
        # 如果使用faiss索引且索引文件存在
        if self.use_faiss and index_file_exists:
            try:
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
            except Exception as e:
                print(f"Failed to load faiss index, falling back to pure numpy implementation: {e}")
                self.use_faiss = False
                self.index = None
        
        # 如果faiss可用但索引文件不存在，抛出异常，让调用者重新构建索引
        if self.use_faiss and not index_file_exists:
            raise FileNotFoundError(f"faiss index file not found: {index_path}")
        
        # 使用纯numpy实现
        self.index_type = 'flat'
        print(f"Loaded vectors from {vectors_path}, using pure numpy implementation")
        return self.vectors
    

