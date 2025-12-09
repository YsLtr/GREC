import numpy as np
import os

class VectorIndex:
    def __init__(self, dimension=1024):
        self.dimension = dimension
        self.vectors = None
    
    def build_index(self, vectors, index_type='flat'):
        """构建向量索引"""
        self.vectors = vectors
        print(f"Built {index_type} index with {vectors.shape[0]} vectors")
        return self.vectors
    
    def search(self, query_vector, k=10):
        """搜索最相似的k个向量"""
        if self.vectors is None:
            raise ValueError("Index not built yet")
        
        # 计算余弦相似度
        # 归一化向量
        norm_vectors = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norm_query = query_vector / np.linalg.norm(query_vector)
        
        # 计算相似度
        similarities = np.dot(norm_vectors, norm_query)
        
        # 获取top k最相似的向量
        indices = np.argsort(similarities)[::-1][:k]  # 降序排列，取前k个
        distances = similarities[indices]
        
        return distances, indices
    
    def save_index(self, index_path):
        """保存索引到文件"""
        if self.vectors is None:
            raise ValueError("Index not built yet")
        
        np.save(index_path, self.vectors)
        print(f"Index saved to {index_path}")
    
    def load_index(self, index_path, vectors=None):
        """从文件加载索引"""
        self.vectors = np.load(index_path)
        print(f"Index loaded from {index_path}")
        return self.vectors
    
    def batch_search(self, query_vectors, k=10):
        """批量搜索"""
        if self.vectors is None:
            raise ValueError("Index not built yet")
        
        results = []
        for query_vector in query_vectors:
            distances, indices = self.search(query_vector, k)
            results.append((distances, indices))
        
        return results
    
    def get_vector_by_index(self, idx):
        """根据索引获取向量"""
        if self.vectors is None:
            raise ValueError("Vectors not loaded")
        
        return self.vectors[idx]
    
    def get_similar_vectors(self, query_vector, k=10):
        """获取最相似的k个向量"""
        distances, indices = self.search(query_vector, k)
        similar_vectors = self.vectors[indices]
        return similar_vectors, distances, indices
