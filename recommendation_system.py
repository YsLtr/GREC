from data_loader import DataLoader
from vector_index import VectorIndex
import numpy as np

class GameRecommendationSystem:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_loader = DataLoader(data_dir)
        self.vector_index = VectorIndex(dimension=1024)
        self.appid_to_vector_index = None
        self.vector_index_to_appid = None
    
    def initialize(self):
        """初始化推荐系统"""
        # 加载数据
        self.data_loader.initialize()
        
        # 加载映射关系
        self.data_loader.load_embedding_maps()
        
        # 构建向量索引
        self.vector_index.build_index(self.data_loader.app_embeddings, index_type='flat')
        
        # 保存映射关系
        self.appid_to_vector_index = self.data_loader.app_id_to_index
        self.vector_index_to_appid = self.data_loader.index_to_app_id
        
        print("Recommendation system initialized successfully")
    
    def get_recommendations_by_appid(self, appid, k=10):
        """基于游戏ID推荐相似游戏"""
        # 获取游戏嵌入向量
        embedding = self.data_loader.get_embedding_by_appid(appid)
        if embedding is None:
            return f"Game with appid {appid} not found"
        
        # 搜索相似向量
        distances, indices = self.vector_index.search(embedding, k+1)  # 多取一个，排除自身
        
        # 构建推荐结果
        recommendations = []
        for i in range(len(distances)):
            vector_idx = indices[i]
            rec_appid = self.vector_index_to_appid[vector_idx]
            
            # 跳过查询的游戏本身
            if rec_appid == appid:
                continue
            
            # 获取游戏信息
            game_info = self.data_loader.get_game_by_appid(rec_appid)
            if game_info is not None:
                # 获取游戏分类
                genres = self.data_loader.appid_to_genres.get(rec_appid, [])
                
                # 获取用户评分
                user_score_data = self.data_loader.appid_to_user_score.get(rec_appid, {
                    'positive_rate': 0.0,
                    'total_reviews': 0,
                    'user_score_label': '评价不足'
                })
                
                recommendations.append({
                    'appid': rec_appid,
                    'name': game_info['name'],
                    'similarity': distances[i],
                    'is_free': game_info['is_free'],
                    'metacritic_score': game_info['metacritic_score'],
                    'short_description': game_info['short_description'],
                    'header_image': game_info['header_image'],
                    'release_date': game_info['release_date'],
                    'genres': genres,
                    'user_score_label': user_score_data['user_score_label'],
                    'user_score_percentage': user_score_data['positive_rate']
                })
            
            # 确保只返回k个推荐
            if len(recommendations) >= k:
                break
        
        # 根据评论数量重排推荐结果
        recommendations = self.reorder_recommendations(recommendations)
        
        return recommendations
    
    def get_recommendations_by_name(self, game_name, k=10):
        """基于游戏名称推荐相似游戏"""
        # 根据名称获取游戏
        game_info = self.data_loader.get_game_by_name(game_name)
        if game_info is None:
            return f"Game with name {game_name} not found"
        
        # 使用appid进行推荐
        return self.get_recommendations_by_appid(game_info['appid'], k)
    
    def get_recommendations_by_game_list(self, game_appids, k=10, weights=None):
        """基于多个游戏推荐相似游戏"""
        if not game_appids:
            return "No game appids provided"
        
        # 确保权重数量与游戏数量匹配
        if weights is None:
            weights = [1.0] * len(game_appids)
        elif len(weights) != len(game_appids):
            return "Number of weights must match number of game appids"
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 计算平均嵌入向量
        avg_embedding = np.zeros(1024)
        valid_count = 0
        
        for appid, weight in zip(game_appids, weights):
            embedding = self.data_loader.get_embedding_by_appid(appid)
            if embedding is not None:
                avg_embedding += embedding * weight
                valid_count += 1
        
        if valid_count == 0:
            return "No valid game appids provided"
        
        # 搜索相似向量
        distances, indices = self.vector_index.search(avg_embedding, k)
        
        # 构建推荐结果
        recommendations = []
        for i in range(len(distances)):
            vector_idx = indices[i]
            rec_appid = self.vector_index_to_appid[vector_idx]
            
            # 跳过查询的游戏
            if rec_appid in game_appids:
                continue
            
            # 获取游戏信息
            game_info = self.data_loader.get_game_by_appid(rec_appid)
            if game_info is not None:
                # 获取游戏分类
                genres = self.data_loader.appid_to_genres.get(rec_appid, [])
                
                # 获取用户评分
                user_score_data = self.data_loader.appid_to_user_score.get(rec_appid, {
                    'positive_rate': 0.0,
                    'total_reviews': 0,
                    'user_score_label': '评价不足'
                })
                
                recommendations.append({
                    'appid': rec_appid,
                    'name': game_info['name'],
                    'similarity': distances[i],
                    'is_free': game_info['is_free'],
                    'metacritic_score': game_info['metacritic_score'],
                    'short_description': game_info['short_description'],
                    'header_image': game_info['header_image'],
                    'release_date': game_info['release_date'],
                    'genres': genres,
                    'user_score_label': user_score_data['user_score_label'],
                    'user_score_percentage': user_score_data['positive_rate']
                })
            
            # 确保只返回k个推荐
            if len(recommendations) >= k:
                break
        
        # 根据评论数量重排推荐结果
        recommendations = self.reorder_recommendations(recommendations)
        
        return recommendations
    
    def filter_recommendations(self, recommendations, filter_criteria=None):
        """过滤推荐结果"""
        if filter_criteria is None:
            return recommendations
        
        filtered = []
        for rec in recommendations:
            match = True
            
            # 检查免费/付费过滤
            if 'is_free' in filter_criteria:
                if rec['is_free'] != filter_criteria['is_free']:
                    match = False
            
            # 检查评分过滤
            if 'min_metacritic_score' in filter_criteria:
                if rec['metacritic_score'] < filter_criteria['min_metacritic_score']:
                    match = False
            
            # 检查价格过滤
            # 注意：价格信息在applications_df中，这里可以扩展
            
            if match:
                filtered.append(rec)
        
        return filtered
    
    def display_recommendations(self, recommendations):
        """打印推荐结果"""
        if isinstance(recommendations, str):
            print(recommendations)
            return
        
        print(f"\nFound {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['name']}")
            print(f"   AppID: {rec['appid']}")
            print(f"   Similarity: {rec['similarity']:.4f}")
            print(f"   Free: {rec['is_free']}")
            print(f"   Metacritic Score: {rec['metacritic_score'] if not np.isnan(rec['metacritic_score']) else 'N/A'}")
            print(f"   Description: {rec['short_description'][:100]}...")
    
    def save_index(self, index_path):
        """保存向量索引"""
        self.vector_index.save_index(index_path)
    
    def load_index(self, index_path):
        """加载向量索引"""
        self.vector_index.load_index(index_path, self.data_loader.app_embeddings)
    
    def get_review_count_by_appid(self, appid):
        """根据appid获取游戏评论数量"""
        game = self.data_loader.get_game_by_appid(appid)
        if game is not None and 'recommendations_total' in game:
            return game['recommendations_total']
        return 0
    
    def calculate_weighted_score(self, similarity, review_count):
        """计算带评论权重的最终评分"""
        # 处理评论数量为nan的情况
        if review_count is None or np.isnan(review_count):
            return similarity
        
        # 获取最大评论数量，用于归一化
        max_review_count = self.data_loader.applications_df['recommendations_total'].max() if not self.data_loader.applications_df.empty else 1
        
        # 避免除零错误
        if max_review_count == 0:
            return similarity
        
        # 使用对数函数平滑评论数量差异
        review_weight = np.log(review_count + 1) / np.log(max_review_count + 1)
        
        # 计算最终评分：相似度 * (1 + 评论权重)
        return similarity * (1 + review_weight)
    
    def reorder_recommendations(self, recommendations):
        """根据评论数量重排推荐结果"""
        # 为每个推荐结果计算带评论权重的最终评分
        for rec in recommendations:
            review_count = self.get_review_count_by_appid(rec['appid'])
            rec['review_count'] = review_count
            rec['weighted_score'] = self.calculate_weighted_score(rec['similarity'], review_count)
        
        # 根据最终评分降序排序
        recommendations.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return recommendations
