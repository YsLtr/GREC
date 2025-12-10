from data_loader import DataLoader
from vector_index import VectorIndex
import numpy as np
import os
from config import GAME_WEIGHT, REVIEW_WEIGHT, REVIEW_COUNT_WEIGHT, USER_SCORE_WEIGHT

class GameRecommendationSystem:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_loader = DataLoader(data_dir)
        self.vector_index = VectorIndex(dimension=1024)  # 内容推荐索引
        self.user_vector_index = VectorIndex(dimension=1024)  # 用户嵌入索引
        self.collaborative_index = VectorIndex(dimension=1024)  # 协同推荐索引
        self.appid_to_vector_index = None
        self.vector_index_to_appid = None
        # 评论嵌入相关
        self.review_embeddings = None
        self.appid_to_review_vectors = None
        # 用户嵌入相关
        self.steamid_to_review_vectors = None
        self.steamid_to_appids = None
        self.userid_to_vector_index = None
        self.vector_index_to_userid = None
        # 游戏平均评论向量相关
        self.game_avg_review_vectors = None
        self.gameid_to_vector_index = None
        self.vector_index_to_gameid = None
    
    def initialize(self, use_content_based=True, use_collaborative=True):
        """初始化推荐系统，根据推荐算法的开关选择性加载数据"""
        print(f"Initializing recommendation system with content_based={use_content_based}, collaborative={use_collaborative}")
        
        # 验证至少启用一种推荐算法
        if not use_content_based and not use_collaborative:
            raise ValueError("至少需要启用一种推荐算法：内容推荐或协同推荐")
        
        # 创建索引目录（如果不存在）
        index_dir = os.path.join(self.data_loader.data_dir, 'indices')
        os.makedirs(index_dir, exist_ok=True)
        
        # 选择性加载数据
        if use_content_based or use_collaborative:
            # 加载基础数据
            self.data_loader.load_applications_data()
            self.data_loader.load_embeddings(load_app_embeddings=use_content_based, load_review_embeddings=use_collaborative)
            self.data_loader.load_embedding_maps(load_content_mappings=use_content_based, load_collaborative_mappings=use_collaborative)
            self.data_loader.load_genres_data()
        
        if use_content_based:
            # 内容推荐索引文件路径
            content_index_path = os.path.join(index_dir, 'content_hnsw.index')
            
            # 检查faiss是否可用
            faiss_available = self.vector_index.use_faiss
            print(f"faiss available: {faiss_available}")
            
            # 尝试加载已保存的内容索引
            index_loaded = False
            try:
                print(f"Attempting to load content index from {content_index_path}...")
                # 加载索引，传入向量数据
                self.vector_index.load_index(content_index_path, self.data_loader.app_embeddings)
                
                # 检查索引类型是否符合预期
                if faiss_available and self.vector_index.index_type == 'hnsw':
                    print(f"Successfully loaded hnsw content index from {content_index_path}")
                    index_loaded = True
                else:
                    print(f"Loaded {self.vector_index.index_type} index, but hnsw index is preferred")
            except Exception as e:
                print(f"Failed to load content index: {e}")
            
            # 如果索引未加载成功，构建新索引
            if not index_loaded:
                print(f"Building new content index...")
                # 根据faiss可用性选择索引类型
                index_type = 'hnsw' if faiss_available else 'flat'
                print(f"Using index type: {index_type}")
                
                # 直接构建向量索引，不重新初始化vector_index
                self.vector_index.build_index(self.data_loader.app_embeddings, index_type=index_type)
                
                # 保存索引到文件
                print(f"Saving content index to {content_index_path}...")
                self.vector_index.save_index(content_index_path)
                print(f"Successfully saved {index_type} content index to {content_index_path}")
            
            # 保存映射关系
            self.appid_to_vector_index = self.data_loader.app_id_to_index
            self.vector_index_to_appid = self.data_loader.index_to_app_id
        
        if use_collaborative:
            # 保存评论嵌入相关数据
            self.review_embeddings = self.data_loader.review_embeddings
            self.appid_to_review_vectors = self.data_loader.appid_to_review_vectors
            
            # 保存用户嵌入相关数据
            self.steamid_to_review_vectors = self.data_loader.steamid_to_review_vectors
            self.steamid_to_appids = self.data_loader.steamid_to_appids
            
            # 构建用户嵌入索引
            if self.steamid_to_review_vectors and len(self.steamid_to_review_vectors) > 0:
                print(f"Building user embedding index for {len(self.steamid_to_review_vectors)} users...")
                # 准备用户嵌入数据
                user_embeddings = []
                user_ids = []
                for steamid, embedding in self.steamid_to_review_vectors.items():
                    user_embeddings.append(embedding)
                    user_ids.append(steamid)
                
                # 转换为numpy数组
                user_embeddings = np.array(user_embeddings)
                
                # 构建用户嵌入映射
                self.userid_to_vector_index = {steamid: idx for idx, steamid in enumerate(user_ids)}
                self.vector_index_to_userid = {idx: steamid for idx, steamid in enumerate(user_ids)}
                
                # 构建用户嵌入索引
                user_index_path = os.path.join(index_dir, 'user_hnsw.index')
                
                # 检查user_vector_index是否已经使用faiss
                print(f"Before user index operation, faiss available: {self.user_vector_index.use_faiss}")
                
                # 根据faiss可用性选择索引类型
                user_index_type = 'hnsw' if self.vector_index.use_faiss else 'flat'
                print(f"Using index type: {user_index_type} for user index")
                
                try:
                    print(f"Attempting to load user index from {user_index_path}...")
                    self.user_vector_index.load_index(user_index_path, user_embeddings)
                    
                    print(f"Successfully loaded user index from {user_index_path}, index_type: {self.user_vector_index.index_type}")
                except Exception as e:
                    print(f"Failed to load user index: {e}, building new index...")
                    
                    # 直接构建索引，不重新初始化user_vector_index
                    self.user_vector_index.build_index(user_embeddings, index_type=user_index_type)
                    print(f"Built user index, index_type: {self.user_vector_index.index_type}, faiss used: {self.user_vector_index.use_faiss}")
                    
                    print(f"Saving user index to {user_index_path}...")
                    self.user_vector_index.save_index(user_index_path)
                    print(f"Successfully saved user index to {user_index_path}")
        
        # 构建协同推荐索引
        if use_collaborative and self.appid_to_review_vectors and len(self.appid_to_review_vectors) > 0:
            print(f"Building collaborative recommendation index...")
            # 准备游戏平均评论向量数据
            game_embeddings = []
            game_ids = []
            for appid, review_vectors in self.appid_to_review_vectors.items():
                # 计算每个游戏的平均评论向量
                avg_vector = np.mean(review_vectors, axis=0)
                game_embeddings.append(avg_vector)
                game_ids.append(appid)
            
            # 转换为numpy数组
            game_embeddings = np.array(game_embeddings)
            
            # 构建游戏平均评论向量映射
            self.gameid_to_vector_index = {appid: idx for idx, appid in enumerate(game_ids)}
            self.vector_index_to_gameid = {idx: appid for idx, appid in enumerate(game_ids)}
            
            # 构建协同推荐索引
            collaborative_index_path = os.path.join(index_dir, 'collaborative_hnsw.index')
            
            # 检查collaborative_index是否已经使用faiss
            print(f"Before collaborative index operation, faiss available: {self.collaborative_index.use_faiss}")
            
            # 根据faiss可用性选择索引类型
            collaborative_index_type = 'hnsw' if self.vector_index.use_faiss else 'flat'
            print(f"Using index type: {collaborative_index_type} for collaborative recommendation index")
            
            try:
                print(f"Attempting to load collaborative index from {collaborative_index_path}...")
                self.collaborative_index.load_index(collaborative_index_path, game_embeddings)
                
                print(f"Successfully loaded collaborative index from {collaborative_index_path}, index_type: {self.collaborative_index.index_type}")
            except Exception as e:
                print(f"Failed to load collaborative index: {e}, building new index...")
                
                # 直接构建索引，不重新初始化collaborative_index
                self.collaborative_index.build_index(game_embeddings, index_type=collaborative_index_type)
                print(f"Built collaborative index, index_type: {self.collaborative_index.index_type}, faiss used: {self.collaborative_index.use_faiss}")
                
                print(f"Saving collaborative index to {collaborative_index_path}...")
                self.collaborative_index.save_index(collaborative_index_path)
                print(f"Successfully saved collaborative index to {collaborative_index_path}")
        
        print("Recommendation system initialized successfully")
    
    def get_content_recommendations(self, target_appid_or_list, k=10, weights=None):
        """基于游戏ID或游戏ID列表推荐相似游戏"""
        # 将单个游戏ID转换为列表
        game_appids = target_appid_or_list if isinstance(target_appid_or_list, list) else [target_appid_or_list]
        
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
        
        # 搜索相似向量，根据游戏数量调整返回结果数量
        search_k = k*2+len(game_appids)
        distances, indices = self.vector_index.search(avg_embedding, search_k)  # 多取一些结果用于后续筛选
        
        # 构建推荐结果
        recommendations = []
        max_recommendations = k*2
        for i in range(len(distances)):
            vector_idx = indices[i]
            rec_appid = self.vector_index_to_appid[vector_idx]
            
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
            
            # 确保只返回指定数量的推荐
            if len(recommendations) >= max_recommendations:
                break
        
        # 直接返回原始推荐结果，不在此重排，由调用方决定是否重排
        return recommendations
    

    
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
    

    
    def get_review_count_by_appid(self, appid):
        """根据appid获取游戏评论数量"""
        game = self.data_loader.get_game_by_appid(appid)
        if game is not None and 'recommendations_total' in game:
            return game['recommendations_total']
        return 0
    
    def calculate_weighted_score(self, similarity, review_count, user_score):
        """计算带评论权重、评分的最终评分"""
        # 基础分数：相似度
        base_score = similarity
        
        # 1. 处理评论数量权重
        if review_count is None or np.isnan(review_count):
            review_weight = 0.0
        else:
            # 获取最大评论数量，用于归一化
            max_review_count = self.data_loader.applications_df['recommendations_total'].max() if not self.data_loader.applications_df.empty else 1
            if max_review_count == 0:
                review_weight = 0.0
            else:
                # 使用对数函数平滑评论数量差异
                review_weight = np.log(review_count + 1) / np.log(max_review_count + 1)
        
        # 2. 处理用户评分权重
        if user_score is None or np.isnan(user_score):
            user_score_weight = 0.0
        else:
            # 用户评分已经是0-1范围
            user_score_weight = user_score
        
        # 计算最终评分：基础分数 * (1 + 评论权重 + 用户评分权重)
        # 调整权重系数，确保各因素贡献合理
        final_score = base_score * (1 + REVIEW_COUNT_WEIGHT * review_weight + USER_SCORE_WEIGHT * user_score_weight)
        
        return final_score
    
    def get_collaborative_recommendations(self, target_appid_or_list, k=10, weights=None):
        """基于评论嵌入向量的协同推荐，支持单个游戏ID或游戏ID列表"""
        # 将单个游戏ID转换为列表
        target_appids = target_appid_or_list if isinstance(target_appid_or_list, list) else [target_appid_or_list]
        
        if not target_appids:
            return "No game appids provided"
        
        # 确保权重数量与游戏数量匹配
        if weights is None:
            weights = [1.0] * len(target_appids)
        elif len(weights) != len(target_appids):
            return "Number of weights must match number of game appids"
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 计算所有游戏的平均评论向量
        total_review_vectors = []
        for i, target_appid in enumerate(target_appids):
            if target_appid in self.appid_to_review_vectors:
                game_review_vectors = self.appid_to_review_vectors[target_appid]
                # 计算当前游戏的平均评论向量
                avg_review_vector = np.mean(game_review_vectors, axis=0)
                # 使用权重加权
                total_review_vectors.append(avg_review_vector * weights[i])
        
        # 如果没有有效的评论向量，返回空列表
        if not total_review_vectors:
            return []
        
        # 计算最终的平均评论向量
        final_avg_review_vector = np.mean(total_review_vectors, axis=0)
        
        # 不传递排除ID，由最终排除逻辑处理
        return self._get_collaborative_recommendations_by_vector(final_avg_review_vector, None, k)
    
    def _get_collaborative_recommendations_by_vector(self, avg_review_vector, exclude_appid, k=10):
        """基于评论向量的协同推荐"""
        # 使用FAISS索引进行搜索
        distances, indices = self.collaborative_index.search(avg_review_vector, k+10)  # 多取一些结果，便于后续过滤
        
        # 构建推荐结果
        recommendations = []
        for i in range(len(distances)):
            vector_idx = indices[i]
            other_appid = self.vector_index_to_gameid[vector_idx]
            
            # 获取游戏信息
            game_info = self.data_loader.get_game_by_appid(other_appid)
            if game_info is not None:
                genres = self.data_loader.appid_to_genres.get(other_appid, [])
                
                recommendations.append({
                    'appid': other_appid,
                    'name': game_info['name'],
                    'similarity': distances[i],
                    'is_free': game_info['is_free'],
                    'metacritic_score': game_info['metacritic_score'],
                    'short_description': game_info['short_description'],
                    'header_image': game_info['header_image'],
                    'release_date': game_info['release_date'],
                    'genres': genres,
                    'user_score_label': '评价不足',
                    'user_score_percentage': 0.0
                })
            
            # 确保只返回k个推荐
            if len(recommendations) >= k:
                break
        
        return recommendations
    
    def get_hybrid_recommendations(self, target_appid_or_list, k=10, game_weight=0.7, review_weight=0.3, weights=None, use_content_based=True, use_collaborative=True):
        """混合推荐：结合游戏嵌入和评论嵌入，支持单个游戏ID或游戏ID列表"""
        # 检查并归一化权重
        total_weight = game_weight + review_weight
        if total_weight > 0:
            # 归一化权重，确保总和为1
            game_weight = game_weight / total_weight
            review_weight = review_weight / total_weight
        else:
            # 如果总权重为0，使用默认值
            game_weight = GAME_WEIGHT
            review_weight = REVIEW_WEIGHT
        
        # 确保权重在合理范围内
        game_weight = max(0.0, min(1.0, game_weight))
        review_weight = max(0.0, min(1.0, review_weight))
        # 创建推荐结果字典，便于融合
        rec_dict = {}
        
        # 获取基于游戏嵌入的推荐（如果启用）
        if use_content_based:
            # 基于单个游戏或多个游戏的内容推荐，统一调用get_content_recommendations函数
            content_recs = self.get_content_recommendations(target_appid_or_list, k=k*2, weights=weights)  # 获取更多结果用于融合
            
            # 检查是否返回了错误信息
            if isinstance(content_recs, str):
                return content_recs
            
            # 处理基于游戏嵌入的推荐
            for rec in content_recs:
                rec_appid = rec['appid']
                if rec_appid not in rec_dict:
                    rec_dict[rec_appid] = {
                        'appid': rec_appid,
                        'name': rec['name'],
                        'content_similarity': rec['similarity'],
                        'collaborative_similarity': 0.0,
                        'is_free': rec['is_free'],
                        'metacritic_score': rec['metacritic_score'],
                        'short_description': rec['short_description'],
                        'header_image': rec['header_image'],
                        'release_date': rec['release_date'],
                        'genres': rec['genres'],
                        'user_score_label': rec['user_score_label'],
                        'user_score_percentage': rec['user_score_percentage']
                    }
                else:
                    rec_dict[rec_appid]['content_similarity'] = rec['similarity']
        
        # 获取基于评论嵌入的推荐（如果启用）
        if use_collaborative:
            # 对于单个游戏或多个游戏，使用统一的协同推荐函数
            collaborative_recs = self.get_collaborative_recommendations(target_appid_or_list, k=k*2, weights=weights)  # 获取更多结果用于融合
            
            # 处理基于评论嵌入的推荐
            for rec in collaborative_recs:
                rec_appid = rec['appid']
                if rec_appid not in rec_dict:
                    rec_dict[rec_appid] = {
                        'appid': rec_appid,
                        'name': rec['name'],
                        'content_similarity': 0.0,
                        'collaborative_similarity': rec['similarity'],
                        'is_free': rec['is_free'],
                        'metacritic_score': rec['metacritic_score'],
                        'short_description': rec['short_description'],
                        'header_image': rec['header_image'],
                        'release_date': rec['release_date'],
                        'genres': rec['genres'],
                        'user_score_label': rec['user_score_label'],
                        'user_score_percentage': rec['user_score_percentage']
                    }
                else:
                    rec_dict[rec_appid]['collaborative_similarity'] = rec['similarity']
        
        # 如果两个推荐都禁用，返回错误信息
        if not use_content_based and not use_collaborative:
            return "At least one recommendation algorithm must be enabled"
        
        # 计算混合相似度
        for rec_appid in rec_dict:
            rec = rec_dict[rec_appid]
            # 根据启用的算法调整权重
            if use_content_based and use_collaborative:
                # 两者都启用，使用加权平均
                rec['similarity'] = (game_weight * rec['content_similarity']) + (review_weight * rec['collaborative_similarity'])
            elif use_content_based:
                # 仅内容推荐
                rec['similarity'] = rec['content_similarity']
            else:
                # 仅协同推荐
                rec['similarity'] = rec['collaborative_similarity']
        
        # 转换为列表并按相似度排序
        hybrid_recs = list(rec_dict.values())
        hybrid_recs.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 排除所有目标游戏
        target_appids = target_appid_or_list if isinstance(target_appid_or_list, list) else [target_appid_or_list]
        hybrid_recs = [rec for rec in hybrid_recs if rec['appid'] not in target_appids]
        
        # 取前k个结果
        hybrid_recs = hybrid_recs[:k]
        
        # 重排序推荐结果
        hybrid_recs = self.reorder_recommendations(hybrid_recs)
        
        return hybrid_recs
    
    def reorder_recommendations(self, recommendations):
        """根据评论数量、用户评分重排推荐结果"""
        # 为每个推荐结果计算带评论权重和评分的最终评分
        for rec in recommendations:
            # 获取评论数量
            review_count = self.get_review_count_by_appid(rec['appid'])
            rec['review_count'] = review_count
            
            # 获取用户评分
            user_score = rec['user_score_percentage']
            
            # 计算最终加权分数
            rec['weighted_score'] = self.calculate_weighted_score(
                rec['similarity'],
                review_count,
                user_score
            )
        
        # 根据最终评分降序排序
        recommendations.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        return recommendations
