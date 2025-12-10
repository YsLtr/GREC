import pandas as pd
import numpy as np
import os
import pickle

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.applications_df = None
        self.app_embeddings = None
        self.review_embeddings = None
        self.app_id_to_embedding = None
        self.app_id_to_index = None
        self.index_to_app_id = None
        self.appid_to_genres = None
        self.genres_df = None
        self.application_genres_df = None
        self.appid_to_user_score = None
        # 评论相关映射
        self.recommendationid_to_appid = None  # 评论ID到appid的映射
        self.appid_to_review_indices = None    # appid到评论向量索引列表的映射
        self.appid_to_review_vectors = None    # appid到评论嵌入向量列表的映射
        # 用户相关映射
        self.steamid_to_recommendationids = None  # 用户ID到评论ID列表的映射
        self.steamid_to_review_vectors = None     # 用户ID到评论嵌入向量列表的映射
        self.steamid_to_appids = None             # 用户ID到游戏ID列表的映射
    
    def load_applications_data(self):
        """加载游戏基本数据"""
        apps_path = os.path.join(self.data_dir, 'steam_dataset_2025_csv_package_v1', 'steam_dataset_2025_csv', 'applications.csv')
        self.applications_df = pd.read_csv(apps_path)
        print(f"Loaded {len(self.applications_df)} applications from CSV")
        return self.applications_df
    
    def load_embeddings(self, load_app_embeddings=True, load_review_embeddings=True):
        """加载嵌入向量数据，支持选择性加载"""
        embeddings_dir = os.path.join(self.data_dir, 'steam_dataset_2025_embeddings_package_v1', 'steam_dataset_2025_embeddings')
        expected_dim = 1024
        
        # 加载应用嵌入向量
        if load_app_embeddings:
            app_embeddings_path = os.path.join(embeddings_dir, 'applications_embeddings.npy')
            # 直接从二进制文件读取，使用float32类型
            app_embeddings_raw = np.fromfile(app_embeddings_path, dtype=np.float32)
            
            # 验证数据完整性
            if len(app_embeddings_raw) % expected_dim != 0:
                raise ValueError(f"Application embeddings data is not aligned with expected dimension {expected_dim}")
            
            # 重塑为 (num_vectors, 1024) 形状
            num_vectors = len(app_embeddings_raw) // expected_dim
            self.app_embeddings = app_embeddings_raw.reshape(num_vectors, expected_dim)
            
            # 验证嵌入向量格式
            self._validate_embeddings(self.app_embeddings, "Application")
            
            # 检查并确保应用嵌入向量是L2归一化的
            app_norms = np.linalg.norm(self.app_embeddings, axis=1)
            app_is_normalized = np.allclose(app_norms, 1.0, atol=0.01)
            if not app_is_normalized:
                print(f"Application embeddings not normalized, normalizing now...")
                self.app_embeddings = self.app_embeddings / app_norms[:, np.newaxis]
            print(f"Loaded application embeddings with shape: {self.app_embeddings.shape}, normalized: {app_is_normalized}")
        
        # 加载评论嵌入向量
        if load_review_embeddings:
            review_embeddings_path = os.path.join(embeddings_dir, 'reviews_embeddings.npy')
            # 直接从二进制文件读取，使用float32类型
            review_embeddings_raw = np.fromfile(review_embeddings_path, dtype=np.float32)
            
            # 验证数据完整性
            if len(review_embeddings_raw) % expected_dim != 0:
                raise ValueError(f"Review embeddings data is not aligned with expected dimension {expected_dim}")
            
            # 重塑为 (num_vectors, 1024) 形状
            num_review_vectors = len(review_embeddings_raw) // expected_dim
            self.review_embeddings = review_embeddings_raw.reshape(num_review_vectors, expected_dim)
            
            # 验证嵌入向量格式
            self._validate_embeddings(self.review_embeddings, "Review")
            
            # 检查并确保评论嵌入向量是L2归一化的
            review_norms = np.linalg.norm(self.review_embeddings, axis=1)
            review_is_normalized = np.allclose(review_norms, 1.0, atol=0.01)
            if not review_is_normalized:
                print(f"Review embeddings not normalized, normalizing now...")
                self.review_embeddings = self.review_embeddings / review_norms[:, np.newaxis]
            print(f"Loaded review embeddings with shape: {self.review_embeddings.shape}, normalized: {review_is_normalized}")
        
        return self.app_embeddings, self.review_embeddings
    
    def _validate_embeddings(self, embeddings, name=""):
        """验证嵌入向量的格式和维度"""
        expected_dim = 1024
        
        # 验证维度
        if embeddings.shape[1] != expected_dim:
            raise ValueError(f"{name} embeddings have wrong dimension: {embeddings.shape[1]}, expected: {expected_dim}")
        
        # 检查是否有NaN值
        if np.any(np.isnan(embeddings)):
            raise ValueError(f"{name} embeddings contain NaN values")
        
        # 检查是否有Inf值
        if np.any(np.isinf(embeddings)):
            raise ValueError(f"{name} embeddings contain Inf values")
        
        # 检查是否有空向量
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms == 0):
            raise ValueError(f"{name} embeddings contain zero vectors")
        
        print(f"{name} embeddings validation passed")
    
    def save_embedding_mappings(self, save_content_mappings=True, save_collaborative_mappings=True):
        """保存嵌入映射到文件，支持选择性保存"""
        mappings_dir = os.path.join(self.data_dir, 'mappings')
        os.makedirs(mappings_dir, exist_ok=True)
        
        # 创建嵌入映射字典
        embedding_mappings = {}
        
        # 保存内容推荐所需的映射
        if save_content_mappings:
            embedding_mappings['app_id_to_index'] = self.app_id_to_index
            embedding_mappings['index_to_app_id'] = self.index_to_app_id
            embedding_mappings['app_id_to_embedding'] = self.app_id_to_embedding
        
        # 保存协同推荐所需的映射
        if save_collaborative_mappings:
            # 只有在需要保存协同推荐映射时才包含这些字段
            embedding_mappings['review_id_to_index'] = self.review_id_to_index if hasattr(self, 'review_id_to_index') else {}
            embedding_mappings['recommendationid_to_appid'] = self.recommendationid_to_appid if hasattr(self, 'recommendationid_to_appid') else {}
            embedding_mappings['appid_to_review_indices'] = self.appid_to_review_indices if hasattr(self, 'appid_to_review_indices') else {}
            embedding_mappings['appid_to_review_vectors'] = self.appid_to_review_vectors if hasattr(self, 'appid_to_review_vectors') else {}
            embedding_mappings['steamid_to_recommendationids'] = self.steamid_to_recommendationids if hasattr(self, 'steamid_to_recommendationids') else {}
            embedding_mappings['steamid_to_appids'] = self.steamid_to_appids if hasattr(self, 'steamid_to_appids') else {}
            embedding_mappings['steamid_to_review_vectors'] = self.steamid_to_review_vectors if hasattr(self, 'steamid_to_review_vectors') else {}
        
        embedding_mappings_path = os.path.join(mappings_dir, 'embedding_mappings.pkl')
        with open(embedding_mappings_path, 'wb') as f:
            pickle.dump(embedding_mappings, f)
        print(f"Saved embedding mappings to {embedding_mappings_path}")
    
    def load_mappings(self, load_content_mappings=True, load_collaborative_mappings=True):
        """从文件加载映射，支持选择性加载"""
        mappings_dir = os.path.join(self.data_dir, 'mappings')
        
        # 加载嵌入映射
        embedding_mappings_path = os.path.join(mappings_dir, 'embedding_mappings.pkl')
        if os.path.exists(embedding_mappings_path):
            with open(embedding_mappings_path, 'rb') as f:
                embedding_mappings = pickle.load(f)
            
            # 加载内容推荐所需的映射
            if load_content_mappings:
                self.app_id_to_index = embedding_mappings['app_id_to_index']
                self.index_to_app_id = embedding_mappings['index_to_app_id']
                self.app_id_to_embedding = embedding_mappings['app_id_to_embedding']
                print(f"Loaded {len(self.app_id_to_embedding)} application embedding mappings from file")
            
            # 加载协同推荐所需的映射
            if load_collaborative_mappings:
                self.review_id_to_index = embedding_mappings['review_id_to_index']
                self.recommendationid_to_appid = embedding_mappings['recommendationid_to_appid']
                self.appid_to_review_indices = embedding_mappings['appid_to_review_indices']
                self.appid_to_review_vectors = embedding_mappings['appid_to_review_vectors']
                self.steamid_to_recommendationids = embedding_mappings['steamid_to_recommendationids']
                self.steamid_to_appids = embedding_mappings['steamid_to_appids']
                self.steamid_to_review_vectors = embedding_mappings['steamid_to_review_vectors']
                
                print(f"Loaded {len(self.review_id_to_index)} review embedding mappings from file")
                print(f"Loaded {len(self.recommendationid_to_appid)} recommendationid to appid mappings from file")
                print(f"Loaded appid to review indices mappings for {len(self.appid_to_review_indices)} apps from file")
                print(f"Loaded appid to review vectors mappings for {len(self.appid_to_review_vectors)} apps from file")
                print(f"Loaded steamid to recommendationids mapping for {len(self.steamid_to_recommendationids)} users from file")
                print(f"Loaded steamid to appids mapping for {len(self.steamid_to_appids)} users from file")
                print(f"Loaded steamid to review vectors mapping for {len(self.steamid_to_review_vectors)} users from file")
            
            return True
        
        return False
    
    def load_genres_mappings(self):
        """从文件加载分类映射"""
        mappings_dir = os.path.join(self.data_dir, 'mappings')
        genres_mappings_path = os.path.join(mappings_dir, 'genres_mappings.pkl')
        
        if os.path.exists(genres_mappings_path):
            with open(genres_mappings_path, 'rb') as f:
                genres_mappings = pickle.load(f)
            
            self.appid_to_genres = genres_mappings['appid_to_genres']
            if self.appid_to_genres is not None:
                print(f"Loaded genres mapping for {len(self.appid_to_genres)} applications from file")
                return True
        
        return False
    
    def load_embedding_maps(self, load_content_mappings=True, load_collaborative_mappings=True):
        """加载嵌入向量映射关系，支持选择性加载"""
        # 尝试从文件加载映射
        if self.load_mappings(load_content_mappings, load_collaborative_mappings):
            return None, None
        
        # 如果文件不存在，构建映射
        embeddings_dir = os.path.join(self.data_dir, 'steam_dataset_2025_embeddings_package_v1', 'steam_dataset_2025_embeddings')
        
        # 加载应用嵌入映射（内容推荐所需）
        review_map_df = None
        if load_content_mappings:
            app_map_path = os.path.join(embeddings_dir, 'applications_embedding_map.csv')
            app_map_df = pd.read_csv(app_map_path)
            
            # 建立appid到向量索引的映射
            self.app_id_to_index = dict(zip(app_map_df['appid'], app_map_df['vector_index']))
            self.index_to_app_id = dict(zip(app_map_df['vector_index'], app_map_df['appid']))
            
            # 建立appid到嵌入向量的直接映射
            self.app_id_to_embedding = {}
            for _, row in app_map_df.iterrows():
                appid = row['appid']
                vector_index = row['vector_index']
                self.app_id_to_embedding[appid] = self.app_embeddings[vector_index]
            
            print(f"Loaded {len(self.app_id_to_embedding)} application embedding mappings from CSV")
        
        # 加载评论嵌入映射（协同推荐所需）
        if load_collaborative_mappings:
            # 加载评论嵌入映射
            review_map_path = os.path.join(embeddings_dir, 'reviews_embedding_map.csv')
            review_map_df = pd.read_csv(review_map_path)
            self.review_id_to_index = dict(zip(review_map_df['recommendationid'], review_map_df['vector_index']))
            
            print(f"Loaded {len(self.review_id_to_index)} review embedding mappings from CSV")
            
            # 1. 加载reviews.csv，构建recommendationid到appid的映射
            reviews_path = os.path.join(self.data_dir, 'steam_dataset_2025_csv_package_v1', 'steam_dataset_2025_csv', 'reviews.csv')
            reviews_df = pd.read_csv(reviews_path, usecols=['recommendationid', 'appid'])
            self.recommendationid_to_appid = dict(zip(reviews_df['recommendationid'], reviews_df['appid']))
            print(f"Loaded {len(self.recommendationid_to_appid)} recommendationid to appid mappings")
            
            # 2. 构建appid到评论向量索引的映射
            self.appid_to_review_indices = {}
            for rec_id, vector_index in self.review_id_to_index.items():
                if rec_id in self.recommendationid_to_appid:
                    appid = self.recommendationid_to_appid[rec_id]
                    if appid not in self.appid_to_review_indices:
                        self.appid_to_review_indices[appid] = []
                    self.appid_to_review_indices[appid].append(vector_index)
            print(f"Built appid to review indices mappings for {len(self.appid_to_review_indices)} apps")
            
            # 3. 构建appid到评论嵌入向量的映射
            self.appid_to_review_vectors = {}
            for appid, indices in self.appid_to_review_indices.items():
                # 获取该appid对应的所有评论向量
                review_vectors = [self.review_embeddings[idx] for idx in indices]
                self.appid_to_review_vectors[appid] = np.array(review_vectors)
            print(f"Built appid to review vectors mappings for {len(self.appid_to_review_vectors)} apps")
            
            # 加载完整的reviews.csv数据，用于构建用户映射
            reviews_path = os.path.join(self.data_dir, 'steam_dataset_2025_csv_package_v1', 'steam_dataset_2025_csv', 'reviews.csv')
            reviews_df = pd.read_csv(reviews_path, usecols=['recommendationid', 'appid', 'author_steamid', 'voted_up'])
            
            # 4. 构建steamid到recommendationids的映射
            self.steamid_to_recommendationids = {}
            for _, row in reviews_df.iterrows():
                steamid = row['author_steamid']
                recommendationid = row['recommendationid']
                if steamid not in self.steamid_to_recommendationids:
                    self.steamid_to_recommendationids[steamid] = []
                self.steamid_to_recommendationids[steamid].append(recommendationid)
            
            print(f"Built steamid to recommendationids mapping for {len(self.steamid_to_recommendationids)} users")
            
            # 5. 构建steamid到appids的映射（仅包含正向推荐）
            self.steamid_to_appids = {}
            for _, row in reviews_df[reviews_df['voted_up']].iterrows():
                steamid = row['author_steamid']
                appid = row['appid']
                if steamid not in self.steamid_to_appids:
                    self.steamid_to_appids[steamid] = []
                if appid not in self.steamid_to_appids[steamid]:
                    self.steamid_to_appids[steamid].append(appid)
            
            print(f"Built steamid to appids mapping for {len(self.steamid_to_appids)} users")
            
            # 6. 构建steamid到review_vectors的映射（用户平均嵌入）
            self.steamid_to_review_vectors = {}
            for steamid, recommendationids in self.steamid_to_recommendationids.items():
                # 获取该用户所有评论的向量索引
                vector_indices = [self.review_id_to_index.get(rec_id) for rec_id in recommendationids if rec_id in self.review_id_to_index]
                if vector_indices:
                    # 获取对应的评论向量
                    review_vectors = [self.review_embeddings[idx] for idx in vector_indices if idx < len(self.review_embeddings)]
                    if review_vectors:
                        # 计算用户平均嵌入向量
                        avg_review_vector = np.mean(review_vectors, axis=0)
                        self.steamid_to_review_vectors[steamid] = avg_review_vector
            
            print(f"Built steamid to review vectors mapping for {len(self.steamid_to_review_vectors)} users")
        
        # 保存嵌入映射到文件
        self.save_embedding_mappings(load_content_mappings, load_collaborative_mappings)
        
        return app_map_df, review_map_df
    
    def get_game_by_appid(self, appid):
        """根据appid获取游戏信息"""
        if self.applications_df is None:
            self.load_applications_data()
        
        game = self.applications_df[self.applications_df['appid'] == appid]
        if len(game) > 0:
            return game.iloc[0]
        else:
            return None
    
    def get_game_by_name(self, name):
        """根据名称获取游戏信息"""
        if self.applications_df is None:
            self.load_applications_data()
        
        game = self.applications_df[self.applications_df['name'].str.contains(name, case=False, na=False)]
        if len(game) > 0:
            return game.iloc[0]
        else:
            return None
    
    def get_embedding_by_appid(self, appid):
        """根据appid获取嵌入向量"""
        if self.app_id_to_embedding is None:
            self.load_embeddings()
            self.load_embedding_maps()
        
        return self.app_id_to_embedding.get(appid)
    
    def load_genres_data(self):
        """加载游戏分类数据"""
        # 尝试从文件加载分类映射
        if self.load_genres_mappings():
            return self.appid_to_genres
        
        # 如果文件不存在，构建分类映射
        # 加载genres.csv
        genres_path = os.path.join(self.data_dir, 'steam_dataset_2025_csv_package_v1', 'steam_dataset_2025_csv', 'genres.csv')
        self.genres_df = pd.read_csv(genres_path)
        print(f"Loaded {len(self.genres_df)} genres from CSV")
        
        # 加载application_genres.csv
        app_genres_path = os.path.join(self.data_dir, 'steam_dataset_2025_csv_package_v1', 'steam_dataset_2025_csv', 'application_genres.csv')
        self.application_genres_df = pd.read_csv(app_genres_path)
        print(f"Loaded {len(self.application_genres_df)} application-genre mappings from CSV")
        
        # 创建appid到genres的映射
        self.appid_to_genres = {}
        
        # 合并application_genres和genres，获取genre名称
        merged_genres = pd.merge(self.application_genres_df, self.genres_df, left_on='genre_id', right_on='id')
        
        # 按appid分组，收集所有genre名称
        for appid, group in merged_genres.groupby('appid'):
            self.appid_to_genres[appid] = group['name'].tolist()
        
        print(f"Created genres mapping for {len(self.appid_to_genres)} applications from CSV")
        
        # 保存分类映射到文件
        mappings_dir = os.path.join(self.data_dir, 'mappings')
        os.makedirs(mappings_dir, exist_ok=True)
        genres_mappings_path = os.path.join(mappings_dir, 'genres_mappings.pkl')
        genres_mappings = {
            'appid_to_genres': self.appid_to_genres
        }
        with open(genres_mappings_path, 'wb') as f:
            pickle.dump(genres_mappings, f)
        print(f"Saved genres mappings to {genres_mappings_path}")
        
        return self.appid_to_genres
    

    
    def __init__(self, data_dir):
        """初始化DataLoader，设置数据目录"""
        self.data_dir = data_dir
        self.applications_df = None
        self.app_embeddings = None
        self.review_embeddings = None
        self.app_id_to_embedding = None
        self.app_id_to_index = None
        self.index_to_app_id = None
        self.appid_to_genres = None
        self.genres_df = None
        self.application_genres_df = None
        # 初始化appid_to_user_score为空字典，实际评分将通过API获取
        self.appid_to_user_score = {}
        # 评论相关映射
        self.recommendationid_to_appid = None  # 评论ID到appid的映射
        self.appid_to_review_indices = None    # appid到评论向量索引列表的映射
        self.appid_to_review_vectors = None    # appid到评论嵌入向量列表的映射
        # 用户相关映射
        self.review_id_to_index = None
        self.steamid_to_recommendationids = None  # 用户ID到评论ID列表的映射
        self.steamid_to_review_vectors = None     # 用户ID到评论嵌入向量列表的映射
        self.steamid_to_appids = None             # 用户ID到游戏ID列表的映射
        
    def initialize(self):
        """初始化所有数据"""
        print("Initializing DataLoader in memory mode...")
        
        self.load_applications_data()
        self.load_embeddings()
        self.load_embedding_maps()
        self.load_genres_data()
        print("DataLoader initialization completed")
