import pandas as pd
import numpy as np
import os

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
    
    def load_applications_data(self):
        """加载游戏基本数据"""
        apps_path = os.path.join(self.data_dir, 'steam_dataset_2025_csv_package_v1', 'steam_dataset_2025_csv', 'applications.csv')
        self.applications_df = pd.read_csv(apps_path)
        print(f"Loaded {len(self.applications_df)} applications from CSV")
        return self.applications_df
    
    def load_embeddings(self):
        """加载嵌入向量数据"""
        embeddings_dir = os.path.join(self.data_dir, 'steam_dataset_2025_embeddings_package_v1', 'steam_dataset_2025_embeddings')
        
        # 加载应用嵌入向量
        app_embeddings_path = os.path.join(embeddings_dir, 'applications_embeddings.npy')
        # 直接从二进制文件读取，使用float32类型
        app_embeddings_raw = np.fromfile(app_embeddings_path, dtype=np.float32)
        # 重塑为 (num_vectors, 1024) 形状
        num_vectors = len(app_embeddings_raw) // 1024
        self.app_embeddings = app_embeddings_raw.reshape(num_vectors, 1024)
        print(f"Loaded application embeddings with shape: {self.app_embeddings.shape}")
        
        # 加载评论嵌入向量
        review_embeddings_path = os.path.join(embeddings_dir, 'reviews_embeddings.npy')
        # 直接从二进制文件读取，使用float32类型
        review_embeddings_raw = np.fromfile(review_embeddings_path, dtype=np.float32)
        # 重塑为 (num_vectors, 1024) 形状
        num_review_vectors = len(review_embeddings_raw) // 1024
        self.review_embeddings = review_embeddings_raw.reshape(num_review_vectors, 1024)
        print(f"Loaded review embeddings with shape: {self.review_embeddings.shape}")
        
        return self.app_embeddings, self.review_embeddings
    
    def load_embedding_maps(self):
        """加载嵌入向量映射关系"""
        embeddings_dir = os.path.join(self.data_dir, 'steam_dataset_2025_embeddings_package_v1', 'steam_dataset_2025_embeddings')
        
        # 加载应用嵌入映射
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
        
        # 加载评论嵌入映射
        review_map_path = os.path.join(embeddings_dir, 'reviews_embedding_map.csv')
        review_map_df = pd.read_csv(review_map_path)
        self.review_id_to_index = dict(zip(review_map_df['recommendationid'], review_map_df['vector_index']))
        
        print(f"Loaded {len(self.review_id_to_index)} review embedding mappings from CSV")
        
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
        return self.appid_to_genres
    
    def initialize(self):
        """初始化所有数据"""
        print("Initializing DataLoader in memory mode...")
        
        self.load_applications_data()
        self.load_embeddings()
        self.load_embedding_maps()
        self.load_genres_data()
        # 初始化appid_to_user_score为空字典，实际评分将通过API获取
        self.appid_to_user_score = {}
        print("DataLoader initialization completed")
