import pandas as pd
import numpy as np
import os
import psycopg2
from psycopg2 import sql

class DataLoader:
    def __init__(self, data_dir, load_mode='memory', db_config=None):
        self.data_dir = data_dir
        # 加载模式：'memory' 或 'database'
        self.load_mode = load_mode
        # 数据库配置
        self.db_config = db_config or {
            'host': 'localhost',
            'port': 5432,
            'database': 'steamfull',
            'user': 'postgres',
            'password': '123456'
        }
        # 数据库连接
        self.db_conn = None
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
    
    def connect_to_db(self):
        """连接到PostgreSQL数据库"""
        if self.db_conn is None or self.db_conn.closed:
            try:
                self.db_conn = psycopg2.connect(**self.db_config)
                print("Connected to PostgreSQL database successfully")
            except psycopg2.Error as e:
                print(f"Error connecting to PostgreSQL: {e}")
                raise
        return self.db_conn
    
    def load_applications_data(self):
        """加载游戏基本数据"""
        if self.load_mode == 'memory':
            apps_path = os.path.join(self.data_dir, 'steam_dataset_2025_csv_package_v1', 'steam_dataset_2025_csv', 'applications.csv')
            self.applications_df = pd.read_csv(apps_path)
            print(f"Loaded {len(self.applications_df)} applications from CSV")
        else:
            # 从数据库加载
            conn = self.connect_to_db()
            query = "SELECT * FROM applications"
            self.applications_df = pd.read_sql(query, conn)
            print(f"Loaded {len(self.applications_df)} applications from database")
        return self.applications_df
    
    def load_embeddings(self):
        """加载嵌入向量数据"""
        if self.load_mode == 'memory':
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
        else:
            # 从数据库加载嵌入向量
            conn = self.connect_to_db()
            
            # 加载应用嵌入向量
            query = "SELECT embedding FROM application_embeddings ORDER BY vector_index"
            app_embeddings_df = pd.read_sql(query, conn)
            # 将embedding列转换为numpy数组
            self.app_embeddings = np.vstack(app_embeddings_df['embedding'].apply(np.array))
            print(f"Loaded application embeddings with shape: {self.app_embeddings.shape} from database")
            
            # 加载评论嵌入向量
            query = "SELECT embedding FROM review_embeddings ORDER BY vector_index"
            review_embeddings_df = pd.read_sql(query, conn)
            # 将embedding列转换为numpy数组
            self.review_embeddings = np.vstack(review_embeddings_df['embedding'].apply(np.array))
            print(f"Loaded review embeddings with shape: {self.review_embeddings.shape} from database")
        
        return self.app_embeddings, self.review_embeddings
    
    def load_embedding_maps(self):
        """加载嵌入向量映射关系"""
        if self.load_mode == 'memory':
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
        else:
            # 从数据库加载嵌入映射
            conn = self.connect_to_db()
            
            # 加载应用嵌入映射
            app_query = "SELECT appid, vector_index FROM application_embeddings"
            app_map_df = pd.read_sql(app_query, conn)
            
            # 建立appid到向量索引的映射
            self.app_id_to_index = dict(zip(app_map_df['appid'], app_map_df['vector_index']))
            self.index_to_app_id = dict(zip(app_map_df['vector_index'], app_map_df['appid']))
            
            # 建立appid到嵌入向量的直接映射
            self.app_id_to_embedding = {}
            for _, row in app_map_df.iterrows():
                appid = row['appid']
                vector_index = row['vector_index']
                self.app_id_to_embedding[appid] = self.app_embeddings[vector_index]
            
            print(f"Loaded {len(self.app_id_to_embedding)} application embedding mappings from database")
            
            # 加载评论嵌入映射
            review_query = "SELECT recommendationid, vector_index FROM review_embeddings"
            review_map_df = pd.read_sql(review_query, conn)
            self.review_id_to_index = dict(zip(review_map_df['recommendationid'], review_map_df['vector_index']))
            
            print(f"Loaded {len(self.review_id_to_index)} review embedding mappings from database")
            
            return app_map_df, review_map_df
    
    def get_game_by_appid(self, appid):
        """根据appid获取游戏信息"""
        if self.load_mode == 'memory' or self.applications_df is not None:
            if self.applications_df is None:
                self.load_applications_data()
            
            game = self.applications_df[self.applications_df['appid'] == appid]
            if len(game) > 0:
                return game.iloc[0]
            else:
                return None
        else:
            # 直接从数据库获取
            conn = self.connect_to_db()
            query = sql.SQL("SELECT * FROM applications WHERE appid = %s")
            with conn.cursor() as cur:
                cur.execute(query, (appid,))
                row = cur.fetchone()
                if row:
                    # 获取列名
                    columns = [desc[0] for desc in cur.description]
                    # 转换为字典
                    game_dict = dict(zip(columns, row))
                    return game_dict
                else:
                    return None
    
    def get_game_by_name(self, name):
        """根据名称获取游戏信息"""
        if self.load_mode == 'memory' or self.applications_df is not None:
            if self.applications_df is None:
                self.load_applications_data()
            
            game = self.applications_df[self.applications_df['name'].str.contains(name, case=False, na=False)]
            if len(game) > 0:
                return game.iloc[0]
            else:
                return None
        else:
            # 直接从数据库获取
            conn = self.connect_to_db()
            query = sql.SQL("SELECT * FROM applications WHERE name ILIKE %s LIMIT 1")
            with conn.cursor() as cur:
                cur.execute(query, (f'%{name}%',))
                row = cur.fetchone()
                if row:
                    # 获取列名
                    columns = [desc[0] for desc in cur.description]
                    # 转换为字典
                    game_dict = dict(zip(columns, row))
                    return game_dict
                else:
                    return None
    
    def get_embedding_by_appid(self, appid):
        """根据appid获取嵌入向量"""
        if self.load_mode == 'memory' or self.app_id_to_embedding is not None:
            if self.app_id_to_embedding is None:
                self.load_embeddings()
                self.load_embedding_maps()
            
            return self.app_id_to_embedding.get(appid)
        else:
            # 直接从数据库获取嵌入向量
            conn = self.connect_to_db()
            query = sql.SQL("SELECT embedding FROM application_embeddings WHERE appid = %s")
            with conn.cursor() as cur:
                cur.execute(query, (appid,))
                row = cur.fetchone()
                if row and row[0] is not None:
                    # 将PostgreSQL数组转换为numpy数组
                    embedding = np.array(row[0], dtype=np.float32)
                    return embedding
                else:
                    return None
    
    def load_genres_data(self):
        """加载游戏分类数据"""
        if self.load_mode == 'memory':
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
        else:
            # 从数据库加载分类数据
            conn = self.connect_to_db()
            
            # 加载genres
            query = "SELECT * FROM genres"
            self.genres_df = pd.read_sql(query, conn)
            print(f"Loaded {len(self.genres_df)} genres from database")
            
            # 加载application_genres
            query = "SELECT * FROM application_genres"
            self.application_genres_df = pd.read_sql(query, conn)
            print(f"Loaded {len(self.application_genres_df)} application-genre mappings from database")
            
            # 创建appid到genres的映射
            self.appid_to_genres = {}
            
            # 合并application_genres和genres，获取genre名称
            merged_genres = pd.merge(self.application_genres_df, self.genres_df, left_on='genre_id', right_on='id')
            
            # 按appid分组，收集所有genre名称
            for appid, group in merged_genres.groupby('appid'):
                self.appid_to_genres[appid] = group['name'].tolist()
            
            print(f"Created genres mapping for {len(self.appid_to_genres)} applications from database")
        return self.appid_to_genres
    
    def initialize(self):
        """初始化所有数据"""
        print(f"Initializing DataLoader in {self.load_mode} mode...")
        
        # 如果是数据库模式，先连接数据库
        if self.load_mode == 'database':
            self.connect_to_db()
        
        self.load_applications_data()
        self.load_embeddings()
        self.load_embedding_maps()
        self.load_genres_data()
        # 初始化appid_to_user_score为空字典，实际评分将通过API获取
        self.appid_to_user_score = {}
        print("DataLoader initialization completed")
