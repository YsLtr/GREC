# GREC - Game Recommendation Engine

## 项目介绍

GREC (Game Recommendation Engine based on embeddings and Collaborative filtering) 是一个基于向量嵌入和协同过滤的游戏推荐系统，能够根据用户提供的游戏列表推荐相似游戏。

### 核心功能

- **智能推荐**：基于游戏嵌入向量和协同过滤算法，提供高精度的游戏推荐
- **高效内存加载**：优化的内存加载机制，快速启动和响应
- **异步数据更新**：先返回基本推荐结果，再异步加载Steam API数据
- **瀑布流布局**：前端采用响应式瀑布流布局，支持无限滚动
- **多游戏权重推荐**：支持为多个游戏设置不同权重进行混合推荐

## 项目结构

```
GREC/
├── front/                  # 前端文件
│   ├── favicon.ico         # 网站图标
│   └── game_recommender.html  # 主页面
├── meta_data/              # 元数据和文档
│   ├── Steam游戏数据集分析报告.md
│   ├── steam-dataset-2025-multi-modal-gaming-analytics-metadata.json
│   ├── steam-dataset-2025-schema-analysis.md
│   └── 嵌入向量使用概要.md
├── steam_dataset_2025/     # Steam数据集（外部数据目录）
│   ├── steam-dataset-2025-v1/
│   ├── steam_dataset_2025_csv_package_v1/
│   ├── steam_dataset_2025_embeddings_package_v1/
│   └── steam_dataset_2025_power_users_dump_v1/
├── test_use/               # 测试和工具脚本
├── app.py                  # Flask应用入口
├── data_loader.py          # 数据加载器
├── recommendation_system.py  # 推荐系统核心
├── vector_index.py         # 向量索引实现
├── requirements.txt        # 依赖列表
└── test_recommendations.py # 推荐系统测试
```

## 数据集
[Steam Dataset 2025: Multi-Modal Gaming Analytics](https://www.kaggle.com/datasets/crainbramp/steam-dataset-2025-multi-modal-gaming-analytics)

### 数据集结构

```
steam_dataset_2025/
├── steam-dataset-2025-v1/
│   ├── notebook-data/
│   │   ├── 01-platform-evolution/
│   │   │   ├── 01_temporal_growth.csv
│   │   │   ├── 02_genre_evolution.csv
│   │   │   ├── 03_platform_support.csv
│   │   │   ├── 04_pricing_strategy.csv
│   │   │   ├── 05_publisher_portfolios.csv
│   │   │   ├── 06_achievement_evolution.csv
│   │   │   └── README.md
│   │   ├── 02-semantic-game-discovery/
│   │   │   ├── 01_game_embeddings_sample.csv
│   │   │   ├── 02_embeddings_appids.csv
│   │   │   ├── 02_embeddings_vectors.npy
│   │   │   ├── 02_genre_representatives.csv
│   │   │   └── 02_semantic_search_examples.json
│   │   └── 03-the-semantic-fingerprint/
│   │       ├── 03-the-semantic-fingerprint-preview.csv
│   │       └── 03-the-semantic-fingerprint.parquet
│   ├── notebooks/
│   │   ├── 01-steam-platform-evolution-and-marketplace/
│   │   │   ├── notebook-01-steam-platform-evolution-and-market-landscape.ipynb
│   │   │   └── notebook-01-steam-platform-evolution.pdf
│   │   ├── 02-semantic-game-discovery/
│   │   │   ├── 02-semantic-game-discovery.ipynb
│   │   │   └── notebook-02-semantic-game-discovery.pdf
│   │   └── 03-the-semantic-fingerprint/
│   │       ├── 03-the-semantic-fingerprint.ipynb
│   │       └── notebook-03-the-semantic-fingerprint.pdf
│   ├── DATASET_CARD.md
│   ├── DATA_DICTIONARY.md
│   ├── README.md
│   └── steam-dataset-2025-full-schema.sql
├── steam_dataset_2025_csv_package_v1/
│   └── steam_dataset_2025_csv/
│       ├── MANIFEST.json
│       ├── applications.csv              # 应用基本信息
│       ├── application_categories.csv    # 应用-分类关联
│       ├── application_developers.csv    # 应用-开发者关联
│       ├── application_genres.csv        # 应用-类型关联
│       ├── application_platforms.csv   # 应用-平台关联
│       ├── application_publishers.csv  # 应用-发行商关联
│       ├── categories.csv                # 分类信息
│       ├── developers.csv                # 开发者信息
│       ├── genres.csv                    # 游戏类型信息
│       ├── platforms.csv                 # 平台信息
│       ├── publishers.csv                # 发行商信息
│       └── reviews.csv                   # 用户评价数据
├── steam_dataset_2025_embeddings_package_v1/
│   └── steam_dataset_2025_embeddings/
│       ├── MANIFEST.json
│       ├── applications_embedding_map.csv    # 应用嵌入向量映射
│       ├── applications_embeddings.npy       # 应用嵌入向量（主要使用）
│       ├── reviews_embedding_map.csv       # 评价嵌入向量映射
│       └── reviews_embeddings.npy          # 评价嵌入向量
└── steam_dataset_2025_power_users_dump_v1/
    └── steam_dataset_2025_power_users/
        ├── dump.toc
        ├── pg_dump.log
        └── steam_dataset_20250929.dump        # PostgreSQL数据库转储文件
```

### 核心数据文件说明

- **`applications_embeddings.npy`**：游戏应用的嵌入向量文件，用于计算游戏相似度
- **`applications_embedding_map.csv`**：嵌入向量与应用ID的映射关系
- **`applications.csv`**：Steam应用的基础信息（名称、发布日期、价格等）
- **`reviews.csv`**：用户评价数据，包含评分和评论数量
- **`genres.csv` / `application_genres.csv`**：游戏类型信息
- **`developers.csv` / `application_developers.csv`**：开发者信息

## 功能实现

### 1. 高效内存加载

- **优化的内存管理**：将所有数据加载到内存中，提供最快的访问速度
- **适合开发和生产环境**：优化的加载机制确保快速启动和响应
- **低资源占用**：高效的数据结构设计，减少内存消耗

### 2. 异步数据获取

- 推荐结果先返回基本信息，减少初始加载时间
- 后续通过异步请求获取Steam API数据（价格、评分等）
- 实现了Steam API数据缓存，减少重复请求

### 3. 瀑布流布局

- 前端采用CSS Grid实现瀑布流布局
- 支持响应式设计，适配不同屏幕尺寸
- 实现了无限滚动，自动加载更多推荐项

### 4. 高精度推荐算法

#### 向量嵌入处理与相似度计算

推荐系统的核心是基于游戏嵌入向量的相似度计算，具体实现位于 `vector_index.py`：

```python
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
```

**技术细节**：
- **向量归一化**：对查询向量和数据库中的向量进行L2归一化，确保余弦相似度计算的准确性
- **余弦相似度**：通过点积计算归一化向量间的余弦相似度，值域为[-1, 1]
- **高效搜索**：使用NumPy的向量化操作，支持大规模向量的快速相似度计算
- **Top-K检索**：按相似度降序排列，返回最相似的K个游戏

#### 多游戏加权推荐

系统支持为多个游戏设置不同权重进行混合推荐：

```python
def get_weighted_recommendations(self, game_weights, k=10):
    """基于多个游戏的加权推荐"""
    weighted_vector = np.zeros(self.vector_dim)
    total_weight = 0
    
    # 计算加权平均向量
    for appid, weight in game_weights.items():
        if appid in self.appid_to_index:
            idx = self.appid_to_index[appid]
            game_vector = self.vectors[idx]
            weighted_vector += weight * game_vector
            total_weight += weight
    
    if total_weight > 0:
        weighted_vector /= total_weight
    
    # 基于加权向量搜索相似游戏
    distances, indices = self.search(weighted_vector, k)
    
    return distances, indices
```

**核心思想**：
- **向量加权平均**：将多个游戏的嵌入向量按权重进行线性组合
- **偏好融合**：通过调整权重比例，平衡用户对不同游戏的偏好强度
- **个性化推荐**：权重越大表示用户对该游戏的偏好越强

#### 基于评论数量的重排序机制

为了提升推荐质量，系统引入评论数量作为重要权重因子进行结果重排序，实现位于 `recommendation_system.py`：

```python
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
```

**重排序策略**：
- **评论数量权重**：评论数量反映游戏的受欢迎程度和社区活跃度
- **加权评分计算**：结合相似度分数和评论数量，计算综合评分
- **质量优先**：优先推荐既有高相似度又有大量评论的游戏
- **冷启动缓解**：避免推荐相似度高但评论极少的冷门游戏

**评分计算公式**：
```python
def calculate_weighted_score(self, similarity, review_count):
    """计算加权评分：结合相似度和评论数量"""
    # 基础相似度分数
    base_score = similarity
    
    # 评论数量权重（使用对数缩放避免极端值影响）
    review_weight = math.log(review_count + 1) / math.log(self.max_review_count + 1)
    
    # 综合评分：70%相似度 + 30%评论权重
    weighted_score = 0.7 * base_score + 0.3 * review_weight
    
    return weighted_score
```

#### 推荐流程总结

1. **输入处理**：接收用户选择的游戏列表和对应权重
2. **向量计算**：基于权重计算加权平均查询向量
3. **相似度搜索**：使用余弦相似度在向量索引中查找最相似的游戏
4. **初步筛选**：获取Top-K相似游戏作为候选集
5. **重排序**：基于评论数量对候选结果进行加权重排序
6. **结果返回**：返回最终的推荐列表，包含相似度、评论数量等信息

这种多层次的推荐机制确保了推荐结果既保持了与用户偏好的高度相关性，又兼顾了游戏的质量和受欢迎程度，提供了更加智能和实用的游戏推荐体验。

## 构建过程中的问题解决

### 1. CORS跨域问题

- **问题**：局域网设备访问时出现跨域请求错误
- **解决**：在Flask应用中配置CORS，允许所有来源的请求
- **代码**：`CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})`

### 2. 数据加载性能问题

- **问题**：直接加载大量数据到内存导致启动缓慢
- **解决**：实现双模式加载，支持从PostgreSQL数据库读取数据
- **代码**：`DataLoader`类支持`load_mode`参数，可选择'memory'或'database'

### 3. Steam API请求延迟问题

- **问题**：Steam API响应慢，导致推荐结果加载延迟
- **解决**：实现异步数据加载，先返回基本结果，再异步更新API数据
- **代码**：前端实现异步请求`/api/game-details`端点



## 依赖要求

- pandas
- requests
- flask
- flask_cors
- psycopg2-binary

## 使用方法

### 0. 数据集准备

在使用本系统之前，需要先下载并配置Steam数据集：

1. **下载数据集**：从 [Kaggle - Steam Dataset 2025](https://www.kaggle.com/datasets/crainbramp/steam-dataset-2025-multi-modal-gaming-analytics) 下载完整数据集

2. **解压数据集**：将下载的数据集解压到项目根目录，确保目录结构如下：
   ```
   GREC/
   ├── steam_dataset_2025/
   │   ├── steam-dataset-2025-v1/
   │   ├── steam_dataset_2025_csv_package_v1/
   │   ├── steam_dataset_2025_embeddings_package_v1/
   │   └── steam_dataset_2025_power_users_dump_v1/
   └── ...（其他项目文件）
   ```

3. **验证数据完整性**：确保以下核心文件存在：
   - `steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embeddings.npy`
   - `steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embedding_map.csv`
   - `steam_dataset_2025_csv_package_v1/steam_dataset_2025_csv/applications.csv`
   - `steam_dataset_2025_csv_package_v1/steam_dataset_2025_csv/reviews.csv`

### 1. 安装依赖

**系统要求**：Python >= 3.9

```bash
pip install -r requirements.txt
```

### 2. 运行应用

```bash
python app.py
```

### 3. 访问应用

在浏览器中访问：http://localhost:5000

输入游戏名或者游戏描述，系统会返回相关游戏列表。添加游戏之后，系统会根据用户输入的游戏列表和权重，计算出用户的游戏偏好向量。



## API接口

### 1. 推荐接口

```
POST /api/recommend
```

**请求体**：
```json
{
  "games": [
    {"appid": 10, "weight": 1.0},
    {"appid": 20, "weight": 0.5}
  ],
  "page": 1,
  "per_page": 10
}
```

**响应**：
```json
{
  "recommendations": [...],
  "total": 100,
  "page": 1,
  "per_page": 10,
  "total_pages": 10
}
```

### 2. 游戏详情接口

```
POST /api/game-details
```

**请求体**：
```json
{
  "appids": [10, 20, 30]
}
```

**响应**：
```json
{
  "details": {
    "10": {...},
    "20": {...},
    "30": {...}
  }
}
```

### 3. 游戏搜索接口

```
GET /api/games?q=game_name&limit=10
```

**响应**：
```json
{
  "games": [...]
}
```

## 开发说明

### 数据准备

1. 下载Steam数据集和嵌入向量文件
2. 放置在`steam_dataset_2025`目录下
3. 或者使用PostgreSQL数据库存储数据

### 测试

运行测试脚本：

```bash
python test_recommendations.py
```

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题或建议，请通过GitHub Issues联系我。