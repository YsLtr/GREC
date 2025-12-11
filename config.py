# 配置文件 - 集中管理所有系统配置

# 缓存配置
CACHE_EXPIRY = 24 * 60 * 60  # 缓存过期时间（24小时）

# 推荐算法配置
USE_CONTENT_BASED = True  # 是否使用内容推荐
USE_COLLABORATIVE = True  # 是否使用协同推荐

# 推荐权重配置
GAME_WEIGHT = 0.3  # 内容推荐权重
REVIEW_WEIGHT = 0.7  # 协同推荐权重

# 索引类型配置 - 支持 FLAT 或 HNSW
CONTENT_INDEX_TYPE = 'HNSW'  # 内容推荐索引类型
COLLABORATIVE_INDEX_TYPE = 'HNSW'  # 协同推荐索引类型

# 重排权重配置
REVIEW_COUNT_WEIGHT = 0.5  # 评论数量权重
USER_SCORE_WEIGHT = 0.5  # 用户评分权重

