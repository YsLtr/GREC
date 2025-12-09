# Steam Dataset 2025: Multi-Modal Gaming Analytics 数据集分析报告

## 概述

**数据集名称：** Steam Dataset 2025: Multi-Modal Gaming Analytics  
**创建者：** CrainBramp  
**发布日期：** 2025-10-19  
**数据规模：** 11.621 GB  
**许可证：** Creative Commons Attribution 4.0 International (CC BY 4.0)  
**数据源：** [Kaggle - Steam Dataset 2025](https://www.kaggle.com/datasets/crainbramp/steam-dataset-2025-multi-modal-gaming-analytics)

## 数据集核心特征

### 🎯 独特优势
- **首个多模态Steam数据集**：集成文本、向量嵌入和结构化数据
- **生产级规模**：239,664个应用 vs 传统数据集的6K-27K
- **语义搜索就绪**：1024维BGE-M3向量嵌入支持内容发现
- **完整评论语料库**：1,048,148条用户评论，包含情感和元数据
- **28年覆盖**：从1997年到2025年的平台演进数据
- **多模态架构**：PostgreSQL + JSONB + pgvector统一结构

## 数据集构成分析

### 1. 核心CSV数据包 (steam_dataset_2025_csv_package_v1)

#### 1.1 主数据表
**applications.csv** (180 MB)
- **规模**：239,664条Steam应用记录
- **关键字段**：
  - `appid`：Steam应用ID（主键）
  - `name, type, is_free, release_date`：基本信息
  - `mat_final_price, mat_currency, mat_discount_percent`：价格信息
  - `mat_supports_windows/mac/linux`：平台兼容性
  - `mat_achievement_count`：成就数量
  - `metacritic_score, recommendations_total`：评分指标

#### 1.2 关系表
**categories.csv** (12 KB)
- Steam功能类别定义（如单人对战、合作、云存档等）

**genres.csv** (2.4 KB)
- 游戏流派定义（动作、独立、RPG等）

**developers.csv** (2.0 MB)
- 开发商信息表，支持开发商网络分析

**publishers.csv** (1.8 MB)
- 发行商信息表

**platforms.csv** (32 B)
- 支持的平台定义（Windows、macOS、Linux）

#### 1.3 连接表
**application_categories.csv** (12.4 MB)
- 应用与功能类别的多对多关系

**application_genres.csv** (6.5 MB)
- 应用与流派的多对多关系

**application_developers.csv** (3.4 MB)
- 应用与开发商的多对多关系

**application_publishers.csv** (3.0 MB)
- 应用与发行商的多对多关系

**application_platforms.csv** (3.2 MB)
- 应用与平台的兼容性关系

#### 1.4 用户评论数据
**reviews.csv** (670 MB)
- **规模**：1,048,148条用户评论
- **关键字段**：
  - `recommendationid`：评论ID（主键）
  - `appid`：关联应用ID
  - `author_steamid, author_playtime_forever`：用户信息
  - `review_text`：评论文本
  - `voted_up, votes_up, votes_funny`：投票和情感数据
  - `timestamp_created/updated`：时间戳
  - `language`：评论语言

### 2. 向量嵌入包 (steam_dataset_2025_embeddings_package_v1)

**applications_embeddings.npy**
- 239,664个应用的1024维向量嵌入
- 基于BGE-M3模型生成，支持语义搜索

**applications_embedding_map.csv**
- 向量嵌入的元数据映射表
- 包含应用ID、名称、类型、发布日期、Metacritic评分等

**reviews_embeddings.npy**
- 100万+评论的1024维向量嵌入

**reviews_embedding_map.csv**
- 评论嵌入的元数据映射表

### 3. 平台演进分析数据 (steam-dataset-2025-v1/notebook-data)

#### 3.1 平台增长分析 (01-platform-evolution/)
**01_temporal_growth.csv**
- 1997-2025年Steam目录年度扩展统计
- 按年份统计总应用数、不同类型应用（游戏、DLC、软件、视频、演示）数量
- 包含增长率计算

**02_genre_evolution.csv**
- 年度流派流行度和市场份额分析
- 追踪玩家兴趣和开发者焦点的长期演进

**03_platform_support.csv**
- 操作系统支持采用率历史数据
- 分析Linux/macOS支持趋势和平台碎片化

**04_pricing_strategy.csv**
- 定价和折扣策略年度分析
- 包含通胀调整后的定价分析和免费游戏趋势

**05_publisher_portfolios.csv**
- 发行商组合分析数据

**06_achievement_evolution.csv**
- 成就系统演进数据分析

#### 3.2 语义游戏发现 (02-semantic-game-discovery/)
**01_game_embeddings_sample.csv**
- 游戏嵌入样本数据

**02_embeddings_appids.csv**
- 嵌入数据的应用ID映射

**02_embeddings_vectors.npy**
- 嵌入向量数据

**02_genre_representatives.csv**
- 各流派代表性游戏

**02_semantic_search_examples.json**
- 语义搜索示例

#### 3.3 语义指纹分析 (03-the-semantic-fingerprint/)
**03-the-semantic-fingerprint.parquet**
- 语义指纹数据集（完整版）

**03-the-semantic-fingerprint-preview.csv**
- 语义指纹数据预览版（用于Kaggle笔记本）

### 4. 分析笔记本

**notebook-01-steam-platform-evolution-and-market-landscape.ipynb**
- Steam平台演进和市场格局分析

**notebook-02-semantic-game-discovery.ipynb**
- 语义游戏发现和推荐系统

**notebook-03-the-semantic-fingerprint.ipynb**
- 语义指纹分析和游戏分类

### 5. 数据库架构

**steam-dataset-2025-full-schema.sql**
- 完整的PostgreSQL数据库架构
- 包含JSONB字段和pgvector扩展支持

## 数据质量评估

### ✅ 优势
1. **规模优势**：覆盖Steam平台近30年完整历史数据
2. **多模态特性**：结合结构化数据、文本数据和向量嵌入
3. **预加工分析**：包含预计算的时序分析和语义特征
4. **开放许可**：CC BY 4.0许可允许商业使用
5. **技术先进**：集成最新的向量数据库和语义搜索技术

### ⚠️ 注意事项
1. **文件大小**：总大小11.6GB，需要相应存储和计算资源
2. **处理复杂性**：多模态数据需要不同的处理方法
3. **语言多样性**：评论包含多种语言，需要语言处理能力

## 潜在应用场景

### 🎮 游戏研究
- 游戏市场趋势分析
- 玩家行为模式研究
- 游戏生命周期分析
- 跨平台兼容性研究

### 🤖 机器学习
- 游戏推荐系统
- 情感分析
- 游戏分类和聚类
- 价格预测模型

### 📊 商业智能
- 开发商/发行商表现分析
- 定价策略优化
- 市场细分分析
- 竞争情报

### 🔍 语义搜索
- 内容相似度搜索
- 游戏发现和推荐
- 跨语言游戏匹配

## 技术规格

- **数据格式**：CSV, NPY, Parquet, JSON
- **数据库**：PostgreSQL with JSONB and pgvector
- **向量维度**：1024维 (BGE-M3 embeddings)
- **编码**：UTF-8
- **时间范围**：1997-11-19 to 2025-10-19
- **更新频率**：静态数据集（基于2025年10月数据）

## 使用建议

1. **资源规划**：确保有足够的存储空间（>15GB）和内存（>8GB推荐）
2. **工具选择**：推荐使用Python生态系统（pandas, numpy, scikit-learn）
3. **数据库集成**：可导入PostgreSQL进行复杂查询分析
4. **并行处理**：大规模数据处理建议使用Spark或Dask
5. **语言处理**：多语言评论需要适当的NLP工具包

---

*分析日期：2025-12-05*  
*数据集版本：v1*  
*报告生成者：Claude Code Assistant*