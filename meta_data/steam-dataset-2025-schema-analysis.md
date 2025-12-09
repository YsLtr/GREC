# Steam Dataset 2025 数据库架构分析报告

## 1. 文件概述

| 属性 | 详细信息 |
|------|----------|
| 文件名 | `steam-dataset-2025-full-schema.sql` |
| 项目 | Steam Dataset 2025 |
| 作者 | Don (vintagedon) |
| 版本 | 1.0 |
| 创建日期 | 2025-08-31 |
| 最后更新 | 2025-09-29 |
| 许可证 | MIT |
| 数据库要求 | PostgreSQL 16.10 + pgvector 0.8.0 |

### 1.1 核心目的

该SQL文件定义了一个完整的PostgreSQL数据库架构，用于存储和管理Steam游戏平台的多模态数据，包括：
- 239,664个应用程序（游戏、DLC、软件等）
- 1,048,148条用户评论
- BGE-M3 1024维嵌入向量，支持语义搜索

### 1.2 架构特点

- **混合架构**：规范化关系表 + JSONB存储原始API响应
- **向量搜索**：使用pgvector扩展实现语义相似度查询
- **多阶段物化**：从JSONB中提取查询优化列
- **引用完整性**：完整的外键约束和级联删除机制

## 2. 数据库扩展与自定义类型

### 2.1 扩展

```sql
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
```
- **vector**：提供向量数据类型和ivfflat、hnsw访问方法，支持1024维嵌入向量存储和语义搜索

### 2.2 自定义枚举类型

#### app_type (应用程序类型)
```sql
CREATE TYPE public.app_type AS ENUM (
    'game', 'dlc', 'software', 'video', 'demo',
    'music', 'advertising', 'mod', 'episode', 'series'
);
```

#### platform_type (平台类型)
```sql
CREATE TYPE public.platform_type AS ENUM (
    'windows', 'mac', 'linux'
);
```

## 3. 表结构分析

### 3.1 核心表

#### applications (应用程序表)
**用途**：存储Steam应用程序的核心信息，包括游戏、DLC、软件等

**关键字段**：
- `appid`：唯一标识符（主键）
- `name`：应用程序名称
- `type`：应用程序类型（app_type枚举）
- `is_free`：是否免费
- `release_date`：发布日期
- `detailed_description`：详细描述
- `short_description`：简短描述
- `about_the_game`：关于游戏的描述
- `combined_text`：生成的组合文本（用于嵌入）
- `price_overview`：价格信息（JSONB）
- `pc_requirements`：PC系统要求（JSONB）
- `mac_requirements`：Mac系统要求（JSONB）
- `linux_requirements`：Linux系统要求（JSONB）
- `description_embedding`：1024维BGE-M3描述嵌入向量
- 大量物化列：从JSONB中提取的查询优化字段（如`mat_final_price`、`mat_supports_windows`等）

**表结构**：
```sql
CREATE TABLE public.applications (
    appid bigint NOT NULL,
    steam_appid bigint,
    name_from_applist text NOT NULL,
    name text,
    type public.app_type,
    is_free boolean DEFAULT false,
    release_date date,
    -- ... 省略部分字段 ...
    description_embedding public.vector(1024),
    -- ... 省略物化列 ...
    CONSTRAINT valid_metacritic_score CHECK (((metacritic_score IS NULL) OR ((metacritic_score >= 0) AND (metacritic_score <= 100))))
);
```

#### reviews (评论表)
**用途**：存储用户对应用程序的评论数据

**关键字段**：
- `recommendationid`：评论唯一标识符（主键）
- `appid`：应用程序ID（外键）
- `author_steamid`：评论者Steam ID
- `author_playtime_forever`：总游戏时间
- `review_text`：评论内容
- `voted_up`：是否推荐
- `votes_up`：有用投票数
- `votes_funny`：有趣投票数
- `review_embedding`：1024维BGE-M3评论嵌入向量

**表结构**：
```sql
CREATE TABLE public.reviews (
    recommendationid text NOT NULL,
    appid bigint NOT NULL,
    author_steamid text,
    author_num_games_owned bigint,
    author_num_reviews bigint,
    author_playtime_forever bigint DEFAULT 0,
    -- ... 省略部分字段 ...
    review_embedding public.vector(1024),
    CONSTRAINT reviews_comment_count_check CHECK ((comment_count >= 0)),
    CONSTRAINT reviews_votes_funny_check CHECK ((votes_funny >= 0)),
    CONSTRAINT reviews_votes_up_check CHECK ((votes_up >= 0))
);
```

#### embedding_runs (嵌入运行表)
**用途**：跟踪嵌入向量的生成历史和参数

**关键字段**：
- `run_id`：运行ID（主键）
- `model_name`：嵌入模型名称
- `dimension`：向量维度
- `normalized`：是否归一化
- `created_at`：创建时间
- `notes`：备注信息

### 3.2 查找表

- **categories**：应用程序类别
- **developers**：开发者信息
- **genres**：游戏类型
- **platforms**：支持的平台
- **publishers**：发行商信息

这些表都有类似的结构：`id`作为主键，`name`作为唯一名称。

### 3.3 关联表（多对多关系）

- **application_categories**：应用程序-类别关联
- **application_developers**：应用程序-开发者关联
- **application_genres**：应用程序-类型关联
- **application_platforms**：应用程序-平台关联
- **application_publishers**：应用程序-发行商关联

这些表都包含两个外键字段，形成复合主键。

## 4. 视图与物化视图

### 4.1 视图

#### application_platforms_view
**用途**：简化应用程序平台信息的查询

**定义**：
```sql
CREATE VIEW public.application_platforms_view AS
 SELECT a.appid, a.name, array_agg(p.name ORDER BY p.name) AS platforms
   FROM ((public.applications a
     LEFT JOIN public.application_platforms ap ON ((ap.appid = a.appid)))
     LEFT JOIN public.platforms p ON ((p.id = ap.platform_id)))
  GROUP BY a.appid, a.name;
```

### 4.2 物化视图

#### game_features_view
**用途**：预计算游戏的关键特征指标，用于分析和推荐

**定义**：
```sql
CREATE MATERIALIZED VIEW public.game_features_view AS
 WITH review_agg AS (
         SELECT reviews.appid,
            sum(CASE WHEN reviews.voted_up THEN 1 ELSE 0 END)::integer AS positive_ratings,
            sum(CASE WHEN NOT reviews.voted_up THEN 1 ELSE 0 END)::integer AS negative_ratings
           FROM public.reviews
          GROUP BY reviews.appid
        ), dlc_agg AS (
         SELECT applications.base_app_id, count(*) AS dlc_count
           FROM public.applications
          WHERE (applications.base_app_id IS NOT NULL)
          GROUP BY applications.base_app_id
        ), dev_agg AS (
         SELECT application_developers.appid, count(*) AS developer_count
           FROM public.application_developers
          GROUP BY application_developers.appid
        )
 SELECT a.appid, a.name,
        CASE WHEN ((r.positive_ratings + r.negative_ratings) > 0) 
             THEN (r.positive_ratings::double precision / (r.positive_ratings + r.negative_ratings)::double precision)
             ELSE 0::double precision END AS review_score,
        ((a.price_overview ->> 'final'::text))::integer AS price,
        COALESCE(dev.developer_count, 0::bigint) AS developer_count,
        COALESCE(dlc.dlc_count, 0::bigint) AS dlc_count
   FROM (((public.applications a
     LEFT JOIN review_agg r ON ((a.appid = r.appid)))
     LEFT JOIN dlc_agg dlc ON ((a.appid = dlc.base_app_id)))
     LEFT JOIN dev_agg dev ON ((a.appid = dev.appid)))
  WHERE ((a.type = 'game'::public.app_type) AND ((r.positive_ratings + r.negative_ratings) >= 50) AND ((a.price_overview ->> 'final'::text) IS NOT NULL))
  WITH NO DATA;
```

## 5. 函数与触发器

### 5.1 函数

#### get_database_stats()
**用途**：获取数据库统计信息的便捷函数

**定义**：
```sql
CREATE FUNCTION public.get_database_stats() RETURNS TABLE(total_applications bigint, total_games bigint, total_dlc bigint, total_reviews bigint, total_developers bigint, total_publishers bigint, applications_with_embeddings bigint, reviews_with_embeddings bigint)
    LANGUAGE plpgsql
    AS $$
BEGIN
    RETURN QUERY SELECT (
        (SELECT COUNT(*) FROM applications), 
        (SELECT COUNT(*) FROM applications WHERE type = 'game'), 
        (SELECT COUNT(*) FROM applications WHERE type = 'dlc'), 
        (SELECT COUNT(*) FROM reviews), 
        (SELECT COUNT(*) FROM developers), 
        (SELECT COUNT(*) FROM publishers), 
        (SELECT COUNT(*) FROM applications WHERE description_embedding IS NOT NULL), 
        (SELECT COUNT(*) FROM reviews WHERE review_embedding IS NOT NULL)
    );
END;
$$;
```

#### sync_application_platforms()
**用途**：自动同步应用程序的平台支持信息

**定义**：
```sql
CREATE FUNCTION public.sync_application_platforms() RETURNS trigger
    LANGUAGE plpgsql
    AS $$
DECLARE
  pid_windows integer;
  pid_mac     integer;
  pid_linux   integer;
BEGIN
  SELECT MAX(id) FILTER (WHERE name = 'windows'::platform_type),
         MAX(id) FILTER (WHERE name = 'mac'::platform_type),
         MAX(id) FILTER (WHERE name = 'linux'::platform_type)
    INTO pid_windows, pid_mac, pid_linux
  FROM public.platforms;

  -- DELETE stale links
  DELETE FROM public.application_platforms
  WHERE appid = NEW.appid
    AND (
      (platform_id = pid_windows AND NOT COALESCE(NEW.supports_windows, FALSE)) OR
      (platform_id = pid_mac     AND NOT COALESCE(NEW.supports_mac,     FALSE)) OR
      (platform_id = pid_linux   AND NOT COALESCE(NEW.supports_linux,   FALSE))
    );

  -- INSERT missing links
  IF COALESCE(NEW.supports_windows, FALSE) THEN
    INSERT INTO public.application_platforms(appid, platform_id)
    VALUES (NEW.appid, pid_windows)
    ON CONFLICT DO NOTHING;
  END IF;
  -- ... 省略mac和linux的类似逻辑 ...

  RETURN NEW;
END$$;
```

#### update_updated_at_column()
**用途**：自动更新记录的updated_at时间戳

**定义**：
```sql
CREATE FUNCTION public.update_updated_at_column() RETURNS trigger
    LANGUAGE plpgsql
    AS $$ BEGIN NEW.updated_at = NOW(); RETURN NEW; END; $$;
```

### 5.2 触发器

#### trg_sync_application_platforms
**用途**：当applications表的平台支持字段更新时，自动同步application_platforms表

```sql
CREATE TRIGGER trg_sync_application_platforms 
AFTER INSERT OR UPDATE OF supports_windows, supports_mac, supports_linux 
ON public.applications 
FOR EACH ROW EXECUTE FUNCTION public.sync_application_platforms();
```

#### update_applications_updated_at
**用途**：当applications表记录更新时，自动更新updated_at字段

```sql
CREATE TRIGGER update_applications_updated_at 
BEFORE UPDATE ON public.applications 
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
```

#### update_reviews_updated_at
**用途**：当reviews表记录更新时，自动更新updated_at字段

```sql
CREATE TRIGGER update_reviews_updated_at 
BEFORE UPDATE ON public.reviews 
FOR EACH ROW EXECUTE FUNCTION public.update_updated_at_column();
```

## 6. 索引与约束

### 6.1 索引

#### 向量索引（支持语义搜索）
```sql
-- 应用程序描述嵌入的HNSW索引
CREATE INDEX idx_applications_description_embedding_hnsw 
ON public.applications USING hnsw (description_embedding public.vector_cosine_ops);

-- 评论嵌入的HNSW索引
CREATE INDEX idx_reviews_review_embedding_hnsw 
ON public.reviews USING hnsw (review_embedding public.vector_cosine_ops);
```

#### 常规索引
```sql
-- 应用程序名称索引
CREATE INDEX idx_applications_name ON public.applications USING btree (name);

-- 应用程序发布日期索引
CREATE INDEX idx_applications_release_date ON public.applications USING btree (release_date);

-- 应用程序类型索引
CREATE INDEX idx_applications_type ON public.applications USING btree (type);

-- 评论appid索引
CREATE INDEX idx_reviews_appid ON public.reviews USING btree (appid);

-- 评论投票索引
CREATE INDEX idx_reviews_voted_up ON public.reviews USING btree (voted_up);
```

### 6.2 约束

#### 主键约束
所有表都有适当的主键约束，例如：
```sql
ALTER TABLE ONLY public.applications ADD CONSTRAINT applications_pkey PRIMARY KEY (appid);
ALTER TABLE ONLY public.reviews ADD CONSTRAINT reviews_pkey PRIMARY KEY (recommendationid);
```

#### 外键约束
所有关联表都有完整的外键约束，例如：
```sql
ALTER TABLE ONLY public.application_categories 
ADD CONSTRAINT application_categories_appid_fkey 
FOREIGN KEY (appid) REFERENCES public.applications(appid) ON DELETE CASCADE;

ALTER TABLE ONLY public.reviews 
ADD CONSTRAINT reviews_appid_fkey 
FOREIGN KEY (appid) REFERENCES public.applications(appid) ON DELETE CASCADE;
```

#### 唯一约束
```sql
ALTER TABLE ONLY public.embedding_runs 
ADD CONSTRAINT uq_embedding_run UNIQUE (model_name, dimension, normalized);
```

#### 检查约束
```sql
ALTER TABLE ONLY public.applications 
ADD CONSTRAINT valid_metacritic_score 
CHECK (((metacritic_score IS NULL) OR ((metacritic_score >= 0) AND (metacritic_score <= 100))));

ALTER TABLE ONLY public.reviews 
ADD CONSTRAINT reviews_comment_count_check CHECK ((comment_count >= 0));
```

## 7. 数据流向与关系

### 7.1 核心数据流

1. **数据来源**：Steam Web API（appdetails和reviews API）
2. **数据存储**：
   - 原始JSON响应存储在applications表的JSONB字段中
   - 评论数据存储在reviews表中
3. **数据处理**：
   - 从JSONB中提取关键信息到物化列
   - 生成combined_text字段用于嵌入
   - 计算BGE-M3 1024维嵌入向量
4. **数据访问**：
   - 通过视图和物化视图简化查询
   - 使用向量索引进行语义搜索
   - 通过关联表访问多对多关系

### 7.2 主要实体关系

- **应用程序与评论**：一对多关系（一个应用程序可以有多个评论）
- **应用程序与开发者/发行商**：多对多关系
- **应用程序与类型/类别**：多对多关系
- **应用程序与平台**：多对多关系
- **应用程序与嵌入向量**：一对一关系

## 8. 技术特点与亮点

### 8.1 多模态数据支持
- 支持结构化数据（表字段）
- 支持半结构化数据（JSONB）
- 支持向量数据（嵌入向量）

### 8.2 高性能设计
- **HNSW向量索引**：支持高效的语义相似性搜索
- **物化列**：从JSONB中提取常用查询字段，提高查询性能
- **适当的B-tree索引**：加速常规查询
- **视图与物化视图**：预计算常用查询结果

### 8.3 数据完整性与自动化
- **完整的外键约束**：确保数据一致性
- **触发器**：自动维护数据关系和时间戳
- **检查约束**：确保数据有效性

### 8.4 可扩展性
- **嵌入运行表**：支持多版本嵌入模型
- **规范化设计**：支持数据模型扩展
- **PostgreSQL扩展**：利用pgvector实现高级功能

## 9. 应用场景

该数据库架构适用于以下场景：

1. **游戏推荐系统**：基于内容（嵌入向量）、协同过滤（评论数据）或混合方法
2. **语义搜索**：通过BGE-M3嵌入实现自然语言查询
3. **数据分析**：分析游戏趋势、用户评价、开发者表现等
4. **游戏发现**：基于类型、平台、价格等筛选条件
5. **用户行为分析**：分析用户评论和游戏偏好

## 10. 总结

Steam Dataset 2025数据库架构是一个设计精良、功能完整的多模态数据存储解决方案，具有以下优点：

- **全面性**：支持多种数据类型和关系
- **性能**：优化的索引和物化设计
- **灵活性**：JSONB存储和向量支持
- **可靠性**：完整的约束和自动化机制
- **可扩展性**：支持未来功能扩展

该架构为构建Steam游戏推荐系统、分析平台或搜索工具提供了坚实的数据基础。