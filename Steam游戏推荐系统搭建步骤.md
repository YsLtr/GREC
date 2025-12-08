# Steam游戏推荐系统搭建步骤

## 系统开发阶段划分

本游戏推荐系统的搭建过程明确分为**Debug阶段**和**Release阶段**，以确保开发效率和生产环境的稳定性。

### Debug阶段（调试阶段）
- **目的**：快速调试代码逻辑、验证模型架构、测试功能完整性
- **特点**：
  - 使用`steam_simple_dataset`简化数据集
  - 直接将数据存储在内存中，无需依赖数据库
  - 较短的训练时间（1-2个epoch）
  - 专注于代码错误排查和功能验证
  - 数据库测试需单独进行

### Release阶段（发布阶段）
- **目的**：构建生产环境可用的推荐系统
- **特点**：
  - 使用完整的`steam_dataset_2025`数据集
  - 数据存储在PostgreSQL数据库中
  - 较长的训练时间（10-20个epoch或更多）
  - 确保数据库连接信息正确且已导入完整数据集
  - 优化性能和稳定性

---

## 1. 环境准备和验证

### 1.1 虚拟环境配置
- 使用Python 3.9创建虚拟环境：
  ```bash
  python -m venv venv
  ```
- 激活虚拟环境：
  - Windows: `venv\Scripts\activate`
  - Linux/macOS: `source venv/bin/activate`

### 1.2 依赖安装
- 安装基础依赖（严格按照指定版本）：
  ```bash
  pip install numpy==1.19.5 pandas==1.3.5 scikit-learn tensorflow-gpu==2.6.0 sqlalchemy psycopg2-binary fastapi uvicorn redis python-dotenv requests==2.26.0 protobuf==3.20.3
  ```
- 安装额外工具：
  ```bash
  pip install pytest pytest-cov locust
  ```

### 1.3 环境验证
- 验证Python版本：`python --version` (应显示3.9.x)

### 1.4 GPU环境配置和验证
- TensorFlow 2.6.0需要CUDA 11.2和cuDNN 8.1，必须确保版本完全匹配
- 安装CUDA 11.2：
  - 下载地址：https://developer.nvidia.com/cuda-11.2.0-download-archive
  - 安装时选择自定义安装，确保勾选以下组件：
    - CUDA Toolkit 11.2
    - CUDA Development Libraries
    - CUDA Runtime Libraries
    - CUDA Documentation (可选)
    - CUDA Samples (可选)
- 安装cuDNN 8.1：
  - 下载地址：https://developer.nvidia.com/rdp/cudnn-archive（需要NVIDIA开发者账号）
  - 选择与CUDA 11.2兼容的cuDNN版本
  - 解压后将文件复制到CUDA安装目录：
    - 将`bin/cudnn64_8.dll`复制到`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
    - 将`include/cudnn.h`复制到`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include`
    - 将`lib/x64/cudnn.lib`复制到`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64`
- 配置环境变量：
  - Windows:
    - 系统变量Path中添加：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin`
    - 系统变量Path中添加：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64`
    - 系统变量Path中添加：`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include`
  - Linux:
    - 编辑~/.bashrc文件，添加以下内容：
      ```bash
      export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:$LD_LIBRARY_PATH
      export PATH=/usr/local/cuda-11.2/bin:$PATH
      export CUDA_HOME=/usr/local/cuda-11.2
      ```
    - 执行`source ~/.bashrc`使配置生效

### 1.5 GPU环境测试
**重要提示：每个需要使用GPU进行训练的脚本都必须添加以下代码来正确导入DLL，否则无法开启GPU加速：**
```python
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')
```

- 使用提供的测试脚本验证GPU环境：
  ```bash
  python c:\Users\28676\Documents\Program\GREC\test_gpu.py
  ```

- 该脚本会测试：
  - GPU设备检测
  - 不同规模矩阵运算的GPU加速效果
  - TensorFlow在GPU上的基本运算功能

### 1.6 完整环境验证
- 验证所有依赖版本是否正确安装：
  ```bash
  pip list | grep -E "numpy|pandas|tensorflow|protobuf|requests"
  ```
- 预期输出应显示：
  - numpy 1.19.5
  - pandas 1.3.5
  - tensorflow-gpu 2.6.0
  - protobuf 3.20.3
  - requests 2.26.0

## 2. 数据处理流程

### 2.1 Debug阶段数据处理（内存模式）
在Debug阶段，我们直接将简化数据集存储在内存中，无需依赖数据库，以加快调试速度。

#### 2.1.1 简化数据集加载
```python
import os
import numpy as np
import pandas as pd

# 确保正确导入CUDA DLL（所有GPU训练脚本都需要添加）
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

# 设置简化数据集路径
dataset_path = 'steam_simple_dataset'

# 加载核心数据到内存
applications = pd.read_csv(os.path.join(dataset_path, 'applications_simple.csv'))
categories = pd.read_csv(os.path.join(dataset_path, 'categories.csv'))
genres = pd.read_csv(os.path.join(dataset_path, 'genres.csv'))
reviews = pd.read_csv(os.path.join(dataset_path, 'reviews_simple.csv'))

# 加载关系数据
app_categories = pd.read_csv(os.path.join(dataset_path, 'application_categories_simple.csv'))
app_genres = pd.read_csv(os.path.join(dataset_path, 'application_genres_simple.csv'))

# 加载嵌入向量（如果已有的话）
# Debug阶段：将简化数据集转换为1024维嵌入向量（与完整数据集结构一致）
embedding_dim = 1024
num_apps = len(applications)

# 生成1024维嵌入向量（基于文本内容）
if not os.path.exists('debug_app_embeddings.npy'):
    print("开始将文本内容转换为1024维嵌入向量...")
    
    # 安装必要的依赖（仅首次运行需要）
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        import subprocess
        subprocess.check_call(["pip", "install", "sentence-transformers"])
        from sentence_transformers import SentenceTransformer
    
    # 组合文本字段（与完整数据集的combined_text字段结构一致）
    applications['combined_text'] = applications.apply(lambda x: f"{x['name']} {x['description']}", axis=1)
    
    # 使用轻量级模型生成嵌入向量，然后调整为1024维
    # 注：完整数据集使用BGE-M3模型，Debug阶段使用轻量化模型加快速度
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 生成基础嵌入向量
    base_embeddings = model.encode(applications['combined_text'].tolist(), show_progress_bar=True)
    
    # 调整为1024维（与完整数据集结构一致）
    from sklearn.decomposition import PCA
    if base_embeddings.shape[1] != 1024:
        pca = PCA(n_components=1024)
        debug_app_embeddings = pca.fit_transform(base_embeddings)
    else:
        debug_app_embeddings = base_embeddings
    
    # 保存嵌入向量
    np.save('debug_app_embeddings.npy', debug_app_embeddings)
    print(f"✅ 成功生成{num_apps}个1024维嵌入向量")
    print(f"嵌入向量形状：{debug_app_embeddings.shape}")
else:
    debug_app_embeddings = np.load('debug_app_embeddings.npy')
    print(f"✅ 加载了预先生成的1024维嵌入向量")
    print(f"嵌入向量形状：{debug_app_embeddings.shape}")

# 创建应用ID到索引的映射
app_id_to_index = {app_id: idx for idx, app_id in enumerate(applications['app_id'])}

# 生成用户嵌入向量（user_embedding）
# Debug阶段：基于用户游戏历史生成1024维用户嵌入向量
if not os.path.exists('debug_user_embeddings.npy'):
    print("\n开始生成用户嵌入向量...")
    
    # 构建用户-游戏交互数据（基于评论数据）
    # 注：完整数据集使用BGE-M3模型直接生成用户嵌入，Debug阶段基于游戏嵌入聚合生成
    
    # 获取所有唯一用户
    unique_users = reviews['author_steamid'].unique()
    num_users = len(unique_users)
    
    # 创建用户ID到索引的映射
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
    
    # 生成用户嵌入向量（基于用户评论的游戏的嵌入向量加权平均）
    debug_user_embeddings = np.zeros((num_users, embedding_dim))
    
    for user_idx, user_id in enumerate(unique_users):
        # 获取用户评论的所有游戏
        user_games = reviews[reviews['author_steamid'] == user_id]['app_id'].tolist()
        valid_games = [gid for gid in user_games if gid in app_id_to_index]
        
        if valid_games:
            # 获取对应游戏的嵌入向量
            game_indices = [app_id_to_index[gid] for gid in valid_games]
            game_vectors = debug_app_embeddings[game_indices]
            
            # 使用评论数量作为权重进行加权平均
            weights = np.ones(len(game_vectors)) / len(game_vectors)  # 简单平均
            debug_user_embeddings[user_idx] = np.average(game_vectors, axis=0, weights=weights)
    
    # 保存用户嵌入向量
    np.save('debug_user_embeddings.npy', debug_user_embeddings)
    np.save('debug_user_id_to_index.npy', user_id_to_index)
    print(f"✅ 成功生成{num_users}个1024维用户嵌入向量")
    print(f"用户嵌入向量形状：{debug_user_embeddings.shape}")
else:
    debug_user_embeddings = np.load('debug_user_embeddings.npy')
    user_id_to_index = np.load('debug_user_id_to_index.npy', allow_pickle=True).item()
    print(f"\n✅ 加载了预先生成的1024维用户嵌入向量")
    print(f"用户嵌入向量形状：{debug_user_embeddings.shape}")
```

#### 2.1.2 Debug阶段数据预处理
- 直接在内存中进行数据清洗和转换
- 将文本内容（名称+描述）组合为`combined_text`字段（与完整数据集结构一致）
- 使用轻量级模型将文本转换为1024维游戏嵌入向量（game_embedding，与完整数据集的BGE-M3嵌入结构一致）
- 基于用户评论的游戏生成1024维用户嵌入向量（user_embedding，与完整数据集结构一致）
- 无需复杂的数据库查询，加快开发速度
- 可以跳过一些非关键的特征工程步骤

#### 2.1.3 Debug阶段数据库测试（内存训练成功后）
在Debug阶段内存训练成功后，建议进行数据库功能测试，以确保Release阶段的数据库操作能够正常工作：

1. **数据库连接测试**：
   ```bash
   python c:\Users\28676\Documents\Program\GREC\postgresql_test.py
   ```
   - 该脚本会测试PostgreSQL连接、版本信息、表创建、数据插入和查询功能
   - 确保输出显示"✅ 连接成功！"和"✅ 测试完成！"，否则需要检查数据库配置

2. **使用完整数据表结构进行测试**：
   使用简化数据集和训练好的嵌入向量测试完整的数据表结构创建和导入流程：
   
   ```bash
   # 创建测试数据库和用户
   psql -U postgres -c "CREATE DATABASE steam_test;
   CREATE USER steam_test_user WITH PASSWORD '123456';
   GRANT ALL PRIVILEGES ON DATABASE steam_test TO steam_test_user;"
   
   # 连接测试数据库并创建扩展
   psql -U steam_test_user -d steam_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
   
   # 使用完整的数据表结构（从schema文件）
   psql -U steam_test_user -d steam_test -f c:\Users\28676\Documents\Program\GREC\steam_dataset_2025\steam-dataset-2025-v1\steam-dataset-2025-full-schema.sql
   
   # 创建Python脚本导入数据和嵌入向量
   python -c "
   import os
   import numpy as np
   import pandas as pd
   import psycopg2
   from psycopg2.extras import execute_values
   
   # 确保在正确的目录
   os.chdir('steam_simple_dataset')
   
   # 连接数据库
   conn = psycopg2.connect(
       host='localhost',
       database='steam_test',
       user='steam_test_user',
       password='123456'
   )
   cur = conn.cursor()
   
   # 加载应用数据和嵌入向量
   applications = pd.read_csv('applications_simple.csv')
   app_embeddings = np.load('../debug_app_embeddings.npy')
   
   # 创建嵌入向量字符串转换函数
   def vector_to_string(vec):
       return '[' + ','.join([str(x) for x in vec]) + ']'
   
   # 导入应用数据和嵌入向量
   insert_query = '''
   INSERT INTO applications(appid, name, type, combined_text, description_embedding)
   VALUES %s
   '''
   
   data_to_insert = []
   for idx, row in applications.iterrows():
       combined_text = f"{row['name']} {row['description']}"
       embedding_str = vector_to_string(app_embeddings[idx])
       data_to_insert.append((
           row['app_id'],
           row['name'],
           'game',
           combined_text,
           embedding_str
       ))
   
   execute_values(cur, insert_query, data_to_insert)
   conn.commit()
   
   print(f"✅ 成功导入{len(applications)}个应用数据和嵌入向量")
   
   # 加载用户数据和嵌入向量
   reviews = pd.read_csv('reviews_simple.csv')
   user_embeddings = np.load('../debug_user_embeddings.npy')
   user_id_to_index = np.load('../debug_user_id_to_index.npy', allow_pickle=True).item()
   
   # 导入用户嵌入数据
   cur.execute('CREATE TEMP TABLE temp_user_embeddings (user_id TEXT, embedding vector(1024))')
   
   for user_id, idx in user_id_to_index.items():
       embedding_str = vector_to_string(user_embeddings[idx])
       cur.execute(
           'INSERT INTO temp_user_embeddings(user_id, embedding) VALUES (%s, %s)',
           (user_id, embedding_str)
       )
   
   conn.commit()
   print(f"✅ 成功导入{len(user_id_to_index)}个用户嵌入向量")
   
   # 关闭连接
   cur.close()
   conn.close()
   ""
   ```


---

### 2.2 Release阶段数据处理（数据库模式）
在Release阶段，我们使用PostgreSQL数据库存储完整数据集，确保数据的持久化和可扩展性。

#### 2.2.1 PostgreSQL数据库配置
- 安装PostgreSQL 18并启动服务
- 创建数据库和用户：
  ```sql
  CREATE DATABASE steam_recommender;
  CREATE USER steam_user WITH PASSWORD '123456';
  GRANT ALL PRIVILEGES ON DATABASE steam_recommender TO steam_user;
  ```
- 启用pgvector扩展：
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```



#### 2.2.4 嵌入向量加载（Release）
```python
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# 确保正确导入CUDA DLL
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

# 加载完整的预计算嵌入向量
applications_embeddings = np.load('steam_dataset_2025/steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embeddings.npy')
applications_map = pd.read_csv('steam_dataset_2025/steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embedding_map.csv')

# 数据库连接
engine = create_engine('postgresql://steam_user:123456@localhost:5432/steam_recommender')

# 从数据库加载应用数据
with engine.connect() as conn:
    applications = pd.read_sql('SELECT * FROM applications', conn)

# 创建应用ID到索引的映射
app_id_to_index = {app_id: idx for idx, app_id in enumerate(applications_map['app_id'])}
```

## 3. 数据预处理和特征工程

### 3.1 嵌入向量加载
在特征工程的数据加载阶段，需要根据不同的部署阶段选择对应的数据集：

#### Debug阶段（使用简化数据集）
Debug阶段使用简化的数据集进行快速验证：

```python
import numpy as np
import pandas as pd

# 加载简化的游戏嵌入向量和映射表
applications_embeddings = np.load('debug_app_embeddings.npy')
applications = pd.read_csv('steam_simple_dataset/applications_simple.csv')

# 创建应用ID到索引的映射
simple_app_map = {row['app_id']: idx for idx, row in applications.iterrows()}
```

#### Release阶段（使用完整数据集）
Release阶段使用完整的数据集进行生产环境部署：

```python
import numpy as np
import pandas as pd

# 加载完整的游戏嵌入向量和映射表
applications_embeddings = np.load('steam_dataset_2025/steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embeddings.npy')
applications_map = pd.read_csv('steam_dataset_2025/steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embedding_map.csv')
```

**数据差异说明：**
- Debug阶段使用`steam_simple_dataset`目录下的简化数据集，包含少量游戏和用户数据
- Release阶段使用`steam_dataset_2025`目录下的完整数据集，包含2025年最新的全部游戏数据
- Debug阶段的嵌入向量文件较小，加载速度快，适合快速验证模型
- Release阶段的嵌入向量文件较大，包含更丰富的语义信息，适合生产环境使用

### 3.2 数据清洗和转换
- 处理缺失值
- 转换数据类型
- 标准化数值特征

### 3.3 用户游戏历史表示
- 构建用户-游戏交互矩阵
- 生成用户画像向量：
  ```python
  def generate_user_profile(user_game_ids, app_embeddings, app_map):
      # 根据用户游戏ID获取对应的嵌入向量
      user_embeddings = []
      for game_id in user_game_ids:
          # 查找游戏ID对应的嵌入向量索引
          idx = app_map[app_map['app_id'] == game_id].index
          if len(idx) > 0:
              user_embeddings.append(app_embeddings[idx[0]])
      
      # 计算用户画像（嵌入向量的平均值）
      if len(user_embeddings) > 0:
          user_profile = np.mean(user_embeddings, axis=0)
      else:
          user_profile = np.zeros(app_embeddings.shape[1])
      
      return user_profile
  ```

## 4. 模型设计和训练

### 4.1 推荐模型架构
- 输入层：用户画像向量 + 游戏嵌入向量
- 融合层：向量拼接 + 全连接层
- 输出层：推荐分数（0-1）

### 4.2 模型实现
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 创建推荐模型
def create_recommendation_model(embedding_dim=1024):
    # 用户输入
    user_input = layers.Input(shape=(embedding_dim,), name='user_input')
    # 游戏输入
    game_input = layers.Input(shape=(embedding_dim,), name='game_input')
    
    # 融合层
    concatenated = layers.Concatenate()([user_input, game_input])
    dense1 = layers.Dense(512, activation='relu')(concatenated)
    dense2 = layers.Dense(256, activation='relu')(dense1)
    dropout = layers.Dropout(0.3)(dense2)
    
    # 输出层
    output = layers.Dense(1, activation='sigmoid', name='output')(dropout)
    
    # 构建模型
    model = models.Model(inputs=[user_input, game_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# 初始化模型
model = create_recommendation_model()
model.summary()
```

### 4.3 GPU测试（训练前必须执行）

**重要提示：所有需要使用GPU的脚本都必须添加CUDA DLL导入语句。**

在进行模型训练之前，必须先执行GPU测试，以确保TensorFlow能够正确使用GPU加速：

```python
# 合并后的GPU测试脚本（适用于所有阶段）
import os
# 确保正确导入CUDA DLL
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.test.is_gpu_available())

# 简单的GPU运算测试
if tf.test.is_gpu_available():
    # 创建一个简单的张量并在GPU上运行
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    
    # 确保操作在GPU上执行
    with tf.device('/GPU:0'):
        c = tf.matmul(a, b)
    
    print("GPU运算结果:", c)
    print("GPU测试通过！")
else:
    print("GPU不可用，请检查环境配置。")
```

或者使用提供的完整测试脚本：
```bash
python c:\Users\28676\Documents\Program\GREC\test_gpu.py
```

该脚本会测试：
- GPU设备检测
- 不同规模矩阵运算的GPU加速效果
- TensorFlow在GPU上的基本运算功能

**注意**：
- 确保CUDA和cuDNN版本与TensorFlow 2.6.0兼容
- 如果GPU测试失败，请检查NVIDIA驱动程序、CUDA和cuDNN的安装配置
- 只有GPU测试通过后，才能进行后续的模型训练

### 4.4 训练数据准备
- 生成正负样本
- 划分训练集和测试集

### 4.5 模型训练

#### 4.5.1 Debug阶段训练（快速调试）
在Debug阶段，我们使用较短的训练时间来快速验证模型架构和代码逻辑：

```python
# Debug阶段训练参数配置
DEBUG_EPOCHS = 2  # 仅训练2个epoch用于调试
DEBUG_BATCH_SIZE = 64

# 确保正确导入CUDA DLL
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

import tensorflow as tf

# 训练模型（Debug模式）
history = model.fit(
    [user_profiles, game_embeddings],
    labels,
    batch_size=DEBUG_BATCH_SIZE,
    epochs=DEBUG_EPOCHS,  # 短训练时间
    validation_split=0.2,
    verbose=1
)

# 保存调试模型
model.save('debug_recommender_model.h5')
print("Debug模型训练完成并保存")
```

#### 4.5.2 Release阶段训练（生产环境）
在Release阶段，我们使用完整的训练时间来优化模型性能：

```python
# Release阶段训练参数配置
RELEASE_EPOCHS = 15  # 完整训练15个epoch
RELEASE_BATCH_SIZE = 64

# 确保正确导入CUDA DLL
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

import tensorflow as tf

# 训练模型（Release模式）
history = model.fit(
    [user_profiles, game_embeddings],
    labels,
    batch_size=RELEASE_BATCH_SIZE,
    epochs=RELEASE_EPOCHS,  # 长训练时间
    validation_split=0.2,
    verbose=1
)

# 保存生产模型
model.save('release_recommender_model.h5')
print("Release模型训练完成并保存")
```

## 5. 模型评估和调优

### 5.1 评估指标
- 准确率（Accuracy）
- 精确率（Precision）
- 召回率（Recall）
- F1分数
- AUC-ROC

### 5.2 评估实现

#### 5.2.1 Debug阶段模型评估
```python
# 确保正确导入CUDA DLL
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

import tensorflow as tf
import numpy as np

# 加载Debug阶段训练的模型
model = tf.keras.models.load_model('debug_recommender_model.h5')

# 评估模型
loss, accuracy = model.evaluate([test_user_profiles, test_game_embeddings], test_labels)
print(f"Debug模型评估结果 - Loss: {loss}, Accuracy: {accuracy}")

# 预测
predictions = model.predict([test_user_profiles, test_game_embeddings])
print(f"Debug模型预测完成，预测结果形状: {predictions.shape}")
```

#### 5.2.2 Release阶段模型评估
```python
# 确保正确导入CUDA DLL
import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

import tensorflow as tf
import numpy as np

# 加载Release阶段训练的模型
model = tf.keras.models.load_model('release_recommender_model.h5')

# 评估模型
loss, accuracy = model.evaluate([test_user_profiles, test_game_embeddings], test_labels)
print(f"Release模型评估结果 - Loss: {loss}, Accuracy: {accuracy}")

# 预测
predictions = model.predict([test_user_profiles, test_game_embeddings])
print(f"Release模型预测完成，预测结果形状: {predictions.shape}")
```

### 5.3 模型调优
- 调整超参数
- 尝试不同的模型架构
- 使用正则化技术防止过拟合

## 6. API开发和部署

### 6.1 FastAPI应用创建

#### 6.1.1 Debug阶段API配置
在Debug阶段，API使用内存数据和Debug模型：

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pandas as pd
import os

# 确保正确导入CUDA DLL
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

# 初始化FastAPI应用
app = FastAPI(title="Steam Game Recommender API (Debug)")

# 加载Debug模型和内存数据
model = tf.keras.models.load_model('debug_recommender_model.h5')

# 加载内存数据（Debug阶段使用）
dataset_path = 'steam_simple_dataset'
applications = pd.read_csv(os.path.join(dataset_path, 'applications_simple.csv'))

# 加载Debug嵌入向量
app_embeddings = np.load('debug_app_embeddings.npy')
app_map = pd.read_csv(os.path.join(dataset_path, 'applications_simple.csv'))
app_id_to_index = {app_id: idx for idx, app_id in enumerate(applications['app_id'])}

# API端点
@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: int, limit: int = 10):
    # Debug阶段：使用内存中的模拟用户数据
    # 生成用户画像
    # 计算推荐分数
    # 返回推荐结果
    pass
```

#### 6.1.2 Release阶段API配置
在Release阶段，API使用数据库和Release模型：

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import pandas as pd
from sqlalchemy import create_engine
import os

# 确保正确导入CUDA DLL
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')

# 初始化FastAPI应用
app = FastAPI(title="Steam Game Recommender API (Release)")

# 加载Release模型
model = tf.keras.models.load_model('release_recommender_model.h5')

# 加载Release嵌入向量
app_embeddings = np.load('steam_dataset_2025/steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embeddings.npy')
app_map = pd.read_csv('steam_dataset_2025/steam_dataset_2025_embeddings_package_v1/steam_dataset_2025_embeddings/applications_embedding_map.csv')

# 数据库连接
engine = create_engine('postgresql://steam_user:123456@localhost:5432/steam_recommender')

# API端点
@app.get("/api/recommendations/{user_id}")
async def get_recommendations(user_id: int, limit: int = 10):#### 2.2.2 数据库表结构创建
- 使用提供的SQL文件创建表结构：
  ```bash
  psql -U steam_user -d steam_recommender -f steam_dataset_2025/steam-dataset-2025-v1/steam-dataset-2025-full-schema.sql
  ```

#### 2.2.3 完整数据集导入
- 导入完整数据集的所有表：
  ```bash
  # 注意：完整数据集导入可能需要较长时间
  cd steam_dataset_2025
  # 导入主表
  psql -U steam_user -d steam_recommender -c "COPY applications FROM 'applications.csv' DELIMITER ',' CSV HEADER;"
  # 导入其他表（根据实际情况调整）
  psql -U steam_user -d steam_recommender -c "COPY categories FROM 'categories.csv' DELIMITER ',' CSV HEADER;"
  psql -U steam_user -d steam_recommender -c "COPY genres FROM 'genres.csv' DELIMITER ',' CSV HEADER;"
  psql -U steam_user -d steam_recommender -c "COPY reviews FROM 'reviews.csv' DELIMITER ',' CSV HEADER;"
  # 导入关系表和其他数据...
  ```
    # 从数据库获取用户游戏历史
    # 生成用户画像
    # 计算推荐分数
    # 返回推荐结果
    pass
```

### 6.2 Redis缓存配置
- 配置Redis连接
- 实现缓存逻辑：
  ```python
  import redis
  
  r = redis.Redis(host='localhost', port=6379, db=0)
  
  # 缓存推荐结果
def cache_recommendations(user_id, recommendations, recommendation_type):
    key = f"recommendations:{user_id}:{recommendation_type}"
    r.setex(key, 3600, str(recommendations))  # 缓存1小时
  ```

### 6.3 API启动
```bash
uvicorn main:app --reload
```

## 7. 前端开发和集成

### 7.1 页面结构
- 顶部区域：Steam ID输入框 + 提交按钮
- 左侧区域：游戏搜索 + 感兴趣游戏列表
- 右侧区域：推荐结果展示 + 推荐类型切换
- 底部区域：加载状态 + 错误提示

### 7.2 交互功能实现
```javascript
// Steam ID导入功能
function importSteamGames() {
    const steamId = document.getElementById('steam-id-input').value;
    // 验证Steam ID
    if (!isValidSteamId(steamId)) {
        showError('Invalid Steam ID');
        return;
    }
    
    showLoading('Importing games...');
    
    // 调用API
    fetch(`/api/users/${userId}/import-steam-games`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ steam_id: steamId })
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        showSuccess('Games imported successfully');
        // 更新游戏列表
        updateGameList(data.games);
        // 获取推荐
        getRecommendations();
    })
    .catch(error => {
        hideLoading();
        showError('Failed to import games');
    });
}
```

## 8. 测试和调试

### 8.1 单元测试
- 使用pytest测试推荐算法
- 测试API函数

### 8.2 集成测试
- 测试API端点
- 测试数据库交互

### 8.3 性能测试
- 使用locust进行性能测试
- 监控API响应时间



## 9. 部署到生产环境

### 9.1 完整数据集导入
- 导入完整的Steam 2025数据集
- 优化数据库索引

### 9.2 模型优化
- 转换为TensorFlow Lite或TensorRT格式
- 优化推理速度

### 9.3 服务部署
- 使用Docker容器化应用
- 配置Nginx反向代理
- 设置监控和日志

## 10. 维护和扩展

### 10.1 数据更新
- 定期更新游戏数据
- 重新训练模型

### 10.2 功能扩展
- 添加新的推荐策略
- 支持多语言
- 集成社交功能

## 11. 故障排除

### 11.1 常见问题
- TensorFlow GPU配置失败
- 数据库连接问题
- API响应慢

### 11.2 解决方案
- 检查CUDA和cuDNN版本兼容性
- 验证数据库连接字符串
- 优化查询和缓存策略

## 12. 性能指标

- 推荐响应时间：< 300ms
- 系统吞吐量：> 200 QPS
- 缓存命中率：> 80%
- GPU使用率：训练时 > 80%，推理时 < 20%
