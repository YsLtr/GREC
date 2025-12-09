import os
import numpy as np

# 检查文件的前几个字节
def check_file_format(file_path):
    print(f"Checking file: {file_path}")
    with open(file_path, 'rb') as f:
        header = f.read(100)
        print(f"First 100 bytes: {header}")
    
    # 尝试用numpy的fromfile函数读取
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        print(f"Fromfile shape: {data.shape}")
        print(f"First 10 values: {data[:10]}")
    except Exception as e:
        print(f"Fromfile failed: {e}")

# 检查应用嵌入向量文件
app_embeddings_path = 'c:\\Users\\28676\\Documents\\Program\\GREC\\steam_dataset_2025\\steam_dataset_2025_embeddings_package_v1\\steam_dataset_2025_embeddings\\applications_embeddings.npy'
check_file_format(app_embeddings_path)
