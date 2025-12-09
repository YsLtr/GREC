# test_gpu_performance.py
"""
TensorFlow GPU 性能测试
"""

import os
os.add_dll_directory(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin')
import time
import numpy as np
import tensorflow as tf

def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"📊 {title}")
    print("=" * 70)

def main():
    print_header("TensorFlow 2.6.0 GPU 性能测试")
    
    
    # 基本信息
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"NumPy版本: {np.__version__}")
    
    # GPU信息
    print_header("GPU设备信息")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ 找到 {len(gpus)} 个GPU设备:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")
            
            # 获取GPU详细信息
            try:
                details = tf.config.experimental.get_device_details(gpu)
                if details:
                    for key, value in details.items():
                        print(f"    {key}: {value}")
            except:
                pass
    else:
        print("❌ 未找到GPU设备")
        return
    
    # 性能测试
    print_header("GPU性能测试")
    
    test_sizes = [
        (100, 100, "小矩阵"),
        (1000, 1000, "中等矩阵"),
        (3000, 3000, "大矩阵"),
    ]
    
    results = []
    
    for rows, cols, description in test_sizes:
        print(f"\n测试: {description} ({rows}x{cols})")
        
        # GPU测试
        with tf.device('/GPU:0'):
            start = time.time()
            a = tf.random.normal([rows, cols])
            b = tf.random.normal([cols, rows])
            c = tf.matmul(a, b)
            result = tf.reduce_sum(c)
            gpu_time = time.time() - start
            gpu_result = result.numpy()
        
        # CPU测试
        with tf.device('/CPU:0'):
            start = time.time()
            a_cpu = tf.random.normal([rows, cols])
            b_cpu = tf.random.normal([cols, rows])
            c_cpu = tf.matmul(a_cpu, b_cpu)
            result_cpu = tf.reduce_sum(c_cpu)
            cpu_time = time.time() - start
        
        # 计算加速比
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        print(f"  GPU时间: {gpu_time:.3f}秒")
        print(f"  CPU时间: {cpu_time:.3f}秒")
        print(f"  加速比: {speedup:.2f}x")
        
        results.append({
            'size': f"{rows}x{cols}",
            'gpu_time': gpu_time,
            'cpu_time': cpu_time,
            'speedup': speedup
        })
    
    # 总结
    print_header("性能测试总结")
    print(f"{'测试':<15} {'GPU时间(秒)':<15} {'CPU时间(秒)':<15} {'加速比':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['size']:<15} {result['gpu_time']:<15.3f} {result['cpu_time']:<15.3f} {result['speedup']:<10.2f}x")
    
    # 运行您的原始测试
    print_header("原始问题测试")
    print("运行: tf.reduce_sum(tf.random.normal([1000, 1000]))")
    
    with tf.device('/GPU:0'):
        result = tf.reduce_sum(tf.random.normal([1000, 1000]))
        print(f"结果: {result.numpy():.6f}")
        print(f"设备: {result.device}")
    
    print_header("✅ 所有测试完成")

if __name__ == "__main__":
    import sys
    main()