import psycopg2

try:
    # 建立连接
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres",
        password="123456"
    )
    
    print("✅ 连接成功！")
    
    # 创建游标
    cur = conn.cursor()
    
    # 执行测试查询
    cur.execute("SELECT version();")
    version = cur.fetchone()
    print(f"PostgreSQL版本: {version[0]}")
    
    # 测试创建表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS test_table (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 测试插入数据
    cur.execute("INSERT INTO test_table (name) VALUES (%s)", ("测试数据",))
    
    # 测试查询数据
    cur.execute("SELECT * FROM test_table")
    rows = cur.fetchall()
    print("测试表中的数据:")
    for row in rows:
        print(row)
    
    # 提交事务
    conn.commit()
    
    # 关闭连接
    cur.close()
    conn.close()
    print("✅ 测试完成！")
    
except Exception as e:
    print(f"❌ 连接失败: {e}")