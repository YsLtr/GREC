import psycopg2

try:
    # 建立连接到postgres数据库
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres",
        password="123456"
    )
    conn.autocommit = True
    
    print("✅ 连接到PostgreSQL成功！")
    
    # 创建游标
    cur = conn.cursor()
    
    # 检查是否存在steam_test数据库
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'steam_test'")
    if cur.fetchone():
        print("🔍 发现steam_test数据库，正在删除...")
        
        # 断开所有连接到steam_test的客户端
        cur.execute("""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = 'steam_test' AND pid <> pg_backend_pid();
        """)
        
        # 删除数据库
        cur.execute("DROP DATABASE IF EXISTS steam_test WITH (FORCE);")
        print("✅ steam_test数据库已删除")
    
    # 重新创建steam_test数据库
    print("🔍 正在创建新的steam_test数据库...")
    cur.execute("CREATE DATABASE steam_test;")
    
    # 创建用户（如果不存在）
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = 'steam_test_user'")
    if not cur.fetchone():
        print("🔍 创建steam_test_user用户...")
        cur.execute("CREATE USER steam_test_user WITH PASSWORD '123456';")
    
    # 授予权限
    cur.execute("GRANT ALL PRIVILEGES ON DATABASE steam_test TO steam_test_user;")
    
    # 连接到新创建的数据库
    cur.close()
    conn.close()
    
    # 连接到steam_test数据库
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="steam_test",
        user="postgres",
        password="123456"
    )
    conn.autocommit = True
    cur = conn.cursor()
    
    # 创建vector扩展
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
    # 授予用户对public schema的所有权限
    cur.execute("GRANT ALL PRIVILEGES ON SCHEMA public TO steam_test_user;")
    
    print("✅ 数据库环境清理完成！")
    print("✅ 已创建干净的steam_test数据库和steam_test_user用户")
    print("✅ 已启用vector扩展")
    
    # 关闭连接
    cur.close()
    conn.close()
    
except Exception as e:
    print(f"❌ 数据库操作失败: {e}")
    import traceback
    traceback.print_exc()