from recommendation_system import GameRecommendationSystem
import time

# 数据目录路径
data_dir = 'c:\\Users\\28676\\Documents\\Program\\GREC\\steam_dataset_2025'

# 创建推荐系统实例
rec_system = GameRecommendationSystem(data_dir)

# 初始化推荐系统
start_time = time.time()
print("Initializing recommendation system...")
rec_system.initialize()
end_time = time.time()
print(f"Initialization completed in {end_time - start_time:.2f} seconds")

print("\n" + "="*50)
print("游戏推荐系统示例")
print("="*50)

# 示例1：基于游戏ID推荐
print("\n1. 基于游戏ID推荐（Half life，appid=50）：")
recs = rec_system.get_recommendations_by_appid(50, k=5)
rec_system.display_recommendations(recs)

# 示例2：基于游戏名称推荐
print("\n2. 基于游戏名称推荐（Dota 2）：")
recs = rec_system.get_recommendations_by_name("Dota 2", k=5)
rec_system.display_recommendations(recs)

# 示例3：基于多个游戏推荐
print("\n3. 基于多个游戏推荐（Counter-Strike和Dota 2）：")
recs = rec_system.get_recommendations_by_game_list([10, 570], k=5)
rec_system.display_recommendations(recs)

# 示例4：带有权重的多个游戏推荐
print("\n4. 带有权重的多个游戏推荐（Counter-Strike权重0.7，Dota 2权重0.3）：")
recs = rec_system.get_recommendations_by_game_list([10, 570], k=5, weights=[0.7, 0.3])
rec_system.display_recommendations(recs)

# 示例5：过滤推荐结果（只推荐免费游戏）
print("\n5. 过滤推荐结果（只推荐免费游戏）：")
recs = rec_system.get_recommendations_by_appid(10, k=10)
filtered_recs = rec_system.filter_recommendations(recs, filter_criteria={'is_free': True})
rec_system.display_recommendations(filtered_recs)

# 示例6：过滤推荐结果（只推荐评分≥80的游戏）
print("\n6. 过滤推荐结果（只推荐评分≥80的游戏）：")
recs = rec_system.get_recommendations_by_appid(10, k=10)
filtered_recs = rec_system.filter_recommendations(recs, filter_criteria={'min_metacritic_score': 80})
rec_system.display_recommendations(filtered_recs)

print("\n" + "="*50)
print("示例演示完成")
print("="*50)