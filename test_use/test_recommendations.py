from recommendation_system import GameRecommendationSystem
import sys

# 初始化推荐系统
data_dir = r'c:\Users\28676\Documents\Program\GREC\steam_dataset_2025'
rec_system = GameRecommendationSystem(data_dir)
rec_system.initialize()

# 测试推荐结果
print("测试推荐系统...")

# 选择一个热门游戏的appid，例如CS:GO的appid是730
appid = 730

# 获取推荐结果
recommendations = rec_system.get_recommendations_by_appid(appid, k=10)

# 打印推荐结果
if isinstance(recommendations, str):
    print(recommendations)
else:
    print(f"\n基于游戏ID {appid} 的推荐结果:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['name']}")
        print(f"   AppID: {rec['appid']}")
        print(f"   相似度: {rec['similarity']:.4f}")
        print(f"   评论数量: {rec.get('review_count', 0)}")
        print(f"   加权评分: {rec.get('weighted_score', rec['similarity']):.4f}")
        print(f"   分类: {', '.join(rec['genres'])}")

print("\n测试完成!")
