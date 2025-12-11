from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from recommendation_system import GameRecommendationSystem
import os
import numpy as np
import pandas as pd
import requests
import time
import uuid
from threading import Lock
from functools import wraps

# 辅助函数：将numpy类型转换为Python原生类型
def convert_numpy_types(data):
    if isinstance(data, dict):
        return {k: convert_numpy_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.float32) or isinstance(data, np.float64):
        return float(data)
    elif isinstance(data, np.int32) or isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    else:
        return data

# Steam API缓存，用于存储游戏价格数据
steam_price_cache = {}
# Steam API缓存，用于存储游戏评分数据
steam_reviews_cache = {}
# 导入配置
from config import CACHE_EXPIRY, USE_CONTENT_BASED, USE_COLLABORATIVE, GAME_WEIGHT, REVIEW_WEIGHT

# 客户端请求状态跟踪
client_states = {
    # 格式: device_id: {
    #         'is_processing': bool,        # 是否有正在执行的请求
    #         'request_timestamps': list,    # 请求时间戳列表
    #         'cool_down_until': float,      # 冷却结束时间
    #         'last_request_time': float     # 上一次请求时间
    #       }
}
# 保护状态字典的锁
state_lock = Lock()

# 请求状态检查装饰器
def check_request_status(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        # 获取或生成设备标识码
        device_id = request.cookies.get('device_id')
        is_new_device = False
        if not device_id:
            # 如果客户端没有发送设备ID Cookie，生成一个
            device_id = f"device_{uuid.uuid4()}"
            is_new_device = True
        
        current_time = time.time()
        
        with state_lock:
            # 初始化设备状态
            if device_id not in client_states:
                client_states[device_id] = {
                    'is_processing': False,
                    'request_timestamps': [],
                    'cool_down_until': 0,
                    'last_request_time': 0
                }
            
            # 清理过期的请求时间戳（5秒时间窗口）
            client_states[device_id]['request_timestamps'] = [
                ts for ts in client_states[device_id]['request_timestamps'] 
                if current_time - ts < 5
            ]

            # 记录当前请求的时间戳（不论请求是否成功）
            client_states[device_id]['request_timestamps'].append(current_time)
            
            # 检查冷却时间
            if current_time < client_states[device_id]['cool_down_until']:
                remaining_time = int(client_states[device_id]['cool_down_until'] - current_time) + 1
                return jsonify({'error': f'请求过于频繁，请{remaining_time}s后重试'}), 429
            
            # 检查请求间隔（1秒内只能请求一次）
            time_since_last_request = current_time - client_states[device_id]['last_request_time']
            if time_since_last_request < 1.0:
                remaining_time = int(1.0 - time_since_last_request) + 1
                return jsonify({'error': f'请求过于频繁，请{remaining_time}s后重试'}), 429
            
            # 检查是否有正在执行的请求
            if client_states[device_id]['is_processing']:
                return jsonify({'error': 'Previous request not completed'}), 429
            
            # 检查请求频率（5秒内最大10个请求）
            if len(client_states[device_id]['request_timestamps']) >= 10:
                # 设置冷却时间为5秒
                client_states[device_id]['cool_down_until'] = current_time + 5
                return jsonify({'error': '请求过于频繁，请5s后重试'}), 429
            
            
            # 更新请求状态和上次请求时间（只有通过所有检查才会执行）
            client_states[device_id]['is_processing'] = True
            client_states[device_id]['last_request_time'] = current_time
        
        try:
            # 执行原函数
            response = f(*args, **kwargs)
            
            # 如果是新设备，添加设备ID到响应的Cookie中
            if is_new_device:
                # 确保响应是Response对象
                if isinstance(response, tuple):
                    resp_obj, status = response
                    # 创建Response对象（如果不是的话）
                    if not hasattr(resp_obj, 'set_cookie'):
                        resp_obj = jsonify(resp_obj)
                    resp_obj.set_cookie('device_id', device_id, max_age=30*24*60*60, path='/')
                    response = resp_obj, status
                elif not hasattr(response, 'set_cookie'):
                    # 如果返回值不是Response对象，转换为Response对象
                    response = jsonify(response)
                    response.set_cookie('device_id', device_id, max_age=30*24*60*60, path='/')
                else:
                    # 直接设置Cookie
                    response.set_cookie('device_id', device_id, max_age=30*24*60*60, path='/')
            
            return response
        finally:
            # 请求完成后，更新状态
            with state_lock:
                client_states[device_id]['is_processing'] = False
    return decorated

# 初始化数据加载器获取游戏用户评分的函数
def get_steam_reviews_score(appid):
    """通过Steam API获取游戏的用户评分信息"""
    # 检查缓存中是否有有效数据
    if appid in steam_reviews_cache:
        cached_data = steam_reviews_cache[appid]
        if time.time() - cached_data['timestamp'] < CACHE_EXPIRY:
            return cached_data['score']
    
    try:
        # 调用Steam API获取评分信息
        url = f"https://store.steampowered.com/appreviews/{appid}?json=1&language=all&review_type=all&purchase_type=all"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        # 初始化默认值
        positive_rate = 0.0
        total_reviews = 0
        positive_reviews = 0
        negative_reviews = 0
        review_score_desc = None
        
        # 解析评分数据
        if data and data['success'] == 1:
            positive_reviews = data.get('query_summary', {}).get('total_positive', 0)
            negative_reviews = data.get('query_summary', {}).get('total_negative', 0)
            total_reviews = positive_reviews + negative_reviews
            review_score_desc = data.get('query_summary', {}).get('review_score_desc', '')
            
            if total_reviews > 0:
                positive_rate = positive_reviews / total_reviews
        
        # 评分标签映射，将API返回的英文标签转换为中文
        score_label_map = {
            'Overwhelmingly Positive': '好评如潮',
            'Very Positive': '特别好评',
            'Mostly Positive': '多半好评',
            'Mixed': '褒贬不一',
            'Mostly Negative': '多半差评',
            'Very Negative': '特别差评',
            'Overwhelmingly Negative': '差评如潮'
        }
        
        # 获取中文评分标签
        if review_score_desc and review_score_desc in score_label_map:
            user_score_label = score_label_map[review_score_desc]
        elif total_reviews < 100:
            user_score_label = "评价不足"
        else:
            user_score_label = "评价不足"
        
        # 构建评分数据
        score_data = {
            'positive_rate': positive_rate,
            'total_reviews': total_reviews,
            'positive_reviews': positive_reviews,
            'negative_reviews': negative_reviews,
            'user_score_label': user_score_label
        }
        
        # 保存到缓存
        steam_reviews_cache[appid] = {
            'timestamp': time.time(),
            'score': score_data
        }
        
        return score_data
    
    except Exception as e:
        app.logger.error(f"Error getting Steam reviews score for appid {appid}: {e}")
        # 失败时返回默认值
        return {
            'positive_rate': 0.0,
            'total_reviews': 0,
            'positive_reviews': 0,
            'negative_reviews': 0,
            'user_score_label': '评价不足'
        }

# 获取游戏实时价格的函数
def get_steam_price(appid):
    """通过Steam API获取游戏的实时价格"""
    # 检查缓存中是否有有效数据
    if appid in steam_price_cache:
        cached_data = steam_price_cache[appid]
        if time.time() - cached_data['timestamp'] < CACHE_EXPIRY:
            return cached_data['price_data']
    
    try:
        # 调用Steam API获取价格信息（中国地区）
        url = f"https://store.steampowered.com/api/appdetails?appids={appid}&cc=cn&l=chinese&filters=price_overview"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        price_data = {
            'price': '暂无价格',
            'currency': 'USD',
            'is_free': False
        }
        
        if str(appid) in data and data[str(appid)]['success']:
            app_data = data[str(appid)]['data']
            
            if 'is_free' in app_data and app_data['is_free']:
                price_data['price'] = '免费'
                price_data['is_free'] = True
            elif 'price_overview' in app_data:
                price_info = app_data['price_overview']
                price_data['price'] = f"{price_info['currency']} {price_info['final_formatted']}"
                price_data['currency'] = price_info['currency']
                price_data['is_free'] = False
        
        # 保存到缓存
        steam_price_cache[appid] = {
            'timestamp': time.time(),
            'price_data': price_data
        }
        
        return price_data
    
    except Exception as e:
        app.logger.error(f"Error getting price for appid {appid}: {e}")
        # 失败时返回默认值
        return {
            'price': '暂无价格',
            'currency': 'USD',
            'is_free': False
        }

# 设置Flask应用，指定静态文件目录
app = Flask(__name__, static_folder='front', static_url_path='/')
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})  # 允许所有跨域请求

# 初始化推荐系统 - 使用相对路径，支持Windows和Linux
data_dir = os.path.join(os.path.dirname(__file__), 'steam_dataset_2025')

# 初始化推荐系统
rec_system = GameRecommendationSystem(data_dir)
rec_system.initialize(use_content_based=USE_CONTENT_BASED, use_collaborative=USE_COLLABORATIVE)

# 优化搜索：预加载游戏数据并创建小写名称列用于快速搜索
games_df = rec_system.data_loader.applications_df
# 创建小写名称列用于快速搜索
if 'name_lower' not in games_df.columns:
    # 处理nan值，确保name列都是字符串
    games_df['name'] = games_df['name'].fillna('')
    games_df['name_lower'] = games_df['name'].str.lower()
print(f"Search optimization: Created lowercase name column for {len(games_df)} games")

@app.route('/api/recommend', methods=['POST'])
@check_request_status
def recommend():
    """处理推荐请求"""
    try:
        data = request.get_json()
        games = data.get('games', [])
        page = data.get('page', 1)
        per_page = data.get('per_page', 10)
        # 使用配置文件中的权重参数
        game_weight = GAME_WEIGHT  # 内容推荐权重
        review_weight = REVIEW_WEIGHT  # 协同推荐权重
        
        if not games:
            return jsonify({'error': 'No games provided'}), 400
        
        # 提取appid和权重
        appids = []
        weights = []
        for i, game in enumerate(games):
            try:
                appid = game['appid']
                weight = game['weight']
                appids.append(appid)
                weights.append(weight)
            except KeyError as ke:
                app.logger.error(f"Error in game {i}: {ke}")
                return jsonify({'error': f'Invalid game data at index {i}: {ke}'}), 400
        
        # 获取全部推荐结果
        total_recommendations = rec_system.get_hybrid_recommendations(appids, k=100, weights=weights, 
                                                                      use_content_based=USE_CONTENT_BASED, 
                                                                      use_collaborative=USE_COLLABORATIVE,
                                                                      game_weight=game_weight,
                                                                      review_weight=review_weight)
        
        # 检查是否返回了错误信息
        if isinstance(total_recommendations, str):
            return jsonify({'error': total_recommendations}), 400
        
        # 计算分页
        start = (page - 1) * per_page
        end = start + per_page
        paginated_recommendations = total_recommendations[start:end]
        
        # 转换为前端需要的格式，先返回基本信息
        results = []
        for i, rec in enumerate(paginated_recommendations):
            try:
                # 处理发行日期
                release_date = rec.get('release_date')
                if isinstance(release_date, float) and np.isnan(release_date):
                    release_date = None
                elif release_date is not None:
                    release_date = str(release_date)
                
                # 处理short_description
                short_description = rec.get('short_description', '')
                if isinstance(short_description, float) and np.isnan(short_description):
                    short_description = ""
                
                # 返回基本信息，Steam API数据后续异步获取
                results.append({
                    'appid': rec.get('appid'),
                    'name': rec.get('name'),
                    'similarity': rec.get('similarity'),
                    'short_description': short_description,
                    'cover': rec.get('header_image'),
                    'release_date': release_date,
                    'genres': rec.get('genres', []),
                    # 初始空值，后续通过异步请求获取
                    'user_score_label': None,
                    'user_score_percentage': 0.0,
                    'total_reviews': 0,
                    'positive_reviews': 0,
                    'negative_reviews': 0,
                    'price': '加载中...',
                    'is_free': False
                })
            except Exception as e:
                app.logger.error(f"Error processing recommendation {i}: {e}")
                import traceback
                app.logger.error(traceback.format_exc())
                return jsonify({'error': f'Error processing recommendation {i}: {e}'}), 500
        
        # 转换numpy类型为Python原生类型，确保JSON可序列化
        results = convert_numpy_types(results)
        
        return jsonify({
            'recommendations': results,
            'total': len(total_recommendations),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(total_recommendations) + per_page - 1) // per_page
        })
    
    except Exception as e:
        app.logger.error(f"Error in recommend: {e}")
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/game-details', methods=['POST'])
def game_details():
    """获取游戏详细信息"""
    try:
        data = request.get_json()
        appids = data.get('appids', [])
        
        if not appids:
            return jsonify({'error': 'No appids provided'}), 400
        
        # 批量获取游戏详细信息
        details = {}
        for appid in appids:
            try:
                # 获取价格信息
                price_data = get_steam_price(appid)
                
                # 获取评分信息
                reviews_score = get_steam_reviews_score(appid)
                
                details[appid] = {
                    'price': price_data['price'],
                    'is_free': price_data['is_free'],
                    'user_score_label': reviews_score['user_score_label'],
                    'user_score_percentage': reviews_score['positive_rate'],
                    'total_reviews': reviews_score['total_reviews'],
                    'positive_reviews': reviews_score['positive_reviews'],
                    'negative_reviews': reviews_score['negative_reviews']
                }
            except Exception as e:
                app.logger.error(f"Error getting details for appid {appid}: {e}")
                continue
        
        return jsonify({'details': details})
    
    except Exception as e:
        app.logger.error(f"Error in game_details: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/games', methods=['GET'])
def get_games():
    """获取游戏列表（用于搜索）"""
    try:
        query = request.args.get('q', '').lower()
        limit = int(request.args.get('limit', 10))
        
        games_df = rec_system.data_loader.applications_df
        filtered_games = pd.DataFrame()
        
        # 检测查询是否为纯数字（游戏ID）
        if query.isdigit():
            appid = int(query)
            # 优先匹配精确的appid
            appid_match = games_df[games_df['appid'] == appid]
            if not appid_match.empty:
                filtered_games = appid_match
            else:
                # 如果appid没有匹配到，继续进行名称搜索
                filtered_games = games_df[games_df['name_lower'].str.contains(query, na=False)]
        else:
            # 使用预创建的name_lower列进行快速搜索
            filtered_games = games_df[games_df['name_lower'].str.contains(query, na=False)]
        
        # 限制返回数量并按appid排序（确保结果稳定）
        filtered_games = filtered_games.sort_values('appid').head(limit)
        
        # 转换为前端需要的格式
        games = []
        for _, game in filtered_games.iterrows():
            # 直接使用数据集提供的header_image字段
            cover_url = game['header_image'] if pd.notna(game['header_image']) else f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{game['appid']}/header.jpg"
            
            games.append({
                'appid': game['appid'],
                'name': game['name'],
                'cover': cover_url
            })
        
        return jsonify({'games': games})
    
    except Exception as e:
        app.logger.error(f"Error in get_games: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/price', methods=['GET'])
def get_price():
    """获取游戏价格"""
    try:
        appid = request.args.get('appid')
        if not appid or not appid.isdigit():
            return jsonify({'error': 'Invalid appid'}), 400
        
        appid = int(appid)
        price_data = get_steam_price(appid)
        
        return jsonify(price_data)
    
    except Exception as e:
        app.logger.error(f"Error in get_price: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    """返回前端主页面"""
    return send_file(os.path.join(app.static_folder, 'game_recommender.html'))

@app.route('/game_recommender.html')
def game_recommender():
    """返回前端主页面"""
    return send_file(os.path.join(app.static_folder, 'game_recommender.html'))

@app.route('/favicon.ico')
def favicon():
    """处理favicon.ico请求"""
    return send_file(os.path.join(app.static_folder, 'favicon.ico'), mimetype='image/x-icon')

if __name__ == '__main__':
    # 关闭调试模式的自动重载功能，避免频繁重启
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)