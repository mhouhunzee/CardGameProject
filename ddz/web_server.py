"""
斗地主Web服务器
提供人机对弈界面，AI辅助决策
"""
import os
import glob
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

from config import (
    MODEL_DIR, WEB_HOST, WEB_PORT, WEB_DEBUG,
    POSITION_FIRST, POSITION_SECOND, POSITION_THIRD,
    ALL_CARDS
)
from model import DouDiZhuAgent
from card_utils import CardPattern, is_valid_play, get_legal_plays

app = Flask(__name__)
CORS(app)

# 全局状态
game_state = {
    'started': False,
    'player_position': None,  # 0=头叫, 1=二叫, 2=三叫
    'player_hand': [],
    'base_cards': [],
    'landlord': None,
    'bids': {0: None, 1: None, 2: None},
    'current_player': None,
    'last_play': None,
    'hands': {0: [], 1: [], 2: []},  # 所有玩家的手牌（用于AI推理）
    'play_history': [],
    'game_over': False,
    'winner': None
}

# AI Agent
agents = {
    0: None,  # 头叫
    1: None,  # 二叫
    2: None   # 三叫
}


def load_agents():
    """加载三个Agent的模型"""
    global agents
    
    for pos, name in [(0, 'A'), (1, 'B'), (2, 'C')]:
        agent = DouDiZhuAgent(pos)
        
        # 首先尝试加载 latest 模型
        model_path = os.path.join(MODEL_DIR, f"agent_{name}_latest.pth")
        
        if not os.path.exists(model_path):
            # 如果没有 latest，寻找最新的 cycle 模型
            pattern = os.path.join(MODEL_DIR, f"agent_{name}_cycle_*.pth")
            cycle_files = glob.glob(pattern)
            
            if cycle_files:
                # 按文件名排序，取最后一个（最新的cycle）
                cycle_files.sort()
                model_path = cycle_files[-1]
        
        if os.path.exists(model_path):
            try:
                agent.load(model_path)
                print(f"[加载] Agent {name}: {model_path}")
            except Exception as e:
                print(f"[错误] 加载Agent {name}失败: {e}")
        else:
            print(f"[警告] 未找到Agent {name}模型")
        
        agents[pos] = agent


def parse_hand(hand_str):
    """解析手牌字符串"""
    hand_str = hand_str.upper().replace(' ', '').replace(',', '')
    return list(hand_str)


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/start', methods=['POST'])
def start_game():
    """开始游戏"""
    global game_state
    
    data = request.json
    player_hand_str = data.get('hand', '')
    player_position = data.get('position', 0)  # 0, 1, 2
    
    # 解析手牌
    player_hand = parse_hand(player_hand_str)
    
    if len(player_hand) != 17:
        return jsonify({'error': '手牌数量必须是17张'}), 400
    
    # 重置游戏状态
    game_state = {
        'started': True,
        'player_position': player_position,
        'player_hand': player_hand,
        'base_cards': [],
        'landlord': None,
        'bids': {0: None, 1: None, 2: None},
        'current_player': 0,  # 从头叫开始叫分
        'last_play': None,
        'pass_count': 0,  # 连续PASS计数
        'hands': {0: [], 1: [], 2: []},
        'play_history': [],
        'game_over': False,
        'winner': None
    }
    
    game_state['hands'][player_position] = player_hand
    
    # 如果是头叫，AI建议叫分
    bid_suggestion = None
    if player_position == 0 and agents[0]:
        bid_suggestion = int(agents[0].select_bid(player_hand, epsilon=0.0))
    
    return jsonify({
        'success': True,
        'player_position': player_position,
        'player_hand': player_hand,
        'current_player': 0,  # 当前轮到叫分的玩家
        'bid_suggestion': bid_suggestion,
        'message': '请输入其他玩家的叫分'
    })


@app.route('/api/bid', methods=['POST'])
def make_bid():
    """叫分"""
    global game_state
    
    if not game_state['started']:
        return jsonify({'error': '游戏未开始'}), 400
    
    data = request.json
    position = data.get('position', 0)
    bid = data.get('bid', 0)  # 0, 1, 2, 3
    
    game_state['bids'][position] = bid
    
    # 检查叫分是否结束
    bids_list = [game_state['bids'][i] for i in [0, 1, 2]]
    
    # 如果有人叫3分，直接成交
    if 3 in bids_list:
        landlord = bids_list.index(3)
        game_state['landlord'] = landlord
        game_state['current_player'] = landlord  # 地主先出牌
        
        return jsonify({
            'success': True,
            'bidding_complete': True,
            'landlord': landlord,
            'final_bid': 3,
            'message': f'玩家{landlord}叫3分成为地主，请输入底牌'
        })
    
    # 如果都叫了分，确定地主
    if all(b is not None for b in bids_list):
        max_bid = max(bids_list)
        if max_bid == 0:
            # 流局
            return jsonify({
                'success': True,
                'bidding_complete': True,
                'landlord': None,
                'final_bid': 0,
                'message': '流局，无人叫分'
            })
        
        # 最高叫分者成为地主
        landlord = bids_list.index(max_bid)
        game_state['landlord'] = landlord
        game_state['current_player'] = landlord
        
        return jsonify({
            'success': True,
            'bidding_complete': True,
            'landlord': landlord,
            'final_bid': max_bid,
            'message': f'玩家{landlord}以{max_bid}分成为地主，请输入底牌'
        })
    
    # 叫分继续，AI建议
    next_position = (position + 1) % 3
    bid_suggestion = None
    
    if agents[next_position]:
        # 简化：假设AI知道前面人的叫分
        bid_suggestion = int(agents[next_position].select_bid(
            game_state['hands'].get(next_position, []), epsilon=0.0
        ))
    
    return jsonify({
        'success': True,
        'bidding_complete': False,
        'next_player': next_position,
        'bid_suggestion': bid_suggestion,
        'current_bids': game_state['bids']
    })


@app.route('/api/base_cards', methods=['POST'])
def set_base_cards():
    """设置底牌"""
    global game_state
    
    data = request.json
    base_cards_str = data.get('base_cards', '')
    landlord = data.get('landlord')  # 前端传递的地主位置
    base_cards = parse_hand(base_cards_str)
    
    if len(base_cards) != 3:
        return jsonify({'error': '底牌必须是3张'}), 400
    
    # 如果前端传递了地主，使用前端的地主；否则使用已存储的
    if landlord is not None:
        game_state['landlord'] = landlord
    else:
        landlord = game_state['landlord']
    
    game_state['base_cards'] = base_cards
    
    # 地主获得底牌
    if landlord is not None:
        game_state['hands'][landlord].extend(base_cards)
        if game_state['player_position'] == landlord:
            game_state['player_hand'].extend(base_cards)
    
    return jsonify({
        'success': True,
        'base_cards': base_cards,
        'landlord': landlord,
        'current_player': landlord,  # 地主先出牌
        'message': '游戏开始，地主先出牌'
    })


@app.route('/api/suggest_play', methods=['POST'])
def suggest_play():
    """AI建议出牌"""
    global game_state
    
    if not game_state['started'] or game_state['game_over']:
        return jsonify({'error': '游戏未开始或已结束'}), 400
    
    player_position = game_state['player_position']
    agent = agents[player_position]
    
    if not agent:
        return jsonify({'error': 'AI模型未加载'}), 500
    
    # 初始化出牌状态（70维向量）
    agent.init_play_state(game_state['player_hand'], player_position)
    
    # 获取上家出牌
    last_pattern = CardPattern(game_state['last_play']) if game_state['last_play'] else None
    
    # 获取合法动作
    legal_plays = get_legal_plays(
        game_state['player_hand'],
        last_pattern
    )
    
    # AI建议 - 传入last_pattern让AI知道上家出了什么
    suggestion = agent.select_play(legal_plays, epsilon=0.0, last_play=last_pattern)
    
    # 判断是否只能PASS（没有其他合法出牌选择）
    can_only_pass = len(legal_plays) == 1 and legal_plays[0] == 'PASS'
    
    return jsonify({
        'success': True,
        'suggestion': suggestion,
        'can_only_pass': can_only_pass,  # 是否只能PASS
        'current_hand': game_state['player_hand'],
        'last_play': game_state['last_play']
    })


@app.route('/api/play', methods=['POST'])
def make_play():
    """玩家出牌"""
    global game_state
    
    if not game_state['started'] or game_state['game_over']:
        return jsonify({'error': '游戏未开始或已结束'}), 400
    
    data = request.json
    play = data.get('play', 'PASS').upper()
    
    player_position = game_state['player_position']
    
    # 检查是否轮到玩家
    if game_state['current_player'] != player_position:
        return jsonify({'error': '不是你的回合'}), 400
    
    # 检查出牌合法性
    last_pattern = CardPattern(game_state['last_play']) if game_state['last_play'] else None
    is_valid, msg = is_valid_play(game_state['player_hand'], play, last_pattern)
    
    if not is_valid:
        return jsonify({'error': msg}), 400
    
    # 跟踪连续PASS次数
    if play == 'PASS':
        game_state['pass_count'] = game_state.get('pass_count', 0) + 1
    else:
        # 从手牌中移除
        from card_utils import remove_cards
        game_state['player_hand'] = remove_cards(game_state['player_hand'], list(play))
        game_state['hands'][player_position] = game_state['player_hand']
        game_state['last_play'] = play
        game_state['pass_count'] = 0
    
    # 记录历史
    game_state['play_history'].append({
        'player': player_position,
        'play': play
    })
    
    # 检查是否连续两人PASS（新一轮开始）
    new_round = False
    if game_state['pass_count'] >= 2:
        game_state['last_play'] = None
        game_state['pass_count'] = 0
        new_round = True
        print(f"[游戏] 连续两人PASS，新一轮开始")
    
    # 检查是否获胜
    if len(game_state['player_hand']) == 0:
        game_state['game_over'] = True
        game_state['winner'] = 'landlord' if player_position == game_state['landlord'] else 'farmers'
        
        return jsonify({
            'success': True,
            'game_over': True,
            'winner': game_state['winner'],
            'message': '游戏结束！'
        })
    
    # 切换到下一个玩家
    game_state['current_player'] = (game_state['current_player'] + 1) % 3
    
    return jsonify({
        'success': True,
        'play': play,
        'remaining_hand': game_state['player_hand'],
        'next_player': game_state['current_player'],
        'game_over': False,
        'new_round': new_round,
        'last_play': game_state['last_play']
    })


@app.route('/api/opponent_play', methods=['POST'])
def record_opponent_play():
    """记录对手出牌"""
    global game_state
    
    data = request.json
    opponent_position = data.get('position')
    if opponent_position is None:
        return jsonify({'error': 'position is required'}), 400
    opponent_position = int(opponent_position)
    play = data.get('play', 'PASS').upper()
    
    # 记录对手出牌
    game_state['play_history'].append({
        'player': opponent_position,
        'play': play
    })
    
    # 跟踪连续PASS次数
    if play == 'PASS':
        game_state['pass_count'] = game_state.get('pass_count', 0) + 1
    else:
        game_state['last_play'] = play
        game_state['pass_count'] = 0
    
    # 检查是否连续两人PASS（新一轮开始）
    if game_state['pass_count'] >= 2:
        game_state['last_play'] = None
        game_state['pass_count'] = 0
        print(f"[游戏] 连续两人PASS，新一轮开始，玩家可以自由出牌")
    
    # 更新当前玩家
    game_state['current_player'] = (opponent_position + 1) % 3
    
    # 检查是否轮到玩家
    is_player_turn = (game_state['current_player'] == game_state['player_position'])
    
    return jsonify({
        'success': True,
        'is_player_turn': is_player_turn,
        'current_player': game_state['current_player'],
        'last_play': game_state['last_play'],
        'new_round': game_state['last_play'] is None  # 告诉前端这是新一轮
    })


@app.route('/api/state', methods=['GET'])
def get_state():
    """获取当前游戏状态"""
    return jsonify({
        'started': game_state['started'],
        'game_state': game_state if game_state['started'] else None
    })


if __name__ == '__main__':
    print("="*60)
    print("启动斗地主Web服务器")
    print("="*60)
    
    # 加载AI模型
    load_agents()
    
    print(f"\n请在浏览器中访问: http://localhost:{WEB_PORT}")
    print("="*60)
    
    app.run(host=WEB_HOST, port=WEB_PORT, debug=WEB_DEBUG)
