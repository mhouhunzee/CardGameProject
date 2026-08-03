"""
斗地主WebServer - MCTS完整版本
支持完整牌型和MCTS搜索
"""

from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import json
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入MCTS
from doudizhu_mcts import (
    LegalActionGenerator, DoudizhuMCTS, MCTSTrainer,
    CARD_TYPES, CARD_TO_IDX, IDX_TO_CARD
)

# 尝试导入高级模型
try:
    from doudizhu_model_advanced import (
        AdvancedDouDizhuNet, AdvancedPPOAgent, GameContextExtractor,
        GameResult, QualityScorer
    )
    ADVANCED_MODEL_AVAILABLE = True
    print("✓ 高级模型已加载")
except ImportError as e:
    ADVANCED_MODEL_AVAILABLE = False
    print(f"✗ 高级模型加载失败: {e}")

app = Flask(__name__)

# 游戏状态存储
game_states = {}


class DouDizhuState:
    """斗地主游戏状态管理 - 完整MCTS版本"""

    def __init__(self, session_id):
        self.session_id = session_id
        self.action_generator = LegalActionGenerator()
        self.mcts = None
        self.model = None
        self.reset()

    def reset(self):
        """重置游戏状态"""
        # 我的手牌（初始17张）
        self.my_hand = []
        
        # 我的位置（叫分时的位置）
        self.my_position = None  # 'east', 'south', 'west'
        
        # 地主位置
        self.landlord_position = None
        
        # 底牌
        self.bottom_cards = []
        
        # 当前手牌（如果是地主则包含底牌）
        self.current_hand = []
        
        # 75维状态向量
        self.state_vector = np.zeros(75, dtype=np.float32)
        
        # 出牌历史
        self.history = []
        
        # 每个玩家剩余的牌数
        self.remaining_cards = {
            'landlord': 20,
            'landlord_up': 17,
            'landlord_down': 17
        }
        
        # 每个玩家已出的牌
        self.played_cards = {
            'landlord': [],
            'landlord_up': [],
            'landlord_down': []
        }
        
        # 上一轮出牌
        self.last_play = None
        self.last_player = None
        
        # 连续pass次数
        self.pass_count = 0
        
        # 当前轮到谁
        self.current_player = None
        
        # 游戏是否开始
        self.game_started = False
        
        # 模型是否已加载
        self.model_loaded = False
        
        # 我的角色
        self.my_role = None  # 'landlord', 'landlord_up', 'landlord_down'
        
        # MCTS设置
        self.use_mcts = False
        self.mcts_simulations = 800

    def set_initial_hand(self, hand: List[str], position: str):
        """设置初始手牌和位置"""
        self.my_hand = hand.copy()
        self.my_position = position
        self.current_hand = hand.copy()
        self._update_state_vector()

    def determine_landlord(self, bids: Dict[str, int]):
        """根据叫分确定地主"""
        max_bid = -1
        landlord_pos = None
        
        for pos, bid in bids.items():
            if bid > max_bid:
                max_bid = bid
                landlord_pos = pos
        
        self.landlord_position = landlord_pos
        
        # 确定我的角色
        if self.my_position == landlord_pos:
            self.my_role = 'landlord'
        else:
            positions = ['east', 'south', 'west']
            my_idx = positions.index(self.my_position)
            landlord_idx = positions.index(landlord_pos)
            diff = (my_idx - landlord_idx) % 3
            if diff == 1:
                self.my_role = 'landlord_down'
            else:
                self.my_role = 'landlord_up'
        
        # 地主先出牌
        self.current_player = 'landlord'
        self._update_state_vector()
        return self.my_role

    def set_bottom_cards(self, cards: List[str]):
        """设置底牌"""
        self.bottom_cards = cards.copy()
        
        # 如果我是地主，底牌加入我的手牌
        if self.my_role == 'landlord':
            self.current_hand.extend(cards)
            self.current_hand.sort(key=lambda x: CARD_TO_IDX[x])
            self.remaining_cards['landlord'] = 20
        
        self._update_state_vector()

    def update_play(self, player: str, cards: List[str]):
        """更新出牌"""
        if cards:  # 出牌
            self.played_cards[player].extend(cards)
            self.remaining_cards[player] -= len(cards)
            self.last_play = cards
            self.last_player = player
            self.pass_count = 0
            
            # 如果是我出牌，从手牌中移除
            if player == self.my_role:
                for card in cards:
                    if card in self.current_hand:
                        self.current_hand.remove(card)
        else:  # pass
            self.pass_count += 1
            if self.pass_count >= 2:
                # 连续两人pass，新一轮开始
                self.last_play = None
                self.pass_count = 0
        
        self.history.append((player, cards))
        
        # 切换到下一个玩家
        players = ['landlord', 'landlord_up', 'landlord_down']
        current_idx = players.index(player)
        self.current_player = players[(current_idx + 1) % 3]
        
        self._update_state_vector()

    def _update_state_vector(self):
        """更新75维状态向量"""
        vec = np.zeros(75, dtype=np.float32)
        
        # [0-14]: 我的手牌 (15种牌，每种数量 0-4)
        hand_counter = Counter(self.current_hand)
        for i, card in enumerate(CARD_TYPES):
            vec[i] = hand_counter.get(card, 0) / 4.0
        
        # 确定上家和下家
        if self.my_role == 'landlord':
            up_player = 'landlord_up'
            down_player = 'landlord_down'
        elif self.my_role == 'landlord_up':
            up_player = 'landlord_down'
            down_player = 'landlord'
        else:  # landlord_down
            up_player = 'landlord'
            down_player = 'landlord_up'
        
        # [15-29]: 上家已出的牌
        up_counter = Counter(self.played_cards.get(up_player, []))
        for i, card in enumerate(CARD_TYPES):
            vec[15 + i] = up_counter.get(card, 0) / 4.0
        
        # [30-44]: 下家已出的牌
        down_counter = Counter(self.played_cards.get(down_player, []))
        for i, card in enumerate(CARD_TYPES):
            vec[30 + i] = down_counter.get(card, 0) / 4.0
        
        # [45-59]: 底牌/未知牌分布
        if self.my_role == 'landlord' and self.bottom_cards:
            bottom_counter = Counter(self.bottom_cards)
            for i, card in enumerate(CARD_TYPES):
                vec[45 + i] = bottom_counter.get(card, 0) / 4.0
        else:
            vec[45:60] = 0.2  # 均匀分布假设
        
        # [60]: 我是地主?
        vec[60] = 1.0 if self.my_role == 'landlord' else 0.0
        
        # [61-63]: 当前轮到谁
        if self.current_player:
            if self.current_player == self.my_role:
                vec[61] = 1.0
            elif self.current_player == up_player:
                vec[62] = 1.0
            else:
                vec[63] = 1.0
        
        # [64]: 上轮出牌类型
        if self.last_play:
            play_type = self._get_play_type(self.last_play)
            vec[64] = play_type / 10.0
        
        # [65-67]: 各玩家剩余牌数
        vec[65] = self.remaining_cards.get('landlord', 20) / 20.0
        vec[66] = self.remaining_cards.get('landlord_up', 17) / 17.0
        vec[67] = self.remaining_cards.get('landlord_down', 17) / 17.0
        
        # [68-74]: 其他特征
        vec[68] = len(self.history) / 50.0
        vec[69] = self.pass_count / 2.0
        vec[70] = 1.0 if self.last_play else 0.0
        
        self.state_vector = vec

    def _get_play_type(self, play: List[str]) -> int:
        """获取出牌类型编码"""
        if not play:
            return 0
        
        counts = Counter(play)
        unique = list(counts.keys())
        
        # 王炸
        if set(play) == {'X', 'D'}:
            return 10
        
        # 炸弹
        if len(play) == 4 and len(unique) == 1:
            return 9
        
        # 单张
        if len(play) == 1:
            return 1
        
        # 对子
        if len(play) == 2 and len(unique) == 1:
            return 2
        
        # 三张
        if len(play) == 3 and len(unique) == 1:
            return 3
        
        # 三带一
        if len(play) == 4 and sorted(counts.values()) == [1, 3]:
            return 4
        
        # 三带二
        if len(play) == 5 and sorted(counts.values()) == [2, 3]:
            return 5
        
        # 顺子
        if len(play) >= 5 and all(counts[c] == 1 for c in unique):
            return 6
        
        # 连对
        if len(play) >= 6 and len(play) % 2 == 0 and all(counts[c] == 2 for c in unique):
            return 7
        
        # 飞机
        if len(play) >= 6 and all(counts[c] == 3 for c in unique):
            return 8
        
        return 0

    def get_legal_actions(self) -> List[List[str]]:
        """获取当前合法动作"""
        if self.current_player != self.my_role:
            return []
        
        hand = self.current_hand
        
        if self.last_play is None or self.pass_count >= 2:
            # 新一轮，可以出任意牌
            return self.action_generator.generate_all_plays(hand)
        else:
            # 必须压过上家的牌
            return self.action_generator.generate_beat_plays(hand, self.last_play)

    def get_mcts_suggestion(self) -> Dict:
        """使用MCTS获取建议"""
        if not self.mcts:
            # 初始化MCTS
            self.mcts = DoudizhuMCTS(
                model=self.model,
                num_simulations=self.mcts_simulations
            )
        
        # 构建MCTS需要的完整状态
        # 注意：这里需要构建3个玩家的完整手牌，但我们只知道自己的
        # 对于对手，我们使用估计（简化处理）
        state = self._build_mcts_state()
        
        # MCTS搜索
        action, pi = self.mcts.search(state, self.my_role)
        
        # 获取备选方案
        legal_actions = self.get_legal_actions()
        alternatives = []
        if len(legal_actions) > 1:
            import random
            other_actions = [a for a in legal_actions if a != action]
            alternatives = random.sample(other_actions, min(3, len(other_actions)))
        
        return {
            'recommended': action,
            'formatted_recommended': self._format_cards(action),
            'confidence': float(pi.max()) if len(pi) > 0 else 0.5,
            'alternatives': alternatives,
            'mcts_visits': self.mcts_simulations,
            'legal_actions_count': len(legal_actions)
        }

    def _build_mcts_state(self) -> Dict:
        """构建MCTS需要的状态"""
        # 简化：只使用我的手牌，对手手牌为空（MCTS会处理）
        state = {
            'hands': {
                'landlord': self.current_hand.copy() if self.my_role == 'landlord' else [],
                'landlord_up': self.current_hand.copy() if self.my_role == 'landlord_up' else [],
                'landlord_down': self.current_hand.copy() if self.my_role == 'landlord_down' else [],
            },
            'current_player': self.my_role,
            'last_play': self.last_play,
            'last_player': self.last_player,
            'pass_count': self.pass_count,
            'history': self.history.copy()
        }
        return state

    def _format_cards(self, cards: List[str]) -> str:
        """格式化牌列表"""
        if not cards:
            return "PASS"
        
        card_names = {
            'X': '小王', 'D': '大王',
            '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
            '8': '8', '9': '9', '10': '10',
            'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A', '2': '2'
        }
        
        return ' '.join([card_names.get(c, c) for c in cards])

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'my_hand': self.my_hand,
            'my_position': self.my_position,
            'my_role': self.my_role,
            'landlord_position': self.landlord_position,
            'bottom_cards': self.bottom_cards,
            'current_hand': self.current_hand,
            'state_vector': self.state_vector.tolist(),
            'history': self.history,
            'remaining_cards': self.remaining_cards,
            'played_cards': self.played_cards,
            'last_play': self.last_play,
            'last_player': self.last_player,
            'current_player': self.current_player,
            'pass_count': self.pass_count,
            'game_started': self.game_started,
            'model_loaded': self.model_loaded,
            'use_mcts': self.use_mcts,
            'mcts_simulations': self.mcts_simulations
        }


def parse_cards(card_str: str) -> List[str]:
    """解析牌字符串"""
    card_str = card_str.upper().strip()
    
    # 替换中文
    replacements = {
        '三': '3', '四': '4', '五': '5', '六': '6', '七': '7',
        '八': '8', '九': '9', '十': '10',
        '小': 'X', '大': 'D', '王': ''
    }
    for cn, en in replacements.items():
        card_str = card_str.replace(cn, en)
    
    if ' ' in card_str:
        cards = card_str.split()
    elif ',' in card_str:
        cards = card_str.split(',')
    else:
        cards = []
        i = 0
        while i < len(card_str):
            if card_str[i:i+2] == '10':
                cards.append('10')
                i += 2
            elif card_str[i] in CARD_TYPES or card_str[i] in ['X', 'D']:
                cards.append(card_str[i])
                i += 1
            else:
                i += 1
    
    valid_cards = []
    for card in cards:
        card = card.strip()
        if card in CARD_TYPES:
            valid_cards.append(card)
    
    return valid_cards


def format_cards(cards: List[str]) -> str:
    """格式化牌列表"""
    if not cards:
        return "PASS"
    
    card_names = {
        'X': '小王', 'D': '大王',
        '3': '3', '4': '4', '5': '5', '6': '6', '7': '7',
        '8': '8', '9': '9', '10': '10',
        'J': 'J', 'Q': 'Q', 'K': 'K', 'A': 'A', '2': '2'
    }
    
    return ' '.join([card_names.get(c, c) for c in cards])


# ============== Flask Routes ==============

@app.route('/')
def index():
    """主页"""
    return render_template('doudizhu_mcts.html')


@app.route('/api/init', methods=['POST'])
def init_game():
    """初始化游戏"""
    session_id = request.json.get('session_id', 'default')
    
    game_state = DouDizhuState(session_id)
    game_states[session_id] = game_state
    
    return jsonify({
        'success': True,
        'message': '游戏已初始化',
        'state': game_state.to_dict()
    })


@app.route('/api/set_hand', methods=['POST'])
def set_hand():
    """设置初始手牌"""
    session_id = request.json.get('session_id', 'default')
    hand_str = request.json.get('hand', '')
    position = request.json.get('position', 'east')
    
    if session_id not in game_states:
        game_states[session_id] = DouDizhuState(session_id)
    
    game_state = game_states[session_id]
    hand = parse_cards(hand_str)
    
    if len(hand) != 17:
        return jsonify({
            'success': False,
            'message': f'手牌数量错误，应为17张，实际{len(hand)}张',
            'parsed_hand': hand
        })
    
    game_state.set_initial_hand(hand, position)
    
    return jsonify({
        'success': True,
        'message': f'手牌已设置，位置: {position}',
        'hand': hand,
        'formatted_hand': format_cards(hand),
        'state': game_state.to_dict()
    })


@app.route('/api/load_model', methods=['POST'])
def load_model():
    """加载MCTS模型"""
    session_id = request.json.get('session_id', 'default')
    use_mcts = request.json.get('use_mcts', True)
    mcts_simulations = request.json.get('mcts_simulations', 800)
    checkpoint_path = request.json.get('checkpoint_path', None)
    
    if session_id not in game_states:
        return jsonify({
            'success': False,
            'message': '游戏未初始化'
        })
    
    game_state = game_states[session_id]
    
    game_state.use_mcts = use_mcts
    game_state.mcts_simulations = mcts_simulations
    game_state.model_loaded = True
    
    loaded_checkpoint = False
    checkpoint_info = {}
    
    if use_mcts:
        # 尝试加载检查点
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # 创建并加载模型
                if ADVANCED_MODEL_AVAILABLE:
                    game_state.model = AdvancedDouDizhuNet()
                    if checkpoint.get('model_state_dict'):
                        game_state.model.load_state_dict(checkpoint['model_state_dict'])
                        loaded_checkpoint = True
                        checkpoint_info = {
                            'iteration': checkpoint.get('iteration', 0),
                            'win_rate': checkpoint.get('best_win_rate', 0.0)
                        }
                        print(f"✓ 从检查点加载模型: {checkpoint_path}")
            except Exception as e:
                print(f"✗ 加载检查点失败: {e}")
        
        # 初始化MCTS
        game_state.mcts = DoudizhuMCTS(
            model=game_state.model,
            num_simulations=mcts_simulations
        )
        
        if loaded_checkpoint:
            message = f'MCTS已加载（检查点: 迭代{checkpoint_info["iteration"]}, 胜率{checkpoint_info["win_rate"]:.1%}, 模拟: {mcts_simulations}）'
        else:
            message = f'MCTS已加载（模拟次数: {mcts_simulations}）'
    else:
        message = '简单模型已加载'
    
    return jsonify({
        'success': True,
        'message': message,
        'use_mcts': use_mcts,
        'checkpoint_loaded': loaded_checkpoint,
        'state': game_state.to_dict()
    })


@app.route('/api/determine_landlord', methods=['POST'])
def determine_landlord():
    """确定地主"""
    session_id = request.json.get('session_id', 'default')
    bids = request.json.get('bids', {})
    
    if session_id not in game_states:
        return jsonify({
            'success': False,
            'message': '游戏未初始化'
        })
    
    game_state = game_states[session_id]
    my_role = game_state.determine_landlord(bids)
    
    return jsonify({
        'success': True,
        'message': f'地主已确定，你的角色: {my_role}',
        'my_role': my_role,
        'landlord_position': game_state.landlord_position,
        'state': game_state.to_dict()
    })


@app.route('/api/set_bottom_cards', methods=['POST'])
def set_bottom_cards():
    """设置底牌"""
    session_id = request.json.get('session_id', 'default')
    cards_str = request.json.get('cards', '')
    
    if session_id not in game_states:
        return jsonify({
            'success': False,
            'message': '游戏未初始化'
        })
    
    game_state = game_states[session_id]
    cards = parse_cards(cards_str)
    
    if len(cards) != 3:
        return jsonify({
            'success': False,
            'message': f'底牌数量错误，应为3张，实际{len(cards)}张',
            'parsed_cards': cards
        })
    
    game_state.set_bottom_cards(cards)
    game_state.game_started = True
    
    message = f'底牌已设置: {format_cards(cards)}'
    if game_state.my_role == 'landlord':
        message += f'，你的手牌已更新为: {format_cards(game_state.current_hand)}'
    else:
        message += f'，地主({game_state.landlord_position})获得底牌'
    
    return jsonify({
        'success': True,
        'message': message,
        'bottom_cards': cards,
        'current_hand': game_state.current_hand,
        'state': game_state.to_dict()
    })


@app.route('/api/play', methods=['POST'])
def play():
    """记录出牌"""
    session_id = request.json.get('session_id', 'default')
    player = request.json.get('player', '')
    cards_str = request.json.get('cards', '')
    
    if session_id not in game_states:
        return jsonify({
            'success': False,
            'message': '游戏未初始化'
        })
    
    game_state = game_states[session_id]
    
    if not game_state.game_started:
        return jsonify({
            'success': False,
            'message': '游戏尚未开始，请先设置底牌'
        })
    
    cards = parse_cards(cards_str) if cards_str.strip() else []
    
    # 验证出牌
    if player == game_state.my_role and cards:
        hand_counter = Counter(game_state.current_hand)
        play_counter = Counter(cards)
        
        for card, count in play_counter.items():
            if hand_counter.get(card, 0) < count:
                return jsonify({
                    'success': False,
                    'message': f'出牌错误: 你没有足够的 {card}'
                })
    
    # 更新游戏状态
    game_state.update_play(player, cards)
    
    # 获取AI建议
    suggestion = None
    if game_state.current_player == game_state.my_role and game_state.model_loaded:
        if game_state.use_mcts:
            suggestion = game_state.get_mcts_suggestion()
            suggestion['type'] = 'mcts'
    
    return jsonify({
        'success': True,
        'message': f'{player} 出了: {format_cards(cards)}',
        'player': player,
        'cards': cards,
        'formatted_cards': format_cards(cards),
        'current_player': game_state.current_player,
        'is_my_turn': game_state.current_player == game_state.my_role,
        'my_current_hand': game_state.current_hand if player == game_state.my_role else None,
        'my_formatted_hand': format_cards(game_state.current_hand) if player == game_state.my_role else None,
        'suggestion': suggestion,
        'state': game_state.to_dict()
    })


@app.route('/api/get_suggestion', methods=['POST'])
def get_suggestion():
    """获取MCTS建议"""
    session_id = request.json.get('session_id', 'default')
    
    if session_id not in game_states:
        return jsonify({
            'success': False,
            'message': '游戏未初始化'
        })
    
    game_state = game_states[session_id]
    
    if not game_state.model_loaded:
        return jsonify({
            'success': False,
            'message': '模型未加载'
        })
    
    if game_state.current_player != game_state.my_role:
        return jsonify({
            'success': False,
            'message': '当前不是你的回合'
        })
    
    # 使用MCTS获取建议
    suggestion = game_state.get_mcts_suggestion()
    
    return jsonify({
        'success': True,
        'recommended': {
            'cards': suggestion['recommended'],
            'formatted': suggestion['formatted_recommended']
        },
        'confidence': suggestion['confidence'],
        'alternatives': [
            {'cards': alt, 'formatted': format_cards(alt)}
            for alt in suggestion['alternatives']
        ],
        'mcts_visits': suggestion['mcts_visits'],
        'legal_actions_count': suggestion['legal_actions_count'],
        'state_vector': game_state.state_vector.tolist()
    })


@app.route('/api/get_legal_actions', methods=['POST'])
def get_legal_actions():
    """获取当前合法动作"""
    session_id = request.json.get('session_id', 'default')
    
    if session_id not in game_states:
        return jsonify({
            'success': False,
            'message': '游戏未初始化'
        })
    
    game_state = game_states[session_id]
    actions = game_state.get_legal_actions()
    
    # 分类统计
    type_counts = {}
    generator = game_state.action_generator
    for action in actions[:100]:
        ptype, _, _ = generator._classify_play(action)
        type_counts[ptype] = type_counts.get(ptype, 0) + 1
    
    return jsonify({
        'success': True,
        'total_actions': len(actions),
        'type_distribution': type_counts,
        'sample_actions': [
            {'cards': a, 'formatted': format_cards(a)}
            for a in actions[:20]
        ]
    })


@app.route('/api/get_state', methods=['POST'])
def get_state():
    """获取当前状态"""
    session_id = request.json.get('session_id', 'default')
    
    if session_id not in game_states:
        return jsonify({
            'success': False,
            'message': '游戏未初始化'
        })
    
    game_state = game_states[session_id]
    
    return jsonify({
        'success': True,
        'state': game_state.to_dict()
    })


@app.route('/api/reset', methods=['POST'])
def reset_game():
    """重置游戏"""
    session_id = request.json.get('session_id', 'default')
    
    if session_id in game_states:
        del game_states[session_id]
    
    return jsonify({
        'success': True,
        'message': '游戏已重置'
    })


if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    
    print("=" * 60)
    print("斗地主WebServer MCTS版本启动")
    print("=" * 60)
    print("特性:")
    print("  - 完整牌型支持（飞机带翅膀、炸弹带牌）")
    print("  - MCTS蒙特卡洛树搜索")
    print("  - 春天/反春天检测")
    print("  - 75维状态向量")
    print("=" * 60)
    print("访问 http://localhost:5000 开始使用")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
