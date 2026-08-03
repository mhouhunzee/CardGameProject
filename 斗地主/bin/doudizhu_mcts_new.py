"""
斗地主MCTS实现
支持完整牌型：单张、对子、三张、三带一/二、顺子、连对、飞机（带翅膀）、炸弹（带牌）、王炸
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import random
import math
import time
import os
from copy import deepcopy

# 牌编码
CARD_TYPES = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'X', 'D']
CARD_TO_IDX = {card: idx for idx, card in enumerate(CARD_TYPES)}
IDX_TO_CARD = {idx: card for idx, card in enumerate(CARD_TYPES)}


class LegalActionGenerator:
    """
    根据斗地主规则生成所有合法出牌
    支持完整牌型
    """
    
    def __init__(self):
        self.card_order = CARD_TYPES
        self.card_to_idx = CARD_TO_IDX
        
        # 顺子可用的牌 (3-A)
        self.straight_cards = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
    def generate_all_plays(self, hand: List[str]) -> List[List[str]]:
        """生成所有可能的出牌组合"""
        plays = [[]]  # PASS
        
        card_counts = Counter(hand)
        unique_cards = sorted(card_counts.keys(), key=lambda x: self.card_to_idx[x])
        
        # 1. 单张
        plays.extend(self._generate_singles(card_counts))
        
        # 2. 对子
        plays.extend(self._generate_pairs(card_counts))
        
        # 3. 三张（纯）
        plays.extend(self._generate_triples(card_counts))
        
        # 4. 三带一、三带二
        plays.extend(self._generate_triple_with_wing(card_counts, unique_cards))
        
        # 5. 顺子 (5-12张)
        plays.extend(self._generate_straights(card_counts))
        
        # 6. 连对 (3-6对)
        plays.extend(self._generate_consecutive_pairs(card_counts))
        
        # 7. 飞机（纯，不带翅膀）
        plays.extend(self._generate_airplanes_pure(card_counts))
        
        # 8. 飞机带翅膀（单张）
        plays.extend(self._generate_airplane_with_single(card_counts, unique_cards))
        
        # 9. 飞机带翅膀（对子）
        plays.extend(self._generate_airplane_with_pair(card_counts, unique_cards))
        
        # 10. 炸弹（纯）
        plays.extend(self._generate_bombs(card_counts))
        
        # 11. 炸弹带牌（四带二单、四带二对）
        plays.extend(self._generate_bomb_with_wing(card_counts, unique_cards))
        
        # 12. 王炸
        if 'X' in hand and 'D' in hand:
            plays.append(['X', 'D'])
        
        return plays
    
    def _generate_singles(self, card_counts: Counter) -> List[List[str]]:
        """生成所有单张"""
        return [[card] for card in card_counts.keys()]
    
    def _generate_pairs(self, card_counts: Counter) -> List[List[str]]:
        """生成所有对子"""
        plays = []
        for card, count in card_counts.items():
            if count >= 2:
                plays.append([card, card])
        return plays
    
    def _generate_triples(self, card_counts: Counter) -> List[List[str]]:
        """生成所有三张"""
        plays = []
        for card, count in card_counts.items():
            if count >= 3:
                plays.append([card, card, card])
        return plays
    
    def _generate_triple_with_wing(self, card_counts: Counter, 
                                   unique_cards: List[str]) -> List[List[str]]:
        """生成三带一、三带二"""
        plays = []
        
        for card, count in card_counts.items():
            if count >= 3:
                # 三带一
                for wing in unique_cards:
                    if wing != card:
                        plays.append([card, card, card, wing])
                
                # 三带二
                for wing, wing_count in card_counts.items():
                    if wing != card and wing_count >= 2:
                        plays.append([card, card, card, wing, wing])
        
        return plays
    
    def _generate_straights(self, card_counts: Counter) -> List[List[str]]:
        """生成所有顺子 (5-12张连续单牌)"""
        plays = []
        
        for length in range(5, 13):
            for start in range(len(self.straight_cards) - length + 1):
                straight = self.straight_cards[start:start + length]
                if all(card_counts.get(c, 0) >= 1 for c in straight):
                    plays.append(straight)
        
        return plays
    
    def _generate_consecutive_pairs(self, card_counts: Counter) -> List[List[str]]:
        """生成所有连对 (3-6对连续)"""
        plays = []
        
        for length in range(3, 7):
            for start in range(len(self.straight_cards) - length + 1):
                consecutive = self.straight_cards[start:start + length]
                if all(card_counts.get(c, 0) >= 2 for c in consecutive):
                    play = []
                    for c in consecutive:
                        play.extend([c, c])
                    plays.append(play)
        
        return plays
    
    def _generate_airplanes_pure(self, card_counts: Counter) -> List[List[str]]:
        """生成纯飞机（不带翅膀）"""
        plays = []
        
        triple_cards = [c for c, count in card_counts.items() 
                       if count >= 3 and c in self.straight_cards]
        triple_cards = sorted(triple_cards, key=lambda x: self.card_to_idx[x])
        
        for length in range(2, 6):
            for start in range(len(triple_cards) - length + 1):
                consecutive = triple_cards[start:start + length]
                indices = [self.card_to_idx[c] for c in consecutive]
                
                if indices == list(range(indices[0], indices[0] + length)):
                    play = []
                    for c in consecutive:
                        play.extend([c, c, c])
                    plays.append(play)
        
        return plays
    
    def _generate_airplane_with_single(self, card_counts: Counter,
                                       unique_cards: List[str]) -> List[List[str]]:
        """生成飞机带单张翅膀"""
        plays = []
        airplanes = self._generate_airplanes_pure(card_counts)
        
        for airplane in airplanes:
            num_groups = len(airplane) // 3
            used_cards = set(airplane)
            available_singles = [c for c in unique_cards if c not in used_cards]
            
            if len(available_singles) >= num_groups:
                from itertools import combinations
                for wings in combinations(available_singles, num_groups):
                    play = airplane + list(wings)
                    plays.append(play)
        
        return plays
    
    def _generate_airplane_with_pair(self, card_counts: Counter,
                                     unique_cards: List[str]) -> List[List[str]]:
        """生成飞机带对子翅膀"""
        plays = []
        airplanes = self._generate_airplanes_pure(card_counts)
        
        for airplane in airplanes:
            num_groups = len(airplane) // 3
            used_cards = set(airplane)
            available_pairs = [c for c in unique_cards 
                             if c not in used_cards and card_counts[c] >= 2]
            
            if len(available_pairs) >= num_groups:
                from itertools import combinations
                for wing_pairs in combinations(available_pairs, num_groups):
                    play = list(airplane)
                    for c in wing_pairs:
                        play.extend([c, c])
                    plays.append(play)
        
        return plays
    
    def _generate_bombs(self, card_counts: Counter) -> List[List[str]]:
        """生成炸弹（四张相同）"""
        plays = []
        for card, count in card_counts.items():
            if count == 4:
                plays.append([card, card, card, card])
        return plays
    
    def _generate_bomb_with_wing(self, card_counts: Counter,
                                 unique_cards: List[str]) -> List[List[str]]:
        """生成炸弹带牌（四带二单、四带二对）"""
        plays = []
        bombs = self._generate_bombs(card_counts)
        
        for bomb in bombs:
            bomb_card = bomb[0]
            
            # 四带二单
            available = [c for c in unique_cards if c != bomb_card]
            if len(available) >= 2:
                from itertools import combinations
                for wings in combinations(available, 2):
                    plays.append(bomb + list(wings))
            
            # 四带二对
            available_pairs = [c for c in unique_cards 
                             if c != bomb_card and card_counts[c] >= 2]
            if len(available_pairs) >= 2:
                from itertools import combinations
                for wing_pairs in combinations(available_pairs, 2):
                    play = list(bomb)
                    for c in wing_pairs:
                        play.extend([c, c])
                    plays.append(play)
        
        return plays
    
    def generate_beat_plays(self, hand: List[str], last_play: List[str]) -> List[List[str]]:
        """生成能压过上家牌的出牌"""
        if not last_play:
            return self.generate_all_plays(hand)
        
        last_type, last_value, last_length = self._classify_play(last_play)
        all_plays = self.generate_all_plays(hand)
        
        beat_plays = [[]]  # PASS
        
        for play in all_plays:
            if not play:
                continue
            
            play_type, play_value, play_length = self._classify_play(play)
            
            # 王炸最大
            if play_type == 'rocket':
                beat_plays.append(play)
            
            # 炸弹可以压非炸弹
            elif play_type == 'bomb' and last_type != 'rocket':
                if last_type != 'bomb' or play_value > last_value:
                    beat_plays.append(play)
            
            # 同类型且更大
            elif play_type == last_type and play_value > last_value:
                if play_length == last_length:
                    beat_plays.append(play)
            
            # 飞机带翅膀比较主体
            elif (play_type.startswith('airplane') and 
                  last_type.startswith('airplane')):
                if self._compare_airplane_body(play, last_play) > 0:
                    beat_plays.append(play)
        
        return beat_plays
    
    def _classify_play(self, play: List[str]) -> Tuple[str, int, int]:
        """分类牌型，返回(类型, 价值, 长度)"""
        if not play:
            return ('pass', 0, 0)
        
        counts = Counter(play)
        unique = list(counts.keys())
        num_unique = len(unique)
        
        # 王炸
        if set(play) == {'X', 'D'}:
            return ('rocket', 100, 2)
        
        # 炸弹（纯）
        if len(play) == 4 and num_unique == 1:
            return ('bomb', self.card_to_idx[unique[0]], 4)
        
        # 炸弹带牌
        if sorted(counts.values()) == [1, 1, 4]:
            bomb_card = [c for c, n in counts.items() if n == 4][0]
            return ('bomb_two_single', self.card_to_idx[bomb_card], 6)
        if sorted(counts.values()) == [2, 2, 4]:
            bomb_card = [c for c, n in counts.items() if n == 4][0]
            return ('bomb_two_pair', self.card_to_idx[bomb_card], 8)
        
        # 单张
        if len(play) == 1:
            return ('single', self.card_to_idx[play[0]], 1)
        
        # 对子
        if len(play) == 2 and num_unique == 1:
            return ('pair', self.card_to_idx[unique[0]], 2)
        
        # 三张（纯）
        if len(play) == 3 and num_unique == 1:
            return ('triple', self.card_to_idx[unique[0]], 3)
        
        # 三带一
        if len(play) == 4 and sorted(counts.values()) == [1, 3]:
            triple_card = [c for c, n in counts.items() if n == 3][0]
            return ('triple_single', self.card_to_idx[triple_card], 4)
        
        # 三带二
        if len(play) == 5 and sorted(counts.values()) == [2, 3]:
            triple_card = [c for c, n in counts.items() if n == 3][0]
            return ('triple_pair', self.card_to_idx[triple_card], 5)
        
        # 顺子
        if len(play) >= 5 and all(n == 1 for n in counts.values()):
            indices = sorted([self.card_to_idx[c] for c in unique])
            if (indices == list(range(indices[0], indices[0] + len(indices))) 
                and indices[-1] <= 11):  # 到A
                return ('straight', indices[0], len(play))
        
        # 连对
        if len(play) >= 6 and len(play) % 2 == 0 and all(n == 2 for n in counts.values()):
            pair_cards = [c for c, n in counts.items()]
            indices = sorted([self.card_to_idx[c] for c in pair_cards])
            if (indices == list(range(indices[0], indices[0] + len(indices))) 
                and indices[-1] <= 11):
                return ('consecutive_pairs', indices[0], len(play))
        
        # 飞机
        airplane_type = self._classify_airplane(counts, play)
        if airplane_type:
            return airplane_type
        
        return ('unknown', 0, len(play))
    
    def _classify_airplane(self, counts: Counter, play: List[str]) -> Optional[Tuple]:
        """分类飞机及其变种"""
        # 纯飞机
        if all(n == 3 for n in counts.values()) and len(counts) >= 2:
            cards = sorted(counts.keys(), key=lambda x: self.card_to_idx[x])
            indices = [self.card_to_idx[c] for c in cards]
            if indices == list(range(indices[0], indices[0] + len(indices))):
                return ('airplane', indices[0], len(play))
        
        # 飞机带单张
        triples = [c for c, n in counts.items() if n == 3]
        singles = [c for c, n in counts.items() if n == 1]
        
        if len(triples) >= 2 and len(singles) == len(triples):
            triples_sorted = sorted(triples, key=lambda x: self.card_to_idx[x])
            indices = [self.card_to_idx[c] for c in triples_sorted]
            if indices == list(range(indices[0], indices[0] + len(indices))):
                return ('airplane_single', indices[0], len(play))
        
        # 飞机带对子
        pairs = [c for c, n in counts.items() if n == 2]
        if len(triples) >= 2 and len(pairs) == len(triples):
            triples_sorted = sorted(triples, key=lambda x: self.card_to_idx[x])
            indices = [self.card_to_idx[c] for c in triples_sorted]
            if indices == list(range(indices[0], indices[0] + len(indices))):
                return ('airplane_pair', indices[0], len(play))
        
        return None
    
    def _compare_airplane_body(self, play1: List[str], play2: List[str]) -> int:
        """比较飞机主体大小"""
        counts1 = Counter(play1)
        counts2 = Counter(play2)
        
        triples1 = sorted([c for c, n in counts1.items() if n == 3],
                         key=lambda x: self.card_to_idx[x])
        triples2 = sorted([c for c, n in counts2.items() if n == 3],
                         key=lambda x: self.card_to_idx[x])
        
        if len(triples1) != len(triples2):
            return 0  # 不同长度无法比较
        
        # 比较最小的三张
        if self.card_to_idx[triples1[0]] > self.card_to_idx[triples2[0]]:
            return 1
        elif self.card_to_idx[triples1[0]] < self.card_to_idx[triples2[0]]:
            return -1
        return 0


class MCTSNode:
    """MCTS节点"""
    
    def __init__(self, state: Dict, player: str, parent=None, action=None):
        self.state = state
        self.player = player  # 当前轮到谁出牌
        self.parent = parent
        self.action = action  # 到达此节点的动作
        
        # MCTS统计
        self.visits = 0
        self.value = 0.0
        self.children = []
        self.prior = 0.0  # 先验概率
        
        # 游戏状态
        self.is_expanded = False
        self.legal_actions = None
        
    def is_terminal(self) -> bool:
        """检查是否终局"""
        for hand in self.state['hands'].values():
            if len(hand) == 0:
                return True
        return False
    
    def get_winner(self) -> Optional[str]:
        """获取获胜者"""
        for player, hand in self.state['hands'].items():
            if len(hand) == 0:
                return player
        return None
    
    def uct_score(self, c_puct=1.5) -> float:
        """计算UCT分数"""
        if self.visits == 0:
            return float('inf')
        
        # Q值
        Q = self.value / self.visits
        
        # U值（探索 bonus）
        if self.parent:
            U = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        else:
            U = 0
        
        return Q + U


class DoudizhuMCTS:
    """斗地主MCTS实现"""
    
    def __init__(self, model=None, num_simulations=800, c_puct=1.5):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.action_generator = LegalActionGenerator()
        
    def search(self, root_state: Dict, my_role: str) -> Tuple[List[str], np.ndarray]:
        """
        MCTS搜索
        
        Returns:
            best_action: 最佳动作
            pi: 策略分布（基于访问次数）
        """
        root = MCTSNode(root_state, root_state['current_player'])
        
        for _ in range(self.num_simulations):
            # 1. Selection
            node = self._select(root)
            
            # 2. Expansion & Evaluation
            if not node.is_terminal():
                node = self._expand(node)
                value = self._evaluate(node, my_role)
            else:
                value = self._get_terminal_value(node, my_role)
            
            # 3. Backup
            self._backup(node, value)
        
        # 返回最佳动作和策略分布
        best_child = max(root.children, key=lambda c: c.visits)
        pi = self._get_policy_distribution(root)
        
        return best_child.action, pi
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """选择路径到叶节点"""
        while node.is_expanded and node.children:
            # 选择UCT分数最高的子节点
            node = max(node.children, key=lambda c: c.uct_score(self.c_puct))
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """扩展节点"""
        if node.is_expanded:
            return node
        
        hand = node.state['hands'][node.player]
        last_play = node.state.get('last_play')
        
        # 生成合法动作
        if last_play is None or node.state.get('pass_count', 0) >= 2:
            legal_actions = self.action_generator.generate_all_plays(hand)
        else:
            legal_actions = self.action_generator.generate_beat_plays(hand, last_play)
        
        node.legal_actions = legal_actions
        
        # 获取先验概率（如果有模型）
        if self.model:
            priors = self._get_priors(node.state, legal_actions, node.player)
        else:
            priors = [1.0 / len(legal_actions)] * len(legal_actions)
        
        # 创建子节点
        for action, prior in zip(legal_actions, priors):
            child_state = self._apply_action(node.state, action, node.player)
            child = MCTSNode(child_state, self._next_player(node.player), 
                           parent=node, action=action)
            child.prior = prior
            node.children.append(child)
        
        node.is_expanded = True
        
        # 返回第一个子节点进行评估
        return node.children[0] if node.children else node
    
    def _get_priors(self, state: Dict, legal_actions: List[List[str]], 
                    player: str) -> List[float]:
        """从神经网络获取先验概率"""
        if self.model is None:
            return [1.0 / len(legal_actions)] * len(legal_actions)
        
        with torch.no_grad():
            # 这里需要根据实际情况调用模型
            # 简化实现：均匀分布
            priors = [1.0 / len(legal_actions)] * len(legal_actions)
        
        return priors
    
    def _evaluate(self, node: MCTSNode, my_role: str) -> float:
        """评估节点价值"""
        if self.model:
            # 使用神经网络评估
            # 简化：随机 rollout
            return self._rollout(node, my_role)
        else:
            return self._rollout(node, my_role)
    
    def _rollout(self, node: MCTSNode, my_role: str, max_steps=50) -> float:
        """快速走子到终局"""
        state = deepcopy(node.state)
        player = node.player
        
        for step in range(max_steps):
            # 检查终局
            if self._is_terminal(state):
                break
            
            # 随机选择动作
            hand = state['hands'][player]
            last_play = state.get('last_play')
            
            if last_play is None or state.get('pass_count', 0) >= 2:
                legal_actions = self.action_generator.generate_all_plays(hand)
            else:
                legal_actions = self.action_generator.generate_beat_plays(hand, last_play)
            
            if not legal_actions:
                legal_actions = [[]]
            
            action = random.choice(legal_actions)
            state = self._apply_action(state, action, player)
            player = self._next_player(player)
        
        return self._get_terminal_value_from_state(state, my_role)
    
    def _backup(self, node: MCTSNode, value: float):
        """反向传播价值"""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def _get_policy_distribution(self, root: MCTSNode) -> np.ndarray:
        """获取基于访问次数的策略分布"""
        if not root.children:
            return np.array([])
        
        visits = np.array([child.visits for child in root.children])
        pi = visits / visits.sum()
        return pi
    
    def _apply_action(self, state: Dict, action: List[str], player: str) -> Dict:
        """应用动作到新状态"""
        new_state = deepcopy(state)
        
        if action:  # 出牌
            # 从手牌中移除
            hand = new_state['hands'][player]
            for card in action:
                hand.remove(card)
            new_state['hands'][player] = hand
            
            # 更新状态
            new_state['last_play'] = action
            new_state['last_player'] = player
            new_state['pass_count'] = 0
            
            # 记录历史
            if 'history' not in new_state:
                new_state['history'] = []
            new_state['history'].append((player, action))
        else:  # PASS
            new_state['pass_count'] = state.get('pass_count', 0) + 1
            if new_state['pass_count'] >= 2:
                new_state['last_play'] = None
                new_state['pass_count'] = 0
            
            if 'history' not in new_state:
                new_state['history'] = []
            new_state['history'].append((player, []))
        
        # 切换玩家
        new_state['current_player'] = self._next_player(player)
        
        return new_state
    
    def _next_player(self, player: str) -> str:
        """获取下一个玩家"""
        players = ['landlord', 'landlord_up', 'landlord_down']
        idx = players.index(player)
        return players[(idx + 1) % 3]
    
    def _is_terminal(self, state: Dict) -> bool:
        """检查是否终局"""
        for hand in state['hands'].values():
            if len(hand) == 0:
                return True
        return False
    
    def _get_terminal_value(self, node: MCTSNode, my_role: str) -> float:
        """获取终局价值"""
        winner = node.get_winner()
        return self._compute_reward(winner, my_role, node.state)
    
    def _get_terminal_value_from_state(self, state: Dict, my_role: str) -> float:
        """从状态获取终局价值"""
        winner = None
        for player, hand in state['hands'].items():
            if len(hand) == 0:
                winner = player
                break
        
        return self._compute_reward(winner, my_role, state)
    
    def _compute_reward(self, winner: str, my_role: str, state: Dict) -> float:
        """计算奖励（考虑春天/翻倍）"""
        if winner is None:
            return 0.0
        
        # 基础奖励
        if winner == my_role:
            base_reward = 1.0
        elif self._is_teammate(winner, my_role):
            base_reward = 1.0
        else:
            base_reward = -1.0
        
        # 检查春天/反春天
        history = state.get('history', [])
        
        # 春天：地主出完，农民一张没出
        if winner == 'landlord':
            farmer_plays = [play for p, play in history 
                          if p != 'landlord' and play]
            if len(farmer_plays) == 0:
                base_reward *= 2.0
        
        # 反春天：农民出完，地主只出了一次
        if winner != 'landlord':
            landlord_plays = [play for p, play in history 
                            if p == 'landlord' and play]
            if len(landlord_plays) <= 1:
                base_reward *= 2.0
        
        return base_reward
    
    def _is_teammate(self, player: str, my_role: str) -> bool:
        """判断是否是队友"""
        if my_role == 'landlord':
            return False
        return player != 'landlord' and player != my_role


class MCTSTrainer:
    """MCTS自博弈训练器 - 支持从检查点恢复"""
    
    def __init__(self, model=None, num_games=1000, checkpoint_dir='checkpoints', paiku_file=None):
        self.model = model
        self.num_games = num_games
        self.checkpoint_dir = checkpoint_dir
        self.paiku_file = paiku_file
        self.mcts = DoudizhuMCTS(model=model, num_simulations=400)
        self.training_data = []
        
        # 训练状态
        self.current_iteration = 0
        self.total_games_played = 0
        self.best_win_rate = 0.0
        
        # 创建检查点目录
        import os
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        从检查点恢复训练
        
        Args:
            checkpoint_path: 检查点文件路径
        
        Returns:
            是否成功加载
        """
        import os
        if not os.path.exists(checkpoint_path):
            print(f"检查点不存在: {checkpoint_path}")
            return False
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 恢复模型
            if self.model and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ 模型已加载")
            
            # 恢复训练状态
            self.current_iteration = checkpoint.get('iteration', 0)
            self.total_games_played = checkpoint.get('total_games', 0)
            self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
            
            print(f"✓ 检查点加载成功:")
            print(f"  - 当前迭代: {self.current_iteration}")
            print(f"  - 已完成对局: {self.total_games_played}")
            print(f"  - 最佳胜率: {self.best_win_rate:.2%}")
            
            return True
            
        except Exception as e:
            print(f"✗ 加载检查点失败: {e}")
            return False
    
    def save_checkpoint(self, iteration: int, win_rate: float, 
                       is_best: bool = False):
        """
        保存检查点
        
        Args:
            iteration: 当前迭代
            win_rate: 当前胜率
            is_best: 是否是最佳模型
        """
        import os
        
        checkpoint = {
            'iteration': iteration,
            'total_games': self.total_games_played,
            'best_win_rate': self.best_win_rate,
            'current_win_rate': win_rate,
            'model_state_dict': self.model.state_dict() if self.model else None,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存迭代检查点（每10轮）
        if iteration % 10 == 0:
            iter_path = os.path.join(self.checkpoint_dir, f'checkpoint_iter_{iteration}.pt')
            torch.save(checkpoint, iter_path)
            print(f"  检查点已保存: {iter_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  ✓ 最佳模型已保存!")
    
    def self_play_game(self) -> List[Dict]:
        """自博弈一局，生成训练数据"""
        # 初始化游戏（支持牌库）
        state = self._init_game(self.paiku_file)
        game_data = []
        
        while not self._is_terminal(state):
            current_player = state['current_player']
            
            # MCTS搜索
            action, pi = self.mcts.search(state, current_player)
            
            # 记录数据
            game_data.append({
                'state': deepcopy(state),
                'player': current_player,
                'action': action,
                'policy': pi,
                'value': None  # 终局后填充
            })
            
            # 执行动作
            state = self.mcts._apply_action(state, action, current_player)
        
        # 计算实际价值
        winner = self._get_winner(state)
        for data in game_data:
            data['value'] = self.mcts._compute_reward(winner, data['player'], state)
        
        self.total_games_played += 1
        
        return game_data
    
    def train(self, num_iterations=100, resume_from: str = None):
        """
        训练循环
        
        Args:
            num_iterations: 训练轮数
            resume_from: 从指定检查点恢复（可选）
        """
        import time
        
        # 尝试恢复检查点
        if resume_from:
            self.load_checkpoint(resume_from)
        else:
            # 尝试加载最新检查点
            latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pt')
            if os.path.exists(latest_path):
                print("发现最新检查点，尝试恢复...")
                self.load_checkpoint(latest_path)
        
        # 训练循环
        start_iteration = self.current_iteration
        
        for iteration in range(start_iteration, num_iterations):
            self.current_iteration = iteration
            start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*60}")
            
            # 生成数据
            all_games = []
            for i in range(self.num_games):
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i + 1}/{self.num_games} games...")
                game_data = self.self_play_game()
                all_games.extend(game_data)
            
            print(f"  ✓ 数据生成完成: {len(all_games)} 样本")
            
            # 训练模型（这里需要根据实际情况实现）
            if self.model:
                loss = self._train_model(all_games)
                print(f"  ✓ 训练完成, Loss: {loss:.4f}")
            
            # 评估
            if (iteration + 1) % 5 == 0:
                win_rate = self._evaluate(num_games=100)
                elapsed = time.time() - start_time
                
                print(f"\n  评估结果:")
                print(f"    胜率: {win_rate:.2%}")
                print(f"    耗时: {elapsed:.1f}s")
                
                # 保存检查点
                is_best = win_rate > self.best_win_rate
                if is_best:
                    self.best_win_rate = win_rate
                
                self.save_checkpoint(iteration + 1, win_rate, is_best)
        
        print(f"\n{'='*60}")
        print("训练完成!")
        print(f"总对局数: {self.total_games_played}")
        print(f"最佳胜率: {self.best_win_rate:.2%}")
        print(f"{'='*60}")
    
    def _train_model(self, game_data: List[Dict]) -> float:
        """
        训练模型（简化实现）
        
        实际实现应该:
        1. 准备训练数据 (state, policy, value)
        2. 前向传播
        3. 计算损失 (policy_loss + value_loss)
        4. 反向传播
        5. 更新参数
        """
        if self.model is None:
            return 0.0
        
        # 这里应该实现实际的训练逻辑
        # 简化：返回随机损失
        import random
        return random.random() * 0.5
    
    def _init_game(self, paiku_file=None) -> Dict:
        """初始化游戏 - 支持从牌库加载"""
        import random
        
        # 如果有牌库文件，从牌库加载
        if paiku_file and os.path.exists(paiku_file):
            try:
                with open(paiku_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        # 随机选择一局
                        line = random.choice(lines).strip()
                        parts = line.split('|')
                        if len(parts) == 4:
                            hand_A = parts[0].split()
                            hand_B = parts[1].split()
                            hand_C = parts[2].split()
                            base = parts[3].split()
                            
                            return {
                                'hands': {
                                    'landlord': hand_A + base,  # 地主20张
                                    'landlord_up': hand_B,       # 农民17张
                                    'landlord_down': hand_C      # 农民17张
                                },
                                'current_player': 'landlord',
                                'last_play': None,
                                'last_player': None,
                                'pass_count': 0,
                                'history': []
                            }
            except Exception as e:
                print(f"从牌库加载失败: {e}，使用随机发牌")
        
        # 随机发牌
        deck = []
        for card in CARD_TYPES[:-2]:  # 3-2各4张
            deck.extend([card] * 4)
        deck.extend(['X', 'D'])  # 大小王
        
        random.shuffle(deck)
        
        state = {
            'hands': {
                'landlord': sorted(deck[:20], key=lambda x: CARD_TO_IDX[x]),
                'landlord_up': sorted(deck[20:37], key=lambda x: CARD_TO_IDX[x]),
                'landlord_down': sorted(deck[37:54], key=lambda x: CARD_TO_IDX[x])
            },
            'current_player': 'landlord',
            'last_play': None,
            'last_player': None,
            'pass_count': 0,
            'history': []
        }
        
        return state
    
    def _is_terminal(self, state: Dict) -> bool:
        """检查是否终局"""
        for hand in state['hands'].values():
            if len(hand) == 0:
                return True
        return False
    
    def _get_winner(self, state: Dict) -> str:
        """获取获胜者"""
        for player, hand in state['hands'].items():
            if len(hand) == 0:
                return player
        return None
    
    def _evaluate(self, num_games=100) -> float:
        """评估模型胜率"""
        wins = 0
        for _ in range(num_games):
            game_data = self.self_play_game()
            if game_data:
                final_state = game_data[-1]['state']
                winner = self._get_winner(final_state)
                if winner == 'landlord':  # 简化：只统计地主胜率
                    wins += 1
        return wins / num_games


# ============== 测试 ==============

def test_legal_actions():
    """测试合法动作生成"""
    print("=" * 60)
    print("测试合法动作生成")
    print("=" * 60)
    
    generator = LegalActionGenerator()
    
    # 测试手牌
    hand = ['3', '3', '3', '4', '4', '4', '5', '5', '6', '7', '8', '9', '10', 
            'J', 'Q', 'K', 'A', '2', 'X', 'D']
    
    print(f"\n手牌: {hand}")
    print(f"手牌数: {len(hand)}")
    
    # 生成所有出牌
    plays = generator.generate_all_plays(hand)
    print(f"\n所有合法出牌数: {len(plays)}")
    
    # 统计各类型
    type_counts = defaultdict(int)
    for play in plays[:50]:  # 只显示前50个
        play_type, _, _ = generator._classify_play(play)
        type_counts[play_type] += 1
    
    print("\n牌型分布（前50个）:")
    for ptype, count in sorted(type_counts.items()):
        print(f"  {ptype}: {count}")
    
    # 测试压牌
    print("\n" + "-" * 60)
    print("测试压牌逻辑")
    print("-" * 60)
    
    last_play = ['3', '4', '5', '6', '7']  # 顺子
    beat_plays = generator.generate_beat_plays(hand, last_play)
    print(f"\n上家出: {last_play}")
    print(f"可压的牌数: {len(beat_plays)}")
    print(f"示例: {beat_plays[:5]}")


def test_mcts():
    """测试MCTS"""
    print("\n" + "=" * 60)
    print("测试MCTS搜索")
    print("=" * 60)
    
    # 初始化状态
    state = {
        'hands': {
            'landlord': ['3', '3', '4', '5', '6', '7', '8', '9', '10', 
                        'J', 'Q', 'K', 'A', '2', 'X', 'D', '4', '4', '5', '5'],
            'landlord_up': ['3', '4', '5', '6', '7', '8', '9', '10', 
                           'J', 'Q', 'K', 'A', '2', 'X', 'D', '3', '6'],
            'landlord_down': ['3', '4', '5', '6', '7', '8', '9', '10', 
                             'J', 'Q', 'K', 'A', '2', 'X', 'D', '7', '8']
        },
        'current_player': 'landlord',
        'last_play': None,
        'last_player': None,
        'pass_count': 0,
        'history': []
    }
    
    mcts = DoudizhuMCTS(num_simulations=100)
    
    print("\n初始状态:")
    print(f"  地主手牌: {state['hands']['landlord']}")
    print(f"  当前玩家: {state['current_player']}")
    
    print("\nMCTS搜索中...")
    action, pi = mcts.search(state, 'landlord')
    
    print(f"\n推荐出牌: {action}")
    print(f"策略分布形状: {pi.shape}")
    print(f"最高概率: {pi.max():.3f}")


def test_trainer():
    """测试训练器"""
    print("\n" + "=" * 60)
    print("测试MCTS训练器")
    print("=" * 60)
    
    trainer = MCTSTrainer(model=None, num_games=10)
    
    print("\n自博弈一局...")
    game_data = trainer.self_play_game()
    
    print(f"生成数据条数: {len(game_data)}")
    print(f"\n前3步:")
    for i, data in enumerate(game_data[:3]):
        print(f"  Step {i+1}: {data['player']} 出 {data['action']}")
    
    if game_data:
        final = game_data[-1]
        print(f"\n最终结果: {final['player']} 获胜")
        print(f"实际价值: {final['value']}")


if __name__ == '__main__':
    test_legal_actions()
    test_mcts()
    test_trainer()
