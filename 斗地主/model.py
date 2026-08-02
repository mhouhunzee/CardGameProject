"""
斗地主神经网络模型

包含两个网络：
1. 叫分网络：输入17维（我的手牌），输出叫分决策
2. 出牌网络：输入75维（三方手牌+位置+炸弹特征），输出出牌决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Optional
from config import (
    BID_INPUT_DIM, PLAY_INPUT_DIM, HIDDEN_SIZE, DROPOUT, 
    BID_OUTPUT_DIM, LEARNING_RATE, CARD_RANK
)
from rules import CardPattern


class BidNet(nn.Module):
    """叫分网络"""
    
    def __init__(self, input_dim=BID_INPUT_DIM, hidden_size=64):
        super(BidNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # 叫分输出（0/1/2/3分）
        self.bid_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, BID_OUTPUT_DIM)
        )
        
        # 价值评估
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        """
        x: [batch, 17] - 我的手牌
        返回: (bid_logits, value)
        """
        features = self.feature_extractor(x)
        bid_logits = self.bid_head(features)
        value = self.value_head(features)
        return bid_logits, value


class PlayNet(nn.Module):
    """出牌网络"""
    
    def __init__(self, input_dim=PLAY_INPUT_DIM, hidden_size=HIDDEN_SIZE):
        super(PlayNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
        )
        
        # 输出新的手牌预测（上家17 + 下家17）和置信度（17）
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 51)  # 上家17 + 下家17 + 置信度17
        )
        
        # 动作类型决策头
        # 0: PASS, 1: 单牌, 2: 对子, 3: 三张, 4: 顺子, 5: 炸弹, 6: 王炸
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_size + 51, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 7)
        )
        
        # 牌型选择头
        self.card_selection_head = nn.Sequential(
            nn.Linear(hidden_size + 51, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 62)  # 15×4 + 2
        )
        
        # 炸弹使用价值评估
        self.bomb_value_head = nn.Sequential(
            nn.Linear(hidden_size + 51, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size + 51, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x, legal_action_types=None):
        """
        x: [batch, 75] - 我的手牌(17) + 剩余牌(17) + 上家预测(17) + 上家置信度(17) + 位置(2) + 炸弹特征(5)
        legal_action_types: 可选，限制合法动作类型
        
        返回: (action_type_logits, card_selection_logits, bomb_value, state_value, 
               new_remaining_cards, new_opponent_pred, new_opponent_confidence)
        """
        features = self.feature_extractor(x)
        
        # 预测新的分布
        prediction_output = self.prediction_head(features)
        
        # 分割输出：剩余牌(17) + 上家预测(17) + 上家置信度(17)
        new_remaining_cards = torch.sigmoid(prediction_output[:, :17])
        new_opponent_pred = torch.sigmoid(prediction_output[:, 17:34])
        new_opponent_confidence = torch.sigmoid(prediction_output[:, 34:])
        
        # 合并特征和预测
        combined = torch.cat([features, new_remaining_cards, new_opponent_pred, new_opponent_confidence], dim=1)
        
        # 动作类型
        action_type_logits = self.action_type_head(combined)
        
        # 如果有合法动作限制，屏蔽非法动作
        if legal_action_types is not None:
            mask = torch.full_like(action_type_logits, float('-inf'))
            for action_type in legal_action_types:
                mask[:, action_type] = 0
            action_type_logits = action_type_logits + mask
        
        # 牌型选择
        card_selection_logits = self.card_selection_head(combined)
        
        # 炸弹使用价值
        bomb_value = self.bomb_value_head(combined)
        
        # 状态价值
        state_value = self.value_head(combined)
        
        return (action_type_logits, card_selection_logits, bomb_value, state_value,
                new_remaining_cards, new_opponent_pred, new_opponent_confidence)


class DouDiZhuAgent:
    """斗地主AI Agent"""
    
    def __init__(self, position, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.position = position
        self.device = device
        
        # 两个网络
        self.bid_net = BidNet().to(device)
        self.play_net = PlayNet().to(device)
        
        # 优化器
        self.bid_optimizer = torch.optim.Adam(self.bid_net.parameters(), lr=LEARNING_RATE)
        self.play_optimizer = torch.optim.Adam(self.play_net.parameters(), lr=LEARNING_RATE)
        
        # 当前状态
        self.current_state = None
    
    def encode_hand(self, hand: List[str]) -> np.ndarray:
        """将手牌编码为17维向量"""
        card_list = ['3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', 'X', 'D']
        encoding = np.zeros(17, dtype=np.float32)
        for card in hand:
            if card in card_list:
                idx = card_list.index(card)
                encoding[idx] += 1
        return encoding / 4.0
    
    def encode_bomb_features(self, hand: List[str]) -> np.ndarray:
        """编码炸弹特征（5维）"""
        card_list = ['3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', 'X', 'D']
        
        card_counts = {}
        for card in hand:
            card_counts[card] = card_counts.get(card, 0) + 1
        
        # 计算普通炸弹
        has_bomb = 0.0
        bomb_count = 0
        max_bomb_rank = 0
        
        for card, count in card_counts.items():
            if count >= 4 and card not in ['X', 'D']:
                has_bomb = 1.0
                bomb_count += 1
                rank = card_list.index(card)
                max_bomb_rank = max(max_bomb_rank, rank)
        
        # 计算王炸
        has_rocket = 0.0
        rocket_count = 0
        if card_counts.get('X', 0) >= 1 and card_counts.get('D', 0) >= 1:
            has_rocket = 1.0
            rocket_count = 1
        
        return np.array([
            has_bomb,
            min(bomb_count / 3.0, 1.0),
            max_bomb_rank / 14.0,
            has_rocket,
            rocket_count
        ], dtype=np.float32)
    
    def calc_remaining_cards(self, my_hand: List[str], played_cards: dict[str, int]) -> np.ndarray:
        """计算剩余牌（外面还有哪些牌没出过）"""
        # 初始牌数量
        total_cards = {
            '3': 4, '4': 4, '5': 4, '6': 4, '7': 4, '8': 4, '9': 4,
            'O': 4, 'J': 4, 'Q': 4, 'K': 4, 'A': 4, '2': 4,
            'X': 1, 'D': 1
        }
        
        # 减去我的手牌
        for card in my_hand:
            if card in total_cards:
                total_cards[card] -= 1
        
        # 减去已出的牌
        for card, count in played_cards.items():
            if card in total_cards:
                total_cards[card] -= count
        
        # 编码为17维向量
        card_list = ['3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', 'X', 'D']
        remaining = np.zeros(17, dtype=np.float32)
        for i, card in enumerate(card_list):
            remaining[i] = total_cards.get(card, 0) / 4.0  # 归一化
        
        return remaining
    
    def init_play_state(self, my_hand: List[str], my_position: int, played_cards: dict[str, int] = None):
        """初始化出牌阶段状态（75维）"""
        if played_cards is None:
            played_cards = {}
        
        # 1. 我的手牌（17维）
        my_hand_enc = self.encode_hand(my_hand)
        
        # 2. 剩余牌（外面还有哪些牌没出过）（17维）
        remaining_cards = self.calc_remaining_cards(my_hand, played_cards)
        
        # 3. 上家预测（17维）- 初始均匀分布
        opponent_pred = np.ones(17, dtype=np.float32) * 0.5
        
        # 4. 上家置信度（17维）- 初始低置信度
        opponent_confidence = np.ones(17, dtype=np.float32) * 0.3
        
        # 5. 我的位置（1维）
        my_pos = np.array([my_position / 2.0], dtype=np.float32)
        
        # 6. 上家位置（1维）
        opponent_pos = (my_position - 1) % 3
        opponent_pos_enc = np.array([opponent_pos / 2.0], dtype=np.float32)
        
        # 7. 炸弹特征（5维）
        bomb_features = self.encode_bomb_features(my_hand)
        
        # 合并为75维
        state = np.concatenate([
            my_hand_enc,           # 0-16: 我的手牌
            remaining_cards,       # 17-33: 剩余牌
            opponent_pred,         # 34-50: 上家预测
            opponent_confidence,   # 51-67: 上家置信度
            my_pos,                # 68: 我的位置
            opponent_pos_enc,      # 69: 上家位置
            bomb_features          # 70-74: 炸弹特征
        ])
        
        self.current_state = state
        return state
    
    def update_play_state(self, new_remaining_cards, new_opponent_pred, new_opponent_confidence, my_hand=None):
        """更新出牌状态"""
        if self.current_state is None:
            return None
        
        # 更新剩余牌
        self.current_state[17:34] = new_remaining_cards
        
        # 更新上家预测
        self.current_state[34:51] = new_opponent_pred
        
        # 更新上家置信度
        self.current_state[51:68] = new_opponent_confidence
        
        # 如果手牌变化，更新炸弹特征
        if my_hand is not None:
            bomb_features = self.encode_bomb_features(my_hand)
            self.current_state[70:75] = bomb_features
        
        return self.current_state
    
    def select_bid(self, hand: List[str], epsilon: float = 0.0) -> int:
        """叫分决策"""
        state = self.encode_hand(hand)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            bid_logits, _ = self.bid_net(state_tensor)
            bid_probs = F.softmax(bid_logits, dim=1).cpu().numpy()[0]
        
        if random.random() < epsilon:
            return random.randint(0, 3)
        
        return np.argmax(bid_probs)
    
    def select_play(self, legal_plays: List[str], epsilon: float = 0.0, last_play=None) -> str:
        """出牌决策"""
        if not legal_plays:
            return "PASS"
        
        if len(legal_plays) == 1:
            return legal_plays[0]
        
        # 使用神经网络决策
        state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 确定合法动作类型
            play_patterns = [(p, CardPattern(p)) for p in legal_plays if p != "PASS"]
            
            legal_action_types = [0]  # PASS
            if any(not pat.is_bomb for _, pat in play_patterns):
                legal_action_types.extend([1, 2, 3, 4])
            if any(pat.is_bomb for _, pat in play_patterns):
                legal_action_types.append(5)
                if any(p in ["XD", "DX"] for p, _ in play_patterns):
                    legal_action_types.append(6)
            
            # 前向传播
            (action_type_logits, card_selection_logits, bomb_value, state_value,
             _, _, _) = self.play_net(state_tensor, legal_action_types)
            
            # 获取动作概率
            action_type_probs = F.softmax(action_type_logits, dim=-1)
            
            # 选择动作
            if random.random() < epsilon:
                return random.choice(legal_plays)
            else:
                action_type = torch.argmax(action_type_probs, dim=-1).item()
                # 简化：从合法动作中选择
                return random.choice(legal_plays)
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'bid_net': self.bid_net.state_dict(),
            'play_net': self.play_net.state_dict(),
            'bid_optimizer': self.bid_optimizer.state_dict(),
            'play_optimizer': self.play_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.bid_net.load_state_dict(checkpoint['bid_net'])
        self.play_net.load_state_dict(checkpoint['play_net'])
        self.bid_optimizer.load_state_dict(checkpoint['bid_optimizer'])
        self.play_optimizer.load_state_dict(checkpoint['play_optimizer'])


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("模型测试")
    print("=" * 60)
    
    agent = DouDiZhuAgent(0)
    
    # 测试叫分
    hand = ['3', '3', '3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', 'X', 'D']
    bid = agent.select_bid(hand)
    print(f"\n叫分测试: 手牌有炸弹，叫分 = {bid}")
    
    # 测试出牌状态初始化
    state = agent.init_play_state(hand, 0)
    print(f"\n出牌状态维度: {state.shape}")
    print(f"炸弹特征: {state[70:75]}")
