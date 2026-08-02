"""
斗地主神经网络模型
包含两个网络：
1. 叫分网络：输入17维（我的手牌），输出叫分决策
2. 出牌网络：输入70维（三方手牌+位置），输出出牌决策+新手牌预测
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from config import BID_INPUT_DIM, PLAY_INPUT_DIM, HIDDEN_SIZE, DROPOUT, BID_OUTPUT_DIM, LEARNING_RATE
from card_utils import CardPattern


class BidNet(nn.Module):
    """叫分网络：仅根据我的手牌决定叫分"""
    
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
    """出牌网络：根据三方手牌预测和位置信息决策
    
    输出动作类型和牌型选择，让AI学会炸弹可以压一切
    """
    
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
        
        # 动作类型决策头（关键：让AI学会什么时候出炸弹）
        # 0: PASS, 1: 单牌, 2: 对子, 3: 三张, 4: 顺子, 5: 炸弹, 6: 王炸
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_size + 51, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 7)  # 7种动作类型
        )
        
        # 牌型选择头（选择具体出什么牌）
        # 15种牌点 × 4种数量（单/对/三/炸）= 60，加上王炸=2，共62种
        self.card_selection_head = nn.Sequential(
            nn.Linear(hidden_size + 51, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 62)  # 15×4 + 2
        )
        
        # 炸弹使用价值评估（专门学习什么时候该用炸弹）
        self.bomb_value_head = nn.Sequential(
            nn.Linear(hidden_size + 51, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # 输出0-1，表示使用炸弹的价值
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
        x: [batch, 70] - 我的手牌(17) + 上家预测(17) + 下家预测(17) + 置信度(17) + 位置(2)
        legal_action_types: 可选，限制合法动作类型
        
        返回: (action_type_logits, card_selection_logits, bomb_value, state_value, 
               opponent_pred, next_pred, confidence)
        """
        features = self.feature_extractor(x)
        
        # 预测新的手牌分布
        prediction_output = self.prediction_head(features)
        
        # 分割输出
        opponent_pred = prediction_output[:, :17]  # 上家手牌预测
        next_pred = prediction_output[:, 17:34]    # 下家手牌预测
        confidence = torch.sigmoid(prediction_output[:, 34:])  # 置信度 0-1
        
        # 合并特征和预测
        combined = torch.cat([features, opponent_pred, next_pred, confidence], dim=1)
        
        # 动作类型（包括炸弹选项）
        action_type_logits = self.action_type_head(combined)
        
        # 如果有合法动作限制，屏蔽非法动作
        if legal_action_types is not None:
            mask = torch.full_like(action_type_logits, float('-inf'))
            for action_type in legal_action_types:
                mask[:, action_type] = 0
            action_type_logits = action_type_logits + mask
        
        # 牌型选择
        card_selection_logits = self.card_selection_head(combined)
        
        # 炸弹使用价值（0-1，越高越应该用炸弹）
        bomb_value = self.bomb_value_head(combined)
        
        # 状态价值
        state_value = self.value_head(combined)
        
        return (action_type_logits, card_selection_logits, bomb_value, state_value,
                opponent_pred, next_pred, confidence)


class DouDiZhuAgent:
    """斗地主AI Agent"""
    
    def __init__(self, position, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.position = position  # 0=头叫, 1=二叫, 2=三叫
        self.device = device
        
        # 两个网络
        self.bid_net = BidNet().to(device)
        self.play_net = PlayNet().to(device)
        
        # 优化器
        self.bid_optimizer = torch.optim.Adam(self.bid_net.parameters(), lr=LEARNING_RATE)
        self.play_optimizer = torch.optim.Adam(self.play_net.parameters(), lr=LEARNING_RATE)
        
        self.training_history = []
        
        # 当前状态（70维向量）
        self.current_state = None
    
    def encode_hand(self, hand):
        """将手牌列表编码为17维向量"""
        card_list = ['3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', 'X', 'D']
        encoding = np.zeros(17, dtype=np.float32)
        for card in hand:
            if card in card_list:
                idx = card_list.index(card)
                encoding[idx] += 1
        return encoding / 4.0  # 归一化
    
    def init_play_state(self, my_hand, my_position):
        """
        初始化出牌阶段状态（70维）
        在游戏开始、确定地主后调用
        """
        card_list = ['3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', 'X', 'D']
        
        # 1. 我的手牌（17维）
        my_hand_enc = self.encode_hand(my_hand)
        
        # 2. 上家手牌初始预测（17维）- 均匀分布
        opponent_pred = np.ones(17, dtype=np.float32) * 0.5  # 平均每种0.5张
        
        # 3. 下家手牌初始预测（17维）
        next_pred = np.ones(17, dtype=np.float32) * 0.5
        
        # 4. 置信度（17维）- 初始低置信度
        confidence = np.ones(17, dtype=np.float32) * 0.3
        
        # 5. 我的位置（1维）
        my_pos = np.array([my_position / 2.0], dtype=np.float32)  # 归一化到0-1
        
        # 6. 上家位置（1维）
        opponent_pos = (my_position - 1) % 3
        opponent_pos_enc = np.array([opponent_pos / 2.0], dtype=np.float32)
        
        # 合并为70维
        state = np.concatenate([
            my_hand_enc,      # 17
            opponent_pred,    # 17
            next_pred,        # 17
            confidence,       # 17
            my_pos,           # 1
            opponent_pos_enc  # 1
        ])
        
        self.current_state = state
        return state
    
    def update_play_state(self, new_opponent_pred, new_next_pred, new_confidence):
        """
        更新出牌状态（使用网络输出的新预测）
        """
        if self.current_state is None:
            return None
        
        # 更新预测部分（保留我的手牌和位置信息）
        self.current_state[17:34] = new_opponent_pred  # 上家预测
        self.current_state[34:51] = new_next_pred      # 下家预测
        self.current_state[51:68] = new_confidence     # 置信度
        
        return self.current_state
    
    def select_bid(self, hand, epsilon=0.0):
        """叫分决策"""
        state = self.encode_hand(hand)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            bid_logits, _ = self.bid_net(state_tensor)
            bid_probs = F.softmax(bid_logits, dim=1).cpu().numpy()[0]
        
        if np.random.random() < epsilon:
            return np.random.randint(4)
        
        return np.argmax(bid_probs)
    
    def select_play(self, legal_plays, epsilon=0.0, last_play=None):
        """出牌决策 - 使用神经网络，理解炸弹可以压制任何牌
        
        Args:
            legal_plays: 合法出牌列表
            epsilon: 探索率
            last_play: 上家出的牌（CardPattern对象），用于判断是否需要炸弹
        """
        if self.current_state is None:
            return "PASS"
        
        if not legal_plays:
            return "PASS"
        
        if len(legal_plays) == 1:
            return legal_plays[0]
        
        # 解析每个出牌选项
        play_patterns = []
        for play in legal_plays:
            if play == "PASS":
                continue
            pattern = CardPattern(play)
            if pattern.is_valid():
                play_patterns.append((play, pattern))
        
        # 分类牌型
        bombs = [(p, pat) for p, pat in play_patterns if pat.is_bomb]
        non_bombs = [(p, pat) for p, pat in play_patterns if not pat.is_bomb]
        
        # 使用神经网络决策
        state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 确定合法动作类型
            legal_action_types = [0]  # PASS 总是合法
            if non_bombs:
                legal_action_types.extend([1, 2, 3, 4])  # 单牌/对子/三张/顺子
            if bombs:
                legal_action_types.append(5)  # 炸弹
                if any(p == "XD" or p == "DX" for p, _ in bombs):
                    legal_action_types.append(6)  # 王炸
            
            # 前向传播
            (action_type_logits, card_selection_logits, bomb_value, state_value,
             opponent_pred, next_pred, confidence) = self.play_net(state_tensor, legal_action_types)
            
            # 更新状态预测
            self.update_play_state(
                opponent_pred.cpu().numpy()[0],
                next_pred.cpu().numpy()[0],
                confidence.cpu().numpy()[0]
            )
            
            # 获取动作类型概率
            action_type_probs = torch.softmax(action_type_logits, dim=-1).cpu().numpy()[0]
            
            # 判断是否必须使用炸弹（没有非炸弹能压制上家）
            must_use_bomb = False
            if last_play is not None and last_play.type != CardPattern.TYPE_PASS:
                can_beat_without_bomb = any(
                    pat.can_beat(last_play) for _, pat in non_bombs
                )
                if not can_beat_without_bomb and bombs:
                    must_use_bomb = True
            
            # 决策逻辑
            if np.random.random() < epsilon:
                # 探索：随机选择
                if must_use_bomb:
                    best_play = random.choice([p for p, _ in bombs])
                else:
                    best_play = random.choice(legal_plays)
            else:
                # 利用：神经网络决策
                if must_use_bomb:
                    # 必须使用炸弹时，选择最小的炸弹
                    bombs.sort(key=lambda x: x[1].main_rank)
                    best_play = bombs[0][0]
                else:
                    # 选择概率最高的动作类型
                    action_type = np.argmax(action_type_probs)
                    
                    if action_type == 0:
                        # PASS
                        best_play = "PASS"
                    elif action_type == 5 and bombs:
                        # 炸弹 - 根据bomb_value决定是否使用
                        if bomb_value.item() > 0.5:
                            bombs.sort(key=lambda x: x[1].main_rank)
                            best_play = bombs[0][0]
                        else:
                            # 炸弹价值不高，选择其他牌
                            if non_bombs:
                                non_bombs.sort(key=lambda x: x[1].main_rank)
                                best_play = non_bombs[0][0]
                            else:
                                best_play = "PASS"
                    else:
                        # 其他牌型，选择最小的
                        if non_bombs:
                            non_bombs.sort(key=lambda x: x[1].main_rank)
                            best_play = non_bombs[0][0]
                        elif bombs:
                            bombs.sort(key=lambda x: x[1].main_rank)
                            best_play = bombs[0][0]
                        else:
                            best_play = "PASS"
        
        return best_play
    
    def get_play_value(self):
        """获取当前出牌状态价值"""
        if self.current_state is None:
            return 0.0
        
        state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, value, _, _, _ = self.play_net(state_tensor)
            return value.cpu().numpy()[0][0]
    
    def train_bid_step(self, states, bid_targets, value_targets):
        """训练叫分网络"""
        states = torch.FloatTensor(states).to(self.device)
        bid_targets = torch.LongTensor(bid_targets).to(self.device)
        value_targets = torch.FloatTensor(value_targets).to(self.device)
        
        bid_logits, values = self.bid_net(states)
        
        bid_loss = F.cross_entropy(bid_logits, bid_targets)
        value_loss = F.mse_loss(values, value_targets)
        
        total_loss = bid_loss + value_loss
        
        self.bid_optimizer.zero_grad()
        total_loss.backward()
        self.bid_optimizer.step()
        
        return {
            'bid_loss': bid_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_play_step(self, states, play_targets, value_targets, opponent_targets, next_targets, conf_targets):
        """训练出牌网络"""
        states = torch.FloatTensor(states).to(self.device)
        value_targets = torch.FloatTensor(value_targets).to(self.device)
        opponent_targets = torch.FloatTensor(opponent_targets).to(self.device)
        next_targets = torch.FloatTensor(next_targets).to(self.device)
        conf_targets = torch.FloatTensor(conf_targets).to(self.device)
        
        play_logits, values, opponent_pred, next_pred, confidence = self.play_net(states)
        
        value_loss = F.mse_loss(values, value_targets)
        opponent_loss = F.mse_loss(opponent_pred, opponent_targets)
        next_loss = F.mse_loss(next_pred, next_targets)
        conf_loss = F.mse_loss(confidence, conf_targets)
        
        total_loss = value_loss + opponent_loss + next_loss + 0.5 * conf_loss
        
        self.play_optimizer.zero_grad()
        total_loss.backward()
        self.play_optimizer.step()
        
        return {
            'value_loss': value_loss.item(),
            'opponent_loss': opponent_loss.item(),
            'next_loss': next_loss.item(),
            'conf_loss': conf_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'bid_net_state_dict': self.bid_net.state_dict(),
            'play_net_state_dict': self.play_net.state_dict(),
            'bid_optimizer_state_dict': self.bid_optimizer.state_dict(),
            'play_optimizer_state_dict': self.play_optimizer.state_dict(),
            'position': self.position,
            'training_history': self.training_history,
            'current_state': self.current_state
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.bid_net.load_state_dict(checkpoint['bid_net_state_dict'])
        self.play_net.load_state_dict(checkpoint['play_net_state_dict'])
        self.bid_optimizer.load_state_dict(checkpoint['bid_optimizer_state_dict'])
        self.play_optimizer.load_state_dict(checkpoint['play_optimizer_state_dict'])
        self.position = checkpoint.get('position', self.position)
        self.training_history = checkpoint.get('training_history', [])
        self.current_state = checkpoint.get('current_state', None)


if __name__ == "__main__":
    # 测试
    agent = DouDiZhuAgent(position=0)
    
    # 测试叫分网络
    hand = ['3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', 'X', 'D', '3', '4']
    bid = agent.select_bid(hand, epsilon=0.0)
    print(f"叫分: {bid}")
    
    # 测试出牌网络
    agent.init_play_state(hand, 0)
    print(f"初始状态维度: {agent.current_state.shape}")  # 应该是(70,)
    
    # 模拟一轮出牌后更新
    state_tensor = torch.FloatTensor(agent.current_state).unsqueeze(0)
    with torch.no_grad():
        _, _, opponent_pred, next_pred, confidence = agent.play_net(state_tensor)
        agent.update_play_state(
            opponent_pred.numpy()[0],
            next_pred.numpy()[0],
            confidence.numpy()[0]
        )
    
    print(f"更新后状态维度: {agent.current_state.shape}")  # 仍然是(70,)
    print("测试完成!")
