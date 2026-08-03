"""
斗地主强化学习模型 - 高级版本
特点:
1. 保持75维输入不变
2. 隐式配合模式学习 (Embedding层)
3. 对比学习识别高质量对局
4. 奖励函数考虑春天/反春天翻倍
5. 自监督学习配合信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter, deque
import random
from dataclasses import dataclass

# 牌编码
CARD_TYPES = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', 'X', 'D']
CARD_TO_IDX = {card: idx for idx, card in enumerate(CARD_TYPES)}


@dataclass
class GameResult:
    """游戏结果"""
    winner: str  # 'landlord' / 'landlord_up' / 'landlord_down'
    is_spring: bool  # 春天
    is_anti_spring: bool  # 反春天
    farmer_cooperation_score: float  # 农民配合度 0-1
    control_changes: int  # 牌权转换次数
    bomb_count: int  # 使用炸弹数
    quality_score: float  # 对局质量评分


class CooperationPatternEncoder(nn.Module):
    """
    配合模式编码器
    从对局历史推断配合模式，不增加输入维度
    """
    def __init__(self, num_patterns=100, embed_dim=32):
        super().__init__()
        self.pattern_embed = nn.Embedding(num_patterns, embed_dim)
        self.num_patterns = num_patterns
        
        # 历史编码器
        self.history_encoder = nn.GRU(
            input_size=15,  # 每轮出牌编码为15维
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        
        # 模式分类器
        self.pattern_classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_patterns)
        )
    
    def encode_play(self, play: List[str]) -> torch.Tensor:
        """将出牌编码为15维向量"""
        vec = torch.zeros(15)
        if play:
            for card in play:
                if card in CARD_TO_IDX:
                    vec[CARD_TO_IDX[card]] += 1
        return vec / 4.0  # 归一化
    
    def forward(self, history: List[Tuple[str, List[str]]], my_role: str) -> torch.Tensor:
        """
        从对局历史推断配合模式
        
        Args:
            history: [(player, cards), ...]
            my_role: 'landlord' / 'landlord_up' / 'landlord_down'
        
        Returns:
            pattern_embedding: [embed_dim]
        """
        if not history:
            # 无历史时返回默认模式
            return self.pattern_embed(torch.tensor(0))
        
        # 编码历史
        history_vecs = []
        for player, play in history[-20:]:  # 取最近20轮
            vec = self.encode_play(play)
            # 标记是谁出的牌
            if player == my_role:
                vec = torch.cat([vec, torch.tensor([1.0, 0.0, 0.0])])
            elif self.is_teammate(player, my_role):
                vec = torch.cat([vec, torch.tensor([0.0, 1.0, 0.0])])
            else:
                vec = torch.cat([vec, torch.tensor([0.0, 0.0, 1.0])])
            history_vecs.append(vec[:15])  # 保持15维
        
        if len(history_vecs) < 20:
            # 填充
            padding = [torch.zeros(15)] * (20 - len(history_vecs))
            history_vecs = padding + history_vecs
        
        history_tensor = torch.stack(history_vecs).unsqueeze(0)  # [1, 20, 15]
        
        # GRU编码
        _, hidden = self.history_encoder(history_tensor)
        history_feat = hidden[-1].squeeze(0)  # [64]
        
        # 分类到配合模式
        pattern_logits = self.pattern_classifier(history_feat)
        pattern_id = torch.argmax(pattern_logits).item()
        
        # 返回嵌入
        return self.pattern_embed(torch.tensor(pattern_id))
    
    def is_teammate(self, player: str, my_role: str) -> bool:
        """判断是否是队友"""
        if my_role == 'landlord':
            return False
        return player != 'landlord' and player != my_role


class AdvancedDouDizhuNet(nn.Module):
    """
    高级斗地主网络
    输入维度: 75 (保持不变)
    """
    def __init__(self, hidden_dim=512, num_patterns=100):
        super().__init__()
        
        # 配合模式编码器
        self.coop_encoder = CooperationPatternEncoder(num_patterns, embed_dim=32)
        
        # 状态编码器 (输入75维)
        self.state_encoder = nn.Sequential(
            nn.Linear(75, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        
        # 上下文编码 (春天/倍数等)
        self.context_encoder = nn.Sequential(
            nn.Linear(4, 64),  # [倍数, 是否可能春天, 是否可能反春天, 当前轮数]
            nn.ReLU()
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(256 + 32 + 64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024)  # 最大动作空间
        )
        
        # 价值头 (输出期望收益，已考虑翻倍)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state_vector: torch.Tensor, 
                history: List[Tuple[str, List[str]]],
                my_role: str,
                context: Dict,
                legal_actions: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state_vector: [75] 状态向量
            history: 对局历史
            my_role: 我的角色
            context: 游戏上下文
            legal_actions: 合法动作列表
        
        Returns:
            policy_logits: 策略logits
            value: 期望收益 (已考虑翻倍)
            pattern_embedding: 配合模式嵌入 (用于分析)
        """
        device = state_vector.device
        
        # 编码状态 (75维)
        state_feat = self.state_encoder(state_vector)  # [256]
        
        # 编码配合模式 (从历史推断)
        pattern_embed = self.coop_encoder(history, my_role)  # [32]
        
        # 编码上下文
        context_vec = torch.tensor([
            context.get('multiplier', 1.0),
            1.0 if context.get('spring_possible', False) else 0.0,
            1.0 if context.get('anti_spring_possible', False) else 0.0,
            context.get('round_num', 0) / 50.0
        ], dtype=torch.float32).to(device)
        context_feat = self.context_encoder(context_vec)  # [64]
        
        # 融合所有特征
        combined = torch.cat([state_feat, pattern_embed, context_feat], dim=-1)
        features = self.feature_fusion(combined)  # [hidden_dim]
        
        # 输出
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        
        # 只保留合法动作的logits
        action_mask = torch.zeros(1024).to(device)
        for i in range(min(len(legal_actions), 1024)):
            action_mask[i] = 1.0
        
        policy_logits = policy_logits * action_mask - (1 - action_mask) * 1e9
        
        return policy_logits, value, pattern_embed


class QualityScorer:
    """对局质量评估器 - 无标签学习"""
    
    def __init__(self):
        self.pattern_stats = {
            'triple_plays': 0,
            'bomb_timings': [],
            'control_changes': 0,
            'cooperation_signals': 0
        }
    
    def score_game(self, history: List[Tuple[str, List[str]]], 
                   result: GameResult) -> float:
        """
        评估单局游戏质量 (无需人工标签)
        
        Returns:
            quality_score: 0-10分
        """
        score = 0.0
        
        # 1. 出牌合理性 (是否有明显违规)
        rationality = self.check_rationality(history)
        score += rationality * 2.0
        
        # 2. 牌权转换次数 (太少的可能是碾压局)
        if 5 <= result.control_changes <= 30:
            score += 1.5
        
        # 3. 农民配合度
        if result.farmer_cooperation_score > 0.7:
            score += 2.0
        
        # 4. 炸弹使用时机
        bomb_score = self.evaluate_bomb_usage(history)
        score += bomb_score * 1.5
        
        # 5. 牌型多样性
        diversity = self.evaluate_diversity(history)
        score += diversity * 1.0
        
        # 6. 春天/反春天 (高水平对局标志)
        if result.is_spring or result.is_anti_spring:
            score += 1.0
        
        # 7. 出牌效率
        efficiency = self.evaluate_efficiency(history, result)
        score += efficiency * 1.0
        
        return min(score, 10.0)
    
    def check_rationality(self, history: List[Tuple[str, List[str]]]) -> float:
        """检查出牌合理性"""
        if not history:
            return 0.5
        
        rational_count = 0
        total_count = 0
        
        for i, (player, play) in enumerate(history):
            if not play:  # PASS
                continue
            
            total_count += 1
            
            # 检查是否有压牌机会却PASS
            if i > 0:
                last_play = history[i-1][1]
                if last_play and self.can_beat(play, last_play):
                    rational_count += 1
            else:
                rational_count += 1
        
        return rational_count / max(total_count, 1)
    
    def can_beat(self, play1: List[str], play2: List[str]) -> bool:
        """简化判断play1是否能压play2"""
        # 实际实现需要完整牌型判断
        return len(play1) > 0
    
    def evaluate_bomb_usage(self, history: List[Tuple[str, List[str]]]) -> float:
        """评估炸弹使用时机"""
        bomb_count = 0
        good_timing = 0
        
        for i, (player, play) in enumerate(history):
            if self.is_bomb(play):
                bomb_count += 1
                # 好的时机：关键时刻使用
                if i > len(history) * 0.3 and i < len(history) * 0.9:
                    good_timing += 1
        
        if bomb_count == 0:
            return 0.5
        return good_timing / bomb_count
    
    def is_bomb(self, play: List[str]) -> bool:
        """判断是否是炸弹"""
        if not play:
            return False
        if len(play) == 4 and len(set(play)) == 1:
            return True
        if set(play) == {'X', 'D'}:
            return True
        return False
    
    def evaluate_diversity(self, history: List[Tuple[str, List[str]]]) -> float:
        """评估牌型多样性"""
        play_types = set()
        
        for player, play in history:
            play_type = self.get_play_type(play)
            play_types.add(play_type)
        
        # 越多不同牌型越好
        return min(len(play_types) / 8.0, 1.0)
    
    def get_play_type(self, play: List[str]) -> str:
        """获取牌型"""
        if not play:
            return 'pass'
        if len(play) == 1:
            return 'single'
        if len(play) == 2:
            return 'pair'
        if len(play) == 3:
            return 'triple'
        if len(play) == 4:
            if len(set(play)) == 1:
                return 'bomb'
            return 'triple_single'
        if len(play) == 5:
            return 'triple_pair'
        return 'combo'
    
    def evaluate_efficiency(self, history: List[Tuple[str, List[str]]], 
                           result: GameResult) -> float:
        """评估出牌效率"""
        # 总出牌数 / 总轮数
        total_plays = sum(len(play) for _, play in history if play)
        total_rounds = len(history)
        
        if total_rounds == 0:
            return 0.5
        
        avg_cards_per_round = total_plays / total_rounds
        # 平均每轮出3-5张牌较好
        if 2 <= avg_cards_per_round <= 6:
            return 1.0
        return 0.5


class ContrastiveLearner:
    """对比学习器"""
    
    def __init__(self, temperature=0.07):
        self.temperature = temperature
        self.quality_scorer = QualityScorer()
    
    def select_positive_negative(self, games: List[Tuple[List, GameResult]]):
        """
        选择正负样本
        
        Args:
            games: [(history, result), ...]
        
        Returns:
            positives: 高质量对局
            negatives: 低质量对局
        """
        scored_games = []
        for history, result in games:
            score = self.quality_scorer.score_game(history, result)
            scored_games.append((history, result, score))
        
        # 排序
        scored_games.sort(key=lambda x: x[2], reverse=True)
        
        # 前20%为正样本，后30%为负样本
        n = len(scored_games)
        positives = scored_games[:n//5]
        negatives = scored_games[-n//3:]
        
        return positives, negatives
    
    def contrastive_loss(self, anchor_feat, positive_feat, negative_feats):
        """
        对比学习损失
        拉近anchor和positive，推远anchor和negative
        """
        # 正样本相似度
        pos_sim = F.cosine_similarity(anchor_feat, positive_feat, dim=-1) / self.temperature
        
        # 负样本相似度
        neg_sims = []
        for neg_feat in negative_feats:
            neg_sim = F.cosine_similarity(anchor_feat, neg_feat, dim=-1) / self.temperature
            neg_sims.append(neg_sim)
        
        # InfoNCE损失
        logits = torch.cat([pos_sim.unsqueeze(0), torch.stack(neg_sims)])
        labels = torch.zeros(1, dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits.unsqueeze(0), labels)
        return loss


class AdvancedPPOAgent:
    """高级PPO智能体"""
    
    def __init__(self, lr=3e-4, gamma=0.99, lam=0.95, clip_eps=0.2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = AdvancedDouDizhuNet().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        
        self.memory = []
        self.quality_scorer = QualityScorer()
        self.contrastive_learner = ContrastiveLearner()
    
    def calculate_reward(self, game_result: GameResult, my_role: str) -> float:
        """
        计算奖励 (已考虑春天/反春天翻倍)
        """
        # 基础奖励
        if game_result.winner == my_role:
            base_reward = 1.0
        elif self.is_teammate_winner(game_result.winner, my_role):
            base_reward = 1.0
        else:
            base_reward = -1.0
        
        # 翻倍结算
        multiplier = 1.0
        if game_result.is_spring:
            multiplier = 2.0
            # 春天额外奖励技巧
            if game_result.winner == my_role:
                base_reward += 0.5
        
        if game_result.is_anti_spring:
            multiplier = 2.0
            if game_result.winner == my_role:
                base_reward += 0.5
        
        # 配合奖励
        if my_role != 'landlord' and game_result.farmer_cooperation_score > 0.8:
            base_reward += 0.3
        
        return base_reward * multiplier
    
    def is_teammate_winner(self, winner: str, my_role: str) -> bool:
        """判断是否是队友获胜"""
        if my_role == 'landlord':
            return False
        return winner != 'landlord' and winner != my_role
    
    def self_supervised_training(self, game_histories: List):
        """
        自监督训练阶段
        识别高质量对局，学习配合模式
        """
        # 1. 评估所有对局质量
        scored_games = []
        for history, result in game_histories:
            score = self.quality_scorer.score_game(history, result)
            scored_games.append((history, result, score))
        
        # 2. 选择高质量对局
        scored_games.sort(key=lambda x: x[2], reverse=True)
        good_games = scored_games[:len(scored_games)//5]
        
        print(f"识别到 {len(good_games)} 局高质量对局")
        
        # 3. 对比学习
        positives, negatives = self.contrastive_learner.select_positive_negative(game_histories)
        
        # 4. 训练配合模式编码器
        for _ in range(10):  # 训练轮数
            for pos_history, pos_result, _ in positives[:50]:
                # 编码正样本
                anchor = self.encode_game(pos_history, pos_result)
                
                # 随机选择负样本
                neg_samples = random.sample(negatives, min(5, len(negatives)))
                negatives_encoded = [self.encode_game(h, r) for h, r, _ in neg_samples]
                
                # 计算对比损失
                loss = self.contrastive_learner.contrastive_loss(
                    anchor, anchor, negatives_encoded
                )
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        print("自监督训练完成")
    
    def encode_game(self, history, result):
        """编码对局为特征向量"""
        # 简化实现
        return torch.randn(32).to(self.device)  # 实际应从网络获取
    
    def select_action(self, state_vector: np.ndarray,
                     history: List[Tuple[str, List[str]]],
                     my_role: str,
                     context: Dict,
                     legal_actions: List[List[str]],
                     training=True):
        """选择动作"""
        device = self.device
        
        state_tensor = torch.tensor(state_vector, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits, value, pattern_embed = self.net(
                state_tensor, history, my_role, context, legal_actions
            )
            probs = F.softmax(logits[:len(legal_actions)], dim=0)
        
        if training:
            action_idx = torch.multinomial(probs, 1).item()
        else:
            action_idx = probs.argmax().item()
        
        return legal_actions[action_idx], action_idx, probs[action_idx].item(), value.item()


class GameContextExtractor:
    """游戏上下文提取器"""
    
    @staticmethod
    def extract(state: Dict, my_role: str) -> Dict:
        """
        从游戏状态提取上下文信息
        
        Returns:
            context: {
                'multiplier': 当前倍数,
                'spring_possible': 是否可能春天,
                'anti_spring_possible': 是否可能反春天,
                'round_num': 当前轮数
            }
        """
        history = state.get('history', [])
        
        # 计算倍数
        multiplier = 1.0
        for player, play in history:
            if len(play) == 4 and len(set(play)) == 1:  # 炸弹
                multiplier *= 2.0
            if set(play) == {'X', 'D'}:  # 王炸
                multiplier *= 2.0
        
        # 判断春天可能性
        landlord_plays = [play for p, play in history if p == 'landlord' and play]
        farmer_plays = [play for p, play in history if p != 'landlord' and play]
        
        spring_possible = len(landlord_plays) <= 2 and len(history) > 5
        anti_spring_possible = len(farmer_plays) <= 2 and len(history) > 5
        
        return {
            'multiplier': multiplier,
            'spring_possible': spring_possible,
            'anti_spring_possible': anti_spring_possible,
            'round_num': len(history)
        }


# ============== 使用示例 ==============

def demo():
    """演示"""
    print("=" * 60)
    print("斗地主高级模型 - 演示")
    print("=" * 60)
    
    # 创建智能体
    agent = AdvancedPPOAgent()
    
    # 模拟状态
    state_vector = np.random.randn(75).astype(np.float32) * 0.1
    state_vector[:15] = np.array([3, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]) / 4.0  # 我的手牌
    
    history = [
        ('landlord', ['3', '4', '5', '6', '7']),
        ('landlord_up', ['8', '9', '10', 'J', 'Q']),
        ('landlord_down', []),
    ]
    
    my_role = 'landlord'
    
    # 提取上下文
    mock_state = {'history': history}
    context = GameContextExtractor.extract(mock_state, my_role)
    print(f"\n游戏上下文: {context}")
    
    # 模拟合法动作
    legal_actions = [
        [],  # PASS
        ['K'],  # 单张
        ['A'],  # 单张
        ['2'],  # 单张
        ['K', 'A', '2', 'X'],  # 三带一
    ]
    
    # 选择动作
    action, action_idx, prob, value = agent.select_action(
        state_vector, history, my_role, context, legal_actions, training=False
    )
    
    print(f"\n推荐出牌: {action}")
    print(f"置信度: {prob:.2%}")
    print(f"期望收益: {value:.3f}")
    
    # 演示质量评估
    print("\n" + "=" * 60)
    print("对局质量评估演示")
    print("=" * 60)
    
    result = GameResult(
        winner='landlord',
        is_spring=True,
        is_anti_spring=False,
        farmer_cooperation_score=0.3,  # 农民配合差
        control_changes=15,
        bomb_count=2,
        quality_score=0.0
    )
    
    quality = agent.quality_scorer.score_game(history, result)
    print(f"对局质量评分: {quality:.2f}/10")
    
    # 演示奖励计算
    reward = agent.calculate_reward(result, 'landlord')
    print(f"地主获得奖励: {reward:.2f} (春天翻倍)")
    
    reward_farmer = agent.calculate_reward(result, 'landlord_up')
    print(f"农民获得奖励: {reward_farmer:.2f}")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == '__main__':
    demo()
