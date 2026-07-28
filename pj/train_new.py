"""斗地主多智能体强化学习训练脚本 - GAN式训练 (新版).

使用MAPPO（Multi-Agent PPO）算法训练三个Agent.
使用独立的train_logger记录日志，visualizer独立生成图表.
训练流程：
1. 初始训练：三个Agent一起玩3万局
2. 轮流强化训练：固定两个，训练一个，各10轮
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import os
import sys
import json
from datetime import datetime

from generator import load_decks_from_file, generate_and_save
from card_utils import (
    CardType, CardPattern, identify_pattern, can_beat,
    filter_legal_patterns, count_cards, hand_to_string,
    has_cards, remove_cards, verify_play
)
from train_logger import TrainLogger


# ============== 超参数配置 ==============
@dataclass
class Config:
    """训练配置."""
    # 网络结构
    state_dim: int = 40
    hidden_dim: int = 256
    action_dim: int = 10000
    
    # MAPPO参数
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # 训练参数
    initial_episodes: int = 30000
    fine_tune_episodes: int = 10000
    fine_tune_rounds: int = 20
    batch_size: int = 32
    update_epochs: int = 4
    
    # 评估
    eval_interval: int = 100
    save_interval: int = 1000
    
    # 强化训练后测试参数
    eval_after_finetune: bool = True  # 是否在强化训练后测试
    eval_games: int = 100  # 测试局数
    
    # 设备
    device: str = "cuda"
    
    # 日志
    log_dir: str = "./train_logs"
    
    # 牌组文件
    decks_file: str = "./decks/decks_data.txt"


# ============== 神经网络 ==============
class ActorCritic(nn.Module):
    """Actor-Critic网络."""
    
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        
    def forward(self, state: torch.Tensor, action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        logits = self.actor(features)
        logits = logits.masked_fill(action_mask == 0, float('-inf'))
        action_probs = torch.softmax(logits, dim=-1)
        value = self.critic(features)
        return action_probs, value
    
    def get_action(self, state: torch.Tensor, action_mask: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            action_probs, value = self.forward(state.unsqueeze(0), action_mask.unsqueeze(0))
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value.squeeze()


# ============== 动作编码器 ==============
class ActionEncoder:
    """将牌型动作编码为固定维度的索引."""
    
    def __init__(self, max_actions: int = 10000):
        self.max_actions = max_actions
        self.action_to_idx: Dict[str, int] = {}
        self.idx_to_action: Dict[int, CardPattern] = {}
        self._build_action_space()
        
    def _build_action_space(self):
        idx = 0
        
        # Pass
        self.action_to_idx["PASS"] = idx
        self.idx_to_action[idx] = CardPattern(CardType.PASS, 0, 0, 0, [])
        idx += 1
        
        # 单张
        for point in range(3, 18):
            key = f"SINGLE_{point}"
            if idx < self.max_actions:
                self.action_to_idx[key] = idx
                self.idx_to_action[idx] = CardPattern(CardType.SINGLE, point, 0, 1, [point])
                idx += 1
        
        # 对子
        for point in range(3, 16):
            key = f"PAIR_{point}"
            if idx < self.max_actions:
                self.action_to_idx[key] = idx
                self.idx_to_action[idx] = CardPattern(CardType.PAIR, point, 0, 2, [point, point])
                idx += 1
        
        # 三张
        for point in range(3, 16):
            key = f"TRIPLE_{point}"
            if idx < self.max_actions:
                self.action_to_idx[key] = idx
                self.idx_to_action[idx] = CardPattern(CardType.TRIPLE, point, 0, 3, [point]*3)
                idx += 1
        
        # 三带一
        for triple in range(3, 16):
            for single in range(3, 18):
                key = f"TRIPLE_SINGLE_{triple}_{single}"
                if idx < self.max_actions:
                    self.action_to_idx[key] = idx
                    self.idx_to_action[idx] = CardPattern(
                        CardType.TRIPLE_SINGLE, triple, single, 4, [triple]*3 + [single]
                    )
                    idx += 1
        
        # 三带二
        for triple in range(3, 16):
            for pair in range(3, 16):
                key = f"TRIPLE_PAIR_{triple}_{pair}"
                if idx < self.max_actions:
                    self.action_to_idx[key] = idx
                    self.idx_to_action[idx] = CardPattern(
                        CardType.TRIPLE_PAIR, triple, pair, 5, [triple]*3 + [pair]*2
                    )
                    idx += 1
        
        # 炸弹
        for point in range(3, 16):
            key = f"BOMB_{point}"
            if idx < self.max_actions:
                self.action_to_idx[key] = idx
                self.idx_to_action[idx] = CardPattern(CardType.BOMB, point, 0, 4, [point]*4)
                idx += 1
        
        # 火箭
        key = "ROCKET"
        if idx < self.max_actions:
            self.action_to_idx[key] = idx
            self.idx_to_action[idx] = CardPattern(CardType.ROCKET, 17, 0, 2, [16, 17])
            idx += 1
        
        # 顺子
        for length in range(5, 13):
            for start in range(3, 15 - length + 1):
                end = start + length - 1
                if end <= 14:
                    key = f"STRAIGHT_{start}_{length}"
                    if idx < self.max_actions:
                        cards = list(range(start, end + 1))
                        self.action_to_idx[key] = idx
                        self.idx_to_action[idx] = CardPattern(CardType.STRAIGHT, end, 0, length, cards)
                        idx += 1
        
        # 连对
        for n_pairs in range(3, 7):
            for start in range(3, 15 - n_pairs + 1):
                end = start + n_pairs - 1
                if end <= 14:
                    key = f"STRAIGHT_PAIR_{start}_{n_pairs}"
                    if idx < self.max_actions:
                        cards = []
                        for p in range(start, end + 1):
                            cards.extend([p, p])
                        self.action_to_idx[key] = idx
                        self.idx_to_action[idx] = CardPattern(CardType.STRAIGHT_PAIR, end, 0, n_pairs*2, cards)
                        idx += 1
        
        # 飞机
        for n_triples in range(2, 5):
            for start in range(3, 15 - n_triples + 1):
                end = start + n_triples - 1
                if end <= 14:
                    key = f"PLANE_{start}_{n_triples}"
                    if idx < self.max_actions:
                        cards = []
                        for p in range(start, end + 1):
                            cards.extend([p, p, p])
                        self.idx_to_action[idx] = CardPattern(CardType.PLANE, end, 0, len(cards), cards)
                        idx += 1
        
        print(f"动作空间大小: {idx}")
        
    def pattern_to_idx(self, pattern: CardPattern) -> int:
        if pattern.card_type == CardType.PASS:
            return self.action_to_idx.get("PASS", 0)
        elif pattern.card_type == CardType.SINGLE:
            return self.action_to_idx.get(f"SINGLE_{pattern.main_point}", 0)
        elif pattern.card_type == CardType.PAIR:
            return self.action_to_idx.get(f"PAIR_{pattern.main_point}", 0)
        elif pattern.card_type == CardType.TRIPLE:
            return self.action_to_idx.get(f"TRIPLE_{pattern.main_point}", 0)
        elif pattern.card_type == CardType.TRIPLE_SINGLE:
            return self.action_to_idx.get(f"TRIPLE_SINGLE_{pattern.main_point}_{pattern.sub_point}", 0)
        elif pattern.card_type == CardType.TRIPLE_PAIR:
            return self.action_to_idx.get(f"TRIPLE_PAIR_{pattern.main_point}_{pattern.sub_point}", 0)
        elif pattern.card_type == CardType.BOMB:
            return self.action_to_idx.get(f"BOMB_{pattern.main_point}", 0)
        elif pattern.card_type == CardType.ROCKET:
            return self.action_to_idx.get("ROCKET", 0)
        elif pattern.card_type == CardType.STRAIGHT:
            start = pattern.main_point - pattern.length + 1
            return self.action_to_idx.get(f"STRAIGHT_{start}_{pattern.length}", 0)
        elif pattern.card_type == CardType.STRAIGHT_PAIR:
            n_pairs = pattern.length // 2
            start = pattern.main_point - n_pairs + 1
            return self.action_to_idx.get(f"STRAIGHT_PAIR_{start}_{n_pairs}", 0)
        elif pattern.card_type == CardType.PLANE:
            n_triples = pattern.length // 3
            start = pattern.main_point - n_triples + 1
            return self.action_to_idx.get(f"PLANE_{start}_{n_triples}", 0)
        return 0
    
    def idx_to_pattern(self, idx: int) -> CardPattern:
        return self.idx_to_action.get(idx, CardPattern(CardType.PASS, 0, 0, 0, []))


# ============== 状态编码 ==============
class StateEncoder:
    """状态编码器."""
    
    def __init__(self):
        self.action_encoder = ActionEncoder()
        
    def encode_hand(self, hand: List[int]) -> np.ndarray:
        vec = np.zeros(15, dtype=np.float32)
        for card in hand:
            if 3 <= card <= 17:
                vec[card - 3] += 1
        return vec
    
    def encode_played(self, played_cards: List[int]) -> np.ndarray:
        return self.encode_hand(played_cards)
    
    def encode_last_action(self, pattern: Optional[CardPattern]) -> np.ndarray:
        if pattern is None:
            return np.zeros(4, dtype=np.float32)
        return np.array(pattern.to_encoding(), dtype=np.float32)
    
    def encode_role(self, is_landlord: bool) -> np.ndarray:
        return np.array([1.0, 0.0] if is_landlord else [0.0, 1.0], dtype=np.float32)
    
    def encode_position(self, position: int) -> np.ndarray:
        vec = np.zeros(3, dtype=np.float32)
        vec[position] = 1.0
        return vec
    
    def encode(self, hand: List[int], played: List[int], 
               last_pattern: Optional[CardPattern], is_landlord: bool, 
               position: int) -> np.ndarray:
        parts = [
            self.encode_hand(hand),
            self.encode_played(played),
            self.encode_last_action(last_pattern),
            self.encode_role(is_landlord),
            self.encode_position(position),
            np.array([len(hand)], dtype=np.float32)
        ]
        return np.concatenate(parts)
    
    def create_action_mask(self, hand: List[int], last_pattern: Optional[CardPattern]) -> np.ndarray:
        mask = np.zeros(self.action_encoder.max_actions, dtype=np.float32)
        legal_patterns = filter_legal_patterns(hand, last_pattern)
        for pattern in legal_patterns:
            idx = self.action_encoder.pattern_to_idx(pattern)
            if 0 <= idx < self.action_encoder.max_actions:
                mask[idx] = 1.0
        return mask


# ============== MAPPO训练器 ==============
class MAPPOTrainer:
    """MAPPO训练器."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device)
        
        self.agents = [
            ActorCritic(config.state_dim, config.hidden_dim, config.action_dim).to(self.device)
            for _ in range(3)
        ]
        
        self.optimizers = [
            optim.Adam(agent.parameters(), lr=config.lr)
            for agent in self.agents
        ]
        
        self.encoder = StateEncoder()
        self.buffer = [[] for _ in range(3)]
        
    def select_action(self, agent_id: int, state: np.ndarray, mask: np.ndarray) -> Tuple[int, float, float]:
        state_tensor = torch.FloatTensor(state).to(self.device)
        mask_tensor = torch.FloatTensor(mask).to(self.device)
        action, log_prob, value = self.agents[agent_id].get_action(state_tensor, mask_tensor)
        return action, log_prob.item(), value.item()
    
    def store_transition(self, agent_id: int, state, action, reward, log_prob, value, mask, done):
        self.buffer[agent_id].append({
            'state': state,
            'action': action,
            'reward': reward,
            'log_prob': log_prob,
            'value': value,
            'mask': mask,
            'done': done
        })
    
    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return np.array(advantages)
    
    def update(self, agent_ids: List[int] = None):
        if agent_ids is None:
            agent_ids = [0, 1, 2]
        
        total_loss = 0
        for agent_id in agent_ids:
            if len(self.buffer[agent_id]) == 0:
                continue
            
            states = torch.FloatTensor(np.array([t['state'] for t in self.buffer[agent_id]])).to(self.device)
            actions = torch.LongTensor([t['action'] for t in self.buffer[agent_id]]).to(self.device)
            old_log_probs = torch.FloatTensor([t['log_prob'] for t in self.buffer[agent_id]]).to(self.device)
            rewards = [t['reward'] for t in self.buffer[agent_id]]
            values = [t['value'] for t in self.buffer[agent_id]]
            masks = torch.FloatTensor(np.array([t['mask'] for t in self.buffer[agent_id]])).to(self.device)
            dones = [t['done'] for t in self.buffer[agent_id]]
            
            advantages = self.compute_gae(rewards, values, dones)
            advantages = torch.FloatTensor(advantages).to(self.device)
            returns = advantages + torch.FloatTensor(values).to(self.device)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for _ in range(self.config.update_epochs):
                action_probs, new_values = self.agents[agent_id](states, masks)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(new_values.squeeze(), returns)
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy.mean()
                
                self.optimizers[agent_id].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agents[agent_id].parameters(), self.config.max_grad_norm)
                self.optimizers[agent_id].step()
                total_loss += loss.item()
        
        for agent_id in agent_ids:
            self.buffer[agent_id] = []
        
        return total_loss
    
    def save_models(self, path: str = "./models"):
        """保存模型到指定目录.
        
        Args:
            path: 模型保存目录，默认为 ./models
        """
        os.makedirs(path, exist_ok=True)
        for i, agent in enumerate(self.agents):
            torch.save(agent.state_dict(), os.path.join(path, f"model_{chr(ord('a') + i)}.pth"))
    
    def save_model(self, agent_id: int, path: str = "./models"):
        """保存单个agent的模型（用于实时更新）.
        
        Args:
            agent_id: agent编号 (0, 1, 2)
            path: 模型保存目录
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.agents[agent_id].state_dict(), 
                  os.path.join(path, f"model_{chr(ord('a') + agent_id)}.pth"))
    
    def load_models(self, path: str):
        for i, agent in enumerate(self.agents):
            model_path = os.path.join(path, f"model_{chr(ord('a') + i)}.pth")
            if os.path.exists(model_path):
                agent.load_state_dict(torch.load(model_path, map_location=self.device))


# ============== 游戏环境 ==============
class DouDizhuEnv:
    """斗地主游戏环境."""
    
    def __init__(self, decks: List[Dict]):
        self.encoder = StateEncoder()
        self.decks = decks
        self.deck_index = 0
        self.game_id_counter = 0
        self.reset()
    
    def reset(self):
        if self.deck_index >= len(self.decks):
            self.deck_index = 0
        
        deck = self.decks[self.deck_index]
        self.deck_index += 1
        
        self.hands = [
            deck["player1"].copy(),
            deck["player2"].copy(),
            deck["player3"].copy()
        ]
        self.bottom = deck["bottom"].copy()
        
        self.landlord = None
        self.bid_score = 0
        self.current_player = 0
        self.last_pattern = None
        self.last_player = None
        self.played_cards = []
        self.bomb_count = 0
        self.rocket_count = 0
        self.done = False
        self.winner = None
        self.history = [[] for _ in range(3)]
        self.step_count = 0
        self.pass_count = 0
        self.game_id = self.game_id_counter
        self.game_id_counter += 1
        
        return self.get_obs()
    
    def get_obs(self):
        obs = []
        for i in range(3):
            is_landlord = (i == self.landlord) if self.landlord is not None else False
            state = self.encoder.encode(
                self.hands[i],
                self.played_cards,
                self.last_pattern,
                is_landlord,
                i
            )
            mask = self.encoder.create_action_mask(self.hands[i], self.last_pattern)
            obs.append((state, mask))
        return obs
    
    def bid_phase(self, bids: List[int]) -> Tuple[bool, int]:
        max_bid = 0
        max_bidder = None
        
        for i, bid in enumerate(bids):
            if bid > max_bid:
                max_bid = bid
                max_bidder = i
        
        if max_bidder is not None and max_bid > 0:
            self.landlord = max_bidder
            self.bid_score = max_bid
            self.hands[self.landlord].extend(self.bottom)
            self.hands[self.landlord].sort()
            self.current_player = self.landlord
            return True, max_bid
        else:
            return False, 0
    
    def step(self, player_id: int, action_idx: int) -> Tuple[List, List[float], bool]:
        pattern = self.encoder.action_encoder.idx_to_pattern(action_idx)
        cards_to_play = pattern.cards if pattern else []
        
        hand_before = self.hands[player_id].copy()
        
        if pattern.card_type != CardType.PASS:
            self.hands[player_id] = remove_cards(self.hands[player_id], cards_to_play)
            self.last_pattern = pattern
            self.last_player = player_id
            self.played_cards.extend(cards_to_play)
            self.history[player_id].append(pattern)
            
            if pattern.card_type == CardType.BOMB:
                self.bomb_count += 1
            elif pattern.card_type == CardType.ROCKET:
                self.rocket_count += 1
            
            if len(self.hands[player_id]) == 0:
                self.done = True
                self.winner = player_id
                rewards = self._compute_rewards()
                return self.get_obs(), rewards, self.done
            
            self.current_player = (player_id + 1) % 3
            self.step_count += 1
        else:
            self.pass_count += 1
            self.current_player = (player_id + 1) % 3
            self.step_count += 1
            
            if self.last_player == (player_id + 2) % 3:
                self.last_pattern = None
        
        return self.get_obs(), [0, 0, 0], self.done
    
    def _compute_rewards(self):
        multiplier = 2 ** (self.bomb_count + self.rocket_count)
        base_score = self.bid_score * multiplier
        
        rewards = [0, 0, 0]
        
        if self.winner == self.landlord:
            # 地主赢：地主获得 base_score，两个农民各失去 base_score/2
            # 这样总和为0
            rewards[self.landlord] = base_score
            for i in range(3):
                if i != self.landlord:
                    rewards[i] = -base_score / 2
        else:
            # 农民赢：两个农民各获得 base_score/2，地主失去 base_score
            # 这样总和为0
            rewards[self.landlord] = -base_score
            for i in range(3):
                if i != self.landlord:
                    rewards[i] = base_score / 2
        
        return [r / 10.0 for r in rewards]


# ============== 训练函数 ==============
def run_training_episodes(env: DouDizhuEnv, trainer: MAPPOTrainer,
                         logger: TrainLogger, num_episodes: int, 
                         start_episode: int, train_agent_ids: List[int],
                         phase_name: str = "") -> Dict:
    """运行训练回合."""
    
    print(f"\n{'='*60}")
    print(f"开始训练: {phase_name}")
    print(f"训练Agent: {train_agent_ids}")
    print(f"总轮数: {num_episodes}")
    print(f"{'='*60}\n")
    
    # 累计得分跟踪
    cumulative_scores = [0.0, 0.0, 0.0]  # 各位置累计得分
    score_history = []  # 用于计算得分变化
    
    # 原有统计信息跟踪
    total_games = 0  # 总局数（不含流局）
    draw_count = 0  # 流局数
    win_count = [0, 0, 0]  # 各位置获胜场数
    total_bids = []  # 所有成交分数（流局为0）
    total_action_counts = []  # 所有出牌次数（不含流局）
    
    # 用于计算报告间隔之间的统计
    last_report_games = 0  # 上次报告时的总局数
    last_report_draws = 0  # 上次报告时的流局数
    last_report_wins = [0, 0, 0]  # 上次报告时的获胜场数
    last_report_bids = []  # 上次报告时的成交分数列表
    last_report_actions = []  # 上次报告时的出牌次数列表
    
    for episode in range(num_episodes):
        global_episode = start_episode + episode
        obs = env.reset()
        
        # 叫分阶段
        bids = [random.randint(0, 3) for _ in range(3)]
        bid_success, final_bid = env.bid_phase(bids)
        
        if not bid_success:
            # 流局 - 记录到日志
            logger.log_game(
                episode=global_episode,
                game_id=env.game_id,
                bids=bids,
                final_bid=0,
                landlord=-1,
                is_draw=True,
                winner=-1,
                game_length=0,
                rewards=[0, 0, 0],
                roles=['none', 'none', 'none'],
                action_counts=[0, 0, 0],
                bomb_count=0,
                rocket_count=0
            )
            # 更新流局统计
            draw_count += 1
            total_bids.append(0)  # 流局成交分数为0
            
            if episode % config.save_interval == 0:
                print(f"Episode {global_episode}: 流局")
            continue
        
        # 确定角色
        roles = ['farmer', 'farmer', 'farmer']
        roles[env.landlord] = 'landlord'
        
        # 游戏循环
        step_count = 0
        max_steps = 100
        action_counts = [0, 0, 0]
        
        while not env.done and step_count < max_steps:
            player_id = env.current_player
            state, mask = obs[player_id]
            
            action, log_prob, value = trainer.select_action(player_id, state, mask)
            next_obs, step_rewards, done = env.step(player_id, action)
            
            pattern = trainer.encoder.action_encoder.idx_to_pattern(action)
            if pattern and pattern.card_type != CardType.PASS:
                action_counts[player_id] += 1
            
            if player_id in train_agent_ids:
                trainer.store_transition(
                    player_id, state, action, 0, log_prob, value, mask, False
                )
            
            obs = next_obs
            step_count += 1
            
            if done:
                # 游戏结束，step_rewards 是最终得分
                final_rewards = step_rewards
                
                for i in range(3):
                    if len(trainer.buffer[i]) > 0:
                        trainer.buffer[i][-1]['reward'] = final_rewards[i]
                        trainer.buffer[i][-1]['done'] = True
                
                # 更新累计得分（使用最终得分）
                for i in range(3):
                    cumulative_scores[i] += final_rewards[i]
                score_history.append(final_rewards[:])
                
                # 更新统计信息
                total_games += 1
                win_count[env.winner] += 1
                total_bids.append(final_bid)
                total_action_counts.append(sum(action_counts))
                
                # 记录游戏结果（使用最终得分）
                logger.log_game(
                    episode=global_episode,
                    game_id=env.game_id,
                    bids=bids,
                    final_bid=final_bid,
                    landlord=env.landlord,
                    is_draw=False,
                    winner=env.winner,
                    game_length=step_count,
                    rewards=final_rewards,
                    roles=roles,
                    action_counts=action_counts,
                    bomb_count=env.bomb_count,
                    rocket_count=env.rocket_count
                )
                break
        
        # 定期更新策略
        if episode % config.batch_size == 0 and episode > 0:
            trainer.update(train_agent_ids)
        
        # 定期记录汇总
        if episode % config.eval_interval == 0 and episode > 0:
            logger.log_episode_summary(global_episode)
        
        # 打印进度
        if episode % config.save_interval == 0 and episode > 0:
            # 计算本报告间隔之间的统计（从上次报告到现在）
            period_games = total_games - last_report_games
            period_draws = draw_count - last_report_draws
            period_attempts = period_games + period_draws
            
            # 计算本期间总得分（从score_history中取出本期间的数据）
            period_scores = [0.0, 0.0, 0.0]
            n_period = len(score_history) - (total_games - period_games)
            if n_period > 0 and len(score_history) >= n_period:
                period_history = score_history[-n_period:]
                for i in range(3):
                    period_scores[i] = sum(h[i] for h in period_history)
            
            # 计算本期间其他统计
            period_bids = total_bids[last_report_games:] if len(total_bids) > last_report_games else []
            period_actions = total_action_counts[last_report_games:] if len(total_action_counts) > last_report_games else []
            
            avg_final_bid = sum(period_bids) / len(period_bids) if period_bids else 0
            avg_action_count = sum(period_actions) / len(period_actions) if period_actions else 0
            
            draw_rate = period_draws / period_attempts if period_attempts > 0 else 0
            
            # 计算本期间胜率
            period_wins = [win_count[i] - last_report_wins[i] for i in range(3)]
            win_rates = [period_wins[i] / period_games if period_games > 0 else 0 for i in range(3)]
            
            print(f"\n{'='*60}")
            print(f"Episode {global_episode} 进度报告")
            print(f"{'='*60}")
            print(f"本期间局数: {period_attempts} | 有效: {period_games} | 流局: {period_draws} ({draw_rate:.1%})")
            print(f"累计局数: {total_games + draw_count}")
            print(f"平均成交分数: {avg_final_bid:.2f} | 平均出牌次数: {avg_action_count:.1f}")
            print(f"-"*60)
            print(f"位置0: 累计得分={cumulative_scores[0]:+.2f}, 本期间得分={period_scores[0]:+.2f}, 胜率={win_rates[0]:.1%} ({period_wins[0]}场)")
            print(f"位置1: 累计得分={cumulative_scores[1]:+.2f}, 本期间得分={period_scores[1]:+.2f}, 胜率={win_rates[1]:.1%} ({period_wins[1]}场)")
            print(f"位置2: 累计得分={cumulative_scores[2]:+.2f}, 本期间得分={period_scores[2]:+.2f}, 胜率={win_rates[2]:.1%} ({period_wins[2]}场)")
            print(f"{'='*60}")
            
            # 更新上次报告的状态
            last_report_games = total_games
            last_report_draws = draw_count
            last_report_wins = win_count.copy()
            last_report_bids = len(total_bids)
            last_report_actions = len(total_action_counts)
    
    return {'final_episode': start_episode + num_episodes}


def evaluate_agent_vs_baseline(env: DouDizhuEnv, trained_trainer: MAPPOTrainer,
                                baseline_trainer: MAPPOTrainer, trained_agent_id: int,
                                num_games: int, episode_offset: int) -> Dict:
    """评估训练后的agent与初始模型的对战表现.
    
    Args:
        env: 游戏环境
        trained_trainer: 包含训练后agent的trainer
        baseline_trainer: 包含初始模型的trainer（作为对手）
        trained_agent_id: 被训练的agent ID (0, 1, 2)
        num_games: 测试局数
        episode_offset: 当前episode偏移量（用于game_id）
        
    Returns:
        评估结果字典
    """
    wins = 0
    total_score = 0.0
    
    for game_idx in range(num_games):
        obs = env.reset()
        bids = [random.randint(0, 3) for _ in range(3)]
        bid_success, final_bid = env.bid_phase(bids)
        
        if not bid_success:
            continue
        
        # 游戏循环
        step_count = 0
        max_steps = 100
        
        while not env.done and step_count < max_steps:
            player_id = env.current_player
            state, mask = obs[player_id]
            
            # 选择模型：如果是被训练的agent，使用trained_trainer，否则使用baseline_trainer
            if player_id == trained_agent_id:
                action, _, _ = trained_trainer.agents[player_id].get_action(
                    torch.FloatTensor(state).to(trained_trainer.device),
                    torch.FloatTensor(mask).to(trained_trainer.device)
                )
            else:
                action, _, _ = baseline_trainer.agents[player_id].get_action(
                    torch.FloatTensor(state).to(baseline_trainer.device),
                    torch.FloatTensor(mask).to(baseline_trainer.device)
                )
            
            next_obs, rewards, done = env.step(player_id, action)
            obs = next_obs
            step_count += 1
            
            if done:
                # 记录结果
                if env.winner == trained_agent_id:
                    wins += 1
                    total_score += rewards[trained_agent_id]
                else:
                    total_score += rewards[trained_agent_id]
                break
    
    valid_games = min(game_idx + 1, num_games)
    return {
        'win_rate': wins / valid_games if valid_games > 0 else 0,
        'avg_score': total_score / valid_games if valid_games > 0 else 0,
        'total_games': valid_games,
        'wins': wins
    }


def save_eval_results(log_dir: str, agent_name: str, round_num: int,
                     episode: int, results: Dict):
    """保存评估结果到日志文件.
    
    Args:
        log_dir: 日志目录
        agent_name: agent名称 (a, b, c)
        round_num: 强化轮次
        episode: 当前episode
        results: 评估结果
    """
    eval_log_path = os.path.join(log_dir, f"eval_results_{agent_name}.jsonl")
    
    eval_data = {
        'type': 'eval',
        'agent': agent_name,
        'round_num': round_num,
        'episode': episode,
        'win_rate': results['win_rate'],
        'avg_score': results['avg_score'],
        'total_games': results['total_games'],
        'wins': results['wins'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(eval_log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(eval_data) + '\n')


def train():
    """主训练函数."""
    config = Config()
    
    # 生成或加载牌组
    if not os.path.exists(config.decks_file):
        print("生成牌组...")
        total_episodes = config.initial_episodes + config.fine_tune_episodes * 3 * config.fine_tune_rounds
        generate_and_save(total_episodes + 1000, config.decks_file, seed=42)
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("错误: CUDA不可用")
        return
    
    print(f"加载牌组: {config.decks_file}")
    decks = load_decks_from_file(config.decks_file)
    
    # 初始化
    trainer = MAPPOTrainer(config)
    env = DouDizhuEnv(decks)
    
    current_episode = 0
    
    # ========== 阶段1: 初始训练 ==========
    print("\n" + "="*60)
    print("阶段1: 初始训练")
    print("="*60)
    
    logger = TrainLogger(log_dir=config.log_dir, phase="initial", round_num=0)
    
    run_training_episodes(
        env, trainer, logger,
        config.initial_episodes, current_episode,
        [0, 1, 2],
        "Initial Training"
    )
    
    current_episode += config.initial_episodes
    logger.close()
    
    # 保存初始模型（作为基准，不会被覆盖）
    os.makedirs("./models/initial", exist_ok=True)
    trainer.save_models("./models/initial")
    print("\n初始模型已保存到: ./models/initial/")
    
    # 加载初始模型作为对手基准
    baseline_trainer = MAPPOTrainer(config)
    baseline_trainer.load_models("./models/initial")
    
    # ========== 阶段2+: 轮流强化训练 ==========
    for round_num in range(1, config.fine_tune_rounds + 1):
        for agent_id, agent_name in [(2, 'c'), (0, 'a'), (1, 'b')]:
            phase_name = f"Round {round_num}: Train Agent {agent_name.upper()}"
            print("\n" + "="*60)
            print(phase_name)
            print("="*60)
            
            logger = TrainLogger(
                log_dir=config.log_dir, 
                phase=f"agent_{agent_name}", 
                round_num=round_num
            )
            
            run_training_episodes(
                env, trainer, logger,
                config.fine_tune_episodes, current_episode,
                [agent_id],
                phase_name
            )
            
            current_episode += config.fine_tune_episodes
            logger.close()
            
            # 实时更新模型到 models/ 目录
            trainer.save_model(agent_id, "./models")
            
            # 强化训练后测试：与初始模型对战
            if config.eval_after_finetune:
                print(f"\n--- 测试 Agent {agent_name.upper()} vs 初始模型 ---")
                eval_results = evaluate_agent_vs_baseline(
                    env, trainer, baseline_trainer, agent_id, 
                    config.eval_games, current_episode
                )
                print(f"胜率: {eval_results['win_rate']:.1%} | 平均得分: {eval_results['avg_score']:+.3f}")
                
                # 保存测试结果到日志
                save_eval_results(config.log_dir, agent_name, round_num, 
                                current_episode, eval_results)
    
    # 最终保存（所有agent的最新模型已在 models/ 中）
    trainer.save_models("./models")
    print("\n模型已保存到: ./models/")
    print("  - model_a.pth (Agent A)")
    print("  - model_b.pth (Agent B)")
    print("  - model_c.pth (Agent C)")
    
    print("\n" + "="*60)
    print("训练完成！")
    print(f"日志目录: {config.log_dir}")
    print("运行 visualizer.py 生成图表")
    print("="*60)


if __name__ == "__main__":
    train()
