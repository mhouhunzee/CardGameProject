"""
斗地主训练脚本

使用MCTS + 深度强化学习训练AI
"""

import os
import json
import random
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Tuple

from config import (
    CYCLE, H, MODEL_DIR, LOG_DIR, LEARNING_RATE, GAMMA,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    ADAPTIVE_TRAINING, WIN_RATE_THRESHOLD, TRAINING_BOOST_FACTOR,
    MAX_BOOST_FACTOR, CONSECUTIVE_LOW_WIN_ROUNDS,
    SAVE_GAME_LOGS, LOG_BUFFER_SIZE, TRAIN_FREQUENCY,
    USE_MCTS, MCTS_NUM_SIMULATIONS, MCTS_TEMPERATURE,
    USE_PARALLEL_MCTS, PARALLEL_BATCH_SIZE,NUM_PAIKU
)
from env import DouDiZhuEnv
from model import DouDiZhuAgent
from generator import generate_paiku, save_paiku, load_paiku, parse_paiku_line


class MAPPOTrainer:
    """MAPPO训练器"""
    
    def __init__(self):
        self.env = DouDiZhuEnv()
        self.agents = [DouDiZhuAgent(i) for i in range(3)]
        
        # 创建目录
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 加载或生成牌库
        self.paiku = self._load_or_generate_paiku()
        
        # 训练统计
        self.stats = {
            'cycle': [],
            'wins': [0, 0, 0],
            'games': 0
        }
        
        # 自适应训练状态
        self.boost_factors = [1.0, 1.0, 1.0]
        self.low_win_counts = [0, 0, 0]
        
        # 日志
        self.log_file = os.path.join(LOG_DIR, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')
        self.log_buffer = []
    
    def _load_or_generate_paiku(self):
        """加载或生成牌库"""
        if os.path.exists('paiku.txt'):
            print("加载现有牌库...")
            return load_paiku('paiku.txt')
        else:
            print("生成新牌库...")
            paiku = generate_paiku(NUM_PAIKU)
            save_paiku(paiku, 'paiku.txt')
            return paiku
    
    def play_one_game(self, training_agent: int, epsilon: float) -> Dict:
        """进行一局游戏"""
        self.env.reset()
        
        # 随机选择牌组
        line = random.choice(self.paiku)
        hand_A, hand_B, hand_C, base = parse_paiku_line(line)
        
        # 发牌
        self.env.deal_cards(hand_A, hand_B, hand_C, base)
        
        # 叫分
        bids = []
        for pos in range(3):
            hand = [hand_A, hand_B, hand_C][pos]
            bid = self.agents[pos].select_bid(hand, epsilon)
            bids.append(bid)
        
        success, landlord, final_bid = self.env.bidding_phase(*bids)
        
        if not success:
            return {'winner': None, 'rewards': [0, 0, 0], 'is_draw': True}
        
        # 初始化出牌状态
        for pos in range(3):
            hand = [hand_A, hand_B, hand_C][pos]
            if pos == landlord:
                hand = hand + base
            self.agents[pos].init_play_state(hand, pos)
        
        # 出牌阶段
        self.env.start_playing()
        
        max_steps = 100
        step = 0
        
        while not self.env.done and step < max_steps:
            player = self.env.current_player
            agent = self.agents[player]
            
            # 获取当前状态（用于训练）
            current_state = agent.current_state.copy() if agent.current_state is not None else None
            
            # 获取合法动作
            legal_actions = self.env.get_legal_actions(player)
            
            # 选择动作
            action = agent.select_play(legal_actions, epsilon=0.1 if player == training_agent else 0.0)
            
            # 执行动作
            success, msg, done, _ = self.env.step(player, action)
            
            if not success:
                # 如果出错，强制PASS
                self.env.step(player, "PASS")
                action = "PASS"
            
            # 更新状态：使用神经网络预测
            if current_state is not None:
                state_tensor = torch.FloatTensor(current_state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    (_, _, _, _, new_remaining, new_opponent_pred, new_opponent_conf) = agent.play_net(state_tensor)
                    
                    # 更新状态，传入当前手牌（如果出牌了则手牌已减少）
                    current_hand = self.env.hands[player]
                    agent.update_play_state(
                        new_remaining.cpu().numpy()[0],
                        new_opponent_pred.cpu().numpy()[0],
                        new_opponent_conf.cpu().numpy()[0],
                        current_hand
                    )
            
            step += 1
        
        # 计算奖励
        rewards = [self.env._calculate_reward(i) for i in range(3)]
        
        return {
            'winner': self.env.winner,
            'rewards': rewards,
            'landlord': landlord,
            'is_draw': False
        }
    
    def train_cycle(self, cycle_num: int):
        """训练一个cycle"""
        print(f"\n{'='*60}")
        print(f"Cycle {cycle_num + 1}/{CYCLE}")
        print(f"{'='*60}")
        
        # 计算epsilon
        epsilon = max(EPSILON_MIN, EPSILON_START * (EPSILON_DECAY ** cycle_num))
        
        # 训练每个agent
        for agent_idx in range(3):
            num_games = int(H * self.boost_factors[agent_idx])
            print(f"\n训练 Agent {agent_idx} ({num_games}局, epsilon={epsilon:.3f})")
            
            wins = 0
            total_reward = 0.0
            
            for game in tqdm(range(num_games), desc=f"Agent {agent_idx}"):
                result = self.play_one_game(agent_idx, epsilon)
                
                if not result['is_draw']:
                    total_reward += result['rewards'][agent_idx]
                    
                    # 判断胜负
                    is_landlord = (result['landlord'] == agent_idx)
                    is_winner = (result['winner'] == 'landlord' and is_landlord) or \
                               (result['winner'] == 'farmers' and not is_landlord)
                    
                    if is_winner:
                        wins += 1
            
            # 统计
            win_rate = wins / max(1, num_games)
            avg_reward = total_reward / max(1, num_games)
            print(f"  胜率: {win_rate:.1%}, 平均奖励: {avg_reward:+.2f}")
            
            # 自适应训练
            if ADAPTIVE_TRAINING:
                if win_rate < WIN_RATE_THRESHOLD:
                    self.low_win_counts[agent_idx] += 1
                    if self.low_win_counts[agent_idx] >= CONSECUTIVE_LOW_WIN_ROUNDS:
                        self.boost_factors[agent_idx] = min(MAX_BOOST_FACTOR, 
                                                          self.boost_factors[agent_idx] * TRAINING_BOOST_FACTOR)
                        print(f"  触发增强训练: boost={self.boost_factors[agent_idx]:.1f}x")
                else:
                    self.low_win_counts[agent_idx] = 0
        
        # 保存模型
        self._save_models(cycle_num)
    
    def _save_models(self, cycle_num: int):
        """保存模型"""
        print("\n[保存模型]")
        for i in range(3):
            model_path = os.path.join(MODEL_DIR, f"agent_{i}_cycle_{cycle_num:03d}.pth")
            self.agents[i].save(model_path)
            print(f"  Agent {i}: {model_path}")
    
    def train(self):
        """完整训练流程"""
        print("=" * 60)
        print("斗地主AI训练")
        print("=" * 60)
        print(f"配置: {CYCLE} cycles, 每cycle每agent训练{H}局")
        print(f"MCTS: {USE_MCTS}, 并行: {USE_PARALLEL_MCTS}")
        print("=" * 60)
        
        for cycle in range(CYCLE):
            self.train_cycle(cycle)
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)


if __name__ == "__main__":
    trainer = MAPPOTrainer()
    trainer.train()
