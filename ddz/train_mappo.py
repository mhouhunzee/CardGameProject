"""
MAPPO 训练脚本 - 完整的强化学习训练流程
"""
import os
import sys
import json
import random
import numpy as np
import torch
from tqdm import tqdm
from datetime import datetime
from typing import List, Tuple, Dict

from config import (CYCLE, H, MODEL_DIR, LOG_DIR, LEARNING_RATE,
    EPSILON_START, EPSILON_MIN,
    ADAPTIVE_TRAINING, WIN_RATE_THRESHOLD, TRAINING_BOOST_FACTOR,
    MAX_BOOST_FACTOR, CONSECUTIVE_LOW_WIN_ROUNDS,
    ENABLE_TRAINING_PROTECTION, WIN_RATE_DROP_THRESHOLD,
    CONSECUTIVE_DROPS_BEFORE_ROLLBACK, LEARNING_RATE_ADJUSTMENT,
    MODEL_KEEP_LAST_N, TRAIN_FREQUENCY, SEGMENT_SIZE)
from model import DouDiZhuAgent
from mappo_trainer import MAPPOTrainer
from env import DouDiZhuEnv
from generator import load_paiku, parse_paiku_line
from card_utils import CardPattern, get_legal_plays


class MAPPOTrainingPipeline:
    """MAPPO 训练流程"""
    
    def __init__(self):
        # 创建目录
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 初始化三个Agent
        self.agents = [DouDiZhuAgent(i) for i in range(3)]
        
        # 初始化MAPPO训练器
        config = {
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'max_grad_norm': 0.5,
            'batch_size': 64,
            'update_epochs': 4,
            'lr': LEARNING_RATE
        }
        self.trainer = MAPPOTrainer(self.agents, config)
        
        # 加载已有模型
        self.trainer.load_models(MODEL_DIR)
        
        # 初始化环境
        self.env = DouDiZhuEnv()
        self.paiku = load_paiku()
        
        # 训练统计
        self.stats = {
            'cycle_rewards': [[] for _ in range(3)],  # 每个agent的奖励
            'cycle_wins': [0, 0, 0],  # 每个agent的胜场
            'cycle_games': 0,  # 总局数
        }
        
        # 自适应训练状态
        self.boost_factors = [1.0, 1.0, 1.0]  # 每个agent的训练量倍增因子
        self.low_win_counts = [0, 0, 0]  # 连续低胜率轮数
        
        # 模型表现跟踪（用于比较新旧模型）
        self.prev_win_rates = [0.0, 0.0, 0.0]  # 上一轮胜率
        self.consecutive_drops = [0, 0, 0]  # 连续胜率下降次数
        self.agent_best_scores = [0.0, 0.0, 0.0]  # 每个agent的历史最佳表现分数
        
        # 分段统计（用于更细粒度的监控）
        self.segment_stats = [[] for _ in range(3)]  # 每个agent的分段统计
        
        # 日志文件
        self.log_file = os.path.join(LOG_DIR, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')
        
        # 加载既有模型（如果存在）
        self._load_existing_models()
    
    def _load_existing_models(self):
        """加载models文件夹中的既有模型"""
        print("\n[加载既有模型]")
        for i in range(3):
            model_path = os.path.join(MODEL_DIR, f"agent_{chr(65+i)}_best.pth")
            if os.path.exists(model_path):
                try:
                    self.agents[i].load(model_path)
                    print(f"  Agent {chr(65+i)}: 已加载 {model_path}")
                    # 设置初始分数为0，等待第一轮训练后更新
                    self.agent_best_scores[i] = 0.0
                except Exception as e:
                    print(f"  Agent {chr(65+i)}: 加载失败 - {e}")
            else:
                print(f"  Agent {chr(65+i)}: 无既有模型，从头训练")
    
    def write_log(self, entry: Dict):
        """写入日志，处理numpy类型"""
        def convert(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        entry = convert(entry)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def select_action_with_logprob(self, agent_idx: int, legal_plays: List[str], 
                                   state: np.ndarray, epsilon: float = 0.0,
                                   last_play=None) -> Tuple[str, float, float]:
        """
        选择动作，并返回log概率和价值
        
        返回: (action, log_prob, value)
        """
        agent = self.agents[agent_idx]
        
        if not legal_plays:
            return "PASS", 0.0, 0.0
        
        if len(legal_plays) == 1:
            # 只能PASS
            return legal_plays[0], 0.0, 0.0
        
        # 使用神经网络决策
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        
        with torch.no_grad():
            # 确定合法动作类型
            from card_utils import CardPattern
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
             _, _, _) = agent.play_net(state_tensor, legal_action_types)
            
            # 获取动作概率
            action_type_probs = torch.softmax(action_type_logits, dim=-1)
            
            # 选择动作
            if random.random() < epsilon:
                # 探索
                action = random.choice(legal_plays)
                action_idx = legal_plays.index(action)
                log_prob = torch.log(action_type_probs[0, min(action_idx, 6)] + 1e-8).item()
            else:
                # 利用：使用select_play的逻辑
                action = agent.select_play(legal_plays, epsilon=0, last_play=last_play)
                
                # 计算log概率（简化）
                action_type_dist = torch.distributions.Categorical(action_type_probs)
                action_type = action_type_dist.sample()
                log_prob = action_type_dist.log_prob(action_type).item()
        
        return action, log_prob, state_value.item()
    
    def play_one_game(self, training_agents: List[int], epsilon: float = 0.0) -> Dict:
        """
        进行一局游戏
        
        training_agents: 正在训练的agent索引列表
        epsilon: 探索率
        
        返回: 游戏结果
        """
        # 重置环境
        self.env.reset()
        
        # 随机选择牌组
        line = random.choice(self.paiku)
        hand_A, hand_B, hand_C, base = parse_paiku_line(line)
        
        # 发牌
        self.env.deal_cards(hand_A, hand_B, hand_C, base)
        
        # 叫分阶段
        bids = []
        for pos in range(3):
            hand = [hand_A, hand_B, hand_C][pos]
            bid = self.agents[pos].select_bid(hand, epsilon)
            bids.append(bid)
        
        success, landlord, final_bid = self.env.bidding_phase(*bids)
        
        if not success:
            # 流局
            return {
                'winner': None,
                'rewards': [0, 0, 0],
                'landlord': None,
                'is_draw': True
            }
        
        # 初始化出牌状态
        for pos in range(3):
            hand = [hand_A, hand_B, hand_C][pos]
            if pos == landlord:
                hand = hand + base
            self.agents[pos].init_play_state(hand, pos)
        
        # 出牌阶段
        self.env.start_playing()
        
        # 记录每个agent的经验
        episode_data = [[] for _ in range(3)]  # 每个agent的(s, a, r, v, done)
        
        max_steps = 100
        step = 0
        
        while not self.env.done and step < max_steps:
            player = self.env.current_player
            agent = self.agents[player]
            
            # 获取合法动作
            legal_actions = self.env.get_legal_actions(player)
            
            # 获取当前状态
            state = agent.current_state.copy() if agent.current_state is not None else np.zeros(70)
            
            # 获取上家出牌（用于判断是否需要炸弹）
            last_pattern = None
            if self.env.last_play is not None and self.env.last_player != player:
                from card_utils import CardPattern
                last_pattern = CardPattern(self.env.last_play.cards_str)
            
            # 选择动作
            if player in training_agents:
                action, log_prob, value = self.select_action_with_logprob(
                    player, legal_actions, state, epsilon, last_pattern
                )
            else:
                # 非训练agent，只选择动作，不记录
                action = agent.select_play(legal_actions, epsilon=0, last_play=last_pattern)
                log_prob, value = 0.0, 0.0
            
            # 执行动作
            success, msg, done, reward = self.env.step(player, action)
            
            if not success:
                action = "PASS"
                success, msg, done, reward = self.env.step(player, action)
            
            # 记录经验（只对训练中的agent）
            if player in training_agents:
                episode_data[player].append({
                    'state': state,
                    'action': 0,  # 简化：动作索引
                    'log_prob': log_prob,
                    'reward': 0,  # 临时奖励，最后统一计算
                    'value': value,
                    'done': False
                })
            
            # 更新状态
            if action != "PASS":
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    (_, _, _, _, opponent_pred, next_pred, confidence) = agent.play_net(state_tensor)
                    agent.update_play_state(
                        opponent_pred.cpu().numpy()[0],
                        next_pred.cpu().numpy()[0],
                        confidence.cpu().numpy()[0]
                    )
            
            step += 1
            
            if done:
                break
        
        # 计算最终奖励
        rewards = [self.env._calculate_reward(i) for i in range(3)]
        winner = self.env.winner
        
        # 更新经验中的奖励
        for i in training_agents:
            for data in episode_data[i]:
                data['reward'] = rewards[i]
                data['done'] = True
            
            # 添加到训练缓冲区
            for data in episode_data[i]:
                self.trainer.buffers[i].add(
                    data['state'],
                    data['action'],
                    data['log_prob'],
                    data['reward'],
                    data['value'],
                    data['done'],
                    np.zeros(70)  # 简化：next_state
                )
        
        return {
            'winner': winner,
            'rewards': rewards,
            'landlord': landlord,
            'is_draw': False,
            'game_length': step
        }
    
    def train_cycle(self, cycle_num: int):
        """训练一个cycle，包含自适应训练和训练保护机制"""
        print(f"\n{'='*60}")
        print(f"Cycle {cycle_num + 1}/{CYCLE}")
        print(f"{'='*60}")
        
        # 重置统计
        cycle_stats = {
            'rewards': [0.0, 0.0, 0.0],
            'wins': [0, 0, 0],
            'games': 0,
            'draws': 0
        }
        
        # 计算epsilon（衰减）
        epsilon = max(EPSILON_MIN, EPSILON_START - (EPSILON_START - EPSILON_MIN) * (cycle_num / (CYCLE * 0.7)))
        
        # 训练每个agent
        for agent_idx in range(3):
            # 根据boost_factor计算实际训练局数
            num_games = int(H * self.boost_factors[agent_idx])
            print(f"\n训练 Agent {chr(65+agent_idx)} ({num_games}局, epsilon={epsilon:.3f}, boost={self.boost_factors[agent_idx]:.1f}x)")
            
            # 分段训练（用于更细粒度的监控）
            segment_games = []
            for segment in range(0, num_games, SEGMENT_SIZE):
                segment_end = min(segment + SEGMENT_SIZE, num_games)
                segment_wins = 0
                segment_reward = 0.0
                
                for game in tqdm(range(segment, segment_end), desc=f"Agent {chr(65+agent_idx)} [{segment}-{segment_end}]", leave=False):
                    result = self.play_one_game([agent_idx], epsilon)
                    
                    cycle_stats['games'] += 1
                    if result['is_draw']:
                        cycle_stats['draws'] += 1
                    else:
                        # 更新奖励统计
                        for i in range(3):
                            cycle_stats['rewards'][i] += result['rewards'][i]
                        segment_reward += result['rewards'][agent_idx]
                        
                        # 更新胜场统计
                        if result['winner'] is not None:
                            if result['winner'] == 'landlord':
                                landlord = result['landlord']
                                cycle_stats['wins'][landlord] += 1
                                if landlord == agent_idx:
                                    segment_wins += 1
                            else:
                                for i in range(3):
                                    if i != result['landlord']:
                                        cycle_stats['wins'][i] += 1
                                        if i == agent_idx:
                                            segment_wins += 1
                
                # 记录分段统计
                segment_win_rate = segment_wins / max(1, SEGMENT_SIZE - (segment_end - segment - segment_wins))
                self.segment_stats[agent_idx].append({
                    'cycle': cycle_num,
                    'segment': segment // SEGMENT_SIZE,
                    'win_rate': segment_win_rate,
                    'avg_reward': segment_reward / max(1, segment_end - segment)
                })
                
                # 每TRAIN_FREQUENCY局更新一次策略
                if len(self.trainer.buffers[agent_idx]) >= TRAIN_FREQUENCY:
                    update_info = self.trainer.update(agent_idx)
                    print(f"  Segment {segment//SEGMENT_SIZE}: loss={update_info.get('total_loss', 0):.4f}, win_rate={segment_win_rate:.1%}")
            
            # 更新剩余的策略
            if len(self.trainer.buffers[agent_idx]) > 0:
                update_info = self.trainer.update(agent_idx)
                print(f"  更新完成: loss={update_info.get('total_loss', 0):.4f}")
        
        # 输出统计
        print(f"\n{'='*60}")
        print(f"Cycle {cycle_num + 1} 统计")
        print(f"{'='*60}")
        valid_games = cycle_stats['games'] - cycle_stats['draws']
        current_win_rates = []
        
        for i in range(3):
            avg_reward = cycle_stats['rewards'][i] / max(1, valid_games)
            win_rate = cycle_stats['wins'][i] / max(1, valid_games)
            current_win_rates.append(win_rate)
            print(f"  Agent {chr(65+i)}: 平均奖励={avg_reward:+.3f}, 胜率={win_rate:.1%}, boost={self.boost_factors[i]:.1f}x")
        print(f"  总局数: {cycle_stats['games']}, 流局: {cycle_stats['draws']}")
        print(f"{'='*60}")
        
        # ========== 自适应训练 ==========
        if ADAPTIVE_TRAINING:
            print("\n[自适应训练]")
            for i in range(3):
                if current_win_rates[i] < WIN_RATE_THRESHOLD:
                    self.low_win_counts[i] += 1
                    if self.low_win_counts[i] >= CONSECUTIVE_LOW_WIN_ROUNDS:
                        old_boost = self.boost_factors[i]
                        self.boost_factors[i] = min(MAX_BOOST_FACTOR, self.boost_factors[i] * TRAINING_BOOST_FACTOR)
                        print(f"  Agent {chr(65+i)}: 胜率{current_win_rates[i]:.1%}低于阈值{WIN_RATE_THRESHOLD:.1%}，"
                              f"连续{self.low_win_counts[i]}轮，训练量从{old_boost:.1f}x提升至{self.boost_factors[i]:.1f}x")
                else:
                    if self.low_win_counts[i] > 0:
                        print(f"  Agent {chr(65+i)}: 胜率恢复至{current_win_rates[i]:.1%}，重置连续低胜率计数")
                    self.low_win_counts[i] = 0
        
        # 更新上一轮胜率
        self.prev_win_rates = current_win_rates
        
        # ========== 模型保存策略：保存新模型→比较表现→保留更优者 ==========
        print("\n[模型保存]")
        for i in range(3):
            # 计算当前模型表现分数（胜率*10 + 平均奖励）
            avg_reward = cycle_stats['rewards'][i] / max(1, valid_games)
            win_rate = current_win_rates[i]
            current_score = win_rate * 10 + avg_reward
            
            # 保存新模型（临时）
            new_model_path = os.path.join(MODEL_DIR, f"agent_{chr(65+i)}_new.pth")
            self.agents[i].save(new_model_path)
            
            # 查找旧模型（最佳模型）
            old_model_path = os.path.join(MODEL_DIR, f"agent_{chr(65+i)}_best.pth")
            
            if os.path.exists(old_model_path):
                # 有旧模型，需要比较
                if current_score > self.agent_best_scores[i]:
                    # 新模型更好，替换旧模型
                    print(f"  Agent {chr(65+i)}: 新模型表现更好 "
                          f"(胜率:{win_rate:.1%}, 奖励:{avg_reward:+.3f}, 分数:{current_score:.2f} > {self.agent_best_scores[i]:.2f})")
                    # 删除旧模型，重命名新模型
                    os.remove(old_model_path)
                    os.rename(new_model_path, old_model_path)
                    self.agent_best_scores[i] = current_score
                    print(f"    ✓ 已保留新模型")
                else:
                    # 旧模型更好，删除新模型
                    print(f"  Agent {chr(65+i)}: 旧模型表现更好 "
                          f"(胜率:{win_rate:.1%}, 奖励:{avg_reward:+.3f}, 分数:{current_score:.2f} <= {self.agent_best_scores[i]:.2f})")
                    os.remove(new_model_path)
                    # 加载旧模型（回退）
                    self.agents[i].load(old_model_path)
                    print(f"    ✓ 已删除新模型，回退到旧模型")
            else:
                # 没有旧模型，直接保留新模型
                print(f"  Agent {chr(65+i)}: 首次保存模型 "
                      f"(胜率:{win_rate:.1%}, 奖励:{avg_reward:+.3f}, 分数:{current_score:.2f})")
                os.rename(new_model_path, old_model_path)
                self.agent_best_scores[i] = current_score
                print(f"    ✓ 已保存为最佳模型")
        
        # 记录日志
        log_entry = {
            'type': 'cycle_complete',
            'cycle': cycle_num,
            'stats': cycle_stats,
            'epsilon': epsilon,
            'timestamp': datetime.now().isoformat()
        }
        self.write_log(log_entry)
    
    def train(self):
        """完整训练流程"""
        print("="*60)
        print("MAPPO 斗地主AI训练")
        print("="*60)
        print(f"配置: {CYCLE} cycles, 每cycle每agent训练{H}局")
        print(f"牌库: {len(self.paiku)}个牌组")
        print(f"模型保存: {MODEL_DIR}")
        print(f"日志文件: {self.log_file}")
        print("="*60)
        
        for cycle in range(CYCLE):
            self.train_cycle(cycle)
        
        print("\n" + "="*60)
        print("训练完成!")
        print("="*60)


if __name__ == "__main__":
    # 生成牌库（如果不存在）
    if not os.path.exists('paiku.txt'):
        print("生成牌库...")
        from generator import generate_paiku, save_paiku
        from config import NUM_PAIKU
        paiku_list = generate_paiku(NUM_PAIKU)
        save_paiku(paiku_list)
    
    # 开始训练
    pipeline = MAPPOTrainingPipeline()
    pipeline.train()
