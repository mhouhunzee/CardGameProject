"""
斗地主训练脚本（适配新的70维状态设计）
"""
import os
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime

from config import (
    CYCLE, H, MODEL_DIR, LOG_DIR,
    ADAPTIVE_TRAINING, WIN_RATE_THRESHOLD, TRAINING_BOOST_FACTOR,
    MAX_BOOST_FACTOR, CONSECUTIVE_LOW_WIN_ROUNDS,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY
)
from env import DouDiZhuEnv
from model import DouDiZhuAgent
from generator import load_paiku, parse_paiku_line


class DouDiZhuTrainer:
    """斗地主训练管理器"""
    
    def __init__(self, paiku_file='paiku.txt'):
        # 三个Agent
        self.agents = [
            DouDiZhuAgent(0),  # 头叫
            DouDiZhuAgent(1),  # 二叫
            DouDiZhuAgent(2)   # 三叫
        ]
        
        # 环境
        self.env = DouDiZhuEnv()
        
        # 加载牌库
        self.paiku_list = load_paiku(paiku_file)
        self.current_paiku_idx = 0
        
        # 创建目录
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # 统计
        self.stats = []
        
        # 自适应训练状态
        self.boost_factors = [1.0, 1.0, 1.0]
        self.low_win_counts = [0, 0, 0]
        
        # 初始化日志文件
        self.log_file = os.path.join(LOG_DIR, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')
        self._write_log_header()
        
        # 加载已有模型
        self._load_existing_models()
    
    def _write_log_header(self):
        """写入日志头部信息"""
        header = {
            'type': 'header',
            'timestamp': datetime.now().isoformat(),
            'config': {
                'CYCLE': CYCLE,
                'H': H,
                'ADAPTIVE_TRAINING': ADAPTIVE_TRAINING,
                'WIN_RATE_THRESHOLD': WIN_RATE_THRESHOLD,
                'EPSILON_START': EPSILON_START,
                'EPSILON_MIN': EPSILON_MIN
            }
        }
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(header, ensure_ascii=False) + '\n')
    
    def write_log(self, entry_type, data):
        """写入日志条目"""
        # 转换 numpy 类型为 Python 原生类型
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
        
        entry = {
            'type': entry_type,
            'timestamp': datetime.now().isoformat(),
            'data': convert(data)
        }
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def _load_existing_models(self):
        """加载已有的模型（加载最新的模型）"""
        print("\n[检测] 检查既有模型...")
        import glob
        
        for i, agent in enumerate(self.agents):
            name = chr(65 + i)  # A, B, C
            pattern = os.path.join(MODEL_DIR, f"agent_{name}_cycle_*.pth")
            model_files = glob.glob(pattern)
            
            if model_files:
                # 按文件名排序，获取最新的模型
                model_files.sort()
                latest_model = model_files[-1]
                
                # 检查是否有多个模型，如果有，保留最新的，删除旧的
                if len(model_files) > 1:
                    print(f"  [整理] Agent {name}: 发现 {len(model_files)} 个模型，保留最新的")
                    for old_model in model_files[:-1]:
                        try:
                            os.remove(old_model)
                            print(f"    已删除旧模型: {os.path.basename(old_model)}")
                        except Exception as e:
                            print(f"    删除旧模型失败: {e}")
                
                try:
                    agent.load(latest_model)
                    print(f"  [加载] Agent {name}: {os.path.basename(latest_model)}")
                except Exception as e:
                    print(f"  [错误] 加载Agent {name}失败: {e}")
            else:
                print(f"  [提示] Agent {name}: 未找到既有模型，将从头训练")
    
    def _get_next_paiku(self):
        """获取下一组牌库"""
        if self.current_paiku_idx >= len(self.paiku_list):
            self.current_paiku_idx = 0
        
        line = self.paiku_list[self.current_paiku_idx]
        self.current_paiku_idx += 1
        return parse_paiku_line(line)
    
    def _get_epsilon(self, game_num, total_games):
        """计算当前探索率"""
        decay = EPSILON_START - (EPSILON_START - EPSILON_MIN) * (game_num / total_games)
        return max(EPSILON_MIN, decay)
    
    def play_one_game(self, agents, epsilon=0.0):
        """进行一局游戏"""
        # 重置环境
        self.env.reset()
        
        # 发牌
        hand_A, hand_B, hand_C, base = self._get_next_paiku()
        self.env.deal_cards(hand_A, hand_B, hand_C, base)
        
        # 叫分阶段（使用bid_net）
        bids = []
        for pos in [0, 1, 2]:
            hand = [hand_A, hand_B, hand_C][pos]
            bid = agents[pos].select_bid(hand, epsilon)
            bids.append(bid)
        
        # 执行叫分
        success, landlord, final_bid = self.env.bidding_phase(*bids)
        
        if not success:
            return 'draw', {0: 0, 1: 0, 2: 0}, {'final_bid': 0, 'training_data': [[], [], []]}
        
        # 初始化出牌状态（70维向量）
        for pos in [0, 1, 2]:
            hand = [hand_A, hand_B, hand_C][pos]
            if pos == landlord:
                hand = hand + base  # 地主获得底牌
            agents[pos].init_play_state(hand, pos)
        
        # 出牌阶段
        self.env.start_playing()
        
        # 记录训练数据
        training_data = [[], [], []]
        
        while not self.env.done:
            player = self.env.current_player
            agent = agents[player]
            
            # 获取合法动作
            legal_actions = self.env.get_legal_actions(player)
            
            # 使用play_net决策（输入70维状态）
            action = agent.select_play(legal_actions, epsilon)
            
            # 记录训练数据（简化版）
            if agent.current_state is not None:
                training_data[player].append({
                    'state': agent.current_state.copy(),
                    'action': action
                })
            
            # 执行动作
            success, msg, done, reward = self.env.step(player, action)
            
            if not success:
                action = "PASS"
                success, msg, done, reward = self.env.step(player, action)
        
        # 计算奖励
        rewards = {}
        for pos in [0, 1, 2]:
            rewards[pos] = self.env._calculate_reward(pos)
        
        # 为训练数据添加奖励
        for pos in [0, 1, 2]:
            for data in training_data[pos]:
                data['reward'] = rewards[pos]
        
        return self.env.winner, rewards, {
            'final_bid': final_bid,
            'training_data': training_data
        }
    
    def train_agent(self, agent_idx, num_games, epsilon_func, stats_collector):
        """训练单个Agent"""
        agent = self.agents[agent_idx]
        
        # 准备其他agent（固定）
        other_agents = [self.agents[i] for i in range(3) if i != agent_idx]
        
        results = {'landlord': 0, 'farmers': 0, 'draw': 0, 'win': 0}
        total_reward = 0
        
        for game in tqdm(range(num_games), desc=f"Agent {chr(65+agent_idx)}"):
            epsilon = epsilon_func(game, num_games)
            
            # 创建agents字典
            agents_dict = {}
            for pos in [0, 1, 2]:
                if pos == agent_idx:
                    agents_dict[pos] = agent
                elif pos == (agent_idx + 1) % 3:
                    agents_dict[pos] = other_agents[0]
                else:
                    agents_dict[pos] = other_agents[1]
            
            winner, rewards, game_data = self.play_one_game(agents_dict, epsilon)
            
            results[winner] += 1
            total_reward += rewards[agent_idx]
            
            # 判断被训练的agent是否获胜
            is_win = rewards[agent_idx] > 0
            if is_win:
                results['win'] += 1
            
            # 收集cycle统计信息
            if 'final_bid' in game_data:
                stats_collector['final_bids'].append(game_data['final_bid'])
            stats_collector['total_games'] = stats_collector.get('total_games', 0) + 1
            if winner == 'draw':
                stats_collector['draws'] = stats_collector.get('draws', 0) + 1
            
            # 写入每局游戏日志
            self.write_log('game', {
                'agent_idx': agent_idx,
                'game_num': game,
                'epsilon': epsilon,
                'winner': winner,
                'rewards': rewards,
                'agent_reward': rewards[agent_idx],
                'is_win': is_win,
                'final_bid': game_data.get('final_bid', 0)
            })
            
            # 训练（简化版）
            if game_data['training_data'][agent_idx]:
                # 这里应该实现真正的训练逻辑
                # 简化版：只收集数据，不实际训练
                pass
        
        # 打印统计
        win_rate = results['win'] / num_games
        avg_reward = total_reward / num_games
        print(f"  胜率: {win_rate:.1%}, 平均奖励: {avg_reward:.2f}")
        print(f"  详细: 地主胜={results['landlord']}, 农民胜={results['farmers']}, 流局={results['draw']}")
        
        return win_rate, avg_reward, results
    
    def train_cycle(self, cycle_num):
        """训练一个cycle"""
        print(f"\n{'='*60}")
        print(f"Cycle {cycle_num + 1}/{CYCLE}")
        print(f"{'='*60}\n")
        
        # 用于统计整个cycle的数据
        cycle_final_bids = []
        cycle_draws = 0
        cycle_total_games = 0
        
        cycle_stats = []
        
        # 训练每个agent
        for agent_idx in [0, 1, 2]:
            num_games = int(H * self.boost_factors[agent_idx])
            print(f"训练Agent {chr(65+agent_idx)} ({num_games}局)")
            
            # 创建统计收集器
            stats_collector = {
                'final_bids': [],
                'draws': 0,
                'total_games': 0
            }
            
            win_rate, avg_reward, agent_results = self.train_agent(
                agent_idx,
                num_games,
                self._get_epsilon,
                stats_collector
            )
            
            # 合并统计结果
            cycle_final_bids.extend(stats_collector['final_bids'])
            cycle_draws += stats_collector['draws']
            cycle_total_games += stats_collector['total_games']
            
            cycle_stats.append({
                'agent': agent_idx,
                'win_rate': win_rate,
                'avg_reward': avg_reward,
                'games': num_games,
                'results': agent_results
            })
            
            # 写入agent训练完成日志
            self.write_log('agent_complete', {
                'cycle': cycle_num,
                'agent_idx': agent_idx,
                'win_rate': win_rate,
                'avg_reward': avg_reward,
                'games': num_games,
                'boost_factor': self.boost_factors[agent_idx]
            })
            
            # 自适应训练
            if ADAPTIVE_TRAINING:
                if win_rate < WIN_RATE_THRESHOLD:
                    self.low_win_counts[agent_idx] += 1
                    if self.low_win_counts[agent_idx] >= CONSECUTIVE_LOW_WIN_ROUNDS:
                        self.boost_factors[agent_idx] = min(
                            MAX_BOOST_FACTOR,
                            self.boost_factors[agent_idx] * TRAINING_BOOST_FACTOR
                        )
                        print(f"  [自适应] 增强训练至{self.boost_factors[agent_idx]:.1f}x")
                else:
                    self.low_win_counts[agent_idx] = 0
        
        self.stats.append(cycle_stats)
        
        # 计算cycle统计
        if cycle_final_bids:
            avg_bid = np.mean([b for b in cycle_final_bids if b > 0])
        else:
            avg_bid = 0.0
        draw_rate = cycle_draws / max(1, cycle_total_games)
        
        # 输出Cycle统计
        print(f"\n{'='*60}")
        print(f"Cycle {cycle_num + 1} 统计")
        print(f"{'='*60}")
        print(f"  平均成交分数: {avg_bid:.2f}")
        print(f"  流局率: {draw_rate:.1%} ({cycle_draws}/{cycle_total_games})")
        print(f"{'='*60}\n")
        
        # 写入cycle完成日志
        self.write_log('cycle_complete', {
            'cycle': cycle_num,
            'avg_bid': avg_bid,
            'draw_rate': draw_rate,
            'total_games': cycle_total_games,
            'draws': cycle_draws,
            'agent_stats': cycle_stats
        })
        
        # 保存模型（带表现比较）
        self._save_models(cycle_num, cycle_stats)
    
    def _save_models(self, cycle_num, cycle_stats):
        """保存模型，保留表现更好的模型"""
        print(f"\n[保存] 评估并保存模型...")
        
        for i, agent in enumerate(self.agents):
            name = chr(65 + i)
            new_path = os.path.join(MODEL_DIR, f"agent_{name}_cycle_{cycle_num:03d}.pth")
            
            # 保存新模型
            agent.save(new_path)
            print(f"  Agent {name}: 新模型已保存到 {new_path}")
            
            # 获取当前agent的统计
            agent_stat = next((s for s in cycle_stats if s['agent'] == i), None)
            if agent_stat:
                current_win_rate = agent_stat['win_rate']
                current_avg_reward = agent_stat['avg_reward']
                
                # 查找之前最新的模型
                import glob
                pattern = os.path.join(MODEL_DIR, f"agent_{name}_cycle_*.pth")
                all_models = glob.glob(pattern)
                
                # 排除当前刚保存的模型
                prev_models = [m for m in all_models if m != new_path]
                
                if prev_models:
                    # 按文件名排序，获取最新的旧模型
                    prev_models.sort()
                    latest_prev = prev_models[-1]
                    
                    # 从文件名提取cycle数
                    try:
                        prev_cycle = int(latest_prev.split('_cycle_')[-1].split('.')[0])
                        
                        # 查找之前cycle的统计
                        prev_stat = None
                        for stat_list in self.stats[:-1]:  # 排除当前cycle
                            for s in stat_list:
                                if s['agent'] == i:
                                    prev_stat = s
                                    break
                        
                        if prev_stat:
                            prev_win_rate = prev_stat['win_rate']
                            prev_avg_reward = prev_stat['avg_reward']
                            
                            # 比较表现：胜率提高或奖励提高
                            is_better = (current_win_rate > prev_win_rate or 
                                       current_avg_reward > prev_avg_reward)
                            
                            if is_better:
                                print(f"    ✓ 新模型表现更好 (胜率: {prev_win_rate:.1%}→{current_win_rate:.1%}, "
                                      f"奖励: {prev_avg_reward:+.2f}→{current_avg_reward:+.2f})")
                                # 删除旧模型
                                try:
                                    os.remove(latest_prev)
                                    print(f"    ✓ 已删除旧模型: {os.path.basename(latest_prev)}")
                                except Exception as e:
                                    print(f"    ! 删除旧模型失败: {e}")
                            else:
                                print(f"    ✗ 新模型表现不如旧模型 (胜率: {prev_win_rate:.1%}→{current_win_rate:.1%}, "
                                      f"奖励: {prev_avg_reward:+.2f}→{current_avg_reward:+.2f})")
                                # 删除新模型，保留旧模型
                                try:
                                    os.remove(new_path)
                                    print(f"    ✓ 已删除新模型，保留旧模型")
                                except Exception as e:
                                    print(f"    ! 删除新模型失败: {e}")
                        else:
                            print(f"    ! 未找到之前cycle的统计，保留新模型")
                    except Exception as e:
                        print(f"    ! 解析旧模型信息失败: {e}")
                else:
                    print(f"    ✓ 首次训练，保留新模型")
    
    def train(self):
        """完整训练流程"""
        print("="*60)
        print("开始斗地主AI训练")
        print("="*60)
        print(f"配置: {CYCLE} cycles, 每cycle {H}局")
        print(f"牌库: {len(self.paiku_list)}个")
        print(f"日志文件: {self.log_file}")
        
        start_time = datetime.now()
        
        for cycle in range(CYCLE):
            self.train_cycle(cycle)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 写入训练完成日志
        self.write_log('training_complete', {
            'total_cycles': CYCLE,
            'duration_seconds': duration,
            'final_stats': self.stats
        })
        
        print("\n" + "="*60)
        print("训练完成!")
        print(f"总用时: {duration/60:.1f}分钟")
        print(f"日志文件: {self.log_file}")
        print("="*60)


if __name__ == "__main__":
    # 首先生成牌库（如果不存在）
    if not os.path.exists('paiku.txt'):
        print("生成牌库...")
        from generator import generate_paiku, save_paiku
        from config import NUM_PAIKU
        paiku_list = generate_paiku(NUM_PAIKU)
        save_paiku(paiku_list)
    
    # 开始训练
    trainer = DouDiZhuTrainer()
    trainer.train()
