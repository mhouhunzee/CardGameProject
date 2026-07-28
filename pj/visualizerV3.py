"""Visualizer V3 - 重新组织的可视化结构

目录结构：
- overall/: 全局指标（跨所有阶段）
- initial/: 初始训练阶段（三Agent对比）
- agent_a/b/c/: 各Agent强化训练阶段

每张图片只包含一个指标
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# 设置样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# 配色方案
AGENT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿


@dataclass
class GameResult:
    episode: int
    game_id: int
    phase: str
    round_num: int
    bids: List[int]
    final_bid: int
    landlord: int
    is_draw: bool
    winner: int
    game_length: int
    rewards: List[float]
    roles: List[str]
    action_counts: List[int]
    bomb_count: int
    rocket_count: int


@dataclass
class EpisodeSummary:
    episode: int
    phase: str
    round_num: int
    win_rates: List[float]
    landlord_win_rate: float
    avg_scores: List[float]
    avg_bids: List[float]
    bid_distribution: List[Dict[int, int]]
    landlord_win_rates: List[float]
    farmer_win_rates: List[float]
    total_games: int
    draw_count: int
    avg_game_length: float
    avg_final_bid: float
    avg_action_count: float


@dataclass
class EvalResult:
    agent: str
    round_num: int
    episode: int
    win_rate: float
    avg_score: float
    total_games: int
    wins: int


class DataLoader:
    @staticmethod
    def load_all_data(log_dir: str) -> Tuple[List[GameResult], List[EpisodeSummary], List[EvalResult]]:
        games = []
        summaries = []
        eval_results = []
        
        # 加载游戏日志
        for file_path in glob.glob(os.path.join(log_dir, "game_log_*.jsonl")):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('type') in ['header', 'summary']:
                            continue
                        games.append(GameResult(
                            episode=data['episode'], game_id=data['game_id'],
                            phase=data['phase'], round_num=data['round_num'],
                            bids=data['bids'], final_bid=data['final_bid'],
                            landlord=data['landlord'], is_draw=data['is_draw'],
                            winner=data['winner'], game_length=data['game_length'],
                            rewards=data['rewards'], roles=data['roles'],
                            action_counts=data['action_counts'],
                            bomb_count=data['bomb_count'], rocket_count=data['rocket_count']
                        ))
                    except:
                        continue
        
        # 加载汇总日志
        for file_path in glob.glob(os.path.join(log_dir, "summary_*.jsonl")):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('type') in ['header', 'summary']:
                            continue
                        summaries.append(EpisodeSummary(
                            episode=data['episode'], phase=data['phase'], round_num=data['round_num'],
                            win_rates=data['win_rates'], landlord_win_rate=data['landlord_win_rate'],
                            avg_scores=data['avg_scores'], avg_bids=data['avg_bids'],
                            bid_distribution=data['bid_distribution'],
                            landlord_win_rates=data['landlord_win_rates'],
                            farmer_win_rates=data['farmer_win_rates'],
                            total_games=data['total_games'], draw_count=data['draw_count'],
                            avg_game_length=data['avg_game_length'],
                            avg_final_bid=data['avg_final_bid'], avg_action_count=data['avg_action_count']
                        ))
                    except:
                        continue
        
        # 加载评估结果
        for file_path in glob.glob(os.path.join(log_dir, "eval_results_*.jsonl")):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('type') != 'eval':
                            continue
                        eval_results.append(EvalResult(
                            agent=data['agent'], round_num=data['round_num'],
                            episode=data['episode'], win_rate=data['win_rate'],
                            avg_score=data['avg_score'], total_games=data['total_games'],
                            wins=data['wins']
                        ))
                    except:
                        continue
        
        games.sort(key=lambda x: x.episode)
        summaries.sort(key=lambda x: x.episode)
        eval_results.sort(key=lambda x: x.episode)
        return games, summaries, eval_results


class VisualizerV3:
    def __init__(self, output_dir: str = "./visualizations_v3",
                 plot_interval: int = 1):
        """初始化可视化器V3.
        
        Args:
            output_dir: 图表输出目录
            plot_interval: 数据采样间隔（默认1，表示所有数据点）
        """
        self.output_dir = output_dir
        self.plot_interval = plot_interval
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子目录
        self.dirs = {
            'overall': os.path.join(output_dir, 'overall'),
            'initial': os.path.join(output_dir, 'initial'),
            'agent_a': os.path.join(output_dir, 'agent_a'),
            'agent_b': os.path.join(output_dir, 'agent_b'),
            'agent_c': os.path.join(output_dir, 'agent_c'),
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
    
    def generate_all(self, log_dir: str = "./train_logs"):
        print("=" * 70)
        print("Visualizer V3 - 生成图表")
        print(f"采样间隔: {self.plot_interval}")
        print("=" * 70)
        
        games, summaries, eval_results = DataLoader.load_all_data(log_dir)
        print(f"\n游戏记录: {len(games):,} 条")
        print(f"汇总记录: {len(summaries):,} 条")
        print(f"评估记录: {len(eval_results):,} 条")
        
        # 数据采样
        if self.plot_interval > 1:
            games = games[::self.plot_interval]
            summaries = summaries[::self.plot_interval]
            print(f"采样后: {len(games):,} 游戏记录, {len(summaries):,} 汇总记录")
        
        df_games = self._games_to_df(games)
        df_summaries = self._summaries_to_df(summaries)
        
        # 按阶段分组
        initial_summaries = [s for s in summaries if s.phase == 'initial']
        agent_a_summaries = [s for s in summaries if s.phase == 'agent_a']
        agent_b_summaries = [s for s in summaries if s.phase == 'agent_b']
        agent_c_summaries = [s for s in summaries if s.phase == 'agent_c']
        
        print("\n1. 生成overall文件夹图表...")
        self._generate_overall(df_games, df_summaries)
        
        print("\n2. 生成initial文件夹图表...")
        self._generate_initial(initial_summaries)
        
        print("\n3. 生成agent文件夹图表...")
        self._generate_agent('a', agent_a_summaries, eval_results)
        self._generate_agent('b', agent_b_summaries, eval_results)
        self._generate_agent('c', agent_c_summaries, eval_results)
        
        print("\n" + "=" * 70)
        print(f"完成! 输出目录: {self.output_dir}")
        print("=" * 70)
    
    def _games_to_df(self, games: List[GameResult]) -> pd.DataFrame:
        data = []
        for g in games:
            data.append({
                'episode': g.episode,
                'phase': g.phase,
                'is_draw': g.is_draw,
                'final_bid': g.final_bid,
                'game_length': g.game_length,
                'total_actions': sum(g.action_counts) if not g.is_draw else 0,
                'bomb_count': g.bomb_count,
                'rocket_count': g.rocket_count,
            })
        return pd.DataFrame(data)
    
    def _summaries_to_df(self, summaries: List[EpisodeSummary]) -> pd.DataFrame:
        data = []
        for s in summaries:
            data.append({
                'episode': s.episode,
                'phase': s.phase,
                'round_num': s.round_num,
                'agent_0_win_rate': s.win_rates[0],
                'agent_1_win_rate': s.win_rates[1],
                'agent_2_win_rate': s.win_rates[2],
                'agent_0_score': s.avg_scores[0],
                'agent_1_score': s.avg_scores[1],
                'agent_2_score': s.avg_scores[2],
                'agent_0_bid': s.avg_bids[0],
                'agent_1_bid': s.avg_bids[1],
                'agent_2_bid': s.avg_bids[2],
                'landlord_win_rate': s.landlord_win_rate,
                'draw_rate': s.draw_count / max(s.total_games, 1),
                'avg_game_length': s.avg_game_length,
                'avg_final_bid': s.avg_final_bid,
            })
        return pd.DataFrame(data)
    
    def _generate_overall(self, df_games: pd.DataFrame, df_summaries: pd.DataFrame):
        """生成overall文件夹图表 - 全局指标，每张图一个指标"""
        
        # 1. 成交分数均值（流局=0）
        fig, ax = plt.subplots(figsize=(12, 6))
        episode_bids = df_games.groupby('episode')['final_bid'].mean()
        ax.plot(episode_bids.index, episode_bids.values, color='blue', linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Final Bid')
        ax.set_title('Overall: Average Final Bid per Game (Draw=0)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['overall'], '01_avg_final_bid.png'))
        plt.close()
        print("   ✓ 01_avg_final_bid.png")
        
        # 2. 出牌次数平均值（不含流局）
        fig, ax = plt.subplots(figsize=(12, 6))
        valid_games = df_games[~df_games['is_draw']]
        episode_actions = valid_games.groupby('episode')['total_actions'].mean()
        ax.plot(episode_actions.index, episode_actions.values, color='green', linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Action Count')
        ax.set_title('Overall: Average Action Count per Game (Excluding Draws)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['overall'], '02_avg_action_count.png'))
        plt.close()
        print("   ✓ 02_avg_action_count.png")
        
        # 3. 游戏长度平均值（不含流局）
        fig, ax = plt.subplots(figsize=(12, 6))
        episode_lengths = valid_games.groupby('episode')['game_length'].mean()
        ax.plot(episode_lengths.index, episode_lengths.values, color='orange', linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Game Length (steps)')
        ax.set_title('Overall: Average Game Length (Excluding Draws)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['overall'], '03_avg_game_length.png'))
        plt.close()
        print("   ✓ 03_avg_game_length.png")
        
        # 4. 流局率
        fig, ax = plt.subplots(figsize=(12, 6))
        draw_rates = df_games.groupby('episode')['is_draw'].mean()
        ax.plot(draw_rates.index, draw_rates.values, color='red', linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Draw Rate')
        ax.set_title('Overall: Draw Rate Over Time')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['overall'], '04_draw_rate.png'))
        plt.close()
        print("   ✓ 04_draw_rate.png")
        
        # 5. 地主胜率
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_summaries['episode'], df_summaries['landlord_win_rate'], 
               color='purple', linewidth=1.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Landlord Win Rate')
        ax.set_title('Overall: Landlord Win Rate')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['overall'], '05_landlord_winrate.png'))
        plt.close()
        print("   ✓ 05_landlord_winrate.png")
        
        # 6. 炸弹+火箭使用率
        fig, ax = plt.subplots(figsize=(12, 6))
        valid_games['special_count'] = valid_games['bomb_count'] + valid_games['rocket_count']
        episode_special = valid_games.groupby('episode')['special_count'].mean()
        ax.plot(episode_special.index, episode_special.values, color='brown', linewidth=1.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Bomb+Rocket Count')
        ax.set_title('Overall: Average Bomb/Rocket Usage per Game')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['overall'], '06_bomb_rocket_usage.png'))
        plt.close()
        print("   ✓ 06_bomb_rocket_usage.png")
    
    def _generate_initial(self, summaries: List[EpisodeSummary]):
        """生成initial文件夹图表 - 三Agent对比，每张图一个指标"""
        if not summaries:
            print("   无initial数据")
            return
        
        episodes = [s.episode for s in summaries]
        
        # 1. 三Agent胜率对比
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(3):
            win_rates = [s.win_rates[i] for s in summaries]
            ax.plot(episodes, win_rates, color=AGENT_COLORS[i], 
                   label=f'Agent {i}', linewidth=2)
        ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title('Initial Training: Win Rate Comparison (3 Agents)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['initial'], '01_win_rate_comparison.png'))
        plt.close()
        print("   ✓ 01_win_rate_comparison.png")
        
        # 2. 三Agent得分对比
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(3):
            scores = [s.avg_scores[i] for s in summaries]
            ax.plot(episodes, scores, color=AGENT_COLORS[i],
                   label=f'Agent {i}', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Score')
        ax.set_title('Initial Training: Score Comparison (3 Agents)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['initial'], '02_score_comparison.png'))
        plt.close()
        print("   ✓ 02_score_comparison.png")
        
        # 3. 三Agent叫分对比
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(3):
            bids = [s.avg_bids[i] for s in summaries]
            ax.plot(episodes, bids, color=AGENT_COLORS[i],
                   label=f'Agent {i}', linewidth=2)
        ax.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Bid')
        ax.set_title('Initial Training: Bid Comparison (3 Agents)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['initial'], '03_bid_comparison.png'))
        plt.close()
        print("   ✓ 03_bid_comparison.png")
        
        # 4. 三Agent地主胜率对比
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(3):
            landlord_rates = [s.landlord_win_rates[i] for s in summaries]
            ax.plot(episodes, landlord_rates, color=AGENT_COLORS[i],
                   label=f'Agent {i}', linewidth=2)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Landlord Win Rate')
        ax.set_title('Initial Training: Landlord Win Rate Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['initial'], '04_landlord_winrate_comparison.png'))
        plt.close()
        print("   ✓ 04_landlord_winrate_comparison.png")
        
        # 5. 三Agent农民胜率对比
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(3):
            farmer_rates = [s.farmer_win_rates[i] for s in summaries]
            ax.plot(episodes, farmer_rates, color=AGENT_COLORS[i],
                   label=f'Agent {i}', linewidth=2)
        ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Farmer Win Rate')
        ax.set_title('Initial Training: Farmer Win Rate Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['initial'], '05_farmer_winrate_comparison.png'))
        plt.close()
        print("   ✓ 05_farmer_winrate_comparison.png")
    
    def _generate_agent(self, agent_name: str, summaries: List[EpisodeSummary], 
                       eval_results: List[EvalResult]):
        """生成agent文件夹图表 - 只显示该agent被训练时的指标"""
        agent_id = ord(agent_name) - ord('a')
        output_dir = self.dirs[f'agent_{agent_name}']
        
        if not summaries:
            print(f"   agent_{agent_name}: 无数据")
            return
        
        # 按round分组
        rounds_data = defaultdict(list)
        for s in summaries:
            rounds_data[s.round_num].append(s)
        
        # 1. 胜率变化（不连续显示，间隔用虚线）
        fig, ax = plt.subplots(figsize=(14, 6))
        prev_last = None
        for round_num in sorted(rounds_data.keys()):
            data = rounds_data[round_num]
            episodes = [s.episode for s in data]
            rates = [s.win_rates[agent_id] for s in data]
            ax.plot(episodes, rates, color=AGENT_COLORS[agent_id], linewidth=2)
            if prev_last and episodes:
                ax.plot([prev_last[0], episodes[0]], [prev_last[1], rates[0]], 
                       'k--', linewidth=1, alpha=0.5)
            if episodes:
                prev_last = (episodes[-1], rates[-1])
        ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title(f'Agent {agent_name.upper()}: Win Rate (Training Periods)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_win_rate.png'))
        plt.close()
        print(f"   ✓ agent_{agent_name}/01_win_rate.png")
        
        # 2. 得分变化
        fig, ax = plt.subplots(figsize=(14, 6))
        prev_last = None
        for round_num in sorted(rounds_data.keys()):
            data = rounds_data[round_num]
            episodes = [s.episode for s in data]
            scores = [s.avg_scores[agent_id] for s in data]
            ax.plot(episodes, scores, color=AGENT_COLORS[agent_id], linewidth=2)
            if prev_last and episodes:
                ax.plot([prev_last[0], episodes[0]], [prev_last[1], scores[0]], 
                       'k--', linewidth=1, alpha=0.5)
            if episodes:
                prev_last = (episodes[-1], scores[-1])
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Score')
        ax.set_title(f'Agent {agent_name.upper()}: Score (Training Periods)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '02_score.png'))
        plt.close()
        print(f"   ✓ agent_{agent_name}/02_score.png")
        
        # 3. 叫分变化
        fig, ax = plt.subplots(figsize=(14, 6))
        prev_last = None
        for round_num in sorted(rounds_data.keys()):
            data = rounds_data[round_num]
            episodes = [s.episode for s in data]
            bids = [s.avg_bids[agent_id] for s in data]
            ax.plot(episodes, bids, color=AGENT_COLORS[agent_id], linewidth=2)
            if prev_last and episodes:
                ax.plot([prev_last[0], episodes[0]], [prev_last[1], bids[0]], 
                       'k--', linewidth=1, alpha=0.5)
            if episodes:
                prev_last = (episodes[-1], bids[-1])
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Bid')
        ax.set_title(f'Agent {agent_name.upper()}: Bid (Training Periods)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '03_bid.png'))
        plt.close()
        print(f"   ✓ agent_{agent_name}/03_bid.png")
        
        # 4. 评估结果（vs初始模型）
        agent_evals = [e for e in eval_results if e.agent == agent_name]
        if agent_evals:
            # 胜率
            fig, ax = plt.subplots(figsize=(12, 6))
            rounds = [e.round_num for e in agent_evals]
            win_rates = [e.win_rate for e in agent_evals]
            ax.plot(rounds, win_rates, 'o-', color=AGENT_COLORS[agent_id], 
                   linewidth=2, markersize=8)
            ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel('Training Round')
            ax.set_ylabel('Win Rate vs Baseline')
            ax.set_title(f'Agent {agent_name.upper()}: Post-Training Win Rate')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '04_eval_winrate.png'))
            plt.close()
            print(f"   ✓ agent_{agent_name}/04_eval_winrate.png")
            
            # 平均得分
            fig, ax = plt.subplots(figsize=(12, 6))
            avg_scores = [e.avg_score for e in agent_evals]
            colors = ['red' if s < 0 else 'green' for s in avg_scores]
            ax.bar(rounds, avg_scores, color=colors, alpha=0.7, edgecolor='black')
            ax.axhline(y=0, color='black', linewidth=1)
            ax.set_xlabel('Training Round')
            ax.set_ylabel('Average Score vs Baseline')
            ax.set_title(f'Agent {agent_name.upper()}: Post-Training Score')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '05_eval_score.png'))
            plt.close()
            print(f"   ✓ agent_{agent_name}/05_eval_score.png")


if __name__ == "__main__":
    """
    终端调用方法:
    python visualizerV3.py [日志目录] [输出目录] [采样间隔]
    
    示例:
    python visualizerV3.py ./train_logs ./visualizations_v3
    python visualizerV3.py ./train_logs ./visualizations_v3 10
    """
    import sys
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "./train_logs"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./visualizations_v3"
    plot_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    viz = VisualizerV3(output_dir=output_dir, plot_interval=plot_interval)
    viz.generate_all(log_dir=log_dir)
