"""独立可视化模块 - 从日志文件生成图表.

此模块完全独立于训练流程，通过读取train_logger生成的日志文件来生成图表.
特性：
1. 长条拼接图表 - 将大量数据分段显示
2. Agent文件夹只显示被训练时的数据，间隔用虚线连接
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class GameResult:
    """单局游戏结果."""
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
    """每轮训练汇总."""
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


class LogReader:
    """日志读取器."""
    
    @staticmethod
    def read_game_logs(log_dir: str, phase: Optional[str] = None, 
                       round_num: Optional[int] = None) -> List[GameResult]:
        """读取游戏日志."""
        games = []
        pattern = os.path.join(log_dir, "game_log_*.jsonl")
        files = glob.glob(pattern)
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('type') in ['header', 'summary']:
                            continue
                        if phase is not None and data.get('phase') != phase:
                            continue
                        if round_num is not None and data.get('round_num') != round_num:
                            continue
                        
                        games.append(GameResult(
                            episode=data['episode'],
                            game_id=data['game_id'],
                            phase=data['phase'],
                            round_num=data['round_num'],
                            bids=data['bids'],
                            final_bid=data['final_bid'],
                            landlord=data['landlord'],
                            is_draw=data['is_draw'],
                            winner=data['winner'],
                            game_length=data['game_length'],
                            rewards=data['rewards'],
                            roles=data['roles'],
                            action_counts=data['action_counts'],
                            bomb_count=data['bomb_count'],
                            rocket_count=data['rocket_count']
                        ))
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        return sorted(games, key=lambda x: (x.phase, x.round_num, x.episode, x.game_id))
    
    @staticmethod
    def read_summary_logs(log_dir: str, phase: Optional[str] = None,
                          round_num: Optional[int] = None) -> List[EpisodeSummary]:
        """读取汇总日志."""
        summaries = []
        pattern = os.path.join(log_dir, "summary_*.jsonl")
        files = glob.glob(pattern)
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('type') in ['header', 'summary']:
                            continue
                        if phase is not None and data.get('phase') != phase:
                            continue
                        if round_num is not None and data.get('round_num') != round_num:
                            continue
                        
                        summaries.append(EpisodeSummary(
                            episode=data['episode'],
                            phase=data['phase'],
                            round_num=data['round_num'],
                            win_rates=data['win_rates'],
                            landlord_win_rate=data['landlord_win_rate'],
                            avg_scores=data['avg_scores'],
                            avg_bids=data['avg_bids'],
                            bid_distribution=data['bid_distribution'],
                            landlord_win_rates=data['landlord_win_rates'],
                            farmer_win_rates=data['farmer_win_rates'],
                            total_games=data['total_games'],
                            draw_count=data['draw_count'],
                            avg_game_length=data['avg_game_length'],
                            avg_final_bid=data['avg_final_bid'],
                            avg_action_count=data['avg_action_count']
                        ))
                    except (json.JSONDecodeError, KeyError):
                        continue
        
        return sorted(summaries, key=lambda x: (x.phase, x.round_num, x.episode))


class Visualizer:
    """可视化器 - 生成各类图表."""
    
    def __init__(self, output_dir: str = "./visualizations", 
                 segment_size: int = 50000,
                 plot_scale: int = 100,
                 plot_interval: int = 100):
        """初始化可视化器.
        
        Args:
            output_dir: 图表输出目录
            segment_size: 每个图表段的大小（episode数）
            plot_scale: 图表包含的训练轮数尺度（默认100）
            plot_interval: 数据点间隔（默认100）
        """
        self.output_dir = output_dir
        self.segment_size = segment_size
        self.plot_scale = plot_scale
        self.plot_interval = plot_interval
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子目录
        self.initial_dir = os.path.join(output_dir, "initial")
        self.overall_dir = os.path.join(output_dir, "overall")
        self.agent_a_dir = os.path.join(output_dir, "agent_a")
        self.agent_b_dir = os.path.join(output_dir, "agent_b")
        self.agent_c_dir = os.path.join(output_dir, "agent_c")
        
        for d in [self.initial_dir, self.overall_dir, self.agent_a_dir, 
                  self.agent_b_dir, self.agent_c_dir]:
            os.makedirs(d, exist_ok=True)
    
    def generate_all(self, log_dir: str = "./train_logs"):
        """生成所有图表."""
        print("=" * 60)
        print("开始生成可视化图表")
        print(f"图表分段大小: {self.segment_size} episodes")
        print("=" * 60)
        
        # 读取所有数据
        print("\n1. 读取日志数据...")
        all_games = LogReader.read_game_logs(log_dir)
        all_summaries = LogReader.read_summary_logs(log_dir)
        
        print(f"   游戏记录: {len(all_games)} 条")
        print(f"   汇总记录: {len(all_summaries)} 条")
        
        # 按阶段分组
        initial_games = [g for g in all_games if g.phase == 'initial']
        initial_summaries = [s for s in all_summaries if s.phase == 'initial']
        
        agent_a_games = [g for g in all_games if g.phase == 'agent_a']
        agent_a_summaries = [s for s in all_summaries if s.phase == 'agent_a']
        
        agent_b_games = [g for g in all_games if g.phase == 'agent_b']
        agent_b_summaries = [s for s in all_summaries if s.phase == 'agent_b']
        
        agent_c_games = [g for g in all_games if g.phase == 'agent_c']
        agent_c_summaries = [s for s in all_summaries if s.phase == 'agent_c']
        
        # 2. 生成initial文件夹图表（分段）
        print("\n2. 生成initial文件夹图表...")
        self._generate_initial_charts_segmented(initial_summaries)
        
        # 3. 生成overall文件夹图表（分段）
        print("\n3. 生成overall文件夹图表...")
        self._generate_overall_charts_segmented(all_games, all_summaries)
        
        # 4. 生成agent文件夹图表（只显示被训练时的数据）
        print("\n4. 生成agent_a文件夹图表...")
        self._generate_agent_charts_discontinuous('a', agent_a_summaries)
        
        print("\n5. 生成agent_b文件夹图表...")
        self._generate_agent_charts_discontinuous('b', agent_b_summaries)
        
        print("\n6. 生成agent_c文件夹图表...")
        self._generate_agent_charts_discontinuous('c', agent_c_summaries)
        
        print("\n" + "=" * 60)
        print("图表生成完成!")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)
    
    def _generate_initial_charts_segmented(self, summaries: List[EpisodeSummary]):
        """生成initial文件夹图表 - 分段显示."""
        if not summaries:
            print("   无initial阶段数据")
            return
        
        # 按episode范围分段
        min_episode = min(s.episode for s in summaries)
        max_episode = max(s.episode for s in summaries)
        
        segment_num = 0
        start_ep = min_episode
        
        while start_ep <= max_episode:
            end_ep = start_ep + self.segment_size
            segment_summaries = [s for s in summaries if start_ep <= s.episode < end_ep]
            
            if not segment_summaries:
                start_ep = end_ep
                continue
            
            segment_num += 1
            self._generate_initial_segment(segment_summaries, segment_num, start_ep, end_ep)
            start_ep = end_ep
    
    def _generate_initial_segment(self, summaries: List[EpisodeSummary], 
                                   segment_num: int, start_ep: int, end_ep: int):
        """生成initial阶段的一个分段图表."""
        episodes = [s.episode for s in summaries]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # 1. 胜率对比
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(3):
            win_rates = [s.win_rates[i] for s in summaries]
            ax.plot(episodes, win_rates, color=colors[i], 
                   label=f'Agent {i}', linewidth=2)
        ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title(f'Initial Training: Win Rate Comparison (Episodes {start_ep}-{end_ep})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.initial_dir, f'win_rate_comparison_seg{segment_num}.png'), dpi=150)
        plt.close()
        print(f"   - win_rate_comparison_seg{segment_num}.png (Episodes {start_ep}-{end_ep})")
        
        # 2. 得分对比
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(3):
            scores = [s.avg_scores[i] for s in summaries]
            ax.plot(episodes, scores, color=colors[i],
                   label=f'Agent {i}', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Score')
        ax.set_title(f'Initial Training: Score Comparison (Episodes {start_ep}-{end_ep})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.initial_dir, f'score_comparison_seg{segment_num}.png'), dpi=150)
        plt.close()
        print(f"   - score_comparison_seg{segment_num}.png")
        
        # 3. 叫分对比
        fig, ax = plt.subplots(figsize=(12, 6))
        for i in range(3):
            bids = [s.avg_bids[i] for s in summaries]
            ax.plot(episodes, bids, color=colors[i],
                   label=f'Agent {i}', linewidth=2)
        ax.axhline(y=1.5, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Bid')
        ax.set_title(f'Initial Training: Bid Comparison (Episodes {start_ep}-{end_ep})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.initial_dir, f'bid_comparison_seg{segment_num}.png'), dpi=150)
        plt.close()
        print(f"   - bid_comparison_seg{segment_num}.png")
    
    def _generate_overall_charts_segmented(self, games: List[GameResult], summaries: List[EpisodeSummary]):
        """生成overall文件夹图表 - 分段显示."""
        if not games or not summaries:
            print("   无数据")
            return
        
        min_episode = min(s.episode for s in summaries)
        max_episode = max(s.episode for s in summaries)
        
        segment_num = 0
        start_ep = min_episode
        
        while start_ep <= max_episode:
            end_ep = start_ep + self.segment_size
            segment_games = [g for g in games if start_ep <= g.episode < end_ep]
            segment_summaries = [s for s in summaries if start_ep <= s.episode < end_ep]
            
            if not segment_summaries:
                start_ep = end_ep
                continue
            
            segment_num += 1
            self._generate_overall_segment(segment_games, segment_summaries, segment_num, start_ep, end_ep)
            start_ep = end_ep
    
    def _generate_overall_segment(self, games: List[GameResult], summaries: List[EpisodeSummary],
                                   segment_num: int, start_ep: int, end_ep: int):
        """生成overall阶段的一个分段图表."""
        
        # 1. 成交分数均值
        fig, ax = plt.subplots(figsize=(12, 6))
        episode_bids = defaultdict(list)
        for g in games:
            episode_bids[g.episode].append(g.final_bid)
        episodes = sorted(episode_bids.keys())
        avg_bids = [np.mean(episode_bids[e]) for e in episodes]
        ax.plot(episodes, avg_bids, color='blue', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Final Bid')
        ax.set_title(f'Overall: Average Final Bid (Episodes {start_ep}-{end_ep})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.overall_dir, f'avg_final_bid_seg{segment_num}.png'), dpi=150)
        plt.close()
        print(f"   - avg_final_bid_seg{segment_num}.png (Episodes {start_ep}-{end_ep})")
        
        # 2. 出牌次数均值
        fig, ax = plt.subplots(figsize=(12, 6))
        episode_actions = defaultdict(list)
        for g in games:
            if not g.is_draw:
                episode_actions[g.episode].append(sum(g.action_counts))
        episodes = sorted(episode_actions.keys())
        avg_actions = [np.mean(episode_actions[e]) for e in episodes]
        ax.plot(episodes, avg_actions, color='green', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Action Count')
        ax.set_title(f'Overall: Average Action Count (Episodes {start_ep}-{end_ep})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.overall_dir, f'avg_action_count_seg{segment_num}.png'), dpi=150)
        plt.close()
        print(f"   - avg_action_count_seg{segment_num}.png")
        
        # 3. 各agent胜率
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(3):
            episodes = [s.episode for s in summaries]
            win_rates = [s.win_rates[i] for s in summaries]
            ax.plot(episodes, win_rates, color=colors[i],
                   label=f'Agent {i}', linewidth=2, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title(f'Overall: Win Rate Trends (Episodes {start_ep}-{end_ep})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.overall_dir, f'overall_win_rates_seg{segment_num}.png'), dpi=150)
        plt.close()
        print(f"   - overall_win_rates_seg{segment_num}.png")
    
    def _generate_agent_charts_discontinuous(self, agent_name: str, summaries: List[EpisodeSummary]):
        """生成agent文件夹图表 - 只显示被训练时的数据，间隔用虚线连接."""
        agent_id = ord(agent_name) - ord('a')
        output_dir = getattr(self, f'agent_{agent_name}_dir')
        
        if not summaries:
            print(f"   无agent_{agent_name}数据")
            return
        
        # 按round_num分组，每个round是一段连续的训练
        rounds_data = defaultdict(list)
        for s in summaries:
            rounds_data[s.round_num].append(s)
        
        # 1. 叫分变化（不连续显示）
        fig, ax = plt.subplots(figsize=(14, 6))
        
        prev_last_episode = None
        for round_num in sorted(rounds_data.keys()):
            round_summaries = rounds_data[round_num]
            episodes = [s.episode for s in round_summaries]
            bids = [s.avg_bids[agent_id] for s in round_summaries]
            
            # 绘制实线（训练期间）
            ax.plot(episodes, bids, color='red', linewidth=2, 
                   label=f'Round {round_num}' if round_num == 1 else "")
            
            # 用虚线连接前一段的结束和下一段的开始
            if prev_last_episode is not None and episodes:
                first_ep = episodes[0]
                first_bid = bids[0]
                ax.plot([prev_last_episode, first_ep], 
                       [prev_last_bid, first_bid], 
                       color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            if episodes:
                prev_last_episode = episodes[-1]
                prev_last_bid = bids[-1]
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Bid')
        ax.set_title(f'Agent {agent_name.upper()}: Bid Evolution (Training Periods Only)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'bid_evolution.png'), dpi=150)
        plt.close()
        print(f"   - bid_evolution.png")
        
        # 2. 得分变化（不连续显示）
        fig, ax = plt.subplots(figsize=(14, 6))
        
        prev_last_episode = None
        for round_num in sorted(rounds_data.keys()):
            round_summaries = rounds_data[round_num]
            episodes = [s.episode for s in round_summaries]
            scores = [s.avg_scores[agent_id] for s in round_summaries]
            
            ax.plot(episodes, scores, color='blue', linewidth=2)
            
            if prev_last_episode is not None and episodes:
                first_ep = episodes[0]
                first_score = scores[0]
                ax.plot([prev_last_episode, first_ep], 
                       [prev_last_score, first_score], 
                       color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            if episodes:
                prev_last_episode = episodes[-1]
                prev_last_score = scores[-1]
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Score')
        ax.set_title(f'Agent {agent_name.upper()}: Score Evolution (Training Periods Only)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'score_evolution.png'), dpi=150)
        plt.close()
        print(f"   - score_evolution.png")
        
        # 3. 胜率变化（不连续显示）
        fig, ax = plt.subplots(figsize=(14, 6))
        
        prev_last_episode = None
        for round_num in sorted(rounds_data.keys()):
            round_summaries = rounds_data[round_num]
            episodes = [s.episode for s in round_summaries]
            win_rates = [s.win_rates[agent_id] for s in round_summaries]
            
            ax.plot(episodes, win_rates, color='green', linewidth=2)
            
            if prev_last_episode is not None and episodes:
                first_ep = episodes[0]
                first_rate = win_rates[0]
                ax.plot([prev_last_episode, first_ep], 
                       [prev_last_rate, first_rate], 
                       color='gray', linestyle='--', linewidth=1, alpha=0.5)
            
            if episodes:
                prev_last_episode = episodes[-1]
                prev_last_rate = win_rates[-1]
        
        ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title(f'Agent {agent_name.upper()}: Win Rate Evolution (Training Periods Only)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'win_rate_evolution.png'), dpi=150)
        plt.close()
        print(f"   - win_rate_evolution.png")


if __name__ == "__main__":
    """
    终端调用方法:
    python visualizer.py [日志目录] [输出目录] [分段大小] [图表尺度] [采样间隔]
    
    示例:
    python visualizer.py ./train_logs ./visualizations
    python visualizer.py ./train_logs ./visualizations 50000 200 50
    """
    import sys
    
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "./train_logs"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./visualizations"
    segment_size = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
    plot_scale = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    plot_interval = int(sys.argv[5]) if len(sys.argv) > 5 else 100
    
    viz = Visualizer(output_dir=output_dir, segment_size=segment_size, 
                    plot_scale=plot_scale, plot_interval=plot_interval)
    viz.generate_all(log_dir=log_dir)
