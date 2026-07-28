"""Visualizer V2 - 高可视性训练数据分析

基于数据可视化最佳实践设计：
1. 使用seaborn美化图表
2. 合理的颜色搭配和对比度
3. 动态范围调整，突出关键变化
4. 多维度数据融合展示
5. 交互式HTML报告（可选）
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 定义专业配色方案
COLORS = {
    'agent_a': '#E74C3C',  # 鲜红
    'agent_b': '#3498DB',  # 亮蓝  
    'agent_c': '#2ECC71',  # 翠绿
    'landlord': '#F39C12', # 橙色
    'farmer': '#9B59B6',   # 紫色
    'neutral': '#95A5A6',  # 灰色
    'highlight': '#E67E22', # 高亮橙
    'background': '#ECF0F1' # 背景色
}

AGENT_COLORS = [COLORS['agent_a'], COLORS['agent_b'], COLORS['agent_c']]


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
    """强化训练后评估结果"""
    agent: str
    round_num: int
    episode: int
    win_rate: float
    avg_score: float
    total_games: int
    wins: int


class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_all_data(log_dir: str) -> Tuple[List[GameResult], List[EpisodeSummary], List[EvalResult]]:
        """加载所有日志数据"""
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
                    except:
                        continue
        
        # 加载评估结果日志
        for file_path in glob.glob(os.path.join(log_dir, "eval_results_*.jsonl")):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get('type') != 'eval':
                            continue
                        eval_results.append(EvalResult(
                            agent=data['agent'],
                            round_num=data['round_num'],
                            episode=data['episode'],
                            win_rate=data['win_rate'],
                            avg_score=data['avg_score'],
                            total_games=data['total_games'],
                            wins=data['wins']
                        ))
                    except:
                        continue
        
        games.sort(key=lambda x: x.episode)
        summaries.sort(key=lambda x: x.episode)
        eval_results.sort(key=lambda x: x.episode)
        return games, summaries, eval_results


class VisualizerV2:
    """可视化器V2"""
    
    def __init__(self, output_dir: str = "./visualizations_v2",
                 plot_scale: int = 100,
                 plot_interval: int = 100):
        """初始化可视化器V2.
        
        Args:
            output_dir: 图表输出目录
            plot_scale: 图表包含的训练轮数尺度（默认100）
            plot_interval: 数据点采样间隔（默认100）
        """
        self.output_dir = output_dir
        self.plot_scale = plot_scale
        self.plot_interval = plot_interval
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建子目录
        self.dirs = {
            'overview': os.path.join(output_dir, '01_overview'),
            'performance': os.path.join(output_dir, '02_performance'),
            'evolution': os.path.join(output_dir, '03_evolution'),
            'strategy': os.path.join(output_dir, '04_strategy'),
            'detailed': os.path.join(output_dir, '05_detailed'),
            'evaluation': os.path.join(output_dir, '06_evaluation')  # 新增评估目录
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
    
    def generate_all(self, log_dir: str = "./train_logs"):
        """生成所有可视化"""
        print("=" * 70)
        print("Visualizer V2 - 生成高可视性图表")
        print(f"图表尺度: {self.plot_scale} | 采样间隔: {self.plot_interval}")
        print("=" * 70)
        
        # 加载数据
        print("\n📊 加载数据...")
        games, summaries, eval_results = DataLoader.load_all_data(log_dir)
        print(f"   游戏记录: {len(games):,} 条")
        print(f"   汇总记录: {len(summaries):,} 条")
        print(f"   评估记录: {len(eval_results):,} 条")
        
        # 数据采样
        if self.plot_interval > 1 and len(summaries) > self.plot_interval:
            summaries = summaries[::self.plot_interval]
            print(f"   采样后: {len(summaries):,} 条")
        
        if not games or not summaries:
            print("❌ 无数据可可视化")
            return
        
        # 数据预处理
        df_games = self._games_to_dataframe(games)
        df_summaries = self._summaries_to_dataframe(summaries)
        
        # 生成各类图表
        print("\n📈 生成概览图表...")
        self._generate_overview(df_games, df_summaries)
        
        print("\n🏆 生成性能分析图表...")
        self._generate_performance(df_games, df_summaries)
        
        print("\n📉 生成进化趋势图表...")
        self._generate_evolution(df_games, df_summaries)
        
        print("\n🎯 生成策略分析图表...")
        self._generate_strategy(df_games, df_summaries)
        
        print("\n🔍 生成详细分析图表...")
        self._generate_detailed(df_games, df_summaries)
        
        # 生成评估结果图表
        if eval_results:
            print("\n📊 生成强化训练评估图表...")
            self._generate_evaluation_charts(eval_results)
        
        print("\n" + "=" * 70)
        print(f"✅ 完成! 输出目录: {self.output_dir}")
        print("=" * 70)
    
    def _games_to_dataframe(self, games: List[GameResult]) -> pd.DataFrame:
        """转换游戏数据为DataFrame"""
        data = []
        for g in games:
            data.append({
                'episode': g.episode,
                'phase': g.phase,
                'round_num': g.round_num,
                'is_draw': g.is_draw,
                'winner': g.winner,
                'landlord': g.landlord,
                'final_bid': g.final_bid,
                'game_length': g.game_length,
                'bomb_count': g.bomb_count,
                'rocket_count': g.rocket_count,
                'total_actions': sum(g.action_counts) if not g.is_draw else 0,
                'agent_0_reward': g.rewards[0] if not g.is_draw else 0,
                'agent_1_reward': g.rewards[1] if not g.is_draw else 0,
                'agent_2_reward': g.rewards[2] if not g.is_draw else 0,
            })
        return pd.DataFrame(data)
    
    def _summaries_to_dataframe(self, summaries: List[EpisodeSummary]) -> pd.DataFrame:
        """转换汇总数据为DataFrame"""
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
    
    def _generate_overview(self, df_games: pd.DataFrame, df_summaries: pd.DataFrame):
        """生成概览图表 - 整体训练情况"""
        
        # 1. 训练进度时间线
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Training Overview Dashboard', fontsize=16, fontweight='bold')
        
        # 左上：三Agent胜率对比（热力图风格）
        ax1 = axes[0, 0]
        win_data = df_summaries[['agent_0_win_rate', 'agent_1_win_rate', 'agent_2_win_rate']].values.T
        im = ax1.imshow(win_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=0.6)
        ax1.set_yticks([0, 1, 2])
        ax1.set_yticklabels(['Agent A', 'Agent B', 'Agent C'])
        ax1.set_xlabel('Training Progress')
        ax1.set_title('Win Rate Heatmap')
        plt.colorbar(im, ax=ax1, label='Win Rate')
        
        # 右上：累计得分趋势（面积图）
        ax2 = axes[0, 1]
        df_summaries['agent_0_cumsum'] = df_summaries['agent_0_score'].cumsum()
        df_summaries['agent_1_cumsum'] = df_summaries['agent_1_score'].cumsum()
        df_summaries['agent_2_cumsum'] = df_summaries['agent_2_score'].cumsum()
        
        ax2.fill_between(df_summaries['episode'], 0, df_summaries['agent_0_cumsum'], 
                        alpha=0.3, color=AGENT_COLORS[0], label='Agent A')
        ax2.fill_between(df_summaries['episode'], 0, df_summaries['agent_1_cumsum'], 
                        alpha=0.3, color=AGENT_COLORS[1], label='Agent B')
        ax2.fill_between(df_summaries['episode'], 0, df_summaries['agent_2_cumsum'], 
                        alpha=0.3, color=AGENT_COLORS[2], label='Agent C')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Score')
        ax2.set_title('Cumulative Score Trends')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 左下：地主vs农民胜率
        ax3 = axes[1, 0]
        ax3.plot(df_summaries['episode'], df_summaries['landlord_win_rate'], 
                color=COLORS['landlord'], linewidth=2, label='Landlord')
        ax3.plot(df_summaries['episode'], 1 - df_summaries['landlord_win_rate'], 
                color=COLORS['farmer'], linewidth=2, label='Farmer')
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Win Rate')
        ax3.set_title('Landlord vs Farmer Win Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 右下：关键指标卡片
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        total_games = len(df_games)
        valid_games = len(df_games[~df_games['is_draw']])
        draw_rate = df_games['is_draw'].mean()
        avg_bid = df_games[df_games['is_draw'] == False]['final_bid'].mean()
        
        metrics = [
            f"Total Games: {total_games:,}",
            f"Valid Games: {valid_games:,}",
            f"Draw Rate: {draw_rate:.1%}",
            f"Avg Final Bid: {avg_bid:.2f}",
            f"Episodes: {df_summaries['episode'].max():,}"
        ]
        
        for i, metric in enumerate(metrics):
            ax4.text(0.1, 0.9 - i*0.15, metric, fontsize=14, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['overview'], '01_training_overview.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 01_training_overview.png")
    
    def _generate_performance(self, df_games: pd.DataFrame, df_summaries: pd.DataFrame):
        """生成性能分析图表"""
        
        # 1. Agent性能雷达图对比
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
        fig.suptitle('Agent Performance Radar', fontsize=16, fontweight='bold')
        
        categories = ['Win Rate', 'Avg Score', 'Bid Aggressiveness', 'Consistency', 'Efficiency']
        
        for idx, (ax, agent_name) in enumerate(zip(axes, ['A', 'B', 'C'])):
            # 计算各维度得分（归一化到0-1）
            win_rate = df_summaries[f'agent_{idx}_win_rate'].mean()
            avg_score = (df_summaries[f'agent_{idx}_score'].mean() + 1) / 2  # 归一化
            bid_agg = df_summaries[f'agent_{idx}_bid'].mean() / 3
            consistency = 1 - df_summaries[f'agent_{idx}_win_rate'].std()
            efficiency = df_summaries[f'agent_{idx}_score'].mean() / (df_summaries[f'agent_{idx}_bid'].mean() + 0.1)
            efficiency = max(0, min(1, (efficiency + 1) / 2))
            
            values = [win_rate, avg_score, bid_agg, consistency, efficiency]
            values += values[:1]  # 闭合
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color=AGENT_COLORS[idx])
            ax.fill(angles, values, alpha=0.25, color=AGENT_COLORS[idx])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=9)
            ax.set_ylim(0, 1)
            ax.set_title(f'Agent {agent_name}', size=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['performance'], '01_performance_radar.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 01_performance_radar.png")
        
        # 2. 胜率分布小提琴图
        fig, ax = plt.subplots(figsize=(12, 6))
        win_data = [
            df_summaries['agent_0_win_rate'].values,
            df_summaries['agent_1_win_rate'].values,
            df_summaries['agent_2_win_rate'].values
        ]
        
        parts = ax.violinplot(win_data, positions=[1, 2, 3], showmeans=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(AGENT_COLORS[i])
            pc.set_alpha(0.7)
        
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Agent A', 'Agent B', 'Agent C'])
        ax.set_ylabel('Win Rate Distribution')
        ax.set_title('Win Rate Distribution Comparison')
        ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['performance'], '02_winrate_distribution.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 02_winrate_distribution.png")
    
    def _generate_evolution(self, df_games: pd.DataFrame, df_summaries: pd.DataFrame):
        """生成进化趋势图表"""
        
        # 1. 多维度进化趋势（使用GridSpec）
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        fig.suptitle('Training Evolution Analysis', fontsize=16, fontweight='bold')
        
        # 胜率趋势（带置信区间）
        ax1 = fig.add_subplot(gs[0, :])
        for i in range(3):
            # 计算移动平均和置信区间
            ma = df_summaries[f'agent_{i}_win_rate'].rolling(window=50, min_periods=1).mean()
            std = df_summaries[f'agent_{i}_win_rate'].rolling(window=50, min_periods=1).std()
            ax1.plot(df_summaries['episode'], ma, color=AGENT_COLORS[i], 
                    linewidth=2, label=f'Agent {chr(65+i)}')
            ax1.fill_between(df_summaries['episode'], ma-std, ma+std, 
                           alpha=0.2, color=AGENT_COLORS[i])
        ax1.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Win Rate')
        ax1.set_title('Win Rate Evolution (with Confidence Interval)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 得分趋势
        ax2 = fig.add_subplot(gs[1, 0])
        for i in range(3):
            ax2.plot(df_summaries['episode'], df_summaries[f'agent_{i}_score'], 
                    color=AGENT_COLORS[i], linewidth=1.5, alpha=0.7)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Average Score')
        ax2.set_title('Score Evolution')
        ax2.grid(True, alpha=0.3)
        
        # 叫分趋势
        ax3 = fig.add_subplot(gs[1, 1])
        for i in range(3):
            ax3.plot(df_summaries['episode'], df_summaries[f'agent_{i}_bid'], 
                    color=AGENT_COLORS[i], linewidth=1.5, alpha=0.7)
        ax3.set_ylabel('Average Bid')
        ax3.set_title('Bid Evolution')
        ax3.grid(True, alpha=0.3)
        
        # 学习曲线（累计得分）
        ax4 = fig.add_subplot(gs[2, :])
        for i in range(3):
            cumsum = df_summaries[f'agent_{i}_score'].cumsum()
            ax4.plot(df_summaries['episode'], cumsum, 
                    color=AGENT_COLORS[i], linewidth=2, label=f'Agent {chr(65+i)}')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Cumulative Score')
        ax4.set_title('Learning Curve (Cumulative Score)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(self.dirs['evolution'], '01_evolution_trends.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 01_evolution_trends.png")
    
    def _generate_strategy(self, df_games: pd.DataFrame, df_summaries: pd.DataFrame):
        """生成策略分析图表"""
        
        # 1. 叫分策略热力图（按episode分段）
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Bid Strategy Analysis', fontsize=16, fontweight='bold')
        
        valid_games = df_games[df_games['is_draw'] == False]
        
        for idx, (ax, agent_name) in enumerate(zip(axes, ['A', 'B', 'C'])):
            # 创建叫分分布热力图
            bid_col = f'agent_{idx}_bid'
            
            # 按episode分段计算叫分分布
            episodes = df_summaries['episode'].values
            bid_data = []
            for ep in episodes[::10]:  # 每10个episode采样
                segment = valid_games[(valid_games['episode'] >= ep) & 
                                     (valid_games['episode'] < ep + 100)]
                if len(segment) > 0:
                    bid_dist = segment['final_bid'].value_counts(normalize=True)
                    bid_data.append([bid_dist.get(i, 0) for i in range(4)])
                else:
                    bid_data.append([0, 0, 0, 0])
            
            if bid_data:
                bid_array = np.array(bid_data).T
                im = ax.imshow(bid_array, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
                ax.set_yticks([0, 1, 2, 3])
                ax.set_yticklabels(['Pass', '1', '2', '3'])
                ax.set_xlabel('Training Progress')
                ax.set_title(f'Agent {agent_name} Bid Distribution')
                plt.colorbar(im, ax=ax, label='Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['strategy'], '01_bid_strategy_heatmap.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 01_bid_strategy_heatmap.png")
        
        # 2. 角色胜率分析
        fig, ax = plt.subplots(figsize=(12, 6))
        
        valid_games = df_games[df_games['is_draw'] == False].copy()
        
        # 计算每个agent作为地主和农民的胜率
        role_win_rates = {'Agent A': {'Landlord': 0, 'Farmer': 0},
                         'Agent B': {'Landlord': 0, 'Farmer': 0},
                         'Agent C': {'Landlord': 0, 'Farmer': 0}}
        
        for idx, name in enumerate(['Agent A', 'Agent B', 'Agent C']):
            landlord_games = valid_games[valid_games['landlord'] == idx]
            farmer_games = valid_games[valid_games['landlord'] != idx]
            
            if len(landlord_games) > 0:
                role_win_rates[name]['Landlord'] = (landlord_games['winner'] == idx).mean()
            if len(farmer_games) > 0:
                role_win_rates[name]['Farmer'] = (farmer_games['winner'] == idx).mean()
        
        x = np.arange(3)
        width = 0.35
        landlord_rates = [role_win_rates[name]['Landlord'] for name in ['Agent A', 'Agent B', 'Agent C']]
        farmer_rates = [role_win_rates[name]['Farmer'] for name in ['Agent A', 'Agent B', 'Agent C']]
        
        bars1 = ax.bar(x - width/2, landlord_rates, width, label='Landlord', 
                      color=COLORS['landlord'], alpha=0.8)
        bars2 = ax.bar(x + width/2, farmer_rates, width, label='Farmer', 
                      color=COLORS['farmer'], alpha=0.8)
        
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate by Role')
        ax.set_xticks(x)
        ax.set_xticklabels(['Agent A', 'Agent B', 'Agent C'])
        ax.legend()
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1%}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['strategy'], '02_role_winrate.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 02_role_winrate.png")
    
    def _generate_detailed(self, df_games: pd.DataFrame, df_summaries: pd.DataFrame):
        """生成详细分析图表"""
        
        # 1. 游戏时长分布
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Game Statistics', fontsize=16, fontweight='bold')
        
        valid_games = df_games[df_games['is_draw'] == False]
        
        # 游戏时长分布
        ax1 = axes[0]
        ax1.hist(valid_games['game_length'], bins=30, color='steelblue', 
                alpha=0.7, edgecolor='black')
        ax1.axvline(valid_games['game_length'].mean(), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {valid_games["game_length"].mean():.1f}')
        ax1.set_xlabel('Game Length (steps)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Game Length Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 炸弹/火箭使用统计
        ax2 = axes[1]
        bomb_data = valid_games.groupby('episode')['bomb_count'].mean().rolling(100).mean()
        rocket_data = valid_games.groupby('episode')['rocket_count'].mean().rolling(100).mean()
        
        ax2.plot(bomb_data.index, bomb_data.values, label='Bombs', linewidth=2, color='red')
        ax2.plot(rocket_data.index, rocket_data.values, label='Rockets', linewidth=2, color='orange')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Count per Game')
        ax2.set_title('Bomb & Rocket Usage Trend')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['detailed'], '01_game_statistics.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 01_game_statistics.png")
        
        # 2. 相关性分析矩阵
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_data = df_summaries[['agent_0_win_rate', 'agent_1_win_rate', 'agent_2_win_rate',
                                 'agent_0_score', 'agent_1_score', 'agent_2_score',
                                 'landlord_win_rate', 'avg_final_bid']].corr()
        
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, vmin=-1, vmax=1, square=True, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        ax.set_title('Metrics Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['detailed'], '02_correlation_matrix.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 02_correlation_matrix.png")
    
    def _generate_evaluation_charts(self, eval_results: List[EvalResult]):
        """生成强化训练后评估结果图表"""
        if not eval_results:
            return
        
        # 转换为DataFrame
        df_eval = pd.DataFrame([{
            'agent': r.agent,
            'round_num': r.round_num,
            'episode': r.episode,
            'win_rate': r.win_rate,
            'avg_score': r.avg_score,
            'total_games': r.total_games,
            'wins': r.wins
        } for r in eval_results])
        
        # 1. 强化训练后胜率 vs 初始模型
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Post-Training Evaluation: Win Rate vs Baseline', 
                    fontsize=16, fontweight='bold')
        
        for idx, (ax, agent_name) in enumerate(zip(axes, ['a', 'b', 'c'])):
            agent_data = df_eval[df_eval['agent'] == agent_name].sort_values('round_num')
            
            if len(agent_data) > 0:
                ax.plot(agent_data['round_num'], agent_data['win_rate'], 
                       'o-', color=AGENT_COLORS[idx], linewidth=2, markersize=8)
                ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5, 
                          label='Random Baseline')
                ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.3, 
                          label='Strong Performance')
                ax.set_xlabel('Training Round')
                ax.set_ylabel('Win Rate vs Initial Model')
                ax.set_title(f'Agent {agent_name.upper()}')
                ax.set_ylim(0, 1)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # 添加数值标签
                for _, row in agent_data.iterrows():
                    ax.text(row['round_num'], row['win_rate'] + 0.03, 
                           f"{row['win_rate']:.1%}", 
                           ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['evaluation'], '01_eval_winrate.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 01_eval_winrate.png")
        
        # 2. 强化训练后平均得分 vs 初始模型
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Post-Training Evaluation: Average Score vs Baseline', 
                    fontsize=16, fontweight='bold')
        
        for idx, (ax, agent_name) in enumerate(zip(axes, ['a', 'b', 'c'])):
            agent_data = df_eval[df_eval['agent'] == agent_name].sort_values('round_num')
            
            if len(agent_data) > 0:
                colors = ['red' if s < 0 else 'green' for s in agent_data['avg_score']]
                ax.bar(agent_data['round_num'], agent_data['avg_score'], 
                      color=colors, alpha=0.7, edgecolor='black')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax.set_xlabel('Training Round')
                ax.set_ylabel('Average Score vs Initial Model')
                ax.set_title(f'Agent {agent_name.upper()}')
                ax.grid(True, alpha=0.3, axis='y')
                
                # 添加数值标签
                for _, row in agent_data.iterrows():
                    y_pos = row['avg_score'] + 0.02 if row['avg_score'] >= 0 else row['avg_score'] - 0.05
                    ax.text(row['round_num'], y_pos, 
                           f"{row['avg_score']:+.3f}", 
                           ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['evaluation'], '02_eval_score.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 02_eval_score.png")
        
        # 3. 综合评估对比（所有agent）
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 准备数据
        agents = ['a', 'b', 'c']
        x_pos = np.arange(len(agents))
        width = 0.35
        
        # 获取每个agent最新一轮的数据
        latest_win_rates = []
        latest_scores = []
        
        for agent in agents:
            agent_data = df_eval[df_eval['agent'] == agent]
            if len(agent_data) > 0:
                latest = agent_data.iloc[-1]
                latest_win_rates.append(latest['win_rate'])
                latest_scores.append(latest['avg_score'])
            else:
                latest_win_rates.append(0)
                latest_scores.append(0)
        
        # 创建双Y轴
        ax2 = ax.twinx()
        
        bars1 = ax.bar(x_pos - width/2, latest_win_rates, width, 
                      label='Win Rate', color='steelblue', alpha=0.8)
        bars2 = ax2.bar(x_pos + width/2, latest_scores, width, 
                       label='Avg Score', color='coral', alpha=0.8)
        
        ax.set_xlabel('Agent')
        ax.set_ylabel('Win Rate', color='steelblue')
        ax2.set_ylabel('Average Score', color='coral')
        ax.set_title('Latest Evaluation: Win Rate & Score vs Baseline')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['Agent A', 'Agent B', 'Agent C'])
        ax.set_ylim(0, 1)
        ax.axhline(y=0.33, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # 添加数值标签
        for i, (wr, sc) in enumerate(zip(latest_win_rates, latest_scores)):
            ax.text(i - width/2, wr + 0.02, f"{wr:.1%}", 
                   ha='center', fontsize=10, fontweight='bold')
            y_pos = sc + 0.02 if sc >= 0 else sc - 0.05
            ax2.text(i + width/2, y_pos, f"{sc:+.3f}", 
                    ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dirs['evaluation'], '03_eval_comparison.png'), 
                   dpi=200, bbox_inches='tight')
        plt.close()
        print("   ✓ 03_eval_comparison.png")


if __name__ == "__main__":
    """
    终端调用方法:
    python visualizerV2.py [日志目录] [输出目录] [图表尺度] [采样间隔]
    
    示例:
    python visualizerV2.py ./train_logs ./visualizations_v2
    python visualizerV2.py ./train_logs ./visualizations_v2 200 50
    """
    import sys
    
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "./train_logs"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./visualizations_v2"
    plot_scale = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    plot_interval = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    
    viz = VisualizerV2(output_dir=output_dir, plot_scale=plot_scale, plot_interval=plot_interval)
    viz.generate_all(log_dir=log_dir)
