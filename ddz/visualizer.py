"""
斗地主训练可视化
生成各种训练图表
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from config import (
    VISUALIZATION_DIR, CHART_DPI, CHART_FIGSIZE,
    NUM_CHART_BARS, BATCH, LOG_DIR
)


class DouDiZhuVisualizer:
    """斗地主训练可视化器"""
    
    def __init__(self, log_dir=LOG_DIR, output_dir=VISUALIZATION_DIR):
        self.log_dir = log_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def load_logs(self):
        """加载训练日志"""
        # 这里需要根据实际的日志格式实现
        # 简化版：假设日志是JSON格式
        log_file = os.path.join(self.log_dir, "training_log.jsonl")
        
        if not os.path.exists(log_file):
            print(f"日志文件不存在: {log_file}")
            return []
        
        logs = []
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))
        
        return logs
    
    def aggregate_by_batch(self, data, batch_size=BATCH):
        """按批次聚合数据"""
        batches = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            batches.append(np.mean(batch) if batch else 0)
        
        # 确保只有100个柱子
        if len(batches) > NUM_CHART_BARS:
            # 重新采样到100个点
            indices = np.linspace(0, len(batches)-1, NUM_CHART_BARS, dtype=int)
            batches = [batches[i] for i in indices]
        
        return batches[:NUM_CHART_BARS]
    
    def plot_win_rates(self, logs):
        """绘制胜率变化图"""
        # 提取数据
        A_wins = []
        B_wins = []
        C_wins = []
        
        for log in logs:
            if 'win_rates' in log:
                A_wins.append(log['win_rates'].get('A', 0))
                B_wins.append(log['win_rates'].get('B', 0))
                C_wins.append(log['win_rates'].get('C', 0))
        
        # 按批次聚合
        A_batches = self.aggregate_by_batch(A_wins)
        B_batches = self.aggregate_by_batch(B_wins)
        C_batches = self.aggregate_by_batch(C_wins)
        
        # 绘图
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        
        x = np.arange(len(A_batches))
        width = 0.25
        
        ax.bar(x - width, A_batches, width, label='Agent A (头叫)', color='#FF6B6B', alpha=0.8)
        ax.bar(x, B_batches, width, label='Agent B (二叫)', color='#4ECDC4', alpha=0.8)
        ax.bar(x + width, C_batches, width, label='Agent C (三叫)', color='#45B7D1', alpha=0.8)
        
        ax.set_xlabel('批次 (每批次600局)', fontsize=12)
        ax.set_ylabel('胜率', fontsize=12)
        ax.set_title('各Agent胜率变化 (每批次平均)', fontsize=14, fontweight='bold')
        ax.set_xticks(x[::10])
        ax.set_xticklabels([f'{i*10}' for i in range(len(x)//10 + 1)])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_win_rates.png'), dpi=CHART_DPI)
        plt.close()
        print(f"已保存: 01_win_rates.png")
    
    def plot_scores(self, logs):
        """绘制得分变化图"""
        A_scores = []
        B_scores = []
        C_scores = []
        
        for log in logs:
            if 'scores' in log:
                A_scores.append(log['scores'].get('A', 0))
                B_scores.append(log['scores'].get('B', 0))
                C_scores.append(log['scores'].get('C', 0))
        
        A_batches = self.aggregate_by_batch(A_scores)
        B_batches = self.aggregate_by_batch(B_scores)
        C_batches = self.aggregate_by_batch(C_scores)
        
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        
        x = np.arange(len(A_batches))
        
        ax.plot(x, A_batches, 'o-', label='Agent A', color='#FF6B6B', linewidth=2, markersize=4)
        ax.plot(x, B_batches, 's-', label='Agent B', color='#4ECDC4', linewidth=2, markersize=4)
        ax.plot(x, C_batches, '^-', label='Agent C', color='#45B7D1', linewidth=2, markersize=4)
        
        ax.set_xlabel('批次', fontsize=12)
        ax.set_ylabel('平均得分', fontsize=12)
        ax.set_title('各Agent得分变化', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_scores.png'), dpi=CHART_DPI)
        plt.close()
        print(f"已保存: 02_scores.png")
    
    def plot_final_bids(self, logs):
        """绘制平均成交分数"""
        final_bids = []
        
        for log in logs:
            if 'final_bid' in log:
                final_bids.append(log['final_bid'])
        
        bid_batches = self.aggregate_by_batch(final_bids)
        
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        
        x = np.arange(len(bid_batches))
        colors = ['#96CEB4' if b < 1.5 else '#FFEAA7' if b < 2.5 else '#DDA0DD' for b in bid_batches]
        
        ax.bar(x, bid_batches, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('批次', fontsize=12)
        ax.set_ylabel('平均成交分数', fontsize=12)
        ax.set_title('平均成交分数变化', fontsize=14, fontweight='bold')
        ax.axhline(y=1.5, color='r', linestyle='--', alpha=0.5, label='低分阈值')
        ax.axhline(y=2.5, color='g', linestyle='--', alpha=0.5, label='高分阈值')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_final_bids.png'), dpi=CHART_DPI)
        plt.close()
        print(f"已保存: 03_final_bids.png")
    
    def plot_bomb_frequency(self, logs):
        """绘制炸弹使用频率"""
        bomb_counts = []
        
        for log in logs:
            if 'bomb_count' in log:
                bomb_counts.append(log['bomb_count'])
        
        bomb_batches = self.aggregate_by_batch(bomb_counts)
        
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        
        x = np.arange(len(bomb_batches))
        
        ax.fill_between(x, bomb_batches, alpha=0.5, color='#E74C3C')
        ax.plot(x, bomb_batches, 'o-', color='#C0392B', linewidth=2, markersize=4)
        
        ax.set_xlabel('批次', fontsize=12)
        ax.set_ylabel('平均炸弹数', fontsize=12)
        ax.set_title('炸弹使用频率', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_bomb_frequency.png'), dpi=CHART_DPI)
        plt.close()
        print(f"已保存: 04_bomb_frequency.png")
    
    def plot_spring_frequency(self, logs):
        """绘制春天/反春天频率"""
        springs = []
        anti_springs = []
        
        for log in logs:
            if 'spring' in log and 'anti_spring' in log:
                springs.append(1 if log['spring'] else 0)
                anti_springs.append(1 if log['anti_spring'] else 0)
        
        spring_batches = self.aggregate_by_batch(springs)
        anti_spring_batches = self.aggregate_by_batch(anti_springs)
        
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        
        x = np.arange(len(spring_batches))
        width = 0.35
        
        ax.bar(x - width/2, spring_batches, width, label='春天', color='#2ECC71', alpha=0.8)
        ax.bar(x + width/2, anti_spring_batches, width, label='反春天', color='#E74C3C', alpha=0.8)
        
        ax.set_xlabel('批次', fontsize=12)
        ax.set_ylabel('频率', fontsize=12)
        ax.set_title('春天/反春天频率', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '05_spring_frequency.png'), dpi=CHART_DPI)
        plt.close()
        print(f"已保存: 05_spring_frequency.png")
    
    def generate_all(self):
        """生成所有图表"""
        print("="*60)
        print("开始生成可视化图表")
        print("="*60)
        
        logs = self.load_logs()
        
        if not logs:
            print("没有日志数据可可视化")
            return
        
        print(f"加载了 {len(logs)} 条日志记录\n")
        
        self.plot_win_rates(logs)
        self.plot_scores(logs)
        self.plot_final_bids(logs)
        self.plot_bomb_frequency(logs)
        self.plot_spring_frequency(logs)
        
        print(f"\n所有图表已保存到: {self.output_dir}")
        print("="*60)


if __name__ == "__main__":
    visualizer = DouDiZhuVisualizer()
    visualizer.generate_all()
