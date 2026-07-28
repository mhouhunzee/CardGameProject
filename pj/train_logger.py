"""训练日志记录模块 - 为独立可视化生成结构化日志.

此模块与train.py一起使用，记录训练过程中的关键指标，
生成的日志文件可以被visualizer独立读取和分析.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class GameResult:
    """单局游戏结果记录."""
    episode: int
    game_id: int
    phase: str  # 'initial', 'agent_a', 'agent_b', 'agent_c'
    round_num: int  # 强化轮次，初始训练为0
    
    # 叫分信息
    bids: List[int]  # [agent0_bid, agent1_bid, agent2_bid]
    final_bid: int  # 最终成交分数（流局为0）
    landlord: int  # 地主位置（流局为-1）
    is_draw: bool  # 是否流局
    
    # 游戏结果
    winner: int  # 获胜者（流局为-1）
    game_length: int  # 游戏步数（流局为0）
    
    # 各agent得分
    rewards: List[float]  # [agent0_reward, agent1_reward, agent2_reward]
    
    # 各agent角色
    roles: List[str]  # [agent0_role, agent1_role, agent2_role] 'landlord' or 'farmer'
    
    # 各agent出牌次数（不含Pass）
    action_counts: List[int]  # [agent0_count, agent1_count, agent2_count]
    
    # 炸弹和火箭统计
    bomb_count: int
    rocket_count: int
    
    timestamp: str


@dataclass
class EvalResult:
    """强化训练后评估结果."""
    agent: str  # 'a', 'b', 'c'
    round_num: int
    episode: int
    win_rate: float  # 与初始模型对战的胜率
    avg_score: float  # 平均得分
    total_games: int  # 测试局数
    wins: int  # 获胜局数
    timestamp: str


@dataclass
class EpisodeSummary:
    """每轮训练的汇总统计."""
    episode: int
    phase: str
    round_num: int
    
    # 胜率统计（最近100局滑动窗口）
    win_rates: List[float]  # [agent0, agent1, agent2]
    landlord_win_rate: float  # 地主胜率
    
    # 得分统计
    avg_scores: List[float]  # [agent0, agent1, agent2]
    
    # 叫分统计
    avg_bids: List[float]  # [agent0, agent1, agent2]
    bid_distribution: List[Dict[int, int]]  # 每个agent的叫分分布 {0: count, 1: count, ...}
    
    # 角色胜率（混淆矩阵用）
    landlord_win_rates: List[float]  # 每个agent作为地主的胜率
    farmer_win_rates: List[float]  # 每个agent作为农民的胜率
    
    # 全局统计
    total_games: int  # 总局数
    draw_count: int  # 流局数
    avg_game_length: float  # 平均游戏长度（不含流局）
    avg_final_bid: float  # 平均成交分数（含流局，流局为0）
    avg_action_count: float  # 平均出牌次数（不含流局）
    
    timestamp: str


class TrainLogger:
    """训练日志记录器 - 生成可供visualizer独立分析的日志."""
    
    def __init__(self, log_dir: str = "./train_logs", phase: str = "initial", round_num: int = 0):
        """初始化日志记录器.
        
        Args:
            log_dir: 日志文件存放目录
            phase: 训练阶段 ('initial', 'agent_a', 'agent_b', 'agent_c')
            round_num: 强化轮次（初始训练为0）
        """
        self.log_dir = log_dir
        self.phase = phase
        self.round_num = round_num
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件路径
        self.game_log_path = os.path.join(log_dir, f"game_log_{phase}_r{round_num}_{self.timestamp}.jsonl")
        self.summary_log_path = os.path.join(log_dir, f"summary_{phase}_r{round_num}_{self.timestamp}.jsonl")
        
        # 缓冲区
        self.game_buffer: List[GameResult] = []
        self.summary_buffer: List[EpisodeSummary] = []
        self.buffer_size = 100
        
        # 统计数据
        self.game_history: List[GameResult] = []  # 保存所有游戏结果用于计算滑动窗口
        
        # 初始化文件
        self._init_files()
    
    def _init_files(self):
        """初始化日志文件."""
        header = {
            'type': 'header',
            'timestamp': self.timestamp,
            'phase': self.phase,
            'round_num': self.round_num,
            'description': f'Training log for {self.phase} phase, round {self.round_num}'
        }
        
        with open(self.game_log_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(header, ensure_ascii=False) + '\n')
        
        with open(self.summary_log_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(header, ensure_ascii=False) + '\n')
    
    def log_game(self, episode: int, game_id: int, bids: List[int],
                 final_bid: int, landlord: int, is_draw: bool,
                 winner: int, game_length: int, rewards: List[float],
                 roles: List[str], action_counts: List[int],
                 bomb_count: int, rocket_count: int):
        """记录单局游戏结果.
        
        Args:
            episode: 训练轮次
            game_id: 游戏ID
            bids: 三个agent的叫分
            final_bid: 最终成交分数
            landlord: 地主位置
            is_draw: 是否流局
            winner: 获胜者
            game_length: 游戏步数
            rewards: 三个agent的得分
            roles: 三个agent的角色
            action_counts: 三个agent的出牌次数
            bomb_count: 炸弹数量
            rocket_count: 火箭数量
        """
        game = GameResult(
            episode=episode,
            game_id=game_id,
            phase=self.phase,
            round_num=self.round_num,
            bids=bids,
            final_bid=final_bid,
            landlord=landlord,
            is_draw=is_draw,
            winner=winner,
            game_length=game_length,
            rewards=rewards,
            roles=roles,
            action_counts=action_counts,
            bomb_count=bomb_count,
            rocket_count=rocket_count,
            timestamp=datetime.now().isoformat()
        )
        
        self.game_buffer.append(game)
        self.game_history.append(game)
        
        # 批量写入
        if len(self.game_buffer) >= self.buffer_size:
            self._flush_game_buffer()
    
    def log_episode_summary(self, episode: int):
        """记录每轮训练的汇总统计（滑动窗口：最近100局）."""
        # 获取最近100局
        recent_games = self.game_history[-100:] if len(self.game_history) >= 100 else self.game_history
        
        if not recent_games:
            return
        
        # 计算胜率
        win_counts = [0, 0, 0]
        for game in recent_games:
            if game.winner >= 0:
                win_counts[game.winner] += 1
        win_rates = [c / len(recent_games) for c in win_counts]
        
        # 计算地主胜率
        landlord_wins = sum(1 for g in recent_games if g.winner == g.landlord and not g.is_draw)
        valid_games = [g for g in recent_games if not g.is_draw]
        landlord_win_rate = landlord_wins / len(valid_games) if valid_games else 0
        
        # 计算平均得分
        avg_scores = [
            sum(g.rewards[i] for g in recent_games) / len(recent_games)
            for i in range(3)
        ]
        
        # 计算平均叫分
        avg_bids = [
            sum(g.bids[i] for g in recent_games) / len(recent_games)
            for i in range(3)
        ]
        
        # 计算叫分分布
        bid_distribution = []
        for i in range(3):
            dist = {0: 0, 1: 0, 2: 0, 3: 0}
            for g in recent_games:
                dist[g.bids[i]] = dist.get(g.bids[i], 0) + 1
            bid_distribution.append(dist)
        
        # 计算角色胜率（混淆矩阵用）
        landlord_win_rates = []
        farmer_win_rates = []
        for i in range(3):
            landlord_games = [g for g in recent_games if g.roles[i] == 'landlord' and not g.is_draw]
            farmer_games = [g for g in recent_games if g.roles[i] == 'farmer' and not g.is_draw]
            
            landlord_wins_i = sum(1 for g in landlord_games if g.winner == i)
            farmer_wins_i = sum(1 for g in farmer_games if g.winner == i)
            
            landlord_win_rates.append(landlord_wins_i / len(landlord_games) if landlord_games else 0)
            farmer_win_rates.append(farmer_wins_i / len(farmer_games) if farmer_games else 0)
        
        # 全局统计
        total_games = len(self.game_history)
        draw_count = sum(1 for g in self.game_history if g.is_draw)
        
        valid_games_all = [g for g in self.game_history if not g.is_draw]
        avg_game_length = sum(g.game_length for g in valid_games_all) / len(valid_games_all) if valid_games_all else 0
        avg_final_bid = sum(g.final_bid for g in self.game_history) / len(self.game_history) if self.game_history else 0
        avg_action_count = sum(sum(g.action_counts) for g in valid_games_all) / len(valid_games_all) if valid_games_all else 0
        
        summary = EpisodeSummary(
            episode=episode,
            phase=self.phase,
            round_num=self.round_num,
            win_rates=win_rates,
            landlord_win_rate=landlord_win_rate,
            avg_scores=avg_scores,
            avg_bids=avg_bids,
            bid_distribution=bid_distribution,
            landlord_win_rates=landlord_win_rates,
            farmer_win_rates=farmer_win_rates,
            total_games=total_games,
            draw_count=draw_count,
            avg_game_length=avg_game_length,
            avg_final_bid=avg_final_bid,
            avg_action_count=avg_action_count,
            timestamp=datetime.now().isoformat()
        )
        
        self.summary_buffer.append(summary)
        
        # 批量写入
        if len(self.summary_buffer) >= self.buffer_size:
            self._flush_summary_buffer()
    
    def _flush_game_buffer(self):
        """刷新游戏日志缓冲区."""
        if not self.game_buffer:
            return
        
        with open(self.game_log_path, 'a', encoding='utf-8') as f:
            for game in self.game_buffer:
                f.write(json.dumps(asdict(game), ensure_ascii=False) + '\n')
        
        self.game_buffer.clear()
    
    def _flush_summary_buffer(self):
        """刷新汇总日志缓冲区."""
        if not self.summary_buffer:
            return
        
        with open(self.summary_log_path, 'a', encoding='utf-8') as f:
            for summary in self.summary_buffer:
                f.write(json.dumps(asdict(summary), ensure_ascii=False) + '\n')
        
        self.summary_buffer.clear()
    
    def close(self):
        """关闭日志记录器."""
        self._flush_game_buffer()
        self._flush_summary_buffer()
        
        # 写入总结
        summary = {
            'type': 'summary',
            'timestamp': datetime.now().isoformat(),
            'phase': self.phase,
            'round_num': self.round_num,
            'total_games': len(self.game_history),
            'file_paths': {
                'game_log': self.game_log_path,
                'summary_log': self.summary_log_path
            }
        }
        
        with open(self.game_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')
        
        with open(self.summary_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')
        
        print(f"\n训练日志已保存:")
        print(f"  游戏日志: {self.game_log_path}")
        print(f"  汇总日志: {self.summary_log_path}")
        print(f"  总游戏数: {len(self.game_history)}")


if __name__ == "__main__":
    # 测试
    logger = TrainLogger(phase="initial", round_num=0)
    
    for episode in range(10):
        for game in range(5):
            game_id = episode * 1000 + game
            
            logger.log_game(
                episode=episode,
                game_id=game_id,
                bids=[random.randint(0, 3) for _ in range(3)],
                final_bid=random.randint(1, 3),
                landlord=random.randint(0, 2),
                is_draw=False,
                winner=random.randint(0, 2),
                game_length=random.randint(15, 40),
                rewards=[random.random() * 10 - 5 for _ in range(3)],
                roles=['landlord', 'farmer', 'farmer'],
                action_counts=[random.randint(5, 15) for _ in range(3)],
                bomb_count=random.randint(0, 2),
                rocket_count=random.randint(0, 1)
            )
        
        logger.log_episode_summary(episode)
    
    logger.close()
    
    print("\n测试完成!")
