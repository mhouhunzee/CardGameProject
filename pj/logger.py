"""日志记录模块.

记录牌库生成和出牌顺序的详细日志.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class DeckRecord:
    """牌局记录."""
    episode: int
    game_id: int
    player1_hand: List[int]
    player2_hand: List[int]
    player3_hand: List[int]
    bottom_cards: List[int]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """转换为字典."""
        return {
            'episode': self.episode,
            'game_id': self.game_id,
            'player1_hand': self.player1_hand,
            'player2_hand': self.player2_hand,
            'player3_hand': self.player3_hand,
            'bottom_cards': self.bottom_cards,
            'timestamp': self.timestamp
        }


@dataclass
class PlayRecord:
    """出牌记录."""
    episode: int
    game_id: int
    step: int
    player_id: int
    role: str  # 'landlord' or 'farmer'
    action_type: str  # 牌型名称
    cards_played: List[int]
    cards_str: str  # 字符串表示
    hand_before: List[int]  # 出牌前手牌
    hand_after: List[int]   # 出牌后手牌
    remaining_count: int    # 剩余牌数
    is_pass: bool
    is_valid: bool
    error_msg: str
    timestamp: str
    
    def to_dict(self) -> Dict:
        """转换为字典."""
        return {
            'episode': self.episode,
            'game_id': self.game_id,
            'step': self.step,
            'player_id': self.player_id,
            'role': self.role,
            'action_type': self.action_type,
            'cards_played': self.cards_played,
            'cards_str': self.cards_str,
            'hand_before_count': len(self.hand_before),
            'hand_after_count': len(self.hand_after),
            'remaining_count': self.remaining_count,
            'is_pass': self.is_pass,
            'is_valid': self.is_valid,
            'error_msg': self.error_msg,
            'timestamp': self.timestamp
        }


class GameLogger:
    """游戏日志记录器."""
    
    def __init__(self, log_dir: str = "./logs"):
        """初始化日志记录器.
        
        Args:
            log_dir: 日志文件存放目录
        """
        self.log_dir = log_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 日志文件路径
        self.deck_log_path = os.path.join(log_dir, f"deck_log_{self.timestamp}.jsonl")
        self.play_log_path = os.path.join(log_dir, f"play_log_{self.timestamp}.jsonl")
        
        # 缓冲区
        self.deck_buffer: List[DeckRecord] = []
        self.play_buffer: List[PlayRecord] = []
        self.buffer_size = 100  # 每100条写入一次
        
        # 统计
        self.total_games = 0
        self.total_plays = 0
        self.violation_count = 0
        
        # 创建文件并写入头部
        self._init_files()
    
    def _init_files(self):
        """初始化日志文件."""
        # 牌库日志头部
        with open(self.deck_log_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                'type': 'header',
                'timestamp': self.timestamp,
                'description': 'Deck generation log - records all randomly generated card distributions',
                'format': 'JSON Lines (JSONL)'
            }, ensure_ascii=False) + '\n')
        
        # 出牌日志头部
        with open(self.play_log_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps({
                'type': 'header',
                'timestamp': self.timestamp,
                'description': 'Play sequence log - records all plays by three virtual players during RL training',
                'format': 'JSON Lines (JSONL)'
            }, ensure_ascii=False) + '\n')
    
    def log_deck(self, episode: int, game_id: int, 
                 player1_hand: List[int], player2_hand: List[int], 
                 player3_hand: List[int], bottom_cards: List[int]):
        """记录牌局.
        
        Args:
            episode: 训练轮次
            game_id: 游戏ID
            player1_hand: 玩家1手牌
            player2_hand: 玩家2手牌
            player3_hand: 玩家3手牌
            bottom_cards: 底牌
        """
        record = DeckRecord(
            episode=episode,
            game_id=game_id,
            player1_hand=sorted(player1_hand),
            player2_hand=sorted(player2_hand),
            player3_hand=sorted(player3_hand),
            bottom_cards=sorted(bottom_cards),
            timestamp=datetime.now().isoformat()
        )
        
        self.deck_buffer.append(record)
        self.total_games += 1
        
        # 批量写入
        if len(self.deck_buffer) >= self.buffer_size:
            self._flush_deck_buffer()
    
    def log_play(self, episode: int, game_id: int, step: int,
                 player_id: int, role: str, action_type: str,
                 cards_played: List[int], cards_str: str,
                 hand_before: List[int], hand_after: List[int],
                 is_pass: bool = False, is_valid: bool = True,
                 error_msg: str = ""):
        """记录出牌.
        
        Args:
            episode: 训练轮次
            game_id: 游戏ID
            step: 步数
            player_id: 玩家ID
            role: 角色 ('landlord' or 'farmer')
            action_type: 牌型名称
            cards_played: 出的牌
            cards_str: 字符串表示
            hand_before: 出牌前手牌
            hand_after: 出牌后手牌
            is_pass: 是否Pass
            is_valid: 是否合法
            error_msg: 错误信息
        """
        record = PlayRecord(
            episode=episode,
            game_id=game_id,
            step=step,
            player_id=player_id,
            role=role,
            action_type=action_type,
            cards_played=sorted(cards_played) if cards_played else [],
            cards_str=cards_str,
            hand_before=sorted(hand_before),
            hand_after=sorted(hand_after),
            remaining_count=len(hand_after),
            is_pass=is_pass,
            is_valid=is_valid,
            error_msg=error_msg,
            timestamp=datetime.now().isoformat()
        )
        
        self.play_buffer.append(record)
        self.total_plays += 1
        
        if not is_valid:
            self.violation_count += 1
        
        # 批量写入
        if len(self.play_buffer) >= self.buffer_size:
            self._flush_play_buffer()
    
    def _flush_deck_buffer(self):
        """刷新牌库日志缓冲区."""
        if not self.deck_buffer:
            return
        
        with open(self.deck_log_path, 'a', encoding='utf-8') as f:
            for record in self.deck_buffer:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
        
        self.deck_buffer.clear()
    
    def _flush_play_buffer(self):
        """刷新出牌日志缓冲区."""
        if not self.play_buffer:
            return
        
        with open(self.play_log_path, 'a', encoding='utf-8') as f:
            for record in self.play_buffer:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + '\n')
        
        self.play_buffer.clear()
    
    def close(self):
        """关闭日志记录器，刷新剩余缓冲区."""
        self._flush_deck_buffer()
        self._flush_play_buffer()
        
        # 写入统计信息
        self._write_summary()
        
        print(f"\n日志记录完成:")
        print(f"  牌库日志: {self.deck_log_path}")
        print(f"  出牌日志: {self.play_log_path}")
        print(f"  总游戏数: {self.total_games}")
        print(f"  总出牌数: {self.total_plays}")
        print(f"  违规次数: {self.violation_count}")
    
    def _write_summary(self):
        """写入统计摘要."""
        summary = {
            'type': 'summary',
            'timestamp': datetime.now().isoformat(),
            'total_games': self.total_games,
            'total_plays': self.total_plays,
            'violation_count': self.violation_count,
            'violation_rate': self.violation_count / max(self.total_plays, 1)
        }
        
        # 添加到两个日志文件末尾
        with open(self.deck_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')
        
        with open(self.play_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')


class LogAnalyzer:
    """日志分析器."""
    
    def __init__(self, deck_log_path: str, play_log_path: str):
        """初始化分析器.
        
        Args:
            deck_log_path: 牌库日志路径
            play_log_path: 出牌日志路径
        """
        self.deck_log_path = deck_log_path
        self.play_log_path = play_log_path
    
    def analyze_deck_distribution(self) -> Dict:
        """分析牌库分布.
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_games': 0,
            'avg_high_cards': [],  # 2, A, K的平均数量
            'bomb_probability': 0,
            'rocket_probability': 0
        }
        
        with open(self.deck_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                if record.get('type') == 'header' or record.get('type') == 'summary':
                    continue
                
                stats['total_games'] += 1
                
                # 统计每个玩家的大牌数量
                for player_key in ['player1_hand', 'player2_hand', 'player3_hand']:
                    hand = record.get(player_key, [])
                    high_cards = sum(1 for c in hand if c >= 13)  # K, A, 2
                    stats['avg_high_cards'].append(high_cards)
                
                # 检查是否有炸弹/火箭
                for player_key in ['player1_hand', 'player2_hand', 'player3_hand']:
                    hand = record.get(player_key, [])
                    from card_utils import count_cards
                    count = count_cards(hand)
                    if any(c >= 4 for c in count.values()):
                        stats['bomb_probability'] += 1
                    if 16 in hand and 17 in hand:
                        stats['rocket_probability'] += 1
        
        if stats['total_games'] > 0:
            stats['bomb_probability'] /= (stats['total_games'] * 3)  # 每玩家概率
            stats['rocket_probability'] /= (stats['total_games'] * 3)
            stats['avg_high_cards'] = sum(stats['avg_high_cards']) / len(stats['avg_high_cards'])
        
        return stats
    
    def analyze_play_patterns(self) -> Dict:
        """分析出牌模式.
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_plays': 0,
            'pass_count': 0,
            'action_type_distribution': {},
            'avg_game_length': [],
            'violation_count': 0
        }
        
        game_lengths = {}
        
        with open(self.play_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                if record.get('type') == 'header' or record.get('type') == 'summary':
                    continue
                
                stats['total_plays'] += 1
                
                if record.get('is_pass'):
                    stats['pass_count'] += 1
                
                action_type = record.get('action_type', 'UNKNOWN')
                stats['action_type_distribution'][action_type] = \
                    stats['action_type_distribution'].get(action_type, 0) + 1
                
                if not record.get('is_valid', True):
                    stats['violation_count'] += 1
                
                # 统计游戏长度
                game_id = (record.get('episode'), record.get('game_id'))
                step = record.get('step', 0)
                if game_id not in game_lengths or step > game_lengths[game_id]:
                    game_lengths[game_id] = step
        
        if game_lengths:
            stats['avg_game_length'] = sum(game_lengths.values()) / len(game_lengths)
        
        return stats


if __name__ == "__main__":
    # 测试
    logger = GameLogger()
    
    # 模拟记录一些数据
    for episode in range(3):
        for game in range(2):
            game_id = episode * 1000 + game
            
            # 记录牌局
            logger.log_deck(
                episode=episode,
                game_id=game_id,
                player1_hand=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 3, 4],
                player2_hand=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3, 4, 5, 6],
                player3_hand=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 3, 4, 5, 6],
                bottom_cards=[16, 17, 3]
            )
            
            # 记录出牌
            for step in range(10):
                logger.log_play(
                    episode=episode,
                    game_id=game_id,
                    step=step,
                    player_id=step % 3,
                    role='landlord' if step % 3 == 0 else 'farmer',
                    action_type='SINGLE',
                    cards_played=[3 + step],
                    cards_str=str(3 + step),
                    hand_before=list(range(3, 20)),
                    hand_after=list(range(3, 20)),
                    is_pass=False,
                    is_valid=True
                )
    
    logger.close()
    
    # 分析
    analyzer = LogAnalyzer(logger.deck_log_path, logger.play_log_path)
    print("\n牌库分布分析:")
    print(analyzer.analyze_deck_distribution())
    print("\n出牌模式分析:")
    print(analyzer.analyze_play_patterns())
