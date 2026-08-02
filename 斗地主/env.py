"""
斗地主游戏环境

实现完整的斗地主游戏规则：
1. 发牌和叫分
2. 出牌和压制
3. 胜负判定
4. 计分规则
"""

import random
from typing import List, Dict, Optional, Tuple
from collections import Counter
from config import (
    HAND_SIZE, BASE_CARD_SIZE, POSITION_FIRST, POSITION_SECOND, POSITION_THIRD,
    ALL_CARDS, CARD_RANK, REWARD_WIN, REWARD_LOSE
)
from rules import CardPattern, has_cards, remove_cards, get_legal_plays


class DouDiZhuEnv:
    """斗地主游戏环境"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置游戏状态"""
        # 手牌
        self.hands = {0: [], 1: [], 2: []}  # 三个玩家的手牌
        self.base_cards = []  # 底牌
        
        # 游戏状态
        self.landlord = None  # 地主位置
        self.bids = {0: 0, 1: 0, 2: 0}  # 叫分结果
        self.final_bid = 0  # 成交分数
        
        # 出牌状态
        self.current_player = None  # 当前出牌玩家
        self.last_play = None  # 上家出的牌（CardPattern对象）
        self.last_player = None  # 上家是谁
        self.pass_count = 0  # 连续过牌次数
        
        # 游戏记录
        self.played_cards = {0: {}, 1: {}, 2: {}}  # 每个人已出的牌计数
        self.bomb_count = 0  # 炸弹使用次数
        self.rocket_count = 0  # 王炸使用次数
        self.bomb_used_by = {0: False, 1: False, 2: False}  # 谁使用了炸弹
        self.spring = False  # 是否春天
        self.anti_spring = False  # 是否反春天
        
        # 游戏结束
        self.done = False
        self.winner = None  # 获胜方（'landlord' 或 'farmers'）
        
        return self.get_state()
    
    def deal_cards(self, hand_A: List[str], hand_B: List[str], hand_C: List[str], base: List[str]):
        """发牌"""
        self.hands[0] = list(hand_A)
        self.hands[1] = list(hand_B)
        self.hands[2] = list(hand_C)
        self.base_cards = list(base)
    
    def get_state(self):
        """获取当前游戏状态"""
        return {
            'hands': {k: list(v) for k, v in self.hands.items()},
            'base_cards': list(self.base_cards),
            'landlord': self.landlord,
            'bids': dict(self.bids),
            'final_bid': self.final_bid,
            'current_player': self.current_player,
            'last_play': self.last_play.cards_str if self.last_play else None,
            'last_player': self.last_player,
            'done': self.done,
            'winner': self.winner
        }
    
    def bidding_phase(self, bid_0: int, bid_1: int, bid_2: int) -> Tuple[bool, Optional[int], int]:
        """
        叫分阶段
        
        Returns:
            (success, landlord, final_bid)
            success: 是否成功确定地主
            landlord: 地主位置
            final_bid: 最终叫分
        """
        self.bids = {0: bid_0, 1: bid_1, 2: bid_2}
        
        # 找出最高叫分
        max_bid = max(self.bids.values())
        
        # 如果都叫0分，流局
        if max_bid == 0:
            return False, None, 0
        
        # 找最高叫分者（先叫者优先）
        for pos in [0, 1, 2]:
            if self.bids[pos] == max_bid:
                self.landlord = pos
                self.final_bid = max_bid
                break
        
        # 地主获得底牌
        self.hands[self.landlord].extend(self.base_cards)
        
        return True, self.landlord, self.final_bid
    
    def start_playing(self):
        """开始出牌阶段，地主先出"""
        self.current_player = self.landlord
        self.last_play = None
        self.last_player = None
        self.pass_count = 0
    
    def get_legal_actions(self, player_id: int) -> List[str]:
        """获取玩家的合法动作"""
        last_pattern = self.last_play if self.last_player != player_id else None
        return get_legal_plays(self.hands[player_id], last_pattern)
    
    def step(self, player_id: int, action: str) -> Tuple[bool, str, bool, int]:
        """
        执行一步动作
        
        Returns:
            (success, message, done, reward)
        """
        # 检查是否轮到该玩家
        if self.current_player != player_id:
            return False, "不是你的回合", self.done, 0
        
        # 检查动作合法性
        legal_actions = self.get_legal_actions(player_id)
        if action not in legal_actions:
            return False, f"非法动作: {action}", self.done, 0
        
        # 执行动作
        if action == "PASS":
            # 过牌
            self.pass_count += 1
            
            # 检查是否新一轮开始（连续两人PASS）
            if self.pass_count >= 2:
                self.last_play = None
                self.last_player = None
                self.pass_count = 0
        else:
            # 出牌
            play_cards = list(action)
            
            # 检查手牌是否足够
            if not has_cards(self.hands[player_id], play_cards):
                return False, "手牌不足", self.done, 0
            
            # 解析牌型
            pattern = CardPattern(action)
            if not pattern.is_valid():
                return False, "非法牌型", self.done, 0
            
            # 从手牌中移除
            self.hands[player_id] = remove_cards(self.hands[player_id], play_cards)
            
            # 记录已出牌
            for card in play_cards:
                self.played_cards[player_id][card] = self.played_cards[player_id].get(card, 0) + 1
            
            # 更新炸弹计数
            if pattern.type == CardPattern.TYPE_BOMB:
                self.bomb_count += 1
                self.bomb_used_by[player_id] = True
            elif pattern.type == CardPattern.TYPE_ROCKET:
                self.rocket_count += 1
                self.bomb_used_by[player_id] = True
            
            # 更新上家出牌
            self.last_play = pattern
            self.last_player = player_id
            self.pass_count = 0
            
            # 检查是否获胜
            if len(self.hands[player_id]) == 0:
                self.done = True
                
                # 判断获胜方
                if player_id == self.landlord:
                    self.winner = 'landlord'
                    # 检查是否春天
                    other_played = sum(self.played_cards[p].get('total', 0) 
                                     for p in [0, 1, 2] if p != self.landlord)
                    if other_played == 0:
                        self.spring = True
                else:
                    self.winner = 'farmers'
                    # 检查是否反春天
                    landlord_played = self.played_cards[self.landlord].get('total', 0)
                    if landlord_played == 1:  # 地主只出过第一手
                        self.anti_spring = True
        
        # 切换到下一个玩家
        if not self.done:
            self.current_player = (self.current_player + 1) % 3
        
        return True, "OK", self.done, 0
    
    def _calculate_reward(self, player_id: int) -> float:
        """计算玩家的奖励"""
        if self.final_bid == 0:
            return 0
        
        # 基础分数
        base_score = self.final_bid
        
        # 炸弹倍数
        bomb_multiplier = 2 ** (self.bomb_count + self.rocket_count)
        
        # 春天/反春天倍数
        spring_multiplier = 2 if (self.spring or self.anti_spring) else 1
        
        # 最终分数
        final_score = base_score * bomb_multiplier * spring_multiplier
        
        # 判断胜负
        is_landlord = (player_id == self.landlord)
        is_winner = (self.winner == 'landlord' and is_landlord) or \
                   (self.winner == 'farmers' and not is_landlord)
        
        if is_winner:
            reward = final_score
        else:
            reward = -final_score
        
        return reward


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("斗地主环境测试")
    print("=" * 60)
    
    env = DouDiZhuEnv()
    
    # 测试发牌
    hand_A = ['3', '3', '3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', 'X', 'D']
    hand_B = ['3', '4', '4', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', 'X', 'D']
    hand_C = ['3', '4', '5', '5', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', 'X', 'D']
    base = ['7', '7', '7']
    
    env.deal_cards(hand_A, hand_B, hand_C, base)
    print(f"\n发牌完成")
    print(f"  头叫: {len(env.hands[0])}张")
    print(f"  二叫: {len(env.hands[1])}张")
    print(f"  三叫: {len(env.hands[2])}张")
    print(f"  底牌: {env.base_cards}")
    
    # 测试叫分
    success, landlord, final_bid = env.bidding_phase(1, 2, 0)
    print(f"\n叫分结果:")
    print(f"  成功: {success}")
    print(f"  地主: 位置{landlord}")
    print(f"  分数: {final_bid}")
    print(f"  地主手牌: {len(env.hands[landlord])}张（补了底牌）")
    
    # 测试出牌
    env.start_playing()
    print(f"\n开始出牌，地主(位置{landlord})先出")
    
    # 地主出333
    success, msg, done, _ = env.step(landlord, "333")
    print(f"  地主出333: {success}, {msg}")
    
    # 获取合法动作
    next_player = (landlord + 1) % 3
    legal = env.get_legal_actions(next_player)
    print(f"  玩家{next_player}合法动作数: {len(legal)}")
    print(f"  包含炸弹: {any(CardPattern(a).is_bomb for a in legal if a != 'PASS')}")
