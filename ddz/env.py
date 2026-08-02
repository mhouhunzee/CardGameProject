"""
斗地主游戏环境
"""
import random
from config import (
    HAND_SIZE, BASE_CARD_SIZE, POSITION_FIRST, POSITION_SECOND, POSITION_THIRD,
    ALL_CARDS, CARD_RANK
)
from card_utils import CardPattern, has_cards, remove_cards, is_valid_play


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
        self.next_play = None  # 下家出的牌（记录最近一轮）
        self.next_player = None  # 下家是谁
        self.pass_count = 0  # 连续过牌次数
        
        # 游戏记录
        self.played_cards = {0: {}, 1: {}, 2: {}}  # 每个人已出的牌计数（用于计算剩余牌）
        self.bomb_count = 0  # 炸弹使用次数
        self.rocket_count = 0  # 王炸使用次数
        self.spring = False  # 是否春天
        self.anti_spring = False  # 是否反春天
        
        # 游戏结束
        self.done = False
        self.winner = None  # 获胜方（地主或农民）
        
        return self.get_state()
    
    def deal_cards(self, hand_A, hand_B, hand_C, base):
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
            'next_play': self.next_play.cards_str if self.next_play else None,
            'next_player': self.next_player,
            'pass_count': self.pass_count,
            'played_cards': {k: dict(v) for k, v in self.played_cards.items()},  # 已出牌记录
            'bomb_count': self.bomb_count,
            'rocket_count': self.rocket_count,
            'done': self.done,
            'winner': self.winner
        }
    
    def bidding_phase(self, bid_0, bid_1, bid_2):
        """
        叫分阶段
        bid_0, bid_1, bid_2: 三个玩家的叫分（0/1/2/3）
        返回: (是否成功, 地主位置, 成交分数)
        """
        self.bids = {0: bid_0, 1: bid_1, 2: bid_2}
        
        # 检查是否有人叫3分（直接成交）
        for pos, bid in self.bids.items():
            if bid == 3:
                self.landlord = pos
                self.final_bid = 3
                self._assign_base_cards()
                return True, self.landlord, self.final_bid
        
        # 找出最高叫分
        max_bid = max(self.bids.values())
        
        # 如果没人叫分（都选择不叫），流局
        if max_bid == 0:
            self.final_bid = 0
            return False, None, 0
        
        # 最高叫分者成为地主（如果有多个，先叫的优先）
        for pos in [0, 1, 2]:  # 按头叫、二叫、三叫顺序
            if self.bids[pos] == max_bid:
                self.landlord = pos
                self.final_bid = max_bid
                break
        
        self._assign_base_cards()
        return True, self.landlord, self.final_bid
    
    def _assign_base_cards(self):
        """将底牌分配给地主"""
        if self.landlord is not None:
            self.hands[self.landlord].extend(self.base_cards)
            self.hands[self.landlord].sort(key=lambda x: ALL_CARDS.index(x))
    
    def start_playing(self):
        """开始出牌阶段"""
        # 由地主先出牌
        self.current_player = self.landlord
        self.last_play = None
        self.last_player = None
        self.pass_count = 0
    
    def step(self, player_id, action):
        """
        执行一步动作
        player_id: 玩家位置（0/1/2）
        action: 出牌字符串（如"34567"或"PASS"）
        
        返回: (是否成功, 信息, 是否游戏结束, 奖励)
        """
        if self.done:
            return False, "游戏已结束", True, 0
        
        # 检查是否是当前玩家回合
        if player_id != self.current_player:
            return False, "不是你的回合", False, 0
        
        action = action.upper()
        
        # 检查出牌合法性
        is_valid, msg = is_valid_play(
            self.hands[player_id],
            action,
            self.last_play if self.last_player != player_id else None
        )
        
        if not is_valid:
            return False, msg, False, 0
        
        # 执行出牌
        if action == "PASS":
            # 过牌
            self.pass_count += 1
            # 过牌记录（简化）
            pass
        else:
            # 出牌
            play_cards = list(action)
            pattern = CardPattern(action)
            
            # 更新手牌
            self.hands[player_id] = remove_cards(self.hands[player_id], play_cards)
            
            # 出牌记录（简化）
            pass
            
            # 【新增】记录已出牌
            for card in play_cards:
                if card not in self.played_cards[player_id]:
                    self.played_cards[player_id][card] = 0
                self.played_cards[player_id][card] += 1
            
            # 更新炸弹计数
            if pattern.type == CardPattern.TYPE_BOMB:
                self.bomb_count += 1
            elif pattern.type == CardPattern.TYPE_ROCKET:
                self.rocket_count += 1
            
            # 更新上家出牌
            self.last_play = pattern
            self.last_player = player_id
            self.pass_count = 0
            
            # 检查是否获胜
            if len(self.hands[player_id]) == 0:
                self.done = True
                self._calculate_winner(player_id)
                reward = self._calculate_reward(player_id)
                return True, "游戏结束", True, reward
        
        # 检查是否连续两人过牌（获得自由出牌权）
        if self.pass_count >= 2:
            self.last_play = None
            self.last_player = None
            self.pass_count = 0
        
        # 切换到下一个玩家
        self.current_player = (self.current_player + 1) % 3
        
        return True, "出牌成功", False, 0
    
    def _calculate_winner(self, winner_id):
        """计算获胜方"""
        if winner_id == self.landlord:
            self.winner = 'landlord'
            # 检查是否春天（农民一张牌没出）
            # 春天判断简化：检查农民是否出过牌
            # 实际应该根据played_cards判断
            self.spring = False  # 简化处理
        else:
            self.winner = 'farmers'
            # 反春天判断简化
            self.anti_spring = False  # 简化处理
    
    def _calculate_reward(self, player_id):
        """
        计算奖励
        返回: 该玩家的奖励值
        """
        if self.final_bid == 0:
            return 0  # 流局无奖励
        
        # 基础分数
        base_score = self.final_bid
        
        # 炸弹倍数
        bomb_multiplier = 2 ** (self.bomb_count + self.rocket_count)
        
        # 春天/反春天倍数
        spring_multiplier = 2 if (self.spring or self.anti_spring) else 1
        
        # 最终分数
        final_score = base_score * bomb_multiplier * spring_multiplier
        
        # 根据玩家角色和胜负分配奖励
        is_landlord = (player_id == self.landlord)
        is_winner = (self.winner == 'landlord' and is_landlord) or \
                   (self.winner == 'farmers' and not is_landlord)
        
        if is_winner:
            return final_score
        else:
            return -final_score
    
    def get_legal_actions(self, player_id):
        """获取玩家的合法动作"""
        from card_utils import get_legal_plays
        
        last_pattern = self.last_play if self.last_player != player_id else None
        return get_legal_plays(self.hands[player_id], last_pattern)


if __name__ == "__main__":
    # 测试
    env = DouDiZhuEnv()
    
    # 发牌
    env.deal_cards(
        ['3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', 'X', 'D', '3', '4'],
        ['3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', '3', '4', '5', '6'],
        ['3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'Q', 'K', 'A', '2', '3', '4', '5', '6'],
        ['7', '8', '9']
    )
    
    # 叫分
    success, landlord, bid = env.bidding_phase(1, 2, 0)
    print(f"叫分结果: 地主={landlord}, 分数={bid}")
    
    # 开始出牌
    env.start_playing()
    
    # 模拟几轮
    for _ in range(10):
        if env.done:
            break
        player = env.current_player
        legal_actions = env.get_legal_actions(player)
        action = random.choice(legal_actions)
        success, msg, done, reward = env.step(player, action)
        print(f"玩家{player}: {action} - {msg}")
    
    if env.done:
        print(f"游戏结束，获胜方: {env.winner}")
