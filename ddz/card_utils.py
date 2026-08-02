"""
斗地主卡牌工具函数
用于解析牌型、比较大小、检查合法性等
"""
from config import ALL_CARDS, CARD_RANK


class CardPattern:
    """牌型类，用于表示和比较牌型"""
    
    # 牌型定义
    TYPE_SINGLE = "single"          # 单牌
    TYPE_PAIR = "pair"              # 对子
    TYPE_TRIPLE = "triple"          # 三张
    TYPE_TRIPLE_SINGLE = "triple_single"    # 三带一
    TYPE_TRIPLE_PAIR = "triple_pair"        # 三带二
    TYPE_STRAIGHT = "straight"      # 顺子
    TYPE_DOUBLE_STRAIGHT = "double_straight"  # 连对
    TYPE_BOMB = "bomb"              # 炸弹
    TYPE_ROCKET = "rocket"          # 王炸
    TYPE_PLAIN = "plain"            # 飞机不带
    TYPE_PLAIN_SINGLE = "plain_single"      # 飞机带单
    TYPE_PLAIN_PAIR = "plain_pair"          # 飞机带对
    TYPE_FOUR_SINGLE = "four_single"        # 四带二（单）
    TYPE_FOUR_PAIR = "four_pair"            # 四带二（对）
    TYPE_PASS = "pass"              # 过牌
    
    def __init__(self, cards_str):
        """
        解析牌型
        cards_str: 牌型字符串，如"34567"、"9999"、"PASS"等
        """
        self.cards_str = cards_str.upper()
        self.cards = list(self.cards_str) if self.cards_str != "PASS" else []
        self.type = None
        self.main_rank = 0      # 主牌点数（用于比较）
        self.length = 0         # 牌的数量
        self.is_bomb = False    # 是否是炸弹
        
        if self.cards_str == "PASS":
            self.type = self.TYPE_PASS
        else:
            self._parse_pattern()
    
    def _parse_pattern(self):
        """解析牌型"""
        if not self.cards:
            return
        
        self.length = len(self.cards)
        
        # 统计每种牌的数量
        card_counts = {}
        for card in self.cards:
            card_counts[card] = card_counts.get(card, 0) + 1
        
        # 按点数排序
        sorted_cards = sorted(self.cards, key=lambda x: CARD_RANK[x])
        unique_cards = sorted(set(self.cards), key=lambda x: CARD_RANK[x])
        
        # 检查王炸
        if self.cards_str == "XD" or self.cards_str == "DX":
            self.type = self.TYPE_ROCKET
            self.main_rank = 100  # 王炸最大
            self.is_bomb = True
            return
        
        # 检查炸弹
        if len(card_counts) == 1 and self.length == 4:
            self.type = self.TYPE_BOMB
            self.main_rank = CARD_RANK[unique_cards[0]]
            self.is_bomb = True
            return
        
        # 检查单牌
        if self.length == 1:
            self.type = self.TYPE_SINGLE
            self.main_rank = CARD_RANK[sorted_cards[0]]
            return
        
        # 检查对子
        if self.length == 2 and len(card_counts) == 1:
            self.type = self.TYPE_PAIR
            self.main_rank = CARD_RANK[unique_cards[0]]
            return
        
        # 检查三张
        if self.length == 3 and len(card_counts) == 1:
            self.type = self.TYPE_TRIPLE
            self.main_rank = CARD_RANK[unique_cards[0]]
            return
        
        # 检查三带一
        if self.length == 4 and len(card_counts) == 2:
            counts = list(card_counts.values())
            if 3 in counts and 1 in counts:
                self.type = self.TYPE_TRIPLE_SINGLE
                for card, count in card_counts.items():
                    if count == 3:
                        self.main_rank = CARD_RANK[card]
                        return
        
        # 检查三带二
        if self.length == 5 and len(card_counts) == 2:
            counts = list(card_counts.values())
            if 3 in counts and 2 in counts:
                self.type = self.TYPE_TRIPLE_PAIR
                for card, count in card_counts.items():
                    if count == 3:
                        self.main_rank = CARD_RANK[card]
                        return
        
        # 检查顺子（至少5张，连续，不含2、X、D）
        if self.length >= 5 and len(card_counts) == self.length:
            if all(card not in ['2', 'X', 'D'] for card in self.cards):
                ranks = [CARD_RANK[card] for card in sorted_cards]
                if ranks == list(range(ranks[0], ranks[0] + self.length)):
                    self.type = self.TYPE_STRAIGHT
                    self.main_rank = ranks[-1]  # 最大牌点数
                    return
        
        # 检查连对（至少3对，连续对子，不含2、X、D）
        if self.length >= 6 and self.length % 2 == 0:
            pair_count = self.length // 2
            if all(count == 2 for count in card_counts.values()):
                if all(card not in ['2', 'X', 'D'] for card in self.cards):
                    pair_starts = sorted([CARD_RANK[card] for card in unique_cards])
                    if pair_starts == list(range(pair_starts[0], pair_starts[0] + pair_count)):
                        self.type = self.TYPE_DOUBLE_STRAIGHT
                        self.main_rank = pair_starts[-1]
                        return
        
        # 检查四带二（单牌）
        if self.length == 6 and len(card_counts) == 3:
            counts = sorted(card_counts.values())
            if counts == [1, 1, 4]:
                self.type = self.TYPE_FOUR_SINGLE
                for card, count in card_counts.items():
                    if count == 4:
                        self.main_rank = CARD_RANK[card]
                        return
        
        # 检查四带二（对子）
        if self.length == 8 and len(card_counts) == 3:
            counts = sorted(card_counts.values())
            if counts == [2, 2, 4]:
                self.type = self.TYPE_FOUR_PAIR
                for card, count in card_counts.items():
                    if count == 4:
                        self.main_rank = CARD_RANK[card]
                        return
        
        # 检查飞机（简化版，仅支持连续两个三张）
        if self.length == 6 and len(card_counts) == 2:
            counts = list(card_counts.values())
            if counts == [3, 3]:
                ranks = sorted([CARD_RANK[card] for card in unique_cards])
                if ranks[1] - ranks[0] == 1 and all(r <= CARD_RANK['A'] for r in ranks):
                    self.type = self.TYPE_PLAIN
                    self.main_rank = ranks[-1]
                    return
        
        # 未知牌型
        self.type = None
    
    def can_beat(self, other):
        """
        判断能否压制另一个牌型
        other: 另一个CardPattern对象
        返回: True/False
        """
        if other.type == self.TYPE_PASS:
            # 上家过牌，任意合法牌型都可以出
            return self.type is not None
        
        if self.type == self.TYPE_ROCKET:
            # 王炸可以压一切
            return True
        
        if self.type == self.TYPE_BOMB:
            # 炸弹可以压非炸弹牌型，或更小的炸弹
            if other.is_bomb:
                return self.main_rank > other.main_rank
            return True
        
        if other.is_bomb or other.type == self.TYPE_ROCKET:
            # 非炸弹不能压炸弹/王炸
            return False
        
        # 同牌型比大小
        if self.type != other.type:
            return False
        
        if self.length != other.length:
            return False
        
        return self.main_rank > other.main_rank
    
    def is_valid(self):
        """是否是合法牌型"""
        return self.type is not None
    
    def __str__(self):
        return f"{self.cards_str}({self.type}, rank={self.main_rank})"


def has_cards(hand, play_cards):
    """
    检查手牌中是否有要出的牌
    hand: 手牌列表
    play_cards: 要出的牌列表
    返回: True/False
    """
    hand_copy = hand.copy()
    for card in play_cards:
        if card in hand_copy:
            hand_copy.remove(card)
        else:
            return False
    return True


def remove_cards(hand, play_cards):
    """
    从手牌中移除出的牌
    返回: 新手牌
    """
    hand_copy = hand.copy()
    for card in play_cards:
        if card in hand_copy:
            hand_copy.remove(card)
    return hand_copy


def is_valid_play(hand, play_str, last_pattern=None):
    """
    检查出牌是否合法
    hand: 手牌列表
    play_str: 出牌字符串（如"34567"或"PASS"）
    last_pattern: 上家出的牌型（None表示自由出牌）
    
    返回: (是否合法, 错误信息)
    """
    play_str = play_str.upper()
    
    # 过牌
    if play_str == "PASS":
        return True, "过牌"
    
    # 解析出牌
    play_cards = list(play_str)
    pattern = CardPattern(play_str)
    
    # 检查是否是合法牌型
    if not pattern.is_valid():
        return False, f"非法牌型: {play_str}"
    
    # 检查手牌中是否有这些牌
    if not has_cards(hand, play_cards):
        return False, "手牌中没有这些牌"
    
    # 自由出牌
    if last_pattern is None or last_pattern.type == CardPattern.TYPE_PASS:
        return True, "合法出牌"
    
    # 检查能否压制上家
    if not pattern.can_beat(last_pattern):
        return False, f"无法压制上家的{last_pattern.type}"
    
    return True, "合法出牌"


def get_legal_plays(hand, last_pattern=None):
    """
    获取所有合法出牌选项
    hand: 手牌列表
    last_pattern: 上家出的牌型（None表示自由出牌）
    
    返回: 合法出牌字符串列表（包含"PASS"）
    """
    legal_plays = ["PASS"]
    
    # 生成所有可能的出牌组合（简化版，仅生成常见牌型）
    from itertools import combinations
    
    # 手牌计数
    card_counts = {}
    for card in hand:
        card_counts[card] = card_counts.get(card, 0) + 1
    
    # 单牌
    for card in set(hand):
        play = card
        if is_valid_play(hand, play, last_pattern)[0]:
            legal_plays.append(play)
    
    # 对子
    for card, count in card_counts.items():
        if count >= 2:
            play = card * 2
            if is_valid_play(hand, play, last_pattern)[0]:
                legal_plays.append(play)
    
    # 三张
    for card, count in card_counts.items():
        if count >= 3:
            play = card * 3
            if is_valid_play(hand, play, last_pattern)[0]:
                legal_plays.append(play)
    
    # 炸弹
    for card, count in card_counts.items():
        if count == 4:
            play = card * 4
            if is_valid_play(hand, play, last_pattern)[0]:
                legal_plays.append(play)
    
    # 王炸
    if 'X' in hand and 'D' in hand:
        play = "XD"
        if is_valid_play(hand, play, last_pattern)[0]:
            legal_plays.append(play)
    
    return list(set(legal_plays))


if __name__ == "__main__":
    # 测试
    test_cases = [
        ("3", "single"),
        ("44", "pair"),
        ("555", "triple"),
        ("5553", "triple_single"),
        ("55533", "triple_pair"),
        ("34567", "straight"),
        ("334455", "double_straight"),
        ("9999", "bomb"),
        ("XD", "rocket"),
        ("999955", "four_single"),
        ("999955KK", "four_pair"),
    ]
    
    for cards, expected in test_cases:
        pattern = CardPattern(cards)
        print(f"{cards}: {pattern.type} {'✓' if pattern.type == expected else '✗'}")
