"""
斗地主规则引擎

作用：
1. 定义所有牌型及其大小关系
2. 根据当前局面，生成合法出牌列表
3. 判断牌型是否压制上家
4. 提供炸弹、王炸等特殊规则支持

使用方式：
    from rules import CardPattern, get_legal_plays
    
    # 获取合法出牌
    legal_plays = get_legal_plays(my_hand, last_play)
    
    # 判断牌型
    pattern = CardPattern("3333")
    print(pattern.is_bomb)  # True
    print(pattern.can_beat(other_pattern))  # 判断是否可压制
"""

from typing import List, Optional, Tuple, Dict
from collections import Counter


# 卡牌点数映射（用于比较大小）
CARD_RANK = {
    '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
    'O': 8,   # 10
    'J': 9, 'Q': 10, 'K': 11, 'A': 12, '2': 13,
    'X': 14,  # 小王
    'D': 15   # 大王
}

# 牌型定义
class CardPattern:
    """牌型类，解析和表示一种出牌"""
    
    # 牌型常量
    TYPE_INVALID = -1      # 非法
    TYPE_PASS = 0          # 过牌
    TYPE_SINGLE = 1        # 单牌
    TYPE_PAIR = 2          # 对子
    TYPE_TRIPLE = 3        # 三张
    TYPE_TRIPLE_SINGLE = 4 # 三带一
    TYPE_TRIPLE_PAIR = 5   # 三带二
    TYPE_STRAIGHT = 6      # 顺子（5张以上）
    TYPE_DOUBLE_STRAIGHT = 7  # 连对（3对以上）
    TYPE_PLANE = 8         # 飞机（2组以上连续三张）
    TYPE_PLANE_SINGLE = 9  # 飞机带翅膀（单牌）
    TYPE_PLANE_PAIR = 10   # 飞机带翅膀（对子）
    TYPE_BOMB = 11         # 炸弹（4张）
    TYPE_ROCKET = 12       # 王炸（XD）
    TYPE_FOUR_SINGLE = 13  # 四带二（单牌）
    TYPE_FOUR_PAIR = 14    # 四带二（对子）
    
    def __init__(self, cards_str: str):
        """
        解析牌型字符串
        
        Args:
            cards_str: 牌型字符串，如 "3333", "34567", "XD" 等
                      "PASS" 表示过牌
        """
        self.cards_str = cards_str.upper()
        self.cards = list(self.cards_str)
        self.type = self.TYPE_INVALID
        self.rank = 0  # 牌型大小（用于比较）
        self.is_bomb = False
        self.is_rocket = False
        self.length = len(self.cards)
        
        self._parse()
    
    def _parse(self):
        """解析牌型"""
        if self.cards_str == "PASS":
            self.type = self.TYPE_PASS
            return
        
        if not self.cards:
            return
        
        # 统计每张牌的数量
        card_counts = Counter(self.cards)
        counts = sorted(card_counts.values(), reverse=True)
        
        # 王炸
        if 'X' in card_counts and 'D' in card_counts and len(self.cards) == 2:
            self.type = self.TYPE_ROCKET
            self.is_rocket = True
            self.rank = 100  # 最大
            return
        
        # 单牌
        if len(self.cards) == 1:
            self.type = self.TYPE_SINGLE
            self.rank = CARD_RANK.get(self.cards[0], 0)
            return
        
        # 对子
        if len(self.cards) == 2 and counts == [2]:
            self.type = self.TYPE_PAIR
            card = [c for c, n in card_counts.items() if n == 2][0]
            self.rank = CARD_RANK.get(card, 0)
            return
        
        # 三张
        if len(self.cards) == 3 and counts == [3]:
            self.type = self.TYPE_TRIPLE
            card = [c for c, n in card_counts.items() if n == 3][0]
            self.rank = CARD_RANK.get(card, 0)
            return
        
        # 三带一
        if len(self.cards) == 4 and counts == [3, 1]:
            self.type = self.TYPE_TRIPLE_SINGLE
            card = [c for c, n in card_counts.items() if n == 3][0]
            self.rank = CARD_RANK.get(card, 0)
            return
        
        # 三带二
        if len(self.cards) == 5 and counts == [3, 2]:
            self.type = self.TYPE_TRIPLE_PAIR
            card = [c for c, n in card_counts.items() if n == 3][0]
            self.rank = CARD_RANK.get(card, 0)
            return
        
        # 炸弹（4张）
        if len(self.cards) == 4 and counts == [4]:
            self.type = self.TYPE_BOMB
            self.is_bomb = True
            card = [c for c, n in card_counts.items() if n == 4][0]
            self.rank = CARD_RANK.get(card, 0) + 50  # 炸弹比普通牌大
            return
        
        # 四带二（单牌）
        if len(self.cards) == 6 and counts == [4, 1, 1]:
            self.type = self.TYPE_FOUR_SINGLE
            card = [c for c, n in card_counts.items() if n == 4][0]
            self.rank = CARD_RANK.get(card, 0)
            return
        
        # 四带二（对子）
        if len(self.cards) == 8 and counts == [4, 2, 2]:
            self.type = self.TYPE_FOUR_PAIR
            card = [c for c, n in card_counts.items() if n == 4][0]
            self.rank = CARD_RANK.get(card, 0)
            return
        
        # 检查顺子、连对、飞机
        self._check_straights(card_counts)
    
    def _check_straights(self, card_counts: Counter):
        """检查顺子、连对、飞机等连续牌型"""
        # 获取主牌（非2、X、D）
        main_cards = [c for c in card_counts.keys() if c not in ['2', 'X', 'D']]
        
        if not main_cards:
            return
        
        # 获取点数排序
        ranks = sorted([CARD_RANK[c] for c in main_cards])
        
        # 检查是否连续
        is_continuous = all(ranks[i+1] - ranks[i] == 1 for i in range(len(ranks)-1))
        
        if not is_continuous:
            return
        
        # 顺子（5张以上连续单牌）
        if len(self.cards) >= 5 and len(card_counts) == len(self.cards) and len(ranks) >= 5:
            self.type = self.TYPE_STRAIGHT
            self.rank = ranks[-1]  # 最大点数
            return
        
        # 连对（3对以上连续对子）
        if all(n == 2 for n in card_counts.values()) and len(card_counts) >= 3:
            self.type = self.TYPE_DOUBLE_STRAIGHT
            self.rank = ranks[-1]
            return
        
        # 找出所有连续的三张（飞机主体）
        triples = [(c, CARD_RANK[c]) for c, n in card_counts.items() if n == 3 and c not in ['2', 'X', 'D']]
        if len(triples) >= 2:
            # 按点数排序
            triples.sort(key=lambda x: x[1])
            triple_cards = [t[0] for t in triples]
            triple_ranks = [t[1] for t in triples]
            
            # 找出最长的连续三张序列
            max_len = 1
            current_len = 1
            end_idx = 0
            
            for i in range(1, len(triple_ranks)):
                if triple_ranks[i] - triple_ranks[i-1] == 1:
                    current_len += 1
                    if current_len > max_len:
                        max_len = current_len
                        end_idx = i
                else:
                    current_len = 1
            
            # 使用最长的连续序列
            start_idx = end_idx - max_len + 1
            plane_cards = triple_cards[start_idx:end_idx+1]
            num_planes = len(plane_cards)
            
            # 飞机主体（不带翅膀）
            if num_planes >= 2 and len(self.cards) == num_planes * 3:
                # 检查是否只有飞机主体，没有其他牌
                if all(card_counts[c] == 3 for c in plane_cards) and len(card_counts) == num_planes:
                    self.type = self.TYPE_PLANE
                    self.rank = triple_ranks[end_idx]
                    return
            
            # 飞机带翅膀
            # 翅膀可以是：num_planes张单牌，或num_planes个对子
            others = [(c, n) for c, n in card_counts.items() if c not in plane_cards]
            
            # 检查翅膀数量是否正确
            num_others = sum(n for _, n in others)
            
            # 带单牌：需要 num_planes 张单牌
            if num_others == num_planes and all(n == 1 for _, n in others):
                self.type = self.TYPE_PLANE_SINGLE
                self.rank = triple_ranks[end_idx]
                return
            
            # 带对子：需要 num_planes 个对子
            if num_others == num_planes * 2 and all(n == 2 for _, n in others):
                self.type = self.TYPE_PLANE_PAIR
                self.rank = triple_ranks[end_idx]
                return
    
    def is_valid(self) -> bool:
        """判断牌型是否合法"""
        return self.type != self.TYPE_INVALID
    
    def can_beat(self, other: 'CardPattern') -> bool:
        """
        判断是否能压制上家
        
        Args:
            other: 上家的牌型
            
        Returns:
            True 如果可以压制，False 否则
        """
        if other.type == self.TYPE_PASS:
            # 上家PASS，可以自由出（只要合法）
            return self.is_valid() and self.type != self.TYPE_PASS
        
        if self.type == self.TYPE_ROCKET:
            # 王炸可以压一切
            return True
        
        if self.type == self.TYPE_BOMB:
            # 炸弹可以压非炸弹
            if not other.is_bomb and not other.is_rocket:
                return True
            # 炸弹之间比较点数
            if other.is_bomb:
                return self.rank > other.rank
            return False
        
        # 非炸弹必须同类型比较
        if self.type != other.type:
            return False
        
        # 同类型比较点数
        if self.length != other.length:
            return False
        
        return self.rank > other.rank
    
    def __repr__(self):
        type_names = {
            self.TYPE_INVALID: "非法",
            self.TYPE_PASS: "PASS",
            self.TYPE_SINGLE: "单牌",
            self.TYPE_PAIR: "对子",
            self.TYPE_TRIPLE: "三张",
            self.TYPE_TRIPLE_SINGLE: "三带一",
            self.TYPE_TRIPLE_PAIR: "三带二",
            self.TYPE_STRAIGHT: "顺子",
            self.TYPE_DOUBLE_STRAIGHT: "连对",
            self.TYPE_PLANE: "飞机",
            self.TYPE_PLANE_SINGLE: "飞机带单",
            self.TYPE_PLANE_PAIR: "飞机带对",
            self.TYPE_BOMB: "炸弹",
            self.TYPE_ROCKET: "王炸",
            self.TYPE_FOUR_SINGLE: "四带二",
            self.TYPE_FOUR_PAIR: "四带二对",
        }
        return f"CardPattern({self.cards_str}, type={type_names.get(self.type, '未知')}, rank={self.rank})"


def has_cards(hand: List[str], cards: List[str]) -> bool:
    """
    检查手牌是否包含指定牌
    
    Args:
        hand: 手牌列表
        cards: 需要出的牌
        
    Returns:
        True 如果手牌足够，False 否则
    """
    hand_counter = Counter(hand)
    cards_counter = Counter(cards)
    
    for card, count in cards_counter.items():
        if hand_counter[card] < count:
            return False
    return True


def remove_cards(hand: List[str], cards: List[str]) -> List[str]:
    """
    从手牌中移除指定牌
    
    Args:
        hand: 手牌列表
        cards: 要移除的牌
        
    Returns:
        移除后的手牌
    """
    hand_counter = Counter(hand)
    cards_counter = Counter(cards)
    
    for card, count in cards_counter.items():
        hand_counter[card] -= count
        if hand_counter[card] < 0:
            raise ValueError(f"手牌中{card}数量不足")
    
    result = []
    for card, count in hand_counter.items():
        result.extend([card] * count)
    
    return result


def generate_all_patterns(hand: List[str]) -> List[str]:
    """
    从手牌生成所有可能的牌型组合
    
    Args:
        hand: 手牌列表
        
    Returns:
        所有可能的出牌字符串列表
    """
    patterns = ["PASS"]  # 总是可以PASS
    
    hand_counter = Counter(hand)
    cards = list(hand_counter.keys())
    
    # 单牌
    for card in cards:
        patterns.append(card)
    
    # 对子
    for card, count in hand_counter.items():
        if count >= 2:
            patterns.append(card * 2)
    
    # 三张
    for card, count in hand_counter.items():
        if count >= 3:
            patterns.append(card * 3)
    
    # 三带一
    for card, count in hand_counter.items():
        if count >= 3:
            for other in cards:
                if other != card:
                    patterns.append(card * 3 + other)
    
    # 三带二
    for card, count in hand_counter.items():
        if count >= 3:
            for other, other_count in hand_counter.items():
                if other != card and other_count >= 2:
                    patterns.append(card * 3 + other * 2)
    
    # 炸弹
    for card, count in hand_counter.items():
        if count >= 4:
            patterns.append(card * 4)
    
    # 王炸
    if 'X' in hand_counter and 'D' in hand_counter:
        patterns.append("XD")
        patterns.append("DX")
    
    # 顺子（简化：只生成5张顺子）
    straight_cards = [c for c in cards if c not in ['2', 'X', 'D']]
    straight_cards.sort(key=lambda x: CARD_RANK[x])
    
    for i in range(len(straight_cards) - 4):
        # 检查5张连续
        if all(CARD_RANK[straight_cards[j+1]] - CARD_RANK[straight_cards[j]] == 1 
               for j in range(i, i+4)):
            pattern = ''.join(straight_cards[i:i+5])
            patterns.append(pattern)
    
    # 连对（简化：只生成3连对）
    pairs = [c for c, n in hand_counter.items() if n >= 2 and c not in ['2', 'X', 'D']]
    pairs.sort(key=lambda x: CARD_RANK[x])
    
    for i in range(len(pairs) - 2):
        if all(CARD_RANK[pairs[j+1]] - CARD_RANK[pairs[j]] == 1 
               for j in range(i, i+2)):
            pattern = pairs[i]*2 + pairs[i+1]*2 + pairs[i+2]*2
            patterns.append(pattern)
    
    # 飞机（2组以上连续三张）
    triples = [(c, CARD_RANK[c]) for c, n in hand_counter.items() if n >= 3 and c not in ['2', 'X', 'D']]
    triples.sort(key=lambda x: x[1])
    
    for i in range(len(triples) - 1):
        # 找连续的三张
        plane_cards = [triples[i][0]]
        for j in range(i + 1, len(triples)):
            if triples[j][1] - triples[j-1][1] == 1:
                plane_cards.append(triples[j][0])
            else:
                break
        
        if len(plane_cards) >= 2:
            # 飞机主体
            plane_body = ''.join(c * 3 for c in plane_cards)
            patterns.append(plane_body)
            
            # 飞机带单牌（需要len(plane_cards)张单牌）
            num_planes = len(plane_cards)
            single_candidates = [c for c, n in hand_counter.items() if c not in plane_cards]
            
            # 简化：只生成部分组合
            if len(single_candidates) >= num_planes:
                import itertools
                for wings in itertools.combinations(single_candidates, num_planes):
                    pattern = plane_body + ''.join(wings)
                    patterns.append(pattern)
            
            # 飞机带对子（需要len(plane_cards)个对子）
            pair_candidates = [c for c, n in hand_counter.items() 
                             if c not in plane_cards and n >= 2]
            
            if len(pair_candidates) >= num_planes:
                import itertools
                for wings in itertools.combinations(pair_candidates, num_planes):
                    pattern = plane_body + ''.join(c * 2 for c in wings)
                    patterns.append(pattern)
    
    return patterns


def get_legal_plays(hand: List[str], last_play: Optional[CardPattern] = None) -> List[str]:
    """
    获取所有合法出牌
    
    Args:
        hand: 当前手牌
        last_play: 上家出的牌（None表示新一轮，可以自由出）
        
    Returns:
        合法出牌字符串列表
    """
    # 生成所有可能的牌型
    all_patterns = generate_all_patterns(hand)
    
    # 检查手牌是否足够
    valid_patterns = []
    for pattern_str in all_patterns:
        if pattern_str == "PASS":
            valid_patterns.append(pattern_str)
            continue
        
        try:
            if has_cards(hand, list(pattern_str)):
                valid_patterns.append(pattern_str)
        except:
            pass
    
    # 如果没有上家出牌，返回所有合法牌型
    if last_play is None or last_play.type == CardPattern.TYPE_PASS:
        return valid_patterns
    
    # 过滤能压制上家的牌型
    legal_plays = ["PASS"]  # 总是可以PASS
    
    for pattern_str in valid_patterns:
        if pattern_str == "PASS":
            continue
        
        pattern = CardPattern(pattern_str)
        if pattern.is_valid() and pattern.can_beat(last_play):
            legal_plays.append(pattern_str)
    
    return legal_plays


if __name__ == "__main__":
    # 测试
    print("=" * 60)
    print("规则引擎测试")
    print("=" * 60)
    
    # 测试牌型解析
    test_patterns = ["3", "33", "333", "3333", "3334", "33344", "34567", "334455", "XD", "PASS"]
    
    print("\n牌型解析测试:")
    for p in test_patterns:
        pattern = CardPattern(p)
        print(f"  {p:10} -> {pattern}")
    
    # 测试压制关系
    print("\n压制关系测试:")
    p1 = CardPattern("4444")  # 炸弹
    p2 = CardPattern("5555")  # 更大的炸弹
    p3 = CardPattern("666")   # 三张
    
    print(f"  5555 能压 4444: {p2.can_beat(p1)}")
    print(f"  4444 能压 666:  {p1.can_beat(p3)}")
    print(f"  666  能压 4444: {p3.can_beat(p1)}")
    
    # 测试合法出牌
    print("\n合法出牌测试:")
    hand = ['3', '3', '3', '3', '4', '5', '6', '7', '8', '9', 'O', 'J', 'X', 'D']
    last = CardPattern("666")
    legal = get_legal_plays(hand, last)
    print(f"  手牌: {''.join(hand)}")
    print(f"  上家: 666")
    print(f"  合法出牌数: {len(legal)}")
    print(f"  包含炸弹: {'3333' in legal}")
    print(f"  包含王炸: {'XD' in legal}")
