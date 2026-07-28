"""斗地主牌型工具模块.

提供牌型识别、合法性检查、大小比较等核心功能.
"""

from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from enum import IntEnum


class CardType(IntEnum):
    """牌型枚举."""
    PASS = 0           # 过牌
    SINGLE = 1         # 单张
    PAIR = 2           # 对子
    TRIPLE = 3         # 三张
    TRIPLE_SINGLE = 4  # 三带一
    TRIPLE_PAIR = 5    # 三带二
    STRAIGHT = 6       # 顺子
    STRAIGHT_PAIR = 7  # 连对
    PLANE = 8          # 飞机（可带翅膀）
    BOMB = 9           # 炸弹
    ROCKET = 10        # 火箭（王炸）
    FOUR_TWO = 11      # 四带二


@dataclass
class CardPattern:
    """牌型数据结构."""
    card_type: CardType
    main_point: int      # 主点数（用于比较大小）
    sub_point: int       # 副点数（用于某些牌型）
    length: int          # 牌的数量
    cards: List[int]     # 具体的牌
    
    def to_encoding(self) -> List[int]:
        """转换为状态编码格式 [牌型, 主点数, 副点数, 长度]."""
        return [int(self.card_type), self.main_point, self.sub_point, self.length]
    
    def __hash__(self):
        return hash((self.card_type, self.main_point, self.sub_point, self.length, tuple(sorted(self.cards))))
    
    def __eq__(self, other):
        if not isinstance(other, CardPattern):
            return False
        return (self.card_type == other.card_type and 
                self.main_point == other.main_point and
                self.sub_point == other.sub_point and
                self.length == other.length and
                sorted(self.cards) == sorted(other.cards))


# 点数映射
POINT_NAMES = {
    3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'O',
    11: 'J', 12: 'Q', 13: 'K', 14: 'A', 15: '2', 16: 'X', 17: 'D'
}

REVERSE_POINT_MAP = {
    '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 
    'O': 10, 'o': 10, '0': 10,
    'J': 11, 'j': 11, 'Q': 12, 'q': 12, 'K': 13, 'k': 13,
    'A': 14, 'a': 14, '1': 14,
    '2': 15,
    'X': 16, 'x': 16,
    'D': 17, 'd': 17,
}


def parse_hand_string(hand_str: str) -> List[int]:
    """将手牌字符串解析为点数列表.
    
    Args:
        hand_str: 手牌字符串，如 "333456789OOKKAA2" 或 "333456789OJDK"
        
    Returns:
        点数列表，如 [3,3,3,4,5,6,7,8,9,10,10,13,13,14,14,15]
        
    Example:
        >>> parse_hand_string("333456789OOKKAA2")
        [3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 10, 13, 13, 14, 14, 15]
        >>> parse_hand_string("XJDJ")  # 小王大王
        [16, 17]
    """
    result = []
    i = 0
    hand_str_upper = hand_str.upper()
    
    while i < len(hand_str_upper):
        # 尝试匹配两位字符（DJ/XJ）
        if i + 1 < len(hand_str_upper):
            two_char = hand_str_upper[i:i+2]
            if two_char == 'XJ':
                result.append(16)  # 小王
                i += 2
                continue
            elif two_char == 'DJ':
                result.append(17)  # 大王
                i += 2
                continue
            elif two_char == 'XG':  # 小怪的另一种表示
                result.append(16)
                i += 2
                continue
            elif two_char == 'DG':  # 大怪的另一种表示
                result.append(17)
                i += 2
                continue
        
        # 单字符匹配
        char = hand_str_upper[i]
        if char in REVERSE_POINT_MAP:
            result.append(REVERSE_POINT_MAP[char])
            i += 1
        else:
            # 跳过未知字符
            i += 1
    
    return sorted(result)


def hand_to_string(cards: List[int]) -> str:
    """将点数列表转换为手牌字符串.
    
    Args:
        cards: 点数列表
        
    Returns:
        手牌字符串
        
    Example:
        >>> hand_to_string([3, 3, 3, 10, 10, 16, 17])
        '333OOXD'
    """
    return ''.join(POINT_NAMES.get(c, str(c)) for c in sorted(cards))


def count_cards(cards: List[int]) -> Dict[int, int]:
    """统计各点数的数量.
    
    Args:
        cards: 牌列表
        
    Returns:
        点数到数量的映射字典
    """
    count = {}
    for card in cards:
        count[card] = count.get(card, 0) + 1
    return count


def has_cards(hand: List[int], cards_to_play: List[int]) -> bool:
    """检查手牌是否包含要出的所有牌（考虑重复）.
    
    Args:
        hand: 手牌
        cards_to_play: 要出的牌
        
    Returns:
        是否足够
    """
    hand_count = count_cards(hand)
    play_count = count_cards(cards_to_play)
    
    for point, needed in play_count.items():
        if hand_count.get(point, 0) < needed:
            return False
    return True


def remove_cards(hand: List[int], cards_to_remove: List[int]) -> List[int]:
    """从手牌中移除指定的牌.
    
    Args:
        hand: 原始手牌
        cards_to_remove: 要移除的牌
        
    Returns:
        移除后的手牌
        
    Raises:
        ValueError: 手牌不足以移除时
    """
    if not has_cards(hand, cards_to_remove):
        raise ValueError(f"手牌不足以移除 {cards_to_remove}，当前手牌: {hand}")
    
    hand_copy = hand.copy()
    for card in cards_to_remove:
        hand_copy.remove(card)
    
    return hand_copy


def identify_pattern(cards: List[int]) -> Optional[CardPattern]:
    """识别牌型.
    
    Args:
        cards: 出的牌列表
        
    Returns:
        牌型对象，若非法则返回None
    """
    if not cards:
        return None
    
    n = len(cards)
    count = count_cards(cards)
    unique_points = sorted(count.keys())
    
    # 1. 火箭（王炸）
    if n == 2 and sorted(cards) == [16, 17]:
        return CardPattern(CardType.ROCKET, 17, 0, 2, sorted(cards))
    
    # 2. 单张
    if n == 1:
        return CardPattern(CardType.SINGLE, cards[0], 0, 1, sorted(cards))
    
    # 3. 对子
    if n == 2 and len(unique_points) == 1 and count[unique_points[0]] == 2:
        return CardPattern(CardType.PAIR, unique_points[0], 0, 2, sorted(cards))
    
    # 4. 三张
    if n == 3 and len(unique_points) == 1 and count[unique_points[0]] == 3:
        return CardPattern(CardType.TRIPLE, unique_points[0], 0, 3, sorted(cards))
    
    # 5. 三带一
    if n == 4:
        triple_point = None
        single_point = None
        for p, c in count.items():
            if c == 3:
                triple_point = p
            elif c == 1:
                single_point = p
        if triple_point is not None and single_point is not None:
            return CardPattern(CardType.TRIPLE_SINGLE, triple_point, single_point, 4, sorted(cards))
    
    # 6. 三带二
    if n == 5:
        triple_point = None
        pair_point = None
        for p, c in count.items():
            if c == 3:
                triple_point = p
            elif c == 2:
                pair_point = p
        if triple_point is not None and pair_point is not None:
            return CardPattern(CardType.TRIPLE_PAIR, triple_point, pair_point, 5, sorted(cards))
    
    # 7. 炸弹
    if n == 4 and len(unique_points) == 1 and count[unique_points[0]] == 4:
        return CardPattern(CardType.BOMB, unique_points[0], 0, 4, sorted(cards))
    
    # 8. 顺子（至少5张，不含2和王，A不能作最小）
    if n >= 5:
        # 检查是否不含2和王
        if all(3 <= p <= 14 for p in unique_points):  # 3到A
            # 检查是否连续
            if len(unique_points) == n and max(unique_points) - min(unique_points) == n - 1:
                # 检查A是否作为最小（非法）
                if min(unique_points) >= 3:  # A=14，不能从A开始
                    return CardPattern(CardType.STRAIGHT, max(unique_points), 0, n, sorted(cards))
    
    # 9. 连对（至少3对，不含2和王）
    if n >= 6 and n % 2 == 0:
        n_pairs = n // 2
        if all(3 <= p <= 14 for p in unique_points):
            if len(unique_points) == n_pairs and all(count[p] == 2 for p in unique_points):
                if max(unique_points) - min(unique_points) == n_pairs - 1:
                    return CardPattern(CardType.STRAIGHT_PAIR, max(unique_points), 0, n, sorted(cards))
    
    # 10. 飞机（至少2组连续三张，不含2和王）
    if n >= 6:
        triple_points = [p for p, c in count.items() if c >= 3]
        # 检查三张是否连续
        if len(triple_points) >= 2 and all(3 <= p <= 14 for p in triple_points):
            triple_points_sorted = sorted(triple_points)
            # 找最长连续序列
            max_seq = []
            current_seq = [triple_points_sorted[0]]
            for i in range(1, len(triple_points_sorted)):
                if triple_points_sorted[i] == triple_points_sorted[i-1] + 1:
                    current_seq.append(triple_points_sorted[i])
                else:
                    if len(current_seq) > len(max_seq):
                        max_seq = current_seq
                    current_seq = [triple_points_sorted[i]]
            if len(current_seq) > len(max_seq):
                max_seq = current_seq
            
            if len(max_seq) >= 2:
                n_triples = len(max_seq)
                expected_cards = n_triples * 3
                remaining = n - expected_cards
                
                # 纯飞机或带翅膀
                if remaining == 0:
                    # 纯飞机
                    cards_used = []
                    for p in max_seq:
                        cards_used.extend([p, p, p])
                    return CardPattern(CardType.PLANE, max(max_seq), 0, n, sorted(cards_used))
                elif remaining == n_triples:
                    # 带单张 - 检查带牌是否合法
                    wing_cards = []
                    for p in cards:
                        if p not in max_seq or cards.count(p) > 3:
                            wing_cards.append(p)
                    # 验证带牌数量
                    if len(wing_cards) == n_triples:
                        return CardPattern(CardType.PLANE, max(max_seq), 0, n, sorted(cards))
                elif remaining == n_triples * 2:
                    # 检查是否都是对子
                    other_cards = [p for p in cards if p not in max_seq]
                    other_count = count_cards(other_cards)
                    if all(c == 2 for c in other_count.values()) and len(other_count) == n_triples:
                        return CardPattern(CardType.PLANE, max(max_seq), 0, n, sorted(cards))
    
    # 11. 四带二
    if n == 6 or n == 8:
        four_point = None
        for p, c in count.items():
            if c == 4:
                four_point = p
                break
        if four_point is not None:
            remaining = [p for p in cards if p != four_point]
            if n == 6 and len(remaining) == 2:
                # 四带两张单
                return CardPattern(CardType.FOUR_TWO, four_point, 0, 6, sorted(cards))
            elif n == 8 and len(remaining) == 4:
                # 检查是否两对
                remaining_count = count_cards(remaining)
                if all(c == 2 for c in remaining_count.values()) and len(remaining_count) == 2:
                    return CardPattern(CardType.FOUR_TWO, four_point, 0, 8, sorted(cards))
    
    return None


def can_beat(pattern1: CardPattern, pattern2: CardPattern) -> bool:
    """判断pattern1能否压制pattern2.
    
    Args:
        pattern1: 要出的牌型
        pattern2: 上家的牌型
        
    Returns:
        能否压制
    """
    # 火箭最大
    if pattern1.card_type == CardType.ROCKET:
        return True
    
    # 炸弹可压非炸弹
    if pattern1.card_type == CardType.BOMB:
        if pattern2.card_type == CardType.ROCKET:
            return False
        if pattern2.card_type == CardType.BOMB:
            return pattern1.main_point > pattern2.main_point
        return True
    
    # 同牌型比较
    if pattern1.card_type != pattern2.card_type:
        return False
    
    # 长度必须相同（顺子、连对、飞机）
    if pattern1.length != pattern2.length:
        return False
    
    # 比较主点数
    return pattern1.main_point > pattern2.main_point


def generate_all_patterns(hand: List[int]) -> List[CardPattern]:
    """从手牌生成所有可能的合法牌型.
    
    Args:
        hand: 手牌列表
        
    Returns:
        所有可能的牌型列表（包含Pass）
    """
    patterns = []
    count = count_cards(hand)
    
    # 1. Pass
    patterns.append(CardPattern(CardType.PASS, 0, 0, 0, []))
    
    # 2. 单张
    for point in count.keys():
        patterns.append(CardPattern(CardType.SINGLE, point, 0, 1, [point]))
    
    # 3. 对子
    for point, cnt in count.items():
        if cnt >= 2:
            patterns.append(CardPattern(CardType.PAIR, point, 0, 2, [point, point]))
    
    # 4. 三张
    for point, cnt in count.items():
        if cnt >= 3:
            patterns.append(CardPattern(CardType.TRIPLE, point, 0, 3, [point]*3))
    
    # 5. 三带一
    for triple_point, triple_cnt in count.items():
        if triple_cnt >= 3:
            for single_point in count.keys():
                if single_point != triple_point or triple_cnt >= 4:
                    patterns.append(CardPattern(
                        CardType.TRIPLE_SINGLE, triple_point, single_point, 4,
                        sorted([triple_point]*3 + [single_point])
                    ))
    
    # 6. 三带二
    for triple_point, triple_cnt in count.items():
        if triple_cnt >= 3:
            for pair_point, pair_cnt in count.items():
                if pair_point != triple_point and pair_cnt >= 2:
                    patterns.append(CardPattern(
                        CardType.TRIPLE_PAIR, triple_point, pair_point, 5,
                        sorted([triple_point]*3 + [pair_point]*2)
                    ))
    
    # 7. 炸弹
    for point, cnt in count.items():
        if cnt >= 4:
            patterns.append(CardPattern(CardType.BOMB, point, 0, 4, [point]*4))
    
    # 8. 火箭
    if 16 in count and 17 in count:
        patterns.append(CardPattern(CardType.ROCKET, 17, 0, 2, [16, 17]))
    
    # 9. 顺子（5-12张，不含2和王）
    valid_straight_points = [p for p in count.keys() if 3 <= p <= 14]
    valid_straight_points.sort()
    
    for start_idx in range(len(valid_straight_points)):
        for length in range(5, 13):  # 5到12张
            end_idx = start_idx + length
            if end_idx > len(valid_straight_points):
                break
            segment = valid_straight_points[start_idx:end_idx]
            if max(segment) - min(segment) == length - 1:
                cards = list(segment)
                patterns.append(CardPattern(CardType.STRAIGHT, max(segment), 0, length, cards))
    
    # 10. 连对（3-10对，不含2和王）
    pairs = [p for p, cnt in count.items() if cnt >= 2 and 3 <= p <= 14]
    pairs.sort()
    
    for start_idx in range(len(pairs)):
        for n_pairs in range(3, 11):  # 3到10对
            end_idx = start_idx + n_pairs
            if end_idx > len(pairs):
                break
            segment = pairs[start_idx:end_idx]
            if max(segment) - min(segment) == n_pairs - 1:
                cards = []
                for p in segment:
                    cards.extend([p, p])
                patterns.append(CardPattern(CardType.STRAIGHT_PAIR, max(segment), 0, n_pairs*2, cards))
    
    # 11. 飞机（2-6组，不含2和王）- 简化版，只生成纯飞机
    triples = [p for p, cnt in count.items() if cnt >= 3 and 3 <= p <= 14]
    triples.sort()
    
    for start_idx in range(len(triples)):
        for n_triples in range(2, 7):  # 2到6组
            end_idx = start_idx + n_triples
            if end_idx > len(triples):
                break
            segment = triples[start_idx:end_idx]
            if max(segment) - min(segment) == n_triples - 1:
                cards = []
                for p in segment:
                    cards.extend([p, p, p])
                patterns.append(CardPattern(CardType.PLANE, max(segment), 0, len(cards), cards))
    
    # 12. 四带二
    fours = [p for p, cnt in count.items() if cnt >= 4]
    for four_point in fours:
        base = [four_point] * 4
        remaining = [p for p in hand if p != four_point]
        # 四带两张单
        for i in range(len(remaining)):
            for j in range(i+1, len(remaining)):
                cards = base + [remaining[i], remaining[j]]
                patterns.append(CardPattern(CardType.FOUR_TWO, four_point, 0, 6, sorted(cards)))
        # 四带两对
        pair_points = [p for p, cnt in count.items() if cnt >= 2 and p != four_point]
        for i in range(len(pair_points)):
            for j in range(i+1, len(pair_points)):
                cards = base + [pair_points[i]]*2 + [pair_points[j]]*2
                patterns.append(CardPattern(CardType.FOUR_TWO, four_point, 0, 8, sorted(cards)))
    
    # 去重
    unique_patterns = []
    seen = set()
    for p in patterns:
        key = (p.card_type, p.main_point, p.sub_point, p.length, tuple(sorted(p.cards)))
        if key not in seen:
            seen.add(key)
            unique_patterns.append(p)
    
    return unique_patterns


def filter_legal_patterns(hand: List[int], last_pattern: Optional[CardPattern]) -> List[CardPattern]:
    """根据上家出牌，过滤出当前手牌的合法动作.
    
    Args:
        hand: 当前手牌
        last_pattern: 上家出的牌型，None表示自由出牌
        
    Returns:
        合法牌型列表
    """
    all_patterns = generate_all_patterns(hand)
    
    # 自由出牌：可以出任何牌型（除了Pass）
    if last_pattern is None:
        return [p for p in all_patterns if p.card_type != CardType.PASS]
    
    # 上家出牌，必须压制或Pass
    legal = [p for p in all_patterns if p.card_type == CardType.PASS or can_beat(p, last_pattern)]
    
    return legal


def verify_play(hand: List[int], played_cards: List[int], last_pattern: Optional[CardPattern]) -> Tuple[bool, str]:
    """验证出牌是否合法.
    
    Args:
        hand: 当前手牌
        played_cards: 要出的牌
        last_pattern: 上家出的牌型
        
    Returns:
        (是否合法, 错误信息)
    """
    # 1. 检查手牌是否足够
    if not has_cards(hand, played_cards):
        return False, f"手牌不足以出 {hand_to_string(played_cards)}，当前手牌: {hand_to_string(hand)}"
    
    # 2. 识别牌型
    pattern = identify_pattern(played_cards)
    if pattern is None:
        return False, f"非法牌型: {hand_to_string(played_cards)}"
    
    # 3. 检查是否能压过上家
    if last_pattern is not None and not can_beat(pattern, last_pattern):
        return False, f"{pattern.card_type.name} 无法压制上家的 {last_pattern.card_type.name}"
    
    return True, "OK"


if __name__ == "__main__":
    # 测试
    test_hand = [3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    print("手牌:", hand_to_string(test_hand))
    
    # 测试牌型识别
    patterns = [
        [3],  # 单张
        [3, 3],  # 对子
        [3, 3, 3],  # 三张
        [3, 3, 3, 4],  # 三带一
        [3, 3, 3, 4, 4],  # 三带二
        [3, 4, 5, 6, 7],  # 顺子
        [3, 3, 4, 4, 5, 5],  # 连对
        [3, 3, 3, 4, 4, 4],  # 飞机
        [3, 3, 3, 3],  # 炸弹
        [16, 17],  # 火箭
    ]
    
    for p in patterns:
        pattern = identify_pattern(p)
        if pattern:
            print(f"{p} -> {pattern.card_type.name}, 主点数:{pattern.main_point}")
        else:
            print(f"{p} -> 非法牌型")
