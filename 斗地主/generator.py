"""
斗地主牌组生成器

作用：
1. 生成大量随机但合理的初始牌组（发牌方案）
2. 每个牌组包含：头叫手牌(17张) + 二叫手牌(17张) + 三叫手牌(17张) + 底牌(3张)
3. 确保牌组合法（共54张牌，不重复）
4. 保存到文件供训练使用

使用方式：
    from generator import generate_paiku, save_paiku
    
    # 生成10000个牌组
    paiku_list = generate_paiku(10000)
    
    # 保存到文件
    save_paiku(paiku_list, 'paiku.txt')
"""

import random
from typing import List, Tuple
from config import NUM_PAIKU

# 一副牌的定义（不区分花色）
# 3-10, J, Q, K, A, 2, 小王(X), 大王(D)
ALL_CARDS = (
    ['3', '4', '5', '6', '7', '8', '9', 'O'] * 4 +  # O代表10
    ['J', 'Q', 'K', 'A', '2'] * 4 +
    ['X', 'D']  # 小王、大王各1张
)

# 卡牌点数映射（用于比较大小）
CARD_RANK = {
    '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7,
    'O': 8,  # 10
    'J': 9, 'Q': 10, 'K': 11, 'A': 12, '2': 13,
    'X': 14,  # 小王
    'D': 15   # 大王
}


def generate_one_hand() -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    生成一局游戏的牌组
    
    返回: (hand_A, hand_B, hand_C, base_cards)
    - hand_A: 头叫手牌，17张
    - hand_B: 二叫手牌，17张  
    - hand_C: 三叫手牌，17张
    - base_cards: 底牌，3张
    """
    # 创建一副牌并洗牌
    deck = ALL_CARDS.copy()
    random.shuffle(deck)
    
    # 发牌
    hand_A = sorted(deck[0:17], key=lambda x: CARD_RANK[x])
    hand_B = sorted(deck[17:34], key=lambda x: CARD_RANK[x])
    hand_C = sorted(deck[34:51], key=lambda x: CARD_RANK[x])
    base_cards = sorted(deck[51:54], key=lambda x: CARD_RANK[x])
    
    return hand_A, hand_B, hand_C, base_cards


def hand_to_string(hand: List[str]) -> str:
    """将手牌列表转换为字符串"""
    return ''.join(hand)


def generate_paiku(NUM_PAIKU: int) -> List[str]:
    """
    生成指定数量的牌组
    
    Args:
        num_paiku: 需要生成的牌组数量
        
    Returns:
        牌组字符串列表，每个格式为: "hand_A,hand_B,hand_C,base"
        例如: "3334445556667788X,3334445556667788D,3334445556667799O,2DX"
    """
    paiku_list = []
    
    for i in range(NUM_PAIKU):
        hand_A, hand_B, hand_C, base = generate_one_hand()
        
        # 转换为字符串格式
        line = f"{hand_to_string(hand_A)},{hand_to_string(hand_B)},{hand_to_string(hand_C)},{hand_to_string(base)}"
        paiku_list.append(line)
        
        # 每生成1000个显示进度
        if (i + 1) % 1000 == 0:
            print(f"已生成 {i + 1}/{NUM_PAIKU} 个牌组")
    
    return paiku_list


def save_paiku(paiku_list: List[str], filename: str = 'paiku.txt'):
    """
    将牌组保存到文件

    Args:
        paiku_list: 牌组列表
        filename: 保存文件名
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for line in paiku_list:
            f.write(line + '\n')

    print(f"牌组已保存到 {filename}，共 {len(paiku_list)} 个牌组")


def load_paiku(filename: str = 'paiku.txt') -> List[str]:
    """
    从文件加载牌组
    
    Args:
        filename: 牌组文件名
        
    Returns:
        牌组字符串列表
    """
    paiku_list = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                paiku_list.append(line)
    
    print(f"从 {filename} 加载了 {len(paiku_list)} 个牌组")
    return paiku_list


def parse_paiku_line(line: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    解析牌组字符串
    
    Args:
        line: 格式为 "hand_A,hand_B,hand_C,base"
        
    Returns:
        (hand_A, hand_B, hand_C, base_cards)
    """
    parts = line.strip().split(',')
    if len(parts) != 4:
        raise ValueError(f"Invalid paiku line: {line}")
    
    hand_A = list(parts[0])
    hand_B = list(parts[1])
    hand_C = list(parts[2])
    base = list(parts[3])
    
    return hand_A, hand_B, hand_C, base


def analyze_paiku(paiku_list: List[str]):
    """
    分析牌组统计信息
    
    打印各种牌型出现的频率
    """
    print("=" * 60)
    print("牌组统计分析")
    print("=" * 60)
    
    bomb_count = 0
    rocket_count = 0
    
    for line in paiku_list:
        hand_A, hand_B, hand_C, base = parse_paiku_line(line)
        
        # 统计炸弹（4张相同）
        for hand in [hand_A, hand_B, hand_C]:
            card_counts = {}
            for card in hand:
                card_counts[card] = card_counts.get(card, 0) + 1
            for count in card_counts.values():
                if count >= 4:
                    bomb_count += 1
        
        # 统计王炸
        for hand in [hand_A, hand_B, hand_C]:
            if 'X' in hand and 'D' in hand:
                rocket_count += 1
    
    total_hands = len(paiku_list) * 3
    print(f"总手牌数: {total_hands}")
    print(f"炸弹数: {bomb_count} ({bomb_count/total_hands*100:.1f}%)")
    print(f"王炸数: {rocket_count} ({rocket_count/total_hands*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    # 从config读取需要生成的牌组数量
    from config import NUM_PAIKU
    
    print(f"生成 {NUM_PAIKU} 个牌组...")
    paiku_list = generate_paiku(NUM_PAIKU)
    save_paiku(paiku_list, 'paiku.txt')
    analyze_paiku(paiku_list)
    
    # 显示前3个牌组
    print("\n前3个牌组示例:")
    for i, line in enumerate(paiku_list[:3]):
        hand_A, hand_B, hand_C, base = parse_paiku_line(line)
        print(f"\n牌组 {i+1}:")
        print(f"  头叫: {''.join(hand_A)} ({len(hand_A)}张)")
        print(f"  二叫: {''.join(hand_B)} ({len(hand_B)}张)")
        print(f"  三叫: {''.join(hand_C)} ({len(hand_C)}张)")
        print(f"  底牌: {''.join(base)} ({len(base)}张)")
