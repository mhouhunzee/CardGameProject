"""
斗地主牌库生成器
生成随机牌库并保存为paiku.txt
格式：A(3,3,4,4,...),B(4,4,6,6...),C(5,5,5,7,7...),D(5,8,O)
"""
import random
import os
from config import (
    ALL_CARDS, CARD_COUNTS, HAND_SIZE, BASE_CARD_SIZE,
    PAIKU_FILE, NUM_PAIKU, GENERATOR_DIR
)


def generate_deck():
    """生成一副完整的牌（54张）"""
    deck = []
    for card, count in CARD_COUNTS.items():
        deck.extend([card] * count)
    return deck


def deal_cards(deck):
    """
    发牌
    返回: (A的手牌, B的手牌, C的手牌, 底牌)
    """
    # 洗牌
    shuffled = deck.copy()
    random.shuffle(shuffled)
    
    # 发牌
    hand_A = sorted(shuffled[:HAND_SIZE], key=lambda x: ALL_CARDS.index(x))
    hand_B = sorted(shuffled[HAND_SIZE:HAND_SIZE*2], key=lambda x: ALL_CARDS.index(x))
    hand_C = sorted(shuffled[HAND_SIZE*2:HAND_SIZE*3], key=lambda x: ALL_CARDS.index(x))
    base = sorted(shuffled[HAND_SIZE*3:], key=lambda x: ALL_CARDS.index(x))
    
    return hand_A, hand_B, hand_C, base


def format_hand(hand):
    """将手牌格式化为字符串"""
    return ','.join(hand)


def generate_paiku(num_pakiu=NUM_PAIKU):
    """生成指定数量的牌库"""
    pakiu_list = []
    
    for i in range(num_pakiu):
        deck = generate_deck()
        hand_A, hand_B, hand_C, base = deal_cards(deck)
        
        # 格式化
        line = f"A({format_hand(hand_A)}),B({format_hand(hand_B)}),C({format_hand(hand_C)}),D({format_hand(base)})"
        pakiu_list.append(line)
    
    return pakiu_list


def save_paiku(paiku_list, filepath=PAIKU_FILE):
    """保存牌库到文件"""
    # 确保目录存在
    dir_path = os.path.dirname(filepath)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# 斗地主牌库\n")
        f.write(f"# 共{len(paiku_list)}个随机牌库\n")
        f.write(f"# 格式: A(头叫手牌),B(二叫手牌),C(三叫手牌),D(底牌)\n")
        f.write("\n")
        for i, line in enumerate(paiku_list, 1):
            f.write(f"{line}\n")
    
    print(f"已生成{len(paiku_list)}个牌库，保存到: {filepath}")


def load_paiku(filepath=PAIKU_FILE):
    """加载牌库"""
    paiku_list = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            paiku_list.append(line)
    
    return paiku_list


def parse_paiku_line(line):
    """
    解析一行牌库
    返回: (A手牌列表, B手牌列表, C手牌列表, 底牌列表)
    """
    # 移除前缀
    line = line.replace('A(', '(').replace('B(', '(').replace('C(', '(').replace('D(', '(')
    
    # 分割
    parts = line.split('),(')
    
    # 解析
    hand_A = parts[0][1:].split(',') if parts[0][0] == '(' else parts[0].split(',')
    hand_B = parts[1].split(',')
    hand_C = parts[2].split(',')
    base = parts[3][:-1].split(')')[0].split(',')
    
    return hand_A, hand_B, hand_C, base


if __name__ == "__main__":
    # 生成牌库
    print("开始生成牌库...")
    paiku_list = generate_paiku(NUM_PAIKU)
    save_paiku(paiku_list)
    print("完成!")
