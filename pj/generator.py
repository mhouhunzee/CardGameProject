"""牌库生成器 - 生成随机斗地主牌局数据并保存到文件.

该模块独立于训练流程，负责生成随机牌局数据并编码保存.
"""

import random
import json
import os
from typing import List, Dict, Any


# 牌面映射常量
CARD_3, CARD_4, CARD_5, CARD_6 = 3, 4, 5, 6
CARD_7, CARD_8, CARD_9, CARD_10 = 7, 8, 9, 10
CARD_J, CARD_Q, CARD_K, CARD_A = 11, 12, 13, 14
CARD_2 = 15
CARD_XW = 16  # 小王
CARD_DW = 17  # 大王

# 完整牌组（54张）
FULL_DECK = (
    [CARD_3] * 4 + [CARD_4] * 4 + [CARD_5] * 4 + [CARD_6] * 4 +
    [CARD_7] * 4 + [CARD_8] * 4 + [CARD_9] * 4 + [CARD_10] * 4 +
    [CARD_J] * 4 + [CARD_Q] * 4 + [CARD_K] * 4 + [CARD_A] * 4 +
    [CARD_2] * 4 + [CARD_XW] + [CARD_DW]
)


def generate_decks(n: int, seed: int = None) -> List[Dict[str, Any]]:
    """生成n组随机牌局数据.
    
    每组牌局包含3个玩家的初始手牌（各17张）和3张底牌。
    所有牌组完全独立，每次重新洗牌。
    
    Args:
        n: 需要生成的牌局数量
        seed: 随机种子，用于结果复现。若为None则使用系统随机
        
    Returns:
        包含n组牌局的列表，每组为字典格式：
        {
            "player1": List[int],  # 17张牌
            "player2": List[int],  # 17张牌
            "player3": List[int],  # 17张牌
            "bottom": List[int]    # 3张底牌
        }
        
    Example:
        >>> decks = generate_decks(2, seed=42)
        >>> len(decks)
        2
        >>> len(decks[0]["player1"])
        17
        >>> len(decks[0]["bottom"])
        3
    """
    if seed is not None:
        random.seed(seed)
    
    decks = []
    for _ in range(n):
        # 创建牌组副本并洗牌
        deck = FULL_DECK.copy()
        random.shuffle(deck)
        
        # 发牌：前17张给player1，17-34给player2，34-51给player3，最后3张为底牌
        player1_hand = sorted(deck[0:17])
        player2_hand = sorted(deck[17:34])
        player3_hand = sorted(deck[34:51])
        bottom_cards = sorted(deck[51:54])
        
        decks.append({
            "player1": player1_hand,
            "player2": player2_hand,
            "player3": player3_hand,
            "bottom": bottom_cards
        })
    
    return decks


def save_decks_to_file(decks: List[Dict[str, Any]], filepath: str):
    """将牌组保存到txt文件.
    
    Args:
        decks: 牌组列表
        filepath: 文件路径
    """
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        # 写入头部信息
        f.write(f"# DouDizhu Decks Data\n")
        f.write(f"# Total decks: {len(decks)}\n")
        f.write(f"# Format: JSON Lines\n")
        f.write(f"# Each line contains one deck: {{'player1': [...], 'player2': [...], 'player3': [...], 'bottom': [...]}}\n")
        f.write("# " + "="*50 + "\n")
        
        # 写入每个牌组
        for i, deck in enumerate(decks):
            deck_data = {
                'id': i,
                'player1': deck['player1'],
                'player2': deck['player2'],
                'player3': deck['player3'],
                'bottom': deck['bottom']
            }
            f.write(json.dumps(deck_data) + '\n')
    
    print(f"牌组已保存到: {filepath}")


def load_decks_from_file(filepath: str) -> List[Dict[str, Any]]:
    """从txt文件加载牌组.
    
    Args:
        filepath: 文件路径
        
    Returns:
        牌组列表
    """
    decks = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            deck_data = json.loads(line)
            decks.append({
                'player1': deck_data['player1'],
                'player2': deck_data['player2'],
                'player3': deck_data['player3'],
                'bottom': deck_data['bottom']
            })
    
    print(f"已从 {filepath} 加载 {len(decks)} 组牌局")
    return decks


def generate_and_save(n: int, filepath: str = "./decks/decks_data.txt", seed: int = None):
    """生成牌组并保存到文件.
    
    Args:
        n: 牌组数量
        filepath: 保存路径
        seed: 随机种子
    """
    decks = generate_decks(n, seed)
    save_decks_to_file(decks, filepath)
    return filepath


if __name__ == "__main__":
    # 简单测试
    test_decks = generate_decks(3, seed=42)
    for i, deck in enumerate(test_decks):
        print(f"\n=== 牌局 {i+1} ===")
        print(f"Player1 ({len(deck['player1'])}张): {deck['player1']}")
        print(f"Player2 ({len(deck['player2'])}张): {deck['player2']}")
        print(f"Player3 ({len(deck['player3'])}张): {deck['player3']}")
        print(f"底牌 ({len(deck['bottom'])}张): {deck['bottom']}")
        
        # 验证牌数正确
        total = len(deck['player1']) + len(deck['player2']) + len(deck['player3']) + len(deck['bottom'])
        print(f"总牌数验证: {total} (应为54)")
    
    # 测试保存和加载
    print("\n=== 测试保存和加载 ===")
    save_decks_to_file(test_decks, "./test_decks.txt")
    loaded_decks = load_decks_from_file("./test_decks.txt")
    print(f"加载了 {len(loaded_decks)} 组牌局")
