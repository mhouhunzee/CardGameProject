"""斗地主推理应用 - 交互式助手.

加载训练好的模型，根据当前手牌和局面给出出牌建议.
支持选择位置（对应不同Agent），指导叫分和打牌.
"""

import argparse
import json
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple

from card_utils import (
    CardType, CardPattern, identify_pattern, can_beat,
    filter_legal_patterns, count_cards, hand_to_string, parse_hand_string
)
from train import ActorCritic, StateEncoder, Config


class DouDizhuAssistant:
    """斗地主AI助手."""
    
    def __init__(self, model_path: str, position: int, device: str = None):
        """初始化助手.
        
        Args:
            model_path: 模型权重文件路径
            position: 玩家位置 (0, 1, 2) 对应第1/2/3个叫分
            device: 计算设备
        """
        self.position = position
        self.agent_id = position  # 位置0对应Agent A，位置1对应Agent B，位置2对应Agent C
        self.config = Config()
        # 强制使用CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            print("警告: CUDA不可用，使用CPU运行")
        
        # 加载对应Agent的模型
        self.model = ActorCritic(
            self.config.state_dim,
            self.config.hidden_dim,
            self.config.action_dim
        ).to(self.device)
        
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.eval()
        
        # 状态编码器
        self.encoder = StateEncoder()
        
        # 游戏状态追踪
        self.reset_state()
    
    def reset_state(self):
        """重置游戏状态."""
        self.hand: List[int] = []
        self.played_cards: List[int] = []
        self.last_pattern: Optional[CardPattern] = None
        self.is_landlord: bool = False
        self.landlord_position: Optional[int] = None  # 地主位置
        self.role: str = ""  # 'landlord', 'landlord_prev', 'landlord_next'
        self.bid_score: int = 0
        self.history: List[Dict] = []
        self.current_player: int = 0  # 当前轮到谁出牌
    
    def set_hand(self, hand_str: str):
        """设置当前手牌.
        
        Args:
            hand_str: 手牌字符串，如 "333456789OOKKAA2" 或 "333456789OJDK"
        """
        self.hand = parse_hand_string(hand_str)
        print(f"手牌已设置: {hand_to_string(self.hand)} ({len(self.hand)}张)")
    
    def determine_role(self, landlord_pos: int):
        """确定角色.
        
        Args:
            landlord_pos: 地主位置 (0, 1, 2)
        """
        self.landlord_position = landlord_pos
        
        if self.position == landlord_pos:
            self.is_landlord = True
            self.role = 'landlord'
            print(f"您的角色: 地主")
        else:
            self.is_landlord = False
            # 判断是地主上家还是下家
            # 位置顺序: 0 -> 1 -> 2 -> 0
            if (self.position - landlord_pos) % 3 == 1:
                self.role = 'landlord_next'  # 地主下家（先出牌）
                print(f"您的角色: 农民（地主下家，先出牌）")
            else:
                self.role = 'landlord_prev'  # 地主上家
                print(f"您的角色: 农民（地主上家）")
    
    def get_bid_advice(self, current_bids: List[int]) -> Dict:
        """获取叫分建议.
        
        Args:
            current_bids: 当前已叫的分数 [bid0, bid1, bid2]，未叫为None
            
        Returns:
            叫分建议
        """
        # 基于手牌强度给出建议
        hand_strength = self._evaluate_hand_strength()
        
        # 当前最高叫分
        max_bid = max([b for b in current_bids if b is not None] + [0])
        
        # 建议叫分
        if hand_strength > 0.8:
            suggested_bid = 3
            reason = "手牌很强，建议叫3分抢地主"
        elif hand_strength > 0.6:
            suggested_bid = 2
            reason = "手牌较好，建议叫2分"
        elif hand_strength > 0.4:
            suggested_bid = 1
            reason = "手牌一般，建议叫1分"
        else:
            suggested_bid = 0
            reason = "手牌较弱，建议不叫"
        
        # 如果建议叫分低于当前最高，则建议不叫
        if suggested_bid <= max_bid:
            suggested_bid = 0
            reason = f"当前最高叫分{max_bid}，建议不叫"
        
        return {
            'suggested_bid': suggested_bid,
            'hand_strength': hand_strength,
            'reason': reason,
            'max_current_bid': max_bid
        }
    
    def _evaluate_hand_strength(self) -> float:
        """评估手牌强度.
        
        Returns:
            强度分数 (0-1)
        """
        if not self.hand:
            return 0
        
        count = count_cards(self.hand)
        strength = 0.0
        
        # 大牌加分
        for point, cnt in count.items():
            if point >= 17:  # 大王
                strength += cnt * 0.2
            elif point >= 16:  # 小王
                strength += cnt * 0.15
            elif point >= 15:  # 2
                strength += cnt * 0.12
            elif point >= 14:  # A
                strength += cnt * 0.08
            elif point >= 13:  # K
                strength += cnt * 0.05
        
        # 炸弹加分
        for cnt in count.values():
            if cnt == 4:
                strength += 0.5
        
        # 火箭
        if 16 in count and 17 in count:
            strength += 1.0
        
        # 顺子潜力
        straight_potential = len([p for p in count.keys() if 3 <= p <= 14])
        if straight_potential >= 5:
            strength += 0.1
        
        return min(strength, 1.0)
    
    def update_game_state(self, last_play: str = None, played_cards_str: str = None):
        """更新游戏状态.
        
        Args:
            last_play: 上家出的牌，如 "34567" 或 "Pass"
            played_cards_str: 已出的所有牌（累计）
        """
        # 更新上家出牌
        if last_play and last_play.lower() != "pass":
            cards = parse_hand_string(last_play)
            self.last_pattern = identify_pattern(cards)
            if self.last_pattern:
                print(f"上家出牌: {last_play} ({self.last_pattern.card_type.name})")
        elif last_play and last_play.lower() == "pass":
            print("上家: Pass")
        
        # 更新已出牌
        if played_cards_str:
            self.played_cards = parse_hand_string(played_cards_str)
    
    def get_recommendation(self, top_k: int = 3) -> Dict:
        """获取出牌建议.
        
        Returns:
            包含建议出牌、备选方案、胜率评估的字典
        """
        # 编码状态
        state = self.encoder.encode(
            self.hand,
            self.played_cards,
            self.last_pattern,
            self.is_landlord,
            self.position
        )
        mask = self.encoder.create_action_mask(self.hand, self.last_pattern)
        
        # 模型推理
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, value = self.model(state_tensor, mask_tensor)
        
        # 获取Top-K动作
        probs = action_probs.squeeze().cpu().numpy()
        legal_indices = np.where(mask > 0)[0]
        
        if len(legal_indices) == 0:
            return {
                "recommended": "Pass",
                "alternatives": [],
                "win_rate": 0.5,
                "reason": "无合法出牌，必须Pass"
            }
        
        # 只考虑合法动作
        legal_probs = probs[legal_indices]
        top_k_indices = legal_indices[np.argsort(legal_probs)[-top_k:][::-1]]
        
        # 构建结果
        recommendations = []
        for idx in top_k_indices:
            pattern = self.encoder.action_encoder.idx_to_pattern(int(idx))
            prob = float(probs[idx])
            
            # 转换为字符串表示
            if pattern.card_type == CardType.PASS:
                cards_str = "Pass"
            else:
                cards_str = hand_to_string(pattern.cards)
            
            recommendations.append({
                "cards": cards_str,
                "win_rate": prob,
                "pattern_type": pattern.card_type.name
            })
        
        # 生成建议理由
        best = recommendations[0] if recommendations else {"cards": "Pass", "win_rate": 0}
        reason = self._generate_reason(best, recommendations[1:] if len(recommendations) > 1 else [])
        
        # 估算胜率
        win_rate = float(torch.sigmoid(value).item())
        
        result = {
            "recommended": best["cards"],
            "alternatives": recommendations[1:] if len(recommendations) > 1 else [],
            "win_rate": win_rate,
            "reason": reason
        }
        
        return result
    
    def _generate_reason(self, best: Dict, alternatives: List[Dict]) -> str:
        """生成建议理由."""
        if best["cards"] == "Pass":
            if self.last_pattern:
                return f"建议Pass，上家出了{self.last_pattern.card_type.name}，手牌难以压制，保留牌权"
            else:
                return "建议Pass（自由出牌时不应选择Pass，请检查输入）"
        
        reason = f"建议出 {best['cards']} ({best['pattern_type']})"
        
        # 根据牌型添加具体理由
        if best["pattern_type"] == "BOMB":
            reason += "，炸弹可以压制任何非炸弹牌型，获得出牌权"
        elif best["pattern_type"] == "ROCKET":
            reason += "，王炸最大，可压一切"
        elif best["pattern_type"] == "STRAIGHT":
            reason += "，顺子一次出多张，快速减少手牌"
        elif best["pattern_type"] in ["TRIPLE_SINGLE", "TRIPLE_PAIR"]:
            reason += "，三带牌型效率高"
        
        # 比较备选方案
        if alternatives:
            alt = alternatives[0]
            if best["win_rate"] - alt["win_rate"] < 0.1:
                reason += f"。备选: {alt['cards']}，胜率接近"
        
        return reason
    
    def interactive_mode(self):
        """交互模式."""
        print("\n" + "="*60)
        print("斗地主AI助手 - 交互模式")
        print("="*60)
        print("命令:")
        print("  hand <牌型>  - 设置手牌 (如: hand 333445566778899OOJJQQKKAA2)")
        print("                 17张牌示例: 333445566778899OOJJQQKKAA2")
        print("  bid <0/1/2/3> - 叫分 (0=不叫, 1=1分, 2=2分, 3=3分)")
        print("  see <牌型>   - 设置对手出牌 (如: see 34567 或 see Pass)")
        print("  ask          - 获取出牌建议")
        print("  R/r          - 重新开始下一局")
        print("  E/e          - 退出程序")
        print("="*60 + "\n")
        
        # 叫分阶段
        print("【叫分阶段】")
        print("请输入您的手牌:")
        
        bids = [None, None, None]
        bidding_complete = False
        
        while not bidding_complete:
            try:
                cmd = input("> ").strip()
                
                if not cmd:
                    continue
                
                parts = cmd.split(maxsplit=1)
                action = parts[0].lower()
                
                if action == 'e':
                    print("退出程序")
                    return False  # 表示退出
                
                elif action == 'r':
                    print("重新开始下一局")
                    return True  # 表示重新开始
                
                elif action == "hand":
                    if len(parts) > 1:
                        self.set_hand(parts[1])
                        # 给出叫分建议
                        advice = self.get_bid_advice(bids)
                        print(f"\n叫分建议: {advice['suggested_bid']}分")
                        print(f"理由: {advice['reason']}")
                        print(f"手牌强度: {advice['hand_strength']:.2f}")
                    else:
                        print("错误: 请提供手牌，例如: hand 333445566778899OOJJQQKKAA2")
                
                elif action == "bid" and len(parts) > 1:
                    bid = int(parts[1])
                    bids[self.position] = bid
                    print(f"您叫了 {bid} 分")
                    
                    # 模拟其他玩家叫分（简化）
                    for i in range(3):
                        if i != self.position and bids[i] is None:
                            # 这里可以接入其他Agent的决策
                            bids[i] = random.randint(0, 3)
                            print(f"玩家{i}叫了 {bids[i]} 分")
                    
                    # 确定地主
                    max_bid = max(bids)
                    if max_bid > 0:
                        landlord = bids.index(max_bid)
                        self.determine_role(landlord)
                        self.bid_score = max_bid
                        bidding_complete = True
                        print(f"\n地主是玩家 {landlord}，叫分 {max_bid}")
                    else:
                        print("\n无人叫分，流局！")
                        print("请输入 'R' 重新开始，或 'E' 退出")
                
                else:
                    print("未知命令，请重试")
            
            except Exception as e:
                print(f"错误: {e}")
        
        # 出牌阶段
        print("\n【出牌阶段】")
        print("请输入对手的出牌，或输入 'ask' 获取建议")
        
        while True:
            try:
                cmd = input("> ").strip()
                
                if not cmd:
                    continue
                
                parts = cmd.split(maxsplit=1)
                action = parts[0].lower()
                
                if action == 'e':
                    print("退出程序")
                    return False
                
                elif action == 'r':
                    print("重新开始下一局")
                    return True
                
                elif action == "see" and len(parts) > 1:
                    self.update_game_state(last_play=parts[1])
                
                elif action == "ask":
                    if not self.hand:
                        print("请先设置手牌 (hand <牌型>)")
                        continue
                    
                    result = self.get_recommendation()
                    print("\n" + "-"*40)
                    print(f"建议出牌: {result['recommended']}")
                    print(f"预估胜率: {result['win_rate']:.1%}")
                    print(f"理由: {result['reason']}")
                    
                    if result['alternatives']:
                        print("\n备选方案:")
                        for i, alt in enumerate(result['alternatives'][:2], 1):
                            print(f"  {i}. {alt['cards']} (胜率: {alt['win_rate']:.1%})")
                    print("-"*40 + "\n")
                
                elif action == "hand":
                    if len(parts) > 1:
                        self.set_hand(parts[1])
                    else:
                        print("错误: 请提供手牌，例如: hand 333445566778899OOJJQQKKAA2")
                
                else:
                    print("未知命令，请重试")
            
            except Exception as e:
                print(f"错误: {e}")


def main():
    """主函数."""
    parser = argparse.ArgumentParser(description="斗地主AI助手")
    parser.add_argument("--model", type=str, default="./models/final/model_a.pth",
                        help="模型权重路径")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备 (cuda/cpu)，默认使用cuda")
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    print(f"PyTorch版本: {torch.__version__}")
    
    # 检查PyTorch是否为CUDA版本
    if '+cpu' in torch.__version__:
        print("\n警告: 您安装的是PyTorch CPU版本！")
        print("当前版本:", torch.__version__)
        print("\n建议安装PyTorch CUDA版本以获得更好的性能:")
        print("  pip uninstall torch torchvision torchaudio -y")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n程序将继续使用CPU运行...\n")
        return
    
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"CUDA可用: {device_name}")
            print(f"CUDA版本: {cuda_version}")
        except Exception as e:
            print(f"CUDA检测警告: {e}")
            print("将尝试使用CPU运行")
    else:
        print("警告: CUDA不可用，将使用CPU运行")
    
    print("="*60)
    print("欢迎使用斗地主AI助手！")
    print("="*60)
    
    while True:
        # 选择位置
        print("\n请选择您的位置：")
        print("  1 - 第一个叫分 (对应Agent A)")
        print("  2 - 第二个叫分 (对应Agent B)")
        print("  3 - 第三个叫分 (对应Agent C)")
        print("  E - 退出")
        
        choice = input("> ").strip().lower()
        
        if choice == 'e':
            print("再见！")
            break
        
        if choice not in ['1', '2', '3']:
            print("无效选择，请重试")
            continue
        
        position = int(choice) - 1  # 转换为0-based索引
        agent_names = ['a', 'b', 'c']
        agent_name = agent_names[position]
        
        # 加载对应Agent的模型
        model_path = f"./models/final/model_{agent_name}.pth"
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            print("尝试使用默认模型...")
            model_path = args.model
        
        print(f"\n已选择位置 {choice}，加载 Agent {agent_name.upper()} 的模型...")
        
        # 创建助手
        assistant = DouDizhuAssistant(model_path, position, args.device)
        
        # 进入交互模式
        restart = assistant.interactive_mode()
        
        if not restart:
            print("再见！")
            break
        # 否则继续循环，重新选择位置


if __name__ == "__main__":
    import random
    import os
    main()
