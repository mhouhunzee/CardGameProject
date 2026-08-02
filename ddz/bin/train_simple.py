"""
斗地主训练脚本（简化版，适配新的70维状态设计）
"""
import os
import numpy as np
import torch
from tqdm import tqdm

from config import CYCLE, H, MODEL_DIR
from env import DouDiZhuEnv
from model import DouDiZhuAgent
from generator import load_paiku, parse_paiku_line


class SimpleTrainer:
    """简化版训练器"""
    
    def __init__(self):
        self.agents = [
            DouDiZhuAgent(0),  # 头叫
            DouDiZhuAgent(1),  # 二叫
            DouDiZhuAgent(2)   # 三叫
        ]
        self.env = DouDiZhuEnv()
        self.paiku = load_paiku()
        
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    def play_game(self, agents):
        """进行一局游戏"""
        # 发牌
        line = self.paiku[np.random.randint(len(self.paiku))]
        hand_A, hand_B, hand_C, base = parse_paiku_line(line)
        
        self.env.reset()
        self.env.deal_cards(hand_A, hand_B, hand_C, base)
        
        # 叫分阶段（使用bid_net）
        bids = []
        for pos in [0, 1, 2]:
            hand = [hand_A, hand_B, hand_C][pos]
            bid = agents[pos].select_bid(hand, epsilon=0.1)
            bids.append(bid)
        
        # 确定地主
        success, landlord, final_bid = self.env.bidding_phase(*bids)
        if not success:
            return None, None  # 流局
        
        # 初始化出牌状态（70维向量）
        for pos in [0, 1, 2]:
            hand = [hand_A, hand_B, hand_C][pos]
            if pos == landlord:
                hand = hand + base  # 地主获得底牌
            agents[pos].init_play_state(hand, pos)
        
        # 出牌阶段
        self.env.start_playing()
        
        while not self.env.done:
            player = self.env.current_player
            agent = agents[player]
            
            # 获取合法动作
            legal_actions = self.env.get_legal_actions(player)
            
            # 使用play_net决策（输入70维状态）
            action = agent.select_play(legal_actions, epsilon=0.1)
            
            # 执行动作
            success, msg, done, reward = self.env.step(player, action)
            
            if not success:
                action = "PASS"
                success, msg, done, reward = self.env.step(player, action)
        
        # 返回结果
        return self.env.winner, self.env._calculate_reward(0)
    
    def train(self):
        """训练"""
        print("开始训练...")
        
        for cycle in range(CYCLE):
            print(f"\nCycle {cycle + 1}/{CYCLE}")
            
            # 训练每个agent
            for agent_idx in [0, 1, 2]:
                print(f"  训练Agent {agent_idx}")
                
                for game in tqdm(range(H), desc=f"Agent {agent_idx}"):
                    # 创建agents字典，固定其他两个agent
                    agents = {}
                    for pos in [0, 1, 2]:
                        if pos == agent_idx:
                            agents[pos] = self.agents[pos]  # 被训练的agent
                        else:
                            # 其他agent使用当前版本
                            agents[pos] = self.agents[pos]
                    
                    winner, reward = self.play_game(agents)
                    
                    # 这里应该收集训练数据并更新网络
                    # 简化版省略了具体的训练步骤
            
            # 保存模型
            for idx, agent in enumerate(self.agents):
                path = os.path.join(MODEL_DIR, f"agent_{chr(65+idx)}_cycle_{cycle:03d}.pth")
                agent.save(path)
        
        print("训练完成!")


if __name__ == "__main__":
    trainer = SimpleTrainer()
    trainer.train()
