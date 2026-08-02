"""
MAPPO (Multi-Agent PPO) 训练器
实现多智能体强化学习训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque


class RolloutBuffer:
    """经验回放缓冲区"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.next_states = []
        
    def add(self, state, action, log_prob, reward, value, done, next_state):
        """添加一条经验"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.next_states.append(next_state)
    
    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.next_states.clear()
    
    def get(self):
        """获取所有经验"""
        return {
            'states': self.states,
            'actions': self.actions,
            'log_probs': self.log_probs,
            'rewards': self.rewards,
            'values': self.values,
            'dones': self.dones,
            'next_states': self.next_states
        }
    
    def __len__(self):
        return len(self.states)


class MAPPOTrainer:
    """MAPPO 训练器"""
    
    def __init__(self, agents: List, config: Dict):
        """
        agents: 三个 DouDiZhuAgent 实例
        config: 训练配置
        """
        self.agents = agents
        self.config = config
        self.device = agents[0].device
        
        # 为每个agent创建经验缓冲区
        self.buffers = [RolloutBuffer() for _ in range(3)]
        
        # PPO 超参数
        self.gamma = config.get('gamma', 0.99)  # 折扣因子
        self.gae_lambda = config.get('gae_lambda', 0.95)  # GAE参数
        self.clip_epsilon = config.get('clip_epsilon', 0.2)  # PPO裁剪参数
        self.value_coef = config.get('value_coef', 0.5)  # 价值损失系数
        self.entropy_coef = config.get('entropy_coef', 0.01)  # 熵奖励系数
        self.max_grad_norm = config.get('max_grad_norm', 0.5)  # 梯度裁剪
        
        # 训练参数
        self.batch_size = config.get('batch_size', 64)
        self.update_epochs = config.get('update_epochs', 4)
        self.lr = config.get('lr', 3e-4)
        
        # 为每个agent创建优化器
        self.optimizers = []
        for agent in agents:
            params = list(agent.bid_net.parameters()) + list(agent.play_net.parameters())
            optimizer = optim.Adam(params, lr=self.lr)
            self.optimizers.append(optimizer)
    
    def compute_gae(self, rewards: List, values: List, dones: List, next_value: float) -> np.ndarray:
        """
        计算 GAE (Generalized Advantage Estimation)
        
        rewards: 奖励序列
        values: 价值估计序列
        dones: 是否结束序列
        next_value: 下一状态的价值
        """
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]
            
            # TD误差
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - values[t]
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return np.array(advantages)
    
    def update(self, agent_idx: int):
        """
        更新指定agent的策略
        
        agent_idx: agent索引 (0, 1, 2)
        """
        buffer = self.buffers[agent_idx]
        agent = self.agents[agent_idx]
        optimizer = self.optimizers[agent_idx]
        
        if len(buffer) == 0:
            return {}
        
        # 获取经验数据
        data = buffer.get()
        states = np.array(data['states'])
        actions = np.array(data['actions'])
        old_log_probs = np.array(data['log_probs'])
        rewards = np.array(data['rewards'])
        values = np.array(data['values'])
        dones = np.array(data['dones'])
        
        # 计算优势函数
        # 这里简化处理，使用最后一个状态的value作为next_value
        next_value = 0 if dones[-1] else values[-1]
        advantages = self.compute_gae(rewards, values, dones, next_value)
        
        # 计算回报 (returns)
        returns = advantages + values
        
        # 转换为tensor
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # 标准化优势
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # 多轮更新
        total_loss_sum = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        entropy_sum = 0
        
        for _ in range(self.update_epochs):
            # 前向传播
            # 注意：这里简化处理，假设所有动作都是叫分动作
            # 实际应该根据动作类型选择不同的网络
            bid_logits, state_values = agent.bid_net(states_tensor[:, :17])  # 简化：只用手牌部分
            
            # 计算新的动作概率
            bid_probs = torch.softmax(bid_logits, dim=-1)
            dist = torch.distributions.Categorical(bid_probs)
            new_log_probs = dist.log_prob(actions_tensor)
            entropy = dist.entropy()
            
            # PPO 损失
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = nn.MSELoss()(state_values.squeeze(), returns_tensor)
            
            # 总损失
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.bid_net.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(agent.play_net.parameters(), self.max_grad_norm)
            optimizer.step()
            
            total_loss_sum += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += entropy.mean().item()
        
        # 清空缓冲区
        buffer.clear()
        
        return {
            'total_loss': total_loss_sum / self.update_epochs,
            'policy_loss': policy_loss_sum / self.update_epochs,
            'value_loss': value_loss_sum / self.update_epochs,
            'entropy': entropy_sum / self.update_epochs
        }
    
    def update_all(self):
        """更新所有agent"""
        results = []
        for i in range(3):
            result = self.update(i)
            results.append(result)
        return results
    
    def save_models(self, path: str, cycle: int):
        """保存所有模型"""
        for i, agent in enumerate(self.agents):
            name = chr(65 + i)  # A, B, C
            filepath = f"{path}/agent_{name}_cycle_{cycle:03d}.pth"
            agent.save(filepath)
    
    def load_models(self, path: str):
        """加载所有模型"""
        import glob
        import os
        
        for i, agent in enumerate(self.agents):
            name = chr(65 + i)
            pattern = f"{path}/agent_{name}_cycle_*.pth"
            files = glob.glob(pattern)
            
            if files:
                files.sort()
                latest = files[-1]
                agent.load(latest)
                print(f"[加载] Agent {name}: {os.path.basename(latest)}")


if __name__ == "__main__":
    # 测试
    from model import DouDiZhuAgent
    
    agents = [DouDiZhuAgent(i) for i in range(3)]
    config = {
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'value_coef': 0.5,
        'entropy_coef': 0.01,
        'max_grad_norm': 0.5,
        'batch_size': 64,
        'update_epochs': 4,
        'lr': 3e-4
    }
    
    trainer = MAPPOTrainer(agents, config)
    print("MAPPO Trainer 创建成功")
    print(f"Agent数量: {len(trainer.agents)}")
    print(f"缓冲区数量: {len(trainer.buffers)}")
