# MAPPO 强化学习训练指南

## 概述

本项目实现了 MAPPO (Multi-Agent PPO) 算法来训练斗地主 AI。三个 Agent 通过自我对弈学习最优策略。

## 训练架构

```
┌─────────────────────────────────────────────────────────────┐
│                    MAPPO Training System                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Agent A    │  │  Agent B    │  │  Agent C    │         │
│  │ (头叫)      │  │ (二叫)      │  │ (三叫)      │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          │                                  │
│              ┌───────────▼───────────┐                      │
│              │    DouDiZhuEnv        │                      │
│              │    (游戏环境)          │                      │
│              └───────────┬───────────┘                      │
│                          │                                  │
│              ┌───────────▼───────────┐                      │
│              │   MAPPOTrainer        │                      │
│              │  (经验收集+策略更新)   │                      │
│              └───────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. 经验回放缓冲区 (RolloutBuffer)

存储每个 Agent 的经验：
- `state`: 当前状态 (70维向量)
- `action`: 执行的动作
- `log_prob`: 动作的对数概率
- `reward`: 获得的奖励
- `value`: 状态价值估计
- `done`: 是否结束

### 2. MAPPO 训练器

实现 PPO 算法的关键步骤：
1. **GAE 计算**: 使用 Generalized Advantage Estimation 计算优势函数
2. **策略更新**: 使用 Clipped Surrogate Objective 更新策略
3. **价值更新**: 更新状态价值函数
4. **熵奖励**: 鼓励探索

### 3. 训练流程

每个 Cycle 包含：
1. **轮流训练**: 三个 Agent 分别进行 H 局游戏
2. **经验收集**: 记录 (s, a, r, s') 经验
3. **策略更新**: 使用 PPO 算法更新神经网络
4. **模型保存**: 保存当前模型

## 训练参数

```python
config = {
    'gamma': 0.99,          # 折扣因子
    'gae_lambda': 0.95,     # GAE 参数
    'clip_epsilon': 0.2,    # PPO 裁剪参数
    'value_coef': 0.5,      # 价值损失系数
    'entropy_coef': 0.01,   # 熵奖励系数
    'max_grad_norm': 0.5,   # 梯度裁剪
    'batch_size': 64,       # 批次大小
    'update_epochs': 4,     # 每次数据更新轮数
    'lr': 0.001             # 学习率
}
```

## 奖励设计

```
奖励 = 基础分数 × 炸弹倍数 × 春天倍数

基础分数 = 叫分 (1/2/3)
炸弹倍数 = 2^(炸弹数 + 火箭数)
春天倍数 = 2 (如果是春天或反春天)

获胜方: +奖励
失败方: -奖励
```

## 使用方法

### 1. 开始训练

```bash
python train_mappo.py
```

### 2. 继续训练

训练会自动加载 `models/` 目录下最新的模型继续训练。

### 3. 查看日志

训练日志保存在 `logs/train_YYYYMMDD_HHMMSS.jsonl`

### 4. 使用训练好的模型

```python
from model import DouDiZhuAgent

agent = DouDiZhuAgent(position=0)
agent.load('models/agent_A_cycle_100.pth')
```

## 训练技巧

### 1. 探索 vs 利用

- **前期**: epsilon 较大 (1.0)，鼓励探索
- **后期**: epsilon 衰减到 0.05，主要利用已学策略

### 2. 轮流训练

每个 Cycle 三个 Agent 分别训练，避免同时更新导致的非平稳环境。

### 3. 奖励归一化

使用 GAE 对优势函数进行归一化，稳定训练。

## 常见问题

### Q: 训练需要多久？

A: 取决于硬件和配置。GPU 上 1000 cycles 可能需要几小时到几天。

### Q: 如何判断训练效果？

A: 观察以下指标：
- 平均奖励是否上升
- 胜率是否提高
- 损失是否稳定下降

### Q: 可以只训练一个 Agent 吗？

A: 可以，修改 `train_cycle` 中的循环，只训练指定 Agent。

### Q: 如何调整学习率？

A: 修改 `config.py` 中的 `LEARNING_RATE` 或 `train_mappo.py` 中的配置。

## 进阶优化

### 1. 添加更多特征

- 已出牌记录
- 对手建模
- 牌型统计

### 2. 改进网络结构

- 使用 Transformer 替代 MLP
- 添加注意力机制
- 使用 RNN 处理时序信息

### 3. 改进奖励函数

- 添加出牌合理性奖励
- 添加炸弹使用时机奖励
- 添加农民配合奖励

### 4. 多进程并行

使用多进程并行收集经验，加速训练。

## 参考

- PPO: Proximal Policy Optimization Algorithms (Schulman et al., 2017)
- MAPPO: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (Yu et al., 2021)
- GAE: High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2015)
