# DouDizhu MARL - 斗地主多智能体强化学习

基于 MAPPO (Multi-Agent PPO) 算法的斗地主 AI 训练框架。
**新版特性：训练与可视化完全分离**

## 项目结构

```
doudizhu_marl/
├── train_new.py         # 新版训练脚本 - 独立日志记录
├── train_logger.py      # 训练日志记录模块
├── visualizer.py        # 独立可视化模块 - 从日志生成图表
├── generator.py         # 牌库生成器
├── card_utils.py        # 牌型工具
├── play.py              # 推理应用
├── requirements.txt     # 依赖包
└── README.md            # 项目说明
```

## 工作流程

### 1. 训练阶段

运行训练脚本，生成日志文件：

```bash
python train_new.py
```

训练完成后，日志文件保存在 `./train_logs/` 目录：
- `game_log_{phase}_r{round}_{timestamp}.jsonl` - 每局游戏详细记录
- `summary_{phase}_r{round}_{timestamp}.jsonl` - 每轮训练汇总统计

### 2. 可视化阶段

独立运行可视化脚本，从日志生成图表：

```bash
python visualizer.py [日志目录] [输出目录]
```

示例：
```bash
python visualizer.py ./train_logs ./visualizations
```

## 图表输出结构

可视化脚本会创建以下目录结构：

```
visualizations/
├── initial/                    # 初始训练阶段（3个agent对比）
│   ├── win_rate_comparison.png    # 胜率对比
│   ├── score_comparison.png       # 得分对比
│   ├── bid_comparison.png         # 叫分对比
│   ├── landlord_win_rate.png      # 地主胜率
│   └── draw_rate.png              # 流局率
│
├── overall/                    # 全局指标（跨所有阶段）
│   ├── avg_final_bid.png          # 成交分数均值
│   ├── avg_action_count.png       # 出牌次数均值
│   ├── avg_game_length.png        # 游戏长度均值
│   ├── avg_bomb_count.png         # 炸弹使用率
│   ├── overall_win_rates.png      # 各agent总胜率趋势
│   └── overall_draw_rate.png      # 流局率趋势
│
├── agent_a/                    # Agent A 的所有训练阶段
│   ├── bid_evolution.png          # 叫分均值变化
│   ├── score_evolution.png        # 得分变化
│   ├── win_rate_evolution.png     # 胜率变化
│   ├── role_confusion_matrix.png  # 角色胜率混淆矩阵
│   └── bid_distribution.png       # 叫分分布变化
│
├── agent_b/                    # Agent B 的所有训练阶段
│   └── (同上)
│
└── agent_c/                    # Agent C 的所有训练阶段
    └── (同上)
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练模型

```bash
python train_new.py
```

### 生成图表

```bash
# 使用默认目录
python visualizer.py

# 指定日志和输出目录
python visualizer.py ./train_logs ./my_charts
```

## 训练流程

1. **初始训练**：三个Agent一起玩3万局
2. **轮流强化训练**：
   - 固定A、B，训练C（1万局）
   - 固定B、C，训练A（1万局）
   - 固定A、C，训练B（1万局）
   - 循环20次

**总训练对局数**：
- 初始训练：30,000局
- 强化训练：20轮 × 3Agent × 10,000局 = 600,000局
- 总计：630,000局

## 日志格式

### 游戏日志 (game_log_*.jsonl)

每行一个JSON对象，记录单局游戏：

```json
{
  "episode": 100,
  "game_id": 12345,
  "phase": "initial",
  "round_num": 0,
  "bids": [0, 2, 3],
  "final_bid": 3,
  "landlord": 2,
  "is_draw": false,
  "winner": 2,
  "game_length": 25,
  "rewards": [-3.0, -3.0, 6.0],
  "roles": ["farmer", "farmer", "landlord"],
  "action_counts": [8, 7, 10],
  "bomb_count": 1,
  "rocket_count": 0,
  "timestamp": "2026-07-27T10:00:00"
}
```

### 汇总日志 (summary_*.jsonl)

每行一个JSON对象，记录每轮训练的滑动窗口统计：

```json
{
  "episode": 100,
  "phase": "initial",
  "round_num": 0,
  "win_rates": [0.32, 0.34, 0.34],
  "landlord_win_rate": 0.52,
  "avg_scores": [-0.5, 0.2, 0.3],
  "avg_bids": [0.8, 1.2, 1.5],
  "bid_distribution": [{"0": 20, "1": 30, "2": 40, "3": 10}, ...],
  "landlord_win_rates": [0.55, 0.50, 0.51],
  "farmer_win_rates": [0.30, 0.35, 0.33],
  "total_games": 10000,
  "draw_count": 500,
  "avg_game_length": 22.5,
  "avg_final_bid": 1.8,
  "avg_action_count": 25.3,
  "timestamp": "2026-07-27T10:00:00"
}
```

## 图表说明

### initial/ 文件夹图表

每张图同时显示3个agent，便于对比：

| 图表 | 说明 |
|------|------|
| win_rate_comparison.png | 三个agent胜率变化对比 |
| score_comparison.png | 三个agent得分变化对比 |
| bid_comparison.png | 三个agent叫分变化对比 |
| landlord_win_rate.png | 地主整体胜率变化 |
| draw_rate.png | 流局率变化 |

### overall/ 文件夹图表

全局指标，跨所有训练阶段：

| 图表 | 说明 |
|------|------|
| avg_final_bid.png | 每局成交分数均值（流局=0） |
| avg_action_count.png | 每局出牌次数均值（不含流局） |
| avg_game_length.png | 每局游戏长度均值（不含流局） |
| avg_bomb_count.png | 每局炸弹/火箭数量均值 |
| overall_win_rates.png | 三个agent胜率总趋势 |
| overall_draw_rate.png | 流局率总趋势 |

### agent_x/ 文件夹图表

单个agent的所有强化学习阶段：

| 图表 | 说明 |
|------|------|
| bid_evolution.png | 叫分均值变化（蓝色=初始训练，红色=强化训练） |
| score_evolution.png | 得分变化 |
| win_rate_evolution.png | 胜率变化 |
| role_confusion_matrix.png | 地主/农民胜率混淆矩阵 |
| bid_distribution.png | 叫分分布变化堆叠图 |

## 独立调试优势

1. **训练失败不丢失可视化**：即使训练中断，已生成的日志仍可可视化
2. **多次可视化**：可以修改可视化代码后重新生成图表，无需重新训练
3. **对比分析**：可以加载不同训练的日志进行对比
4. **灵活配置**：可视化参数（图表样式、指标选择）可随时调整

## 算法参数

- 网络：256 → 256 → 128 (ReLU)
- 学习率：3e-4
- 折扣因子 γ：0.99
- GAE λ：0.95
- PPO clip：0.2
- 计算设备：CUDA

## 状态编码 (40维)

1. 我的手牌统计：15维
2. 已出牌统计：15维
3. 上家出牌：4维
4. 角色：2维
5. 位置：3维
6. 剩余牌数：1维
