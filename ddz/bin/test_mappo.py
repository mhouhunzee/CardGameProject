"""
测试 MAPPO 训练流程
"""
import sys
import os

# 测试导入
print("Testing imports...")
from train_mappo import MAPPOTrainingPipeline
print("[OK] Imports successful")

# 创建训练管道
print("\nCreating training pipeline...")
pipeline = MAPPOTrainingPipeline()
print("[OK] Pipeline created")

# 测试一局游戏
print("\nTesting one game...")
result = pipeline.play_one_game([0], epsilon=1.0)  # 只训练agent 0，完全探索
print("[OK] Game completed")
print(f"  Winner: {result['winner']}")
print(f"  Rewards: {result['rewards']}")
print(f"  Is draw: {result['is_draw']}")

# 检查经验缓冲区
print("\nChecking rollout buffer...")
for i in range(3):
    buffer_size = len(pipeline.trainer.buffers[i])
    print(f"  Agent {i} buffer size: {buffer_size}")

# 测试策略更新（如果有经验）
if len(pipeline.trainer.buffers[0]) > 0:
    print("\nTesting policy update...")
    update_info = pipeline.trainer.update(0)
    print("[OK] Update completed")
    print(f"  Loss: {update_info.get('total_loss', 0):.4f}")
else:
    print("\nNo experience to update")

print("\n" + "="*60)
print("All tests passed!")
print("="*60)
