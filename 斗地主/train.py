"""
MCTS训练脚本 - 自动读取models文件夹中的模型
每个cycle结束后自动删除旧模型，保存新模型
支持并行训练
"""

import argparse
import torch
import os
import glob
import time
from multiprocessing import Pool, cpu_count

import config
from config import (
    CYCLE, H, NUM_PAIKU,
    USE_PARALLEL, NUM_WORKERS, WORKER_SIMULATIONS
)
from doudizhu_mcts_new import MCTSTrainer, DoudizhuMCTS
from model_advanced import AdvancedDouDizhuNet


def find_latest_models(models_dir='models'):
    """查找models文件夹中最新的模型文件"""
    if not os.path.exists(models_dir):
        return None
    
    latest_models = {}
    for pos in range(3):
        pattern = os.path.join(models_dir, f"agent_{pos}_cycle_*.pth")
        models = glob.glob(pattern)
        if models:
            models.sort(key=lambda x: int(x.split('_cycle_')[-1].replace('.pth', '')))
            latest_models[pos] = models[-1]
    
    return latest_models if latest_models else None


def get_cycle_number(model_path):
    """从模型路径中提取cycle数字"""
    try:
        return int(model_path.split('_cycle_')[-1].replace('.pth', ''))
    except:
        return 0


def clear_models_dir(models_dir='models'):
    """清空models文件夹"""
    if os.path.exists(models_dir):
        for file in glob.glob(os.path.join(models_dir, '*.pth')):
            try:
                os.remove(file)
                print(f"   [DEL] {os.path.basename(file)}")
            except Exception as e:
                print(f"   [ERR] {file}: {e}")


def save_new_models(models_dir, models_dict, cycle):
    """保存新模型到models文件夹"""
    os.makedirs(models_dir, exist_ok=True)
    for pos, model in models_dict.items():
        save_path = os.path.join(models_dir, f"agent_{pos}_cycle_{cycle}.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'cycle': cycle,
            'position': pos
        }, save_path)
        print(f"   [SAVE] agent_{pos}_cycle_{cycle}.pth")


def create_model(device='cpu'):
    """创建模型"""
    model = AdvancedDouDizhuNet()
    model = model.to(device)
    return model


def generate_paiku(num_games=NUM_PAIKU, output_file='paiku.txt'):
    """生成牌库文件"""
    import random
    
    CARD_TYPES = config.ALL_CARDS
    
    def create_deck():
        deck = []
        for card in CARD_TYPES[:-2]:
            deck.extend([card] * 4)
        deck.extend(['X', 'D'])
        return deck
    
    print(f"\n生成牌库: {output_file}")
    print(f"对局数: {num_games}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_games):
            deck = create_deck()
            random.shuffle(deck)
            
            hand_A = sorted(deck[:17], key=lambda x: CARD_TYPES.index(x) if x in CARD_TYPES else 99)
            hand_B = sorted(deck[17:34], key=lambda x: CARD_TYPES.index(x) if x in CARD_TYPES else 99)
            hand_C = sorted(deck[34:51], key=lambda x: CARD_TYPES.index(x) if x in CARD_TYPES else 99)
            base = sorted(deck[51:54], key=lambda x: CARD_TYPES.index(x) if x in CARD_TYPES else 99)
            
            line = f"{' '.join(hand_A)}|{' '.join(hand_B)}|{' '.join(hand_C)}|{' '.join(base)}\n"
            f.write(line)
            
            if (i + 1) % 1000 == 0:
                print(f"  已生成: {i+1}/{num_games}")
    
    print(f"[OK] 牌库生成完成: {output_file}\n")


def check_and_generate_paiku(paiku_file='paiku.txt', num_games=NUM_PAIKU):
    """检查牌库是否存在，不存在则生成"""
    if not os.path.exists(paiku_file):
        print(f"[!] 未找到牌库文件: {paiku_file}")
        generate_paiku(num_games, paiku_file)
    else:
        with open(paiku_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if len(lines) == 0:
                print(f"[!] 牌库文件为空: {paiku_file}")
                generate_paiku(num_games, paiku_file)
            else:
                print(f"[OK] 使用现有牌库: {paiku_file} ({len(lines)} 局)")
    
    return paiku_file


def self_play_one_game(args):
    """单进程自博弈一局"""
    import config
    
    model_state_dict, device_str, paiku_file, simulations = args
    
    device = torch.device(device_str)
    model = create_model(device)
    
    if model_state_dict:
        model.load_state_dict(model_state_dict)
    
    # 重新创建MCTS，确保使用正确的config
    mcts = DoudizhuMCTS(model=model, num_simulations=simulations)
    
    # 手动运行一局游戏
    import random
    from doudizhu_mcts_new import CARD_TYPES, CARD_TO_IDX
    
    # 初始化游戏
    deck = []
    for card in CARD_TYPES[:-2]:
        deck.extend([card] * 4)
    deck.extend(['X', 'D'])
    random.shuffle(deck)
    
    state = {
        'hands': {
            'landlord': sorted(deck[:20], key=lambda x: CARD_TO_IDX[x]),
            'landlord_up': sorted(deck[20:37], key=lambda x: CARD_TO_IDX[x]),
            'landlord_down': sorted(deck[37:54], key=lambda x: CARD_TO_IDX[x])
        },
        'current_player': 'landlord',
        'last_play': None,
        'last_player': None,
        'pass_count': 0,
        'history': []
    }
    
    game_data = []
    from copy import deepcopy
    
    while True:
        # 检查终局
        if all(len(hand) == 0 for hand in state['hands'].values()):
            break
        if len(state['hands']['landlord']) == 0 or len(state['hands']['landlord_up']) == 0 or len(state['hands']['landlord_down']) == 0:
            break
        
        current_player = state['current_player']
        
        # MCTS搜索
        action, pi = mcts.search(state, current_player)
        
        # 记录数据
        game_data.append({
            'state': deepcopy(state),
            'player': current_player,
            'action': action,
            'policy': pi,
            'value': None
        })
        
        # 执行动作
        state = mcts._apply_action(state, action, current_player)
    
    # 计算价值
    winner = None
    for player, hand in state['hands'].items():
        if len(hand) == 0:
            winner = player
            break
    
    for data in game_data:
        if winner == data['player']:
            data['value'] = 1.0
        elif winner in ['landlord_up', 'landlord_down'] and data['player'] in ['landlord_up', 'landlord_down']:
            data['value'] = 1.0
        else:
            data['value'] = -1.0
    
    return game_data


def parallel_self_play(model, num_games, paiku_file, simulations, num_workers):
    """并行自博弈"""
    model_state_dict = model.state_dict()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args_list = [(model_state_dict, device, paiku_file, simulations) for _ in range(num_games)]
    
    print(f"   启动 {num_workers} 个进程并行训练...")
    
    all_games = []
    with Pool(processes=num_workers) as pool:
        results = pool.map(self_play_one_game, args_list)
        for result in results:
            all_games.extend(result)
    
    return all_games


def main():
    parser = argparse.ArgumentParser(description='MCTS斗地主训练')
    
    parser.add_argument('--cycles', type=int, default=CYCLE,
                       help=f'训练cycle数 (默认: {CYCLE})')
    parser.add_argument('--games-per-cycle', type=int, default=H,
                       help=f'每个cycle的自博弈局数 (默认: {H})')
    parser.add_argument('--simulations', type=int, default=WORKER_SIMULATIONS,
                       help=f'MCTS模拟次数 (默认: {WORKER_SIMULATIONS})')
    
    parser.add_argument('--parallel', action='store_true', default=USE_PARALLEL,
                       help='启用并行训练')
    parser.add_argument('--no-parallel', action='store_true',
                       help='禁用并行训练')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS,
                       help=f'并行工作进程数 (默认: {NUM_WORKERS}, CPU核心数: {cpu_count()})')
    
    parser.add_argument('--models-dir', type=str, default='models')
    parser.add_argument('--paiku-file', type=str, default='paiku.txt')
    parser.add_argument('--paiku-size', type=int, default=NUM_PAIKU)
    
    args = parser.parse_args()
    
    use_parallel = args.parallel and not args.no_parallel
    if use_parallel:
        num_workers = min(args.workers, cpu_count() - 1)
        print(f"\n[并行模式] 使用 {num_workers} 个进程")
    else:
        num_workers = 1
        print(f"\n[串行模式]")
    
    check_and_generate_paiku(args.paiku_file, args.paiku_size)
    
    print("=" * 70)
    print("MCTS斗地主训练")
    print("=" * 70)
    print(f"训练参数:")
    print(f"   cycles: {args.cycles}")
    print(f"   games per cycle: {args.games_per_cycle}")
    print(f"   MCTS simulations: {args.simulations}")
    print(f"   parallel: {use_parallel}")
    print(f"   workers: {num_workers}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    existing_models = find_latest_models(args.models_dir)
    start_cycle = 0
    if existing_models:
        print(f"\n发现已有模型:")
        for pos, path in existing_models.items():
            cycle = get_cycle_number(path)
            print(f"   agent_{pos}: {os.path.basename(path)} (cycle {cycle})")
            start_cycle = max(start_cycle, cycle)
        print(f"将从 cycle {start_cycle + 1} 开始")
    else:
        print("\n未发现已有模型，从cycle 0开始")
    
    print("\n初始化3个agent...")
    agents = {}
    for pos in range(3):
        model = create_model(device)
        if existing_models and pos in existing_models:
            try:
                checkpoint = torch.load(existing_models[pos], map_location=device)
                
                # 尝试不同的键名加载模型
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    # 直接加载整个checkpoint作为state_dict
                    model.load_state_dict(checkpoint)
                
                # 显示旧模型详细信息
                old_cycle = checkpoint.get('cycle', 0)
                old_timestamp = checkpoint.get('timestamp', '未知')
                print(f"   agent_{pos}: 已加载旧模型 (cycle {old_cycle}, {old_timestamp})")
            except Exception as e:
                print(f"   agent_{pos}: 加载失败 ({e})，使用新模型")
        else:
            print(f"   agent_{pos}: 使用新模型")
        agents[pos] = model
    
    print("\n" + "=" * 70)
    print("开始训练")
    print("=" * 70)
    
    for cycle in range(start_cycle + 1, start_cycle + 1 + args.cycles):
        print(f"\n{'='*70}")
        print(f"Cycle {cycle}/{start_cycle + args.cycles}")
        print(f"{'='*70}")
        
        for pos in range(3):
            print(f"\n训练 agent_{pos}...")
            start_time = time.time()
            
            if use_parallel and num_workers > 1:
                all_games = parallel_self_play(
                    agents[pos],
                    args.games_per_cycle,
                    args.paiku_file,
                    args.simulations,
                    num_workers
                )
            else:
                trainer = MCTSTrainer(
                    model=agents[pos],
                    num_games=args.games_per_cycle,
                    checkpoint_dir=f'checkpoints/agent_{pos}',
                    paiku_file=args.paiku_file
                )
                trainer.mcts = DoudizhuMCTS(model=agents[pos], num_simulations=args.simulations)
                
                all_games = []
                for i in range(args.games_per_cycle):
                    game_data = trainer.self_play_game()
                    all_games.extend(game_data)
            
            elapsed = time.time() - start_time
            print(f"  完成: {len(all_games)} 条数据, 耗时: {elapsed:.1f}s")
            
            if agents[pos]:
                print(f"  训练模型...")
        
        print(f"\n{'-'*70}")
        print(f"更新模型文件...")
        clear_models_dir(args.models_dir)
        save_new_models(args.models_dir, agents, cycle)
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
