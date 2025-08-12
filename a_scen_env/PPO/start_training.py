#!/usr/bin/env python3
"""
PPO训练启动脚本

快速启动PPO训练的便捷脚本
"""

import os
import sys

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_ppo import train_ppo


def main():
    """主函数"""
    print("=" * 60)
    print("PPO Training System for TrajectoryReplayEnv")
    print("=" * 60)
    
    # 询问训练参数
    print("\nTraining Configuration:")
    print("-" * 30)
    
    # 总训练步数
    try:
        total_steps = input("Total training steps (default: 500000): ").strip()
        total_steps = int(total_steps) if total_steps else 500000
    except ValueError:
        total_steps = 500000
        print(f"Using default: {total_steps}")
    
    # 并行环境数
    try:
        n_envs = input("Number of parallel environments (default: 4): ").strip()
        n_envs = int(n_envs) if n_envs else 4
    except ValueError:
        n_envs = 4
        print(f"Using default: {n_envs}")
    
    # Episode长度
    try:
        horizon = input("Maximum steps per episode (default: 2000): ").strip()
        horizon = int(horizon) if horizon else 2000
    except ValueError:
        horizon = 2000
        print(f"Using default: {horizon}")
    
    # 地图长度
    try:
        map_length = input("Map length (number of straight segments, default: 30): ").strip()
        map_length = int(map_length) if map_length else 30
    except ValueError:
        map_length = 30
        print(f"Using default: {map_length}")
    
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    
    # 训练配置
    training_config = {
        "horizon": horizon,
        "end_on_crash": True,
        "end_on_out_of_road": True,
        "end_on_arrive_dest": False,
        "end_on_horizon": True,
        "background_vehicle_update_mode": "position",
        "map": "S" * map_length,
        "traffic_density": 0.0,
        "use_render": False,  # 训练时禁用渲染
        "manual_control": False,
        "enable_realtime": False,
    }
    
    # 开始训练
    try:
        model, model_dir = train_ppo(
            total_timesteps=total_steps,
            n_envs=n_envs,
            config=training_config
        )
        
        print("\n" + "=" * 60)
        print("Training Completed Successfully!")
        print("=" * 60)
        print(f"Model saved to: {model_dir}")
        print("\nNext steps:")
        print("1. Evaluate the model: python evaluate_ppo.py")
        print("2. Use in trajectory_replay.py (will auto-load latest model)")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 