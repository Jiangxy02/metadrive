#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append('.')

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory

def verify_ppo_effect():
    print("🔍 验证PPO对主车速度的影响")
    print("="*60)
    
    # 加载数据
    csv_path = "scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    traj_data = load_trajectory(csv_path=csv_path, normalize_position=False, max_duration=5, 
                               use_original_position=False, translate_to_origin=True, target_fps=50.0)
    
    # 创建环境
    env = TrajectoryReplayEnv(traj_data, config=dict(use_render=False))
    env.reset()
    
    print(f"\n实验1: 使用零动作 [0, 0]")
    initial_pos = [env.agent.position[0], env.agent.position[1]]
    
    for step in range(3):
        obs, reward, done, info = env.step([0, 0])  # 零动作
        new_pos = [env.agent.position[0], env.agent.position[1]]
        distance = ((new_pos[0] - initial_pos[0])**2 + (new_pos[1] - initial_pos[1])**2)**0.5
        print(f"  步骤 {step+1}: 动作=[0,0], 移动={distance:.3f}m, 速度={env.agent.speed:.2f}m/s")
    
    # 重置环境测试负动作
    env.reset()
    print(f"\n实验2: 使用刹车动作 [0, -1]")
    initial_pos = [env.agent.position[0], env.agent.position[1]]
    
    for step in range(3):
        obs, reward, done, info = env.step([0, -1])  # 刹车动作
        new_pos = [env.agent.position[0], env.agent.position[1]]
        distance = ((new_pos[0] - initial_pos[0])**2 + (new_pos[1] - initial_pos[1])**2)**0.5
        print(f"  步骤 {step+1}: 动作=[0,-1], 移动={distance:.3f}m, 速度={env.agent.speed:.2f}m/s")
    
    # 重置环境测试加速动作
    env.reset()
    print(f"\n实验3: 使用加速动作 [0, 1]")
    initial_pos = [env.agent.position[0], env.agent.position[1]]
    
    for step in range(3):
        obs, reward, done, info = env.step([0, 1])  # 加速动作
        new_pos = [env.agent.position[0], env.agent.position[1]]
        distance = ((new_pos[0] - initial_pos[0])**2 + (new_pos[1] - initial_pos[1])**2)**0.5
        print(f"  步骤 {step+1}: 动作=[0,1], 移动={distance:.3f}m, 速度={env.agent.speed:.2f}m/s")
    
    env.close()

if __name__ == "__main__":
    verify_ppo_effect()
