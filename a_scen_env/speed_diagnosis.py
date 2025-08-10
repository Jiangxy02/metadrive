#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append('.')

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory
import time

def diagnose_speed_issue():
    print("🔍 MetaDrive速度诊断工具")
    print("="*60)
    
    # 加载数据
    csv_path = "scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    traj_data = load_trajectory(
        csv_path=csv_path, 
        normalize_position=False,
        max_duration=10,  # 只测试前10秒
        use_original_position=False,
        translate_to_origin=True,
        target_fps=50.0
    )
    
    # 创建环境
    env = TrajectoryReplayEnv(traj_data, config=dict(use_render=False))
    env.reset()
    
    print(f"\n📊 初始状态对比:")
    print(f"主车初始状态:")
    print(f"  位置: ({env.agent.position[0]:.2f}, {env.agent.position[1]:.2f})")
    print(f"  速度: {env.agent.speed:.2f} m/s")
    print(f"  速度向量: ({env.agent.velocity[0]:.2f}, {env.agent.velocity[1]:.2f})")
    
    # 检查物理属性
    if hasattr(env.agent, '_body') and env.agent._body:
        mass = env.agent._body.getMass()
        print(f"  主车质量: {mass:.2f}")
        
    # 运行几步并监控位置变化
    print(f"\n🏃 运动测试 (5步):")
    initial_pos = [env.agent.position[0], env.agent.position[1]]
    
    for step in range(5):
        env.step([0, 0])  # 空动作
        new_pos = [env.agent.position[0], env.agent.position[1]]
        distance = ((new_pos[0] - initial_pos[0])**2 + (new_pos[1] - initial_pos[1])**2)**0.5
        print(f"  步骤 {step+1}: 位置=({new_pos[0]:.2f}, {new_pos[1]:.2f}), 累计移动={distance:.2f}m, 当前速度={env.agent.speed:.2f}m/s")
        
        # 检查第一个背景车
        if env.ghost_vehicles:
            first_bg_id = list(env.ghost_vehicles.keys())[0]
            bg_vehicle = env.ghost_vehicles[first_bg_id]
            if hasattr(bg_vehicle, 'position') and hasattr(bg_vehicle, 'speed'):
                bg_pos = bg_vehicle.position
                bg_speed = bg_vehicle.speed
                print(f"        背景车{first_bg_id}: 位置=({bg_pos[0]:.2f}, {bg_pos[1]:.2f}), 速度={bg_speed:.2f}m/s")
                
                # 检查背景车物理属性
                if step == 0 and hasattr(bg_vehicle, '_body') and bg_vehicle._body:
                    bg_mass = bg_vehicle._body.getMass()
                    print(f"        背景车质量: {bg_mass:.2f}")
    
    # 计算每步的理论移动距离
    physics_dt = env.physics_world_step_size
    theoretical_distance_per_step = 22.0 * physics_dt  # 假设22m/s速度
    print(f"\n📐 理论计算:")
    print(f"  物理时间步长: {physics_dt:.6f}s")
    print(f"  理论每步移动距离(22m/s): {theoretical_distance_per_step:.4f}m")
    print(f"  5步理论总移动: {theoretical_distance_per_step * 5:.4f}m")
    print(f"  实际总移动: {distance:.4f}m")
    print(f"  移动比率: {distance / (theoretical_distance_per_step * 5):.3f}")
    
    env.close()

if __name__ == "__main__":
    diagnose_speed_issue()
