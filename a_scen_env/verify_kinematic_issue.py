#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append('.')

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory

def verify_kinematic_issue():
    print("🔍 验证Kinematic模式问题")
    print("="*60)
    
    # 加载数据
    csv_path = "scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    traj_data = load_trajectory(csv_path=csv_path, normalize_position=False, max_duration=5, 
                               use_original_position=False, translate_to_origin=True, target_fps=50.0)
    
    # 创建环境
    env = TrajectoryReplayEnv(traj_data, config=dict(use_render=False))
    env.reset()
    
    print(f"\n🚙 背景车运动分析:")
    
    # 找一个有速度的背景车
    bg_vehicle = None
    bg_id = None
    for vid, vehicle in env.ghost_vehicles.items():
        # 检查轨迹数据中的速度
        if vid in env.trajectory_dict and len(env.trajectory_dict[vid]) > 1:
            traj = env.trajectory_dict[vid]
            if traj[0]["speed"] > 1.0:  # 找一个有明显速度的背景车
                bg_vehicle = vehicle
                bg_id = vid
                print(f"选择背景车 {vid} 进行分析，CSV速度: {traj[0]['speed']:.2f} m/s")
                break
    
    if bg_vehicle is None:
        print("❌ 未找到合适的背景车进行分析")
        env.close()
        return
    
    # 记录初始状态
    initial_pos = [bg_vehicle.position[0], bg_vehicle.position[1]]
    traj = env.trajectory_dict[bg_id]
    print(f"初始位置: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f})")
    print(f"初始CSV数据: 位置=({traj[0]['x']:.3f}, {traj[0]['y']:.3f}), 速度={traj[0]['speed']:.3f}")
    
    # 运行几步观察背景车移动
    print(f"\n📊 背景车逐步运动分析:")
    dt = env.physics_world_step_size
    
    for step in range(5):
        env.step([0, 0])  # 主车零动作
        
        # 检查背景车位置变化
        if env._step_count < len(traj):
            current_csv = traj[env._step_count]
            new_pos = [bg_vehicle.position[0], bg_vehicle.position[1]]
            distance = ((new_pos[0] - initial_pos[0])**2 + (new_pos[1] - initial_pos[1])**2)**0.5
            
            print(f"  步骤 {step+1}:")
            print(f"    CSV期望位置: ({current_csv['x']:.3f}, {current_csv['y']:.3f})")
            print(f"    实际位置: ({new_pos[0]:.3f}, {new_pos[1]:.3f})")
            print(f"    累计移动: {distance:.3f} m")
            print(f"    CSV速度: {current_csv['speed']:.3f} m/s")
            print(f"    显示速度: {bg_vehicle.speed:.3f} m/s")
            print(f"    理论移动(CSV速度×时间): {current_csv['speed'] * dt * (step+1):.3f} m")
            print(f"    位置差异: {abs(new_pos[0] - current_csv['x']) + abs(new_pos[1] - current_csv['y']):.3f}")
    
    print(f"\n🔍 背景车运动模式分析:")
    print(f"  背景车使用kinematic模式: {bg_vehicle._body.isKinematic()}")
    print(f"  背景车实际移动方式: 直接set_position()到CSV坐标")
    print(f"  背景车速度显示方式: set_velocity()调用，但不影响实际运动")
    print(f"  背景车运动独立于物理引擎时间步长")
    
    print(f"\n🚗 主车vs背景车对比:")
    print(f"  主车: 物理引擎驱动，PPO控制，速度×时间=位移")
    print(f"  背景车: CSV数据驱动，kinematic模式，直接跳跃到目标位置")
    print(f"  视觉差异原因: 两种完全不同的运动机制！")
    
    env.close()

if __name__ == "__main__":
    verify_kinematic_issue()
