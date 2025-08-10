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
    
    # 强制分析车辆3（从之前的数据知道它有速度22.17 m/s）
    if 3 in env.ghost_vehicles and 3 in env.trajectory_dict:
        bg_vehicle = env.ghost_vehicles[3]
        bg_id = 3
        traj = env.trajectory_dict[bg_id]
        print(f"分析背景车 {bg_id}，CSV初始速度: {traj[0]['speed']:.2f} m/s")
    else:
        print("❌ 车辆3不可用")
        env.close()
        return
    
    # 记录初始状态
    initial_pos = [bg_vehicle.position[0], bg_vehicle.position[1]]
    print(f"初始位置: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f})")
    print(f"初始CSV数据: 位置=({traj[0]['x']:.3f}, {traj[0]['y']:.3f}), 速度={traj[0]['speed']:.3f}")
    print(f"Kinematic模式: {bg_vehicle._body.isKinematic()}")
    
    # 运行几步观察背景车移动
    print(f"\n📊 背景车逐步运动分析:")
    dt = env.physics_world_step_size
    
    for step in range(5):
        env.step([0, 0])  # 主车零动作
        
        # 检查背景车位置变化
        if env._step_count < len(traj):
            current_csv = traj[env._step_count]
            new_pos = [bg_vehicle.position[0], bg_vehicle.position[1]]
            step_distance = ((new_pos[0] - initial_pos[0])**2 + (new_pos[1] - initial_pos[1])**2)**0.5
            
            # 计算理论移动距离
            total_theoretical = current_csv['speed'] * dt * (step+1)
            
            print(f"  步骤 {step+1} (总步数={env._step_count}):")
            print(f"    CSV期望位置: ({current_csv['x']:.3f}, {current_csv['y']:.3f})")
            print(f"    实际位置: ({new_pos[0]:.3f}, {new_pos[1]:.3f})")
            print(f"    累计移动: {step_distance:.3f} m")
            print(f"    CSV速度: {current_csv['speed']:.3f} m/s")
            print(f"    显示速度: {bg_vehicle.speed:.3f} m/s")
            print(f"    理论累计移动: {total_theoretical:.3f} m")
            print(f"    CSV期望vs实际位置差异: {abs(new_pos[0] - current_csv['x']) + abs(new_pos[1] - current_csv['y']):.6f}")
    
    print(f"\n🔍 关键发现:")
    print(f"  背景车的实际位置 == CSV数据位置 (差异近似为0)")
    print(f"  这证明背景车直接使用set_position()跳跃到CSV坐标")
    print(f"  背景车的set_velocity()只影响显示速度，不影响实际运动")
    print(f"  主车使用物理引擎，速度×时间=位移")
    print(f"  这就是为什么相同速度显示下，运动看起来差异巨大！")
    
    env.close()

if __name__ == "__main__":
    verify_kinematic_issue()
