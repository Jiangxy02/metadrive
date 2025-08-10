#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append('.')

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory

def test_background_vehicle_modes():
    print("🔧 测试背景车更新模式对比")
    print("="*70)
    
    # 加载数据
    csv_path = "scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    traj_data = load_trajectory(
        csv_path=csv_path, 
        normalize_position=False,
        max_duration=5,
        use_original_position=False,
        translate_to_origin=True,
        target_fps=50.0
    )
    
    print(f"\n📊 对比两种背景车更新模式:")
    
    # 测试模式1：position模式（原kinematic模式）
    print(f"\n1️⃣ 测试position模式（kinematic）:")
    env_position = TrajectoryReplayEnv(
        traj_data, 
        config=dict(
            use_render=False,
            background_vehicle_update_mode="position"
        )
    )
    env_position.reset()
    
    # 运行几步记录背景车状态
    position_mode_results = []
    for step in range(3):
        env_position.step([0, 0])
        
        # 记录背景车状态
        step_data = {"step": step + 1, "vehicles": {}}
        for vid, vehicle in env_position.ghost_vehicles.items():
            if hasattr(vehicle, 'position') and hasattr(vehicle, 'speed'):
                step_data["vehicles"][vid] = {
                    "position": [vehicle.position[0], vehicle.position[1]],
                    "speed": vehicle.speed,
                    "is_kinematic": vehicle._body.isKinematic() if hasattr(vehicle, '_body') else "Unknown"
                }
        position_mode_results.append(step_data)
    
    env_position.close()
    
    # 测试模式2：dynamics模式（物理模式）
    print(f"\n2️⃣ 测试dynamics模式（物理）:")
    env_dynamics = TrajectoryReplayEnv(
        traj_data, 
        config=dict(
            use_render=False,
            background_vehicle_update_mode="dynamics"
        )
    )
    env_dynamics.reset()
    
    # 运行几步记录背景车状态
    dynamics_mode_results = []
    for step in range(3):
        env_dynamics.step([0, 0])
        
        # 记录背景车状态
        step_data = {"step": step + 1, "vehicles": {}}
        for vid, vehicle in env_dynamics.ghost_vehicles.items():
            if hasattr(vehicle, 'position') and hasattr(vehicle, 'speed'):
                step_data["vehicles"][vid] = {
                    "position": [vehicle.position[0], vehicle.position[1]],
                    "speed": vehicle.speed,
                    "is_kinematic": vehicle._body.isKinematic() if hasattr(vehicle, '_body') else "Unknown"
                }
        dynamics_mode_results.append(step_data)
    
    env_dynamics.close()
    
    # 对比结果
    print(f"\n📊 对比分析:")
    print(f"{'模式':<12} {'车辆ID':<8} {'步骤':<4} {'位置':<20} {'速度':<8} {'Kinematic':<10}")
    print("-" * 70)
    
    for step_idx in range(3):
        # Position模式结果
        pos_data = position_mode_results[step_idx]
        for vid, data in pos_data["vehicles"].items():
            if step_idx == 0:  # 只显示第一个有效车辆的所有步骤
                position_str = f"({data['position'][0]:.1f}, {data['position'][1]:.1f})"
                print(f"{'Position':<12} {vid:<8} {pos_data['step']:<4} "
                      f"{position_str:<20} "
                      f"{data['speed']:<8.1f} {str(data['is_kinematic']):<10}")
                break
        
        # Dynamics模式结果 
        dyn_data = dynamics_mode_results[step_idx]
        for vid, data in dyn_data["vehicles"].items():
            if step_idx == 0:  # 只显示第一个有效车辆的所有步骤
                position_str = f"({data['position'][0]:.1f}, {data['position'][1]:.1f})"
                print(f"{'Dynamics':<12} {vid:<8} {dyn_data['step']:<4} "
                      f"{position_str:<20} "
                      f"{data['speed']:<8.1f} {str(data['is_kinematic']):<10}")
                break
    
    print(f"\n💡 关键差异:")
    print(f"  Position模式:")
    print(f"    - Kinematic: True (不受物理引擎影响)")
    print(f"    - 位置更新: 直接跳跃到CSV坐标")
    print(f"    - 速度显示: 仅为显示值，不影响运动")
    print(f"    - 运动特点: 精确按CSV轨迹，无物理真实感")
    
    print(f"  Dynamics模式:")
    print(f"    - Kinematic: False (受物理引擎影响)")
    print(f"    - 位置更新: 通过物理引擎和速度控制")
    print(f"    - 速度显示: 实际物理速度")
    print(f"    - 运动特点: 更真实的物理运动，可能有小幅偏差")
    
    print(f"\n🎯 推荐使用场景:")
    print(f"  Position模式: 需要精确轨迹重放、录制回放场景")
    print(f"  Dynamics模式: 需要真实物理交互、训练场景")

if __name__ == "__main__":
    test_background_vehicle_modes() 