#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append('.')

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory

def test_speed_fix():
    print("🔧 测试速度修复效果")
    print("="*60)
    
    # 加载数据
    csv_path = "scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    traj_data = load_trajectory(csv_path=csv_path, normalize_position=False, max_duration=5, 
                               use_original_position=False, translate_to_origin=True, target_fps=50.0)
    
    # 创建环境但修改主车初始速度设置
    env = TrajectoryReplayEnv(traj_data, config=dict(use_render=False))
    
    # 手动修改主车速度设置
    def modified_reset():
        obs = env.reset()
        if env.main_vehicle_trajectory and len(env.main_vehicle_trajectory) > 0:
            initial_state = env.main_vehicle_trajectory[0]
            # 重新设置位置和朝向
            env.agent.set_position([initial_state["x"], initial_state["y"]])
            env.agent.set_heading_theta(initial_state["heading"])
            
            # 应用速度缩放因子
            direction = [np.cos(initial_state["heading"]), np.sin(initial_state["heading"])]
            speed_scale_factor = 0.18  # 缩放因子
            scaled_speed = initial_state["speed"] * speed_scale_factor
            env.agent.set_velocity(direction, scaled_speed)
            
            print(f"🔧 修复后的主车设置:")
            print(f"  原始速度: {initial_state['speed']:.2f} m/s")
            print(f"  缩放速度: {scaled_speed:.2f} m/s (缩放因子: {speed_scale_factor})")
            print(f"  实际速度: {env.agent.speed:.2f} m/s")
        return obs
    
    modified_reset()
    
    print(f"\n🏃 修复后的运动测试:")
    initial_pos = [env.agent.position[0], env.agent.position[1]]
    
    for step in range(5):
        env.step([0, 0])  # 零动作
        new_pos = [env.agent.position[0], env.agent.position[1]]
        distance = ((new_pos[0] - initial_pos[0])**2 + (new_pos[1] - initial_pos[1])**2)**0.5
        print(f"  步骤 {step+1}: 移动={distance:.3f}m, 当前速度={env.agent.speed:.2f}m/s")
        
        # 同时显示背景车
        if env.ghost_vehicles:
            first_bg_id = list(env.ghost_vehicles.keys())[0]
            bg_vehicle = env.ghost_vehicles[first_bg_id]
            if hasattr(bg_vehicle, 'position') and hasattr(bg_vehicle, 'speed'):
                bg_pos = bg_vehicle.position
                bg_speed = bg_vehicle.speed
                print(f"        背景车{first_bg_id}: 位置=({bg_pos[0]:.2f}, {bg_pos[1]:.2f}), 速度={bg_speed:.2f}m/s")
    
    # 计算理论值
    physics_dt = env.physics_world_step_size
    theoretical_distance_per_step = 22.0 * physics_dt
    print(f"\n📐 对比结果:")
    print(f"  理论每步移动(22m/s): {theoretical_distance_per_step:.4f}m")
    print(f"  实际总移动: {distance:.4f}m")
    print(f"  移动比率: {distance / (theoretical_distance_per_step * 5):.3f}")
    
    if abs(distance / (theoretical_distance_per_step * 5) - 1.0) < 0.5:
        print(f"  ✅ 速度匹配改善！")
    else:
        print(f"  ⚠️  仍需调整缩放因子")
    
    env.close()

if __name__ == "__main__":
    test_speed_fix()
