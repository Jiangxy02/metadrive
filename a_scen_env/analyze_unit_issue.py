#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append('.')

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory

def analyze_unit_issue():
    print("🔍 分析单位和渲染差异问题")
    print("="*60)
    
    # 加载数据
    csv_path = "scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    traj_data = load_trajectory(csv_path=csv_path, normalize_position=False, max_duration=5, 
                               use_original_position=False, translate_to_origin=True, target_fps=50.0)
    
    # 创建环境
    env = TrajectoryReplayEnv(traj_data, config=dict(use_render=False))
    env.reset()
    
    print(f"\n🚗 主车分析:")
    print(f"  类型: {type(env.agent)}")
    print(f"  初始位置: ({env.agent.position[0]:.3f}, {env.agent.position[1]:.3f})")
    print(f"  初始速度: {env.agent.speed:.3f} m/s")
    print(f"  速度向量: ({env.agent.velocity[0]:.3f}, {env.agent.velocity[1]:.3f})")
    
    # 检查主车的物理属性
    if hasattr(env.agent, '_body') and env.agent._body:
        body = env.agent._body
        print(f"  物理体质量: {body.getMass():.3f}")
        linear_vel = body.getLinearVelocity()
        print(f"  物理引擎速度向量: ({linear_vel[0]:.3f}, {linear_vel[1]:.3f}, {linear_vel[2]:.3f})")
        print(f"  物理引擎速度大小: {(linear_vel[0]**2 + linear_vel[1]**2)**0.5:.3f}")
    
    # 运行一步看变化
    print(f"\n📊 单步运动分析:")
    initial_pos = [env.agent.position[0], env.agent.position[1]]
    
    env.step([0, 0])  # 零动作
    
    new_pos = [env.agent.position[0], env.agent.position[1]]
    distance_moved = ((new_pos[0] - initial_pos[0])**2 + (new_pos[1] - initial_pos[1])**2)**0.5
    
    print(f"  步骤后位置: ({new_pos[0]:.3f}, {new_pos[1]:.3f})")
    print(f"  位置变化: ({new_pos[0] - initial_pos[0]:.3f}, {new_pos[1] - initial_pos[1]:.3f})")
    print(f"  移动距离: {distance_moved:.3f} m")
    print(f"  步骤后速度: {env.agent.speed:.3f} m/s")
    print(f"  步骤后速度向量: ({env.agent.velocity[0]:.3f}, {env.agent.velocity[1]:.3f})")
    
    # 物理时间步长
    dt = env.physics_world_step_size
    print(f"  物理时间步长: {dt:.6f} s")
    print(f"  理论移动距离(基于显示速度): {env.agent.speed * dt:.3f} m")
    print(f"  实际/理论比率: {distance_moved / (env.agent.speed * dt):.3f}")
    
    print(f"\n🚙 背景车分析:")
    if env.ghost_vehicles:
        for vid, vehicle in list(env.ghost_vehicles.items())[:3]:  # 分析前3个背景车
            print(f"  背景车 {vid}:")
            print(f"    类型: {type(vehicle)}")
            print(f"    位置: ({vehicle.position[0]:.3f}, {vehicle.position[1]:.3f})")
            if hasattr(vehicle, 'speed'):
                print(f"    显示速度: {vehicle.speed:.3f} m/s")
            if hasattr(vehicle, 'velocity'):
                print(f"    速度向量: ({vehicle.velocity[0]:.3f}, {vehicle.velocity[1]:.3f})")
            
            # 检查背景车的物理属性
            if hasattr(vehicle, '_body') and vehicle._body:
                body = vehicle._body
                print(f"    物理体质量: {body.getMass():.3f}")
                linear_vel = body.getLinearVelocity()
                print(f"    物理引擎速度向量: ({linear_vel[0]:.3f}, {linear_vel[1]:.3f}, {linear_vel[2]:.3f})")
                print(f"    物理引擎速度大小: {(linear_vel[0]**2 + linear_vel[1]**2)**0.5:.3f}")
                
                # 检查kinematic状态
                try:
                    is_kinematic = body.isKinematic() if hasattr(body, 'isKinematic') else "未知"
                    print(f"    Kinematic模式: {is_kinematic}")
                except:
                    print(f"    Kinematic模式: 无法检测")
    
    print(f"\n🔍 关键差异分析:")
    print(f"  主车移动距离: {distance_moved:.3f} m")
    print(f"  主车显示速度: {env.agent.speed:.3f} m/s")
    print(f"  主车实际移动速度: {distance_moved / dt:.3f} m/s")
    
    if env.ghost_vehicles:
        first_bg = list(env.ghost_vehicles.values())[0]
        if hasattr(first_bg, 'speed'):
            bg_speed = first_bg.speed
            print(f"  背景车显示速度: {bg_speed:.3f} m/s")
            print(f"  速度比(主车实际/背景车显示): {(distance_moved / dt) / bg_speed:.3f}" if bg_speed > 0 else "  背景车速度为0")
    
    env.close()

if __name__ == "__main__":
    analyze_unit_issue()
