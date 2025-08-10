#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append('.')

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory

def investigate_speed_discrepancy():
    print("🔍 深度分析：主车速度显示vs实际移动的差异")
    print("="*70)
    
    # 加载数据
    csv_path = "scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    traj_data = load_trajectory(csv_path=csv_path, normalize_position=False, max_duration=3, 
                               use_original_position=False, translate_to_origin=True, target_fps=50.0)
    
    # 创建环境
    env = TrajectoryReplayEnv(traj_data, config=dict(use_render=False))
    env.reset()
    
    print(f"\n📊 初始状态详细分析:")
    print(f"  主车位置: ({env.agent.position[0]:.6f}, {env.agent.position[1]:.6f})")
    print(f"  主车速度显示: {env.agent.speed:.6f} m/s")
    print(f"  主车速度向量: ({env.agent.velocity[0]:.6f}, {env.agent.velocity[1]:.6f})")
    
    # 检查物理引擎的具体数值
    if hasattr(env.agent, '_body') and env.agent._body:
        body = env.agent._body
        linear_vel = body.getLinearVelocity()
        print(f"  物理引擎线性速度: ({linear_vel[0]:.6f}, {linear_vel[1]:.6f}, {linear_vel[2]:.6f})")
        print(f"  物理引擎速度大小: {(linear_vel[0]**2 + linear_vel[1]**2)**0.5:.6f}")
    
    dt = env.physics_world_step_size
    print(f"  物理时间步长: {dt:.6f} 秒")
    
    print(f"\n🏃 逐步运动分析:")
    initial_pos = [env.agent.position[0], env.agent.position[1]]
    
    for step in range(3):
        print(f"\n  --- 步骤 {step+1} ---")
        print(f"  步骤前位置: ({env.agent.position[0]:.6f}, {env.agent.position[1]:.6f})")
        print(f"  步骤前速度显示: {env.agent.speed:.6f} m/s")
        print(f"  步骤前速度向量: ({env.agent.velocity[0]:.6f}, {env.agent.velocity[1]:.6f})")
        
        # 执行一步
        obs, reward, done, info = env.step([0, 0])  # 零动作
        
        print(f"  步骤后位置: ({env.agent.position[0]:.6f}, {env.agent.position[1]:.6f})")
        print(f"  步骤后速度显示: {env.agent.speed:.6f} m/s")
        print(f"  步骤后速度向量: ({env.agent.velocity[0]:.6f}, {env.agent.velocity[1]:.6f})")
        
        # 计算位置变化
        new_pos = [env.agent.position[0], env.agent.position[1]]
        dx = new_pos[0] - initial_pos[0]
        dy = new_pos[1] - initial_pos[1]
        step_distance = (dx**2 + dy**2)**0.5
        step_speed = step_distance / dt
        
        print(f"  位置变化: dx={dx:.6f}, dy={dy:.6f}")
        print(f"  此步移动距离: {step_distance:.6f} m")
        print(f"  此步计算速度: {step_speed:.6f} m/s")
        print(f"  速度差异倍数: {step_speed / env.agent.speed:.3f}x" if env.agent.speed > 0 else "  无法计算倍数")
        
        # 理论vs实际
        theoretical_distance = env.agent.speed * dt
        print(f"  理论移动距离: {theoretical_distance:.6f} m")
        print(f"  实际/理论比率: {step_distance / theoretical_distance:.3f}" if theoretical_distance > 0 else "  无法计算比率")
    
    print(f"\n🔬 问题深度分析:")
    
    # 检查MetaDrive的速度计算方法
    print(f"1️⃣ MetaDrive速度计算方法:")
    if hasattr(env.agent, '_body') and env.agent._body:
        body = env.agent._body
        linear_vel = body.getLinearVelocity()
        computed_speed = (linear_vel[0]**2 + linear_vel[1]**2)**0.5
        print(f"   物理引擎速度向量: ({linear_vel[0]:.3f}, {linear_vel[1]:.3f})")
        print(f"   计算得出的速度: {computed_speed:.3f} m/s")
        print(f"   agent.speed属性: {env.agent.speed:.3f} m/s")
        print(f"   两者是否一致: {'✅ 是' if abs(computed_speed - env.agent.speed) < 0.001 else '❌ 否'}")
    
    print(f"\n2️⃣ 可能的原因分析:")
    print(f"   理论A: 时间步长不一致 - dt={dt:.6f}s")
    print(f"   理论B: PPO动作叠加效应")
    print(f"   理论C: 物理引擎内部缩放")
    print(f"   理论D: MetaDrive坐标系统问题")
    
    print(f"\n3️⃣ 关键发现:")
    total_distance = ((env.agent.position[0] - 200)**2 + (env.agent.position[1] - 7)**2)**0.5
    total_time = 3 * dt
    average_speed = total_distance / total_time
    print(f"   3步总移动距离: {total_distance:.3f} m")
    print(f"   3步总时间: {total_time:.6f} s")
    print(f"   平均实际速度: {average_speed:.3f} m/s")
    print(f"   显示速度平均: ~26.6 m/s")
    print(f"   实际/显示比率: {average_speed / 26.6:.3f}x")
    
    print(f"\n💡 可能的解释:")
    print(f"   1. PPO控制器可能在每个时间步内多次更新物理状态")
    print(f"   2. MetaDrive的decision_repeat可能>1，导致实际物理步数更多")
    print(f"   3. 物理引擎的内部时间步可能与显示的不一致")
    print(f"   4. speed属性可能是平滑化或滞后的显示值")
    
    # 检查MetaDrive的配置
    print(f"\n⚙️ MetaDrive配置检查:")
    try:
        config = env.engine.global_config
        print(f"   decision_repeat: {config.get('decision_repeat', '未设置')}")
        print(f"   physics_world_step_size: {config.get('physics_world_step_size', '未设置')}")
        print(f"   其他时间相关配置: {[k for k in config.keys() if 'time' in k.lower() or 'step' in k.lower() or 'dt' in k.lower()]}")
    except:
        print(f"   无法获取配置信息")
    
    env.close()

if __name__ == "__main__":
    investigate_speed_discrepancy()
