#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append('.')

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory

def final_diagnosis():
    print("🎯 最终诊断：速度显示vs实际运动差异")
    print("="*60)
    
    # 加载数据
    csv_path = "scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    traj_data = load_trajectory(csv_path=csv_path, normalize_position=False, max_duration=5, 
                               use_original_position=False, translate_to_origin=True, target_fps=50.0)
    
    # 创建环境
    env = TrajectoryReplayEnv(traj_data, config=dict(use_render=False))
    env.reset()
    
    print(f"\n📋 问题总结:")
    print(f"用户观察: '主车显示速度20多，但比背景车的20多速度快很多倍'")
    
    print(f"\n🔍 技术原因分析:")
    
    # 分析主车
    print(f"1️⃣ 主车 (Vehicle -1):")
    print(f"   - 类型: 正常物理车辆 (kinematic=False)")
    print(f"   - 运动方式: 物理引擎驱动")
    print(f"   - 速度计算: 实时物理速度")
    print(f"   - 初始速度设置: {27.71:.1f} m/s")
    print(f"   - 受PPO控制影响: 是")
    
    # 运行一步分析主车
    env.step([0, 0])
    main_distance = ((env.agent.position[0] - 200)**2 + (env.agent.position[1] - 7)**2)**0.5
    print(f"   - 一步移动距离: {main_distance:.3f} m")
    print(f"   - 实际移动速度: {main_distance / 0.02:.1f} m/s")
    print(f"   - 显示速度: {env.agent.speed:.1f} m/s")
    
    # 分析背景车
    print(f"\n2️⃣ 背景车:")
    print(f"   - 类型: Kinematic车辆 (kinematic=True)")
    print(f"   - 运动方式: 直接位置跳跃")
    print(f"   - 速度计算: set_velocity()设置值")
    print(f"   - 实际运动: 完全按CSV数据位置")
    
    # 查看轨迹数据
    for vid in [3, 5]:
        if vid in env.trajectory_dict:
            traj = env.trajectory_dict[vid]
            if len(traj) > 1:
                csv_speed = traj[0]["speed"]
                pos1 = [traj[0]["x"], traj[0]["y"]]
                pos2 = [traj[1]["x"], traj[1]["y"]]
                csv_distance = ((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)**0.5
                csv_actual_speed = csv_distance / 0.02
                
                print(f"   - 车辆{vid} CSV速度: {csv_speed:.1f} m/s")
                print(f"   - 车辆{vid} CSV位置移动速度: {csv_actual_speed:.1f} m/s")
    
    print(f"\n🎯 根本问题:")
    print(f"   ❌ 不是单位问题")
    print(f"   ❌ 不是CARLA vs MetaDrive差异")
    print(f"   ✅ 是两种完全不同的运动机制:")
    print(f"      - 主车: 物理引擎 + PPO控制 = 复杂运动")
    print(f"      - 背景车: Kinematic模式 = CSV数据跳跃")
    
    print(f"\n💡 视觉效果差异原因:")
    print(f"   1. 主车的'速度显示'是物理引擎实时计算")
    print(f"   2. 背景车的'速度显示'是CSV数据中的记录值")
    print(f"   3. 主车实际受PPO动作和物理引擎双重影响")
    print(f"   4. 背景车实际运动完全独立于速度显示")
    print(f"   5. 两者的'速度'概念完全不同！")
    
    print(f"\n🔧 解决方案:")
    print(f"   方案1: 调整主车初始速度缩放因子")
    print(f"   方案2: 修改背景车为非Kinematic模式")
    print(f"   方案3: 统一运动机制（推荐）")
    
    env.close()

if __name__ == "__main__":
    final_diagnosis()
