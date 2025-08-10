#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append('.')

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory

def analyze_decision_repeat_effect():
    print("🔍 重新分析：decision_repeat应该影响距离而不是速度")
    print("="*70)
    
    # 加载数据
    csv_path = "scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    traj_data = load_trajectory(csv_path=csv_path, normalize_position=False, max_duration=3, 
                               use_original_position=False, translate_to_origin=True, target_fps=50.0)
    
    # 创建环境
    env = TrajectoryReplayEnv(traj_data, config=dict(use_render=False))
    env.reset()
    
    print(f"\n📋 理论分析:")
    print(f"  decision_repeat = 5")
    print(f"  物理步长 = 0.02s")
    print(f"  每次env.step()实际时间 = 0.02s × 5 = 0.1s")
    print(f"  但这应该影响移动距离，而不是速度本身")
    
    print(f"\n🧮 数学期望:")
    print(f"  如果速度是26.6 m/s")
    print(f"  每次step()理论移动距离 = 26.6 × 0.1s = 2.66m")
    print(f"  但我们观察到的距离约是2.19m")
    print(f"  说明还有其他因素在起作用")
    
    print(f"\n🔬 深度检查decision_repeat的实际效果:")
    
    # 检查MetaDrive内部时间
    initial_time = env.engine.episode_step
    initial_physics_time = getattr(env.engine, 'global_time', 0)
    
    print(f"  初始episode_step: {initial_time}")
    print(f"  初始物理时间: {initial_physics_time}")
    
    # 执行一步
    initial_pos = [env.agent.position[0], env.agent.position[1]]
    initial_speed = env.agent.speed
    
    print(f"\n📊 执行一步的详细分析:")
    print(f"  步骤前位置: ({initial_pos[0]:.6f}, {initial_pos[1]:.6f})")
    print(f"  步骤前速度: {initial_speed:.6f} m/s")
    
    env.step([0, 0])  # 零动作
    
    final_pos = [env.agent.position[0], env.agent.position[1]]
    final_speed = env.agent.speed
    final_time = env.engine.episode_step
    final_physics_time = getattr(env.engine, 'global_time', 0)
    
    print(f"  步骤后位置: ({final_pos[0]:.6f}, {final_pos[1]:.6f})")
    print(f"  步骤后速度: {final_speed:.6f} m/s")
    print(f"  episode_step变化: {initial_time} → {final_time} (增加了{final_time - initial_time})")
    print(f"  物理时间变化: {initial_physics_time} → {final_physics_time}")
    
    # 计算实际移动
    distance_moved = ((final_pos[0] - initial_pos[0])**2 + (final_pos[1] - initial_pos[1])**2)**0.5
    print(f"  实际移动距离: {distance_moved:.6f} m")
    
    # 计算各种时间基准下的期望距离
    single_step_expected = initial_speed * 0.02  # 单个物理步
    five_step_expected = initial_speed * 0.1      # 5个物理步
    
    print(f"\n🔍 距离对比分析:")
    print(f"  基于单步物理时间的期望距离: {single_step_expected:.6f} m")
    print(f"  基于5步物理时间的期望距离: {five_step_expected:.6f} m")
    print(f"  实际移动距离: {distance_moved:.6f} m")
    print(f"  实际/单步期望比率: {distance_moved / single_step_expected:.3f}")
    print(f"  实际/五步期望比率: {distance_moved / five_step_expected:.3f}")
    
    print(f"\n🤔 问题分析:")
    if abs(distance_moved / five_step_expected - 1.0) < 0.1:
        print(f"  ✅ 实际距离接近5步期望，decision_repeat解释正确")
    else:
        print(f"  ❌ 实际距离不符合5步期望，还有其他因素")
        
        print(f"\n🔍 可能的其他因素:")
        print(f"  1. PPO控制器修改了速度")
        print(f"  2. 物理引擎的加速度/减速度效应")
        print(f"  3. 初始速度设置的累积效应")
        print(f"  4. MetaDrive内部的速度平滑或限制")
    
    # 检查PPO是否真的影响了速度
    print(f"\n🎯 速度变化分析:")
    speed_change = final_speed - initial_speed
    print(f"  速度变化: {initial_speed:.3f} → {final_speed:.3f} (变化: {speed_change:.3f} m/s)")
    
    if abs(speed_change) > 0.1:
        print(f"  ⚠️  PPO确实改变了速度！这解释了距离差异")
        print(f"  实际情况: 速度在decision_repeat期间发生了变化")
        print(f"  距离计算: 需要考虑变化的速度，而不是恒定速度")
    else:
        print(f"  ✅ 速度基本不变，距离差异可能确实来自decision_repeat")
    
    print(f"\n💡 结论:")
    print(f"  你的观点是正确的：decision_repeat理论上应该只影响距离")
    print(f"  但实际情况是：PPO在这些重复的物理步中改变了速度")
    print(f"  所以最终的距离差异来自：")
    print(f"    1. decision_repeat增加了时间")
    print(f"    2. PPO在这段时间内改变了速度")
    print(f"    3. 最终距离 = 变化的速度 × 延长的时间")
    
    env.close()

if __name__ == "__main__":
    analyze_decision_repeat_effect()
