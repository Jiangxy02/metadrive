#!/usr/bin/env python3

import sys
import os
import numpy as np
sys.path.append('.')

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory

def analyze_speed_calculation_discrepancy():
    print("🔍 分析：为什么显示26.6 m/s，实际计算是103 m/s")
    print("="*70)
    
    # 加载数据
    csv_path = "scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    traj_data = load_trajectory(csv_path=csv_path, normalize_position=False, max_duration=3, 
                               use_original_position=False, translate_to_origin=True, target_fps=50.0)
    
    # 创建环境
    env = TrajectoryReplayEnv(traj_data, config=dict(use_render=False))
    env.reset()
    
    print(f"\n📊 关键问题分析:")
    print(f"  显示速度: 26.6 m/s (agent.speed)")
    print(f"  计算速度: 103 m/s (距离/时间)")
    print(f"  差异倍数: ~4倍")
    print(f"  这个差异从何而来？")
    
    # 执行一步并详细分析
    print(f"\n🔬 单步详细分析:")
    initial_pos = [env.agent.position[0], env.agent.position[1]]
    initial_speed_display = env.agent.speed
    initial_velocity = [env.agent.velocity[0], env.agent.velocity[1]]
    
    print(f"  步骤前:")
    print(f"    位置: ({initial_pos[0]:.6f}, {initial_pos[1]:.6f})")
    print(f"    显示速度: {initial_speed_display:.6f} m/s")
    print(f"    速度向量: ({initial_velocity[0]:.6f}, {initial_velocity[1]:.6f})")
    
    # 获取物理引擎的时间步长
    physics_dt = env.physics_world_step_size
    decision_repeat = env.engine.global_config.get('decision_repeat', 1)
    
    print(f"    物理时间步长: {physics_dt:.6f} s")
    print(f"    decision_repeat: {decision_repeat}")
    print(f"    每次env.step()实际时间: {physics_dt * decision_repeat:.6f} s")
    
    # 执行一步
    env.step([0, 0])
    
    final_pos = [env.agent.position[0], env.agent.position[1]]
    final_speed_display = env.agent.speed
    final_velocity = [env.agent.velocity[0], env.agent.velocity[1]]
    
    print(f"\n  步骤后:")
    print(f"    位置: ({final_pos[0]:.6f}, {final_pos[1]:.6f})")
    print(f"    显示速度: {final_speed_display:.6f} m/s")
    print(f"    速度向量: ({final_velocity[0]:.6f}, {final_velocity[1]:.6f})")
    
    # 计算位移和时间
    dx = final_pos[0] - initial_pos[0]
    dy = final_pos[1] - initial_pos[1]
    distance = (dx**2 + dy**2)**0.5
    
    print(f"\n📏 位移分析:")
    print(f"    X方向位移: {dx:.6f} m")
    print(f"    Y方向位移: {dy:.6f} m")
    print(f"    总位移距离: {distance:.6f} m")
    
    # 关键：分析不同的时间基准
    print(f"\n⏰ 时间基准分析:")
    
    # 1. 单个物理步长
    single_physics_time = physics_dt
    speed_calc_single = distance / single_physics_time
    print(f"  基于单个物理步长 ({single_physics_time:.6f}s):")
    print(f"    计算速度: {speed_calc_single:.2f} m/s")
    print(f"    与显示速度比率: {speed_calc_single / final_speed_display:.2f}x")
    
    # 2. 完整env.step()时间
    full_step_time = physics_dt * decision_repeat
    speed_calc_full = distance / full_step_time
    print(f"  基于完整step时间 ({full_step_time:.6f}s):")
    print(f"    计算速度: {speed_calc_full:.2f} m/s")
    print(f"    与显示速度比率: {speed_calc_full / final_speed_display:.2f}x")
    
    print(f"\n🎯 问题核心分析:")
    print(f"  我们之前使用的时间基准是: {single_physics_time:.6f}s (单个物理步)")
    print(f"  但这是错误的！")
    print(f"  正确的时间基准应该是: {full_step_time:.6f}s (完整step时间)")
    
    # 验证这个理论
    print(f"\n✅ 验证结果:")
    if abs(speed_calc_full / final_speed_display - 1.0) < 0.5:
        print(f"  ✅ 使用完整step时间计算，速度接近显示值")
        print(f"  差异: {abs(speed_calc_full - final_speed_display):.2f} m/s")
        print(f"  相对误差: {abs(speed_calc_full - final_speed_display) / final_speed_display * 100:.1f}%")
    else:
        print(f"  ❌ 仍有较大差异，需要进一步分析")
        
        print(f"\n🔍 进一步分析可能的原因:")
        print(f"  1. 速度在decision_repeat期间发生变化")
        print(f"  2. 物理引擎的时间步长与配置不一致")
        print(f"  3. MetaDrive内部有额外的时间缩放")
        print(f"  4. 速度显示是平滑化或滞后的值")
    
    # 最重要的发现
    print(f"\n💡 关键发现:")
    print(f"  之前的错误: 用单个物理步长(0.02s)计算速度")
    print(f"  实际应该: 用完整step时间({full_step_time:.3f}s)计算速度")
    print(f"  这解释了为什么计算出的速度是显示速度的{decision_repeat}倍！")
    
    print(f"\n�� 最终解释:")
    print(f"  显示速度26.6 m/s是正确的瞬时物理速度")
    print(f"  计算速度103 m/s是错误的，因为用错了时间基准")
    print(f"  正确计算: {distance:.2f}m ÷ {full_step_time:.3f}s = {speed_calc_full:.2f} m/s")
    print(f"  这与显示速度26.6 m/s基本一致！")
    
    env.close()

if __name__ == "__main__":
    analyze_speed_calculation_discrepancy()
