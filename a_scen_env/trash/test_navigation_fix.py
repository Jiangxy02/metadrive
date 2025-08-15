#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导航修复测试脚本
用于验证导航路径修复功能是否正常工作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trajectory_replay import TrajectoryReplayEnv
from fix_navigation_route import fix_navigation_for_env
import pandas as pd


def test_navigation_fix():
    """测试导航修复功能"""
    
    print("🧪 开始测试导航修复功能...")
    
    # 模拟轨迹数据（您可以用实际的CSV文件替换）
    sample_trajectory = {
        'main_vehicle': [
            {'timestamp': 0.0, 'x': 200.0, 'y': 7.0, 'speed': 10.0, 'heading': 0.0},
            {'timestamp': 1.0, 'x': 250.0, 'y': 7.0, 'speed': 10.0, 'heading': 0.0},
            {'timestamp': 2.0, 'x': 300.0, 'y': 7.0, 'speed': 10.0, 'heading': 0.0},
            {'timestamp': 3.0, 'x': 350.0, 'y': 7.0, 'speed': 10.0, 'heading': 0.0},
            {'timestamp': 4.0, 'x': 400.0, 'y': 7.0, 'speed': 10.0, 'heading': 0.0},
        ]
    }
    
    # 环境配置
    config = {
        "use_render": False,  # 不启用渲染以加快测试
        "manual_control": False,
        "enable_background_vehicles": False,  # 只关注主车
        "disable_ppo_expert": False,
        "map_config": {
            "type": "straight"  # 使用直路
        }
    }
    
    try:
        print("🏗️ 创建测试环境...")
        
        # 创建环境
        env = TrajectoryReplayEnv(
            trajectory_dict=sample_trajectory,
            config=config
        )
        
        print("✅ 环境创建成功")
        
        # 重置环境
        obs = env.reset()
        print("✅ 环境重置成功")
        
        # 检查导航状态
        print("\n📊 检查初始导航状态...")
        env._debug_navigation_info()
        
        # 手动调用修复功能
        print("\n🔧 手动测试导航修复...")
        success = fix_navigation_for_env(env)
        
        if success:
            print("\n✅ 导航修复测试成功!")
            
            # 再次检查导航状态
            print("\n📊 检查修复后导航状态...")
            env._debug_navigation_info()
            
            # 测试几步仿真
            print("\n🚗 测试几步仿真...")
            for step in range(5):
                action = [0.0, 0.5]  # 直行，轻微油门
                obs, reward, done, info = env.step(action)
                
                agent_pos = env.agent.position
                agent_speed = env.agent.speed
                route_completion = getattr(env.agent.navigation, 'route_completion', -1)
                
                print(f"  步骤 {step+1}: 位置=({agent_pos[0]:.1f}, {agent_pos[1]:.1f}), " +
                      f"速度={agent_speed:.2f}, 路径完成度={route_completion:.3f}, " +
                      f"奖励={reward:.3f}")
                
                if done:
                    print(f"  仿真结束: {info}")
                    break
            
            print("\n🎉 导航修复测试完成!")
            
        else:
            print("\n❌ 导航修复测试失败")
            
        env.close()
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
        return False
    
    return success


def test_with_real_data(csv_file_path):
    """使用真实数据测试导航修复"""
    
    if not os.path.exists(csv_file_path):
        print(f"❌ CSV文件不存在: {csv_file_path}")
        return False
    
    print(f"📂 使用真实数据测试: {csv_file_path}")
    
    try:
        # 加载CSV数据
        from trajectory_loader import load_trajectory
        trajectory_dict = load_trajectory(csv_file_path)
        
        print(f"✅ 轨迹数据加载成功，包含 {len(trajectory_dict)} 个车辆")
        
        # 环境配置
        config = {
            "use_render": False,
            "manual_control": False,
            "enable_background_vehicles": True,
            "disable_ppo_expert": False,
        }
        
        # 创建环境
        env = TrajectoryReplayEnv(
            trajectory_dict=trajectory_dict,
            config=config
        )
        
        # 重置环境
        obs = env.reset()
        
        # 检查和修复导航
        env._debug_navigation_info()
        
        # 测试仿真
        print("\n🚗 使用真实数据测试仿真...")
        for step in range(10):
            action = [0.0, 0.3]  # 保守的前进动作
            obs, reward, done, info = env.step(action)
            
            agent_pos = env.agent.position
            agent_speed = env.agent.speed
            route_completion = getattr(env.agent.navigation, 'route_completion', -1)
            
            print(f"  步骤 {step+1}: 位置=({agent_pos[0]:.1f}, {agent_pos[1]:.1f}), " +
                  f"速度={agent_speed:.2f}, 路径完成度={route_completion:.3f}")
            
            if done:
                print(f"  仿真结束: {info}")
                break
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ 真实数据测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 导航修复功能测试")
    print("=" * 50)
    
    # 基础测试
    print("\n1️⃣ 基础功能测试")
    success1 = test_navigation_fix()
    
    # 真实数据测试（如果有的话）
    print("\n2️⃣ 真实数据测试")
    
    # 您可以在这里指定您的CSV文件路径
    csv_files = [
        "../data/selected_scenarios/scenario_vehicles_9_0_1_2_3_4_5_6_7_8_test3_20250304_150943_row001_filtered.csv",
        "../data/selected_scenarios/scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    ]
    
    success2 = False
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"\n测试文件: {csv_file}")
            success2 = test_with_real_data(csv_file)
            break
    else:
        print("⚠️ 未找到真实数据文件，跳过真实数据测试")
    
    # 总结
    print("\n" + "=" * 50)
    print("📋 测试总结:")
    print(f"  基础功能测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"  真实数据测试: {'✅ 通过' if success2 else '⚠️ 跳过/失败'}")
    
    if success1:
        print("\n🎉 导航修复功能正常!")
        print("💡 您现在可以正常使用PPO训练了")
    else:
        print("\n😞 导航修复功能需要进一步调试")
        print("💡 请检查错误信息并联系开发者") 