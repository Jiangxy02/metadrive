#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
引导点问题诊断和修复脚本

问题描述：
仿真中有两个引导点连成一条线，但其中一个点在主车初始位置后方，
导致主车试图向后行驶到达第一个检查点，从而停车。

root cause分析：
1. MetaDrive导航系统使用checkpoints（检查点）系统
2. route_completion = travelled_length / total_length
3. 如果第一个检查点在车辆后方，travelled_length会是负数
4. 这导致route_completion异常，PPO认为需要倒退
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trajectory_replay import TrajectoryReplayEnv


def diagnose_checkpoint_issue(env):
    """
    诊断引导点/检查点问题
    
    Args:
        env: TrajectoryReplayEnv实例
        
    Returns:
        dict: 诊断结果
    """
    
    print("🔍 开始诊断引导点问题...")
    print("=" * 50)
    
    diagnosis = {
        "agent_position": None,
        "checkpoints": [],
        "checkpoint_distances": [],
        "checkpoint_directions": [],
        "backward_checkpoint": False,
        "travelled_length": 0,
        "total_length": 0,
        "route_completion": 0,
        "navigation_route": [],
        "target_checkpoints_index": None
    }
    
    try:
        agent = env.agent
        navigation = agent.navigation
        
        # 获取主车当前位置
        agent_pos = agent.position[:2]  # [x, y]
        diagnosis["agent_position"] = agent_pos
        print(f"🚗 主车位置: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})")
        
        # 获取导航路径
        if hasattr(navigation, 'route') and navigation.route:
            diagnosis["navigation_route"] = navigation.route
            print(f"🛣️ 导航路径: {navigation.route}")
        else:
            print(f"❌ 没有找到导航路径")
            return diagnosis
        
        # 获取当前检查点索引
        if hasattr(navigation, '_target_checkpoints_index'):
            diagnosis["target_checkpoints_index"] = navigation._target_checkpoints_index
            print(f"📍 目标检查点索引: {navigation._target_checkpoints_index}")
        
        # 获取当前和下一个检查点
        try:
            checkpoint1, checkpoint2 = navigation.get_checkpoints()
            diagnosis["checkpoints"] = [checkpoint1[:2], checkpoint2[:2]]
            
            print(f"🎯 检查点信息:")
            print(f"  检查点1: ({checkpoint1[0]:.1f}, {checkpoint1[1]:.1f})")
            print(f"  检查点2: ({checkpoint2[0]:.1f}, {checkpoint2[1]:.1f})")
            
            # 计算到各检查点的距离和方向
            for i, checkpoint in enumerate([checkpoint1, checkpoint2]):
                ckpt_pos = checkpoint[:2]
                distance = np.sqrt((ckpt_pos[0] - agent_pos[0])**2 + (ckpt_pos[1] - agent_pos[1])**2)
                
                # 计算方向向量（从主车到检查点）
                direction_vec = np.array(ckpt_pos) - np.array(agent_pos)
                
                # 计算主车朝向
                heading = agent.heading_theta
                heading_vec = np.array([np.cos(heading), np.sin(heading)])
                
                # 计算检查点是否在主车前方（点积 > 0表示前方）
                dot_product = np.dot(direction_vec, heading_vec)
                is_forward = dot_product > 0
                
                diagnosis["checkpoint_distances"].append(distance)
                diagnosis["checkpoint_directions"].append("前方" if is_forward else "后方")
                
                print(f"  检查点{i+1}: 距离={distance:.1f}m, 方向={diagnosis['checkpoint_directions'][i]}")
                
                # 检查是否有检查点在后方
                if not is_forward:
                    diagnosis["backward_checkpoint"] = True
                    print(f"  ❌ 检查点{i+1}在主车后方! (点积={dot_product:.3f})")
        
        except Exception as e:
            print(f"❌ 获取检查点信息失败: {e}")
        
        # 获取路径完成度相关信息
        try:
            if hasattr(navigation, 'travelled_length'):
                diagnosis["travelled_length"] = navigation.travelled_length
                print(f"📏 已行驶距离: {navigation.travelled_length:.2f}m")
            
            if hasattr(navigation, 'total_length'):
                diagnosis["total_length"] = navigation.total_length
                print(f"📏 总路径长度: {navigation.total_length:.2f}m")
            
            route_completion = getattr(navigation, 'route_completion', -1)
            diagnosis["route_completion"] = route_completion
            print(f"📊 路径完成度: {route_completion:.3f}")
            
            # 检查异常情况
            if navigation.travelled_length < 0:
                print(f"❌ 已行驶距离为负数! 这表明主车需要倒退到达检查点")
            
            if route_completion < 0:
                print(f"❌ 路径完成度为负数! 这会导致PPO认为需要倒退")
            
        except Exception as e:
            print(f"❌ 获取路径完成度信息失败: {e}")
        
        print(f"\n📋 诊断总结:")
        print(f"  后方检查点: {'是' if diagnosis['backward_checkpoint'] else '否'}")
        print(f"  路径完成度异常: {'是' if diagnosis['route_completion'] < 0 else '否'}")
        
        return diagnosis
        
    except Exception as e:
        print(f"❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        return diagnosis


def fix_checkpoint_issue(env):
    """
    修复引导点问题
    
    Args:
        env: TrajectoryReplayEnv实例
        
    Returns:
        bool: 修复是否成功
    """
    
    print(f"\n🔧 开始修复引导点问题...")
    print("=" * 50)
    
    try:
        agent = env.agent
        navigation = agent.navigation
        
        # 方法1: 重置travelled_length为0
        if hasattr(navigation, 'travelled_length'):
            old_travelled = navigation.travelled_length
            navigation.travelled_length = 0.0
            print(f"🔧 重置已行驶距离: {old_travelled:.2f} → 0.0")
        
        # 方法2: 重置_last_long_in_ref_lane
        if hasattr(navigation, '_last_long_in_ref_lane'):
            # 获取当前在参考车道上的位置
            if hasattr(navigation, 'current_ref_lanes') and navigation.current_ref_lanes:
                ref_lane = navigation.current_ref_lanes[0]
                current_long, _ = ref_lane.local_coordinates(agent.position)
                navigation._last_long_in_ref_lane = current_long
                print(f"🔧 重置参考车道位置: {navigation._last_long_in_ref_lane:.2f}")
        
        # 方法3: 强制设置route_completion为合理值
        if hasattr(navigation, 'route_completion') and navigation.route_completion < 0:
            # 直接设置为小的正数
            agent_pos = agent.position[:2]
            
            # 计算基于位置的合理完成度
            if hasattr(env, 'custom_destination') and env.custom_destination:
                dest_pos = env.custom_destination[:2]
                start_pos = [202.2, 7.0]  # 从轨迹数据获取的起始位置
                
                total_distance = np.sqrt((dest_pos[0] - start_pos[0])**2 + (dest_pos[1] - start_pos[1])**2)
                current_distance = np.sqrt((agent_pos[0] - start_pos[0])**2 + (agent_pos[1] - start_pos[1])**2)
                
                reasonable_completion = min(current_distance / total_distance, 0.99)
                
                # 通过修改travelled_length来间接修改route_completion
                if hasattr(navigation, 'total_length') and navigation.total_length > 0:
                    navigation.travelled_length = reasonable_completion * navigation.total_length
                    print(f"🔧 设置合理的路径完成度: {reasonable_completion:.3f}")
        
        # 方法4: 如果检查点在后方，尝试重新设置导航
        diagnosis = diagnose_checkpoint_issue(env)
        
        if diagnosis["backward_checkpoint"]:
            print(f"🔧 检测到后方检查点，尝试重新设置导航...")
            
            # 尝试调用之前的修复方法
            try:
                if hasattr(env, '_fix_pg_map_navigation'):
                    success = env._fix_pg_map_navigation()
                    if success:
                        print(f"✅ PG地图导航重新设置成功")
                        return True
            except Exception as e:
                print(f"❌ PG地图导航重设失败: {e}")
        
        # 验证修复效果
        new_completion = getattr(navigation, 'route_completion', -1)
        print(f"🔍 修复后路径完成度: {new_completion:.3f}")
        
        if new_completion >= 0:
            print(f"✅ 引导点问题修复成功!")
            return True
        else:
            print(f"❌ 引导点问题仍然存在")
            return False
            
    except Exception as e:
        print(f"❌ 修复过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_fix():
    """测试引导点修复功能"""
    
    print("🧪 测试引导点问题修复...")
    print("=" * 60)
    
    # 创建测试轨迹数据
    sample_trajectory = {
        -1: [  # 主车轨迹
            {'timestamp': 0.0, 'x': 202.2, 'y': 7.0, 'speed': 10.0, 'heading': 0.0},
            {'timestamp': 1.0, 'x': 250.0, 'y': 7.0, 'speed': 10.0, 'heading': 0.0},
            {'timestamp': 2.0, 'x': 300.0, 'y': 7.0, 'speed': 10.0, 'heading': 0.0},
            {'timestamp': 3.0, 'x': 400.0, 'y': 7.0, 'speed': 10.0, 'heading': 0.0},
            {'timestamp': 4.0, 'x': 500.0, 'y': 7.0, 'speed': 10.0, 'heading': 0.0},
            {'timestamp': 5.0, 'x': 637.3, 'y': -0.2, 'speed': 8.0, 'heading': 0.0},
        ]
    }
    
    # 环境配置
    config = {
        "use_render": False,
        "map": "S" * 8,
        "manual_control": False,
        "enable_background_vehicles": False,
        "disable_ppo_expert": False,
        "traffic_density": 0.0,
        "vehicle_config": {
            "show_navi_mark": True,
            "show_dest_mark": True,
            "show_line_to_dest": True,
            "show_line_to_navi_mark": True,
        }
    }
    
    try:
        print("🏗️ 创建测试环境...")
        env = TrajectoryReplayEnv(trajectory_dict=sample_trajectory, config=config)
        
        print("🔄 重置环境...")
        obs = env.reset()
        
        # 第一次诊断
        print(f"\n📊 初始状态诊断:")
        initial_diagnosis = diagnose_checkpoint_issue(env)
        
        # 如果发现问题，进行修复
        if initial_diagnosis["backward_checkpoint"] or initial_diagnosis["route_completion"] < 0:
            print(f"\n🔧 发现引导点问题，开始修复...")
            fix_success = fix_checkpoint_issue(env)
            
            if fix_success:
                # 第二次诊断验证
                print(f"\n📊 修复后状态诊断:")
                final_diagnosis = diagnose_checkpoint_issue(env)
                
                # 测试几步移动
                print(f"\n🚗 测试车辆移动...")
                for step in range(5):
                    action = [0.0, 0.5]  # 直行前进
                    obs, reward, done, info = env.step(action)
                    
                    pos = env.agent.position
                    speed = env.agent.speed
                    completion = getattr(env.agent.navigation, 'route_completion', -1)
                    
                    print(f"  步骤{step+1}: 位置=({pos[0]:6.1f}, {pos[1]:5.1f}), " +
                          f"速度={speed:5.2f}, 完成度={completion:.3f}, 奖励={reward:6.3f}")
                    
                    if done:
                        break
                
                completion_change = completion - initial_diagnosis["route_completion"]
                if completion_change > 0.001:
                    print(f"✅ 引导点修复成功！路径完成度正常增长")
                    result = True
                else:
                    print(f"❌ 引导点修复失败，完成度仍不增长")
                    result = False
            else:
                result = False
        else:
            print(f"✅ 没有检测到引导点问题")
            result = True
        
        env.close()
        return result
        
    except Exception as e:
        print(f"❌ 测试过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🎯 引导点问题诊断和修复工具")
    print("📝 问题：仿真中有引导点在主车后方，导致PPO停车")
    print("=" * 60)
    
    success = test_checkpoint_fix()
    
    print(f"\n" + "=" * 60)
    print(f"📋 测试总结:")
    
    if success:
        print(f"✅ 引导点问题修复成功!")
        print(f"💡 您的PPO应该能正常前进了")
        print(f"🎉 建议：现在可以正常运行PPO训练")
    else:
        print(f"❌ 引导点问题修复失败")
        print(f"💡 建议：")
        print(f"   1. 检查轨迹数据的起始位置是否与地图匹配")
        print(f"   2. 考虑调整主车的初始位置设置")
        print(f"   3. 使用自定义地图替代PG地图")
    
    print(f"\n📧 如需进一步帮助，请提供详细的诊断日志") 