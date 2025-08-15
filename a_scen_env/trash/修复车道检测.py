#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复车道检测问题 - 根据主车实际位置重新检测正确的当前车道
"""

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def find_next_lane(current_lane, road_network):
    """寻找当前车道的下一个车道"""
    
    try:
        current_index = current_lane.index
        
        # 解析当前车道索引
        if len(current_index) >= 3:
            current_start = current_index[0]
            current_end = current_index[1]
            lane_idx = current_index[2]
            
            # 在道路网络中寻找以current_end为起点的车道
            if current_end in road_network.graph:
                for next_end in road_network.graph[current_end].keys():
                    lanes = road_network.graph[current_end][next_end]
                    
                    # 处理不同的lanes数据结构
                    if hasattr(lanes, 'items'):
                        lane_items = lanes.items()
                    elif isinstance(lanes, (list, tuple)):
                        lane_items = enumerate(lanes)
                    else:
                        continue
                    
                    for next_lane_idx, next_lane in lane_items:
                        if next_lane and next_lane_idx == lane_idx:  # 保持相同的车道编号
                            return next_lane
                            
                    # 如果没有找到相同编号的车道，返回第一个可用车道
                    for next_lane_idx, next_lane in lane_items:
                        if next_lane:
                            return next_lane
        
        return None
        
    except Exception as e:
        print(f"⚠️ 查找下一个车道失败: {e}")
        return None


def fix_lane_detection(env):
    """修复车道检测问题"""
    
    print("🔧 开始修复车道检测...")
    
    try:
        agent = env.agent
        navigation = agent.navigation
        current_map = env.engine.current_map
        road_network = current_map.road_network
        agent_pos = agent.position
        
        print(f"📍 主车实际位置: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})")
        print(f"❌ 错误检测车道: {navigation.current_lane.index}")
        
        # 找到主车真正所在的车道
        best_lane = None
        min_distance = float('inf')
        
        for road_start in road_network.graph.keys():
            for road_end in road_network.graph[road_start].keys():
                lanes = road_network.graph[road_start][road_end]
                
                # 处理不同的lanes数据结构
                if hasattr(lanes, 'items'):
                    lane_items = lanes.items()
                elif isinstance(lanes, (list, tuple)):
                    lane_items = enumerate(lanes)
                else:
                    continue
                
                for lane_idx, lane in lane_items:
                    if lane:
                        try:
                            # 计算主车在此车道上的位置
                            local_coords = lane.local_coordinates(agent_pos)
                            longitudinal = local_coords[0]
                            lateral = local_coords[1]
                            
                            # 检查主车是否在此车道上
                            is_on_lane = (0 <= longitudinal <= lane.length) and (abs(lateral) < 5)
                            
                            if is_on_lane:
                                # 计算距离车道中心的距离作为优先级
                                distance = abs(lateral)
                                if distance < min_distance:
                                    min_distance = distance
                                    best_lane = lane
                                    
                        except Exception as e:
                            continue
        
        if best_lane:
            print(f"✅ 找到正确车道: {best_lane.index}")
            print(f"🎯 车道位置: ({best_lane.position(0, 0)[0]:.1f}, {best_lane.position(0, 0)[1]:.1f}) → ({best_lane.position(best_lane.length, 0)[0]:.1f}, {best_lane.position(best_lane.length, 0)[1]:.1f})")
            
            # 1. 强制更新当前车道
            navigation._current_lane = best_lane
            
            # 2. 更新参考车道 - 这是检查点计算的基础
            navigation.current_ref_lanes = [best_lane]
            print(f"✅ 更新当前参考车道: {best_lane.index}")
            
            # 3. 寻找下一个车道作为next_ref_lanes
            next_lane = find_next_lane(best_lane, road_network)
            if next_lane:
                navigation.next_ref_lanes = [next_lane]
                print(f"✅ 更新下一个参考车道: {next_lane.index}")
            else:
                navigation.next_ref_lanes = [best_lane]  # 如果没有下一个车道，使用当前车道
                print(f"⚠️ 未找到下一个车道，使用当前车道")
            
            # 4. 重置检查点索引
            if hasattr(navigation, '_target_checkpoints_index'):
                navigation._target_checkpoints_index = [0, 1]
                print(f"✅ 重置检查点索引: [0, 1]")
            
            # 5. 更新导航状态
            navigation.update_localization(agent)
            
            print(f"✅ 车道检测和导航路径修复成功!")
            return True
        else:
            print(f"❌ 无法找到主车所在的正确车道")
            return False
            
    except Exception as e:
        print(f"❌ 车道检测修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    from trajectory_replay import TrajectoryReplayEnv
    
    print("🎯 车道检测修复测试")
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
    }
    
    try:
        print("🏗️ 创建测试环境...")
        env = TrajectoryReplayEnv(trajectory_dict=sample_trajectory, config=config)
        
        print("🔄 重置环境...")
        obs = env.reset()
        
        print("\n修复前:")
        agent = env.agent
        navigation = agent.navigation
        
        print(f"  当前车道: {navigation.current_lane.index}")
        
        try:
            checkpoint1, checkpoint2 = navigation.get_checkpoints()
            print(f"  检查点1: ({checkpoint1[0]:.1f}, {checkpoint1[1]:.1f})")
            print(f"  检查点2: ({checkpoint2[0]:.1f}, {checkpoint2[1]:.1f})")
        except:
            print(f"  无法获取检查点")
        
        print(f"  路径完成度: {navigation.route_completion:.6f}")
        
        # 执行修复
        success = fix_lane_detection(env)
        
        if success:
            print("\n修复后:")
            print(f"  当前车道: {navigation.current_lane.index}")
            
            try:
                checkpoint1, checkpoint2 = navigation.get_checkpoints()
                print(f"  检查点1: ({checkpoint1[0]:.1f}, {checkpoint1[1]:.1f})")
                print(f"  检查点2: ({checkpoint2[0]:.1f}, {checkpoint2[1]:.1f})")
                
                # 检查检查点是否在前方
                agent_pos = agent.position
                dx1 = checkpoint1[0] - agent_pos[0]
                dx2 = checkpoint2[0] - agent_pos[0]
                
                print(f"  检查点1方向: {'前方' if dx1 > 0 else '后方'} ({dx1:.1f}m)")
                print(f"  检查点2方向: {'前方' if dx2 > 0 else '后方'} ({dx2:.1f}m)")
                
            except Exception as e:
                print(f"  无法获取检查点: {e}")
            
            print(f"  路径完成度: {navigation.route_completion:.6f}")
            
            print("\n🎉 车道检测修复成功！")
        else:
            print("\n❌ 车道检测修复失败")
        
        env.close()
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc() 