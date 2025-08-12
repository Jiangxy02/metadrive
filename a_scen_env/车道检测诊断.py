#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车道检测诊断 - 检查主车是否错误检测为地图第一条车道
"""

import sys
import os
import numpy as np
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trajectory_replay import TrajectoryReplayEnv


def diagnose_lane_detection():
    """诊断车道检测问题"""
    
    print("🔍 车道检测诊断")
    print("=" * 80)
    
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
        "map": "S" * 8,  # 当前的地图配置
        "manual_control": False,
        "enable_background_vehicles": False,
        "disable_ppo_expert": False,
        "traffic_density": 0.0,
    }
    
    try:
        print("🏗️ 创建环境...")
        env = TrajectoryReplayEnv(trajectory_dict=sample_trajectory, config=config)
        
        print("🔄 重置环境...")
        obs = env.reset()
        
        agent = env.agent
        navigation = agent.navigation
        current_map = env.engine.current_map
        road_network = current_map.road_network
        
        print("\n" + "=" * 80)
        print("📊 主车位置信息")
        print("=" * 80)
        
        agent_pos = agent.position
        agent_heading = agent.heading_theta
        
        print(f"🚗 主车实际位置: ({agent_pos[0]:.3f}, {agent_pos[1]:.3f})")
        print(f"🚗 主车朝向角度: {agent_heading:.3f} rad ({np.degrees(agent_heading):.1f}°)")
        
        print("\n" + "=" * 80)
        print("🛣️ 当前车道检测分析")
        print("=" * 80)
        
        current_lane = navigation.current_lane
        if current_lane:
            print(f"📍 检测到的当前车道:")
            print(f"  车道索引: {current_lane.index}")
            print(f"  车道类型: {type(current_lane).__name__}")
            
            # 车道起点和终点
            lane_start = current_lane.position(0, 0)
            lane_end = current_lane.position(current_lane.length, 0)
            
            print(f"  车道起点: ({lane_start[0]:.3f}, {lane_start[1]:.3f})")
            print(f"  车道终点: ({lane_end[0]:.3f}, {lane_end[1]:.3f})")
            print(f"  车道长度: {current_lane.length:.3f}m")
            
            # 计算主车在车道上的位置
            try:
                local_coords = current_lane.local_coordinates(agent_pos)
                longitudinal = local_coords[0]  # 沿车道方向的距离
                lateral = local_coords[1]       # 垂直车道的距离
                
                print(f"  主车在车道上的位置:")
                print(f"    纵向位置: {longitudinal:.3f}m (0为起点, {current_lane.length:.1f}为终点)")
                print(f"    横向偏移: {lateral:.3f}m (0为车道中心)")
                
                # 检查主车是否真的在这个车道上
                is_on_lane = (0 <= longitudinal <= current_lane.length) and (abs(lateral) < 10)
                print(f"  主车是否在此车道上: {'是' if is_on_lane else '否'}")
                
                if not is_on_lane:
                    print(f"  ❌ 车道检测错误！主车不在检测到的车道上")
                    if longitudinal < 0:
                        print(f"     主车在车道起点前方 {-longitudinal:.1f}m")
                    elif longitudinal > current_lane.length:
                        print(f"     主车在车道终点后方 {longitudinal - current_lane.length:.1f}m")
                    if abs(lateral) > 10:
                        print(f"     主车横向偏离车道中心 {abs(lateral):.1f}m")
                
            except Exception as e:
                print(f"  ❌ 无法计算主车在车道上的位置: {e}")
        else:
            print(f"❌ 未检测到当前车道!")
        
        print("\n" + "=" * 80)
        print("🗺️ 全地图车道分析")
        print("=" * 80)
        
        print(f"📈 地图信息:")
        print(f"  地图类型: {type(current_map).__name__}")
        print(f"  道路段数量: {len(road_network.graph.keys())}")
        
        # 遍历所有车道，找出最接近主车的车道
        closest_lane = None
        min_distance = float('inf')
        lane_distances = []
        
        print(f"\n🔍 分析所有车道与主车的距离:")
        lane_count = 0
        
        for road_start in road_network.graph.keys():
            for road_end in road_network.graph[road_start].keys():
                lanes = road_network.graph[road_start][road_end]
                
                # 处理不同的lanes数据结构
                if hasattr(lanes, 'items'):
                    # lanes是字典
                    lane_items = lanes.items()
                elif isinstance(lanes, (list, tuple)):
                    # lanes是列表
                    lane_items = enumerate(lanes)
                else:
                    print(f"  ⚠️ 未知的lanes数据结构: {type(lanes)}")
                    continue
                
                for lane_idx, lane in lane_items:
                    if lane:
                        lane_count += 1
                        
                        # 计算车道起点、中点、终点
                        lane_start = lane.position(0, 0)
                        lane_mid = lane.position(lane.length/2, 0)
                        lane_end = lane.position(lane.length, 0)
                        
                        # 计算主车到车道各点的距离
                        dist_start = np.sqrt((agent_pos[0] - lane_start[0])**2 + (agent_pos[1] - lane_start[1])**2)
                        dist_mid = np.sqrt((agent_pos[0] - lane_mid[0])**2 + (agent_pos[1] - lane_mid[1])**2)
                        dist_end = np.sqrt((agent_pos[0] - lane_end[0])**2 + (agent_pos[1] - lane_end[1])**2)
                        
                        min_lane_dist = min(dist_start, dist_mid, dist_end)
                        
                        # 计算主车在此车道上的位置
                        try:
                            local_coords = lane.local_coordinates(agent_pos)
                            longitudinal = local_coords[0]
                            lateral = local_coords[1]
                            
                            # 检查主车是否在此车道上
                            is_on_this_lane = (0 <= longitudinal <= lane.length) and (abs(lateral) < 5)
                            
                            lane_info = {
                                'index': lane.index,
                                'road_segment': f"{road_start}→{road_end}",
                                'lane_idx': lane_idx,
                                'start': lane_start,
                                'end': lane_end,
                                'length': lane.length,
                                'min_distance': min_lane_dist,
                                'longitudinal': longitudinal,
                                'lateral': lateral,
                                'is_on_lane': is_on_this_lane,
                                'lane_object': lane
                            }
                            
                            lane_distances.append(lane_info)
                            
                            if min_lane_dist < min_distance:
                                min_distance = min_lane_dist
                                closest_lane = lane_info
                                
                        except Exception as e:
                            print(f"  ⚠️ 车道{lane_count}坐标计算失败: {e}")
        
        print(f"  总共找到 {lane_count} 条车道")
        
        # 排序车道距离
        lane_distances.sort(key=lambda x: x['min_distance'])
        
        print(f"\n📏 距离主车最近的前5条车道:")
        for i, lane_info in enumerate(lane_distances[:5]):
            status = "✅ 在车道上" if lane_info['is_on_lane'] else "❌ 不在车道上"
            current_marker = " [当前检测]" if (current_lane and lane_info['index'] == current_lane.index) else ""
            
            print(f"  {i+1}. 车道{lane_info['index']} ({lane_info['road_segment']}){current_marker}")
            print(f"     起点: ({lane_info['start'][0]:.1f}, {lane_info['start'][1]:.1f})")
            print(f"     终点: ({lane_info['end'][0]:.1f}, {lane_info['end'][1]:.1f})")
            print(f"     最小距离: {lane_info['min_distance']:.1f}m")
            print(f"     纵向位置: {lane_info['longitudinal']:.1f}m")
            print(f"     横向偏移: {lane_info['lateral']:.1f}m")
            print(f"     状态: {status}")
            print()
        
        print("\n" + "=" * 80)
        print("🔎 车道检测问题诊断结果")
        print("=" * 80)
        
        if current_lane and closest_lane:
            is_correct_detection = current_lane.index == closest_lane['index']
            
            print(f"📊 检测结果分析:")
            print(f"  当前检测车道: {current_lane.index}")
            print(f"  实际最近车道: {closest_lane['index']}")
            print(f"  检测是否正确: {'是' if is_correct_detection else '否'}")
            
            if not is_correct_detection:
                print(f"\n❌ 车道检测错误!")
                print(f"  问题原因分析:")
                
                # 检查是否检测成了地图第一条车道
                first_lane_info = lane_distances[-1]  # 最远的车道通常是第一条
                if current_lane.index == first_lane_info['index']:
                    print(f"  ❌ 主车被错误检测为地图第一条车道!")
                    print(f"     第一条车道: {first_lane_info['road_segment']}")
                    print(f"     第一条车道位置: ({first_lane_info['start'][0]:.1f}, {first_lane_info['start'][1]:.1f}) → ({first_lane_info['end'][0]:.1f}, {first_lane_info['end'][1]:.1f})")
                    print(f"     主车到第一条车道距离: {first_lane_info['min_distance']:.1f}m")
                
                print(f"  💡 正确的车道应该是:")
                print(f"     车道索引: {closest_lane['index']}")
                print(f"     道路段: {closest_lane['road_segment']}")
                print(f"     距离主车: {closest_lane['min_distance']:.1f}m")
                print(f"     主车在此车道上: {closest_lane['is_on_lane']}")
            else:
                print(f"✅ 车道检测正确!")
        
        # 分析车道检测算法
        print(f"\n🔧 车道检测算法分析:")
        try:
            # 尝试手动查找正确的车道
            print(f"  尝试手动查找主车所在车道...")
            
            correct_lanes = []
            for lane_info in lane_distances:
                if lane_info['is_on_lane']:
                    correct_lanes.append(lane_info)
            
            if correct_lanes:
                print(f"  找到 {len(correct_lanes)} 条可能的正确车道:")
                for lane_info in correct_lanes:
                    print(f"    - 车道{lane_info['index']}: 纵向{lane_info['longitudinal']:.1f}m, 横向{lane_info['lateral']:.1f}m")
            else:
                print(f"  ❌ 没有找到主车真正所在的车道!")
                print(f"     这表明主车位置与地图坐标系不匹配")
        
        except Exception as e:
            print(f"  ❌ 手动车道查找失败: {e}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ 车道检测诊断失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    diagnose_lane_detection() 