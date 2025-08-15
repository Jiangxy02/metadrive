#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析检查点坐标和位置关系
"""

import sys
import os
import numpy as np
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trajectory_replay import TrajectoryReplayEnv


def analyze_checkpoints():
    """详细分析检查点坐标"""
    
    print("🎯 检查点详细分析")
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
        
        print("\n" + "=" * 80)
        print("📊 基础坐标信息")
        print("=" * 80)
        
        # 主车位置
        agent_pos = agent.position
        print(f"🚗 主车当前位置: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})")
        print(f"🚗 主车朝向角度: {agent.heading_theta:.3f} rad ({np.degrees(agent.heading_theta):.1f}°)")
        
        # 当前车道信息
        current_lane = navigation.current_lane
        if current_lane:
            lane_start = current_lane.position(0, 0)
            lane_end = current_lane.position(current_lane.length, 0)
            print(f"🛣️ 当前车道起点: ({lane_start[0]:.1f}, {lane_start[1]:.1f})")
            print(f"🛣️ 当前车道终点: ({lane_end[0]:.1f}, {lane_end[1]:.1f})")
            print(f"🛣️ 当前车道长度: {current_lane.length:.1f}m")
            print(f"🛣️ 当前车道索引: {current_lane.index}")
        
        print("\n" + "=" * 80)
        print("🎯 检查点详细信息")
        print("=" * 80)
        
        try:
            checkpoint1, checkpoint2 = navigation.get_checkpoints()
            
            print(f"\n📍 检查点1详细信息:")
            print(f"  坐标: ({checkpoint1[0]:.3f}, {checkpoint1[1]:.3f})")
            print(f"  X坐标: {checkpoint1[0]:.3f}")
            print(f"  Y坐标: {checkpoint1[1]:.3f}")
            
            print(f"\n📍 检查点2详细信息:")
            print(f"  坐标: ({checkpoint2[0]:.3f}, {checkpoint2[1]:.3f})")
            print(f"  X坐标: {checkpoint2[0]:.3f}")
            print(f"  Y坐标: {checkpoint2[1]:.3f}")
            
            # 距离计算
            print(f"\n📏 距离分析:")
            dist1 = np.sqrt((checkpoint1[0] - agent_pos[0])**2 + (checkpoint1[1] - agent_pos[1])**2)
            dist2 = np.sqrt((checkpoint2[0] - agent_pos[0])**2 + (checkpoint2[1] - agent_pos[1])**2)
            
            print(f"  主车到检查点1距离: {dist1:.3f}m")
            print(f"  主车到检查点2距离: {dist2:.3f}m")
            
            # 方向分析
            print(f"\n🧭 方向分析:")
            
            # 计算检查点1相对于主车的方向
            dx1 = checkpoint1[0] - agent_pos[0]
            dy1 = checkpoint1[1] - agent_pos[1]
            angle1 = math.atan2(dy1, dx1)
            angle1_deg = np.degrees(angle1)
            
            print(f"  检查点1相对主车:")
            print(f"    X偏移: {dx1:.3f}m ({'向前' if dx1 > 0 else '向后'})")
            print(f"    Y偏移: {dy1:.3f}m ({'向右' if dy1 > 0 else '向左'})")
            print(f"    角度: {angle1:.3f} rad ({angle1_deg:.1f}°)")
            print(f"    方位: {'后方' if abs(angle1_deg) > 90 else '前方'}")
            
            # 计算检查点2相对于主车的方向
            dx2 = checkpoint2[0] - agent_pos[0]
            dy2 = checkpoint2[1] - agent_pos[1]
            angle2 = math.atan2(dy2, dx2)
            angle2_deg = np.degrees(angle2)
            
            print(f"  检查点2相对主车:")
            print(f"    X偏移: {dx2:.3f}m ({'向前' if dx2 > 0 else '向后'})")
            print(f"    Y偏移: {dy2:.3f}m ({'向右' if dy2 > 0 else '向左'})")
            print(f"    角度: {angle2:.3f} rad ({angle2_deg:.1f}°)")
            print(f"    方位: {'后方' if abs(angle2_deg) > 90 else '前方'}")
            
            # 检查点之间的关系
            print(f"\n🔗 检查点之间关系:")
            checkpoint_dist = np.sqrt((checkpoint2[0] - checkpoint1[0])**2 + (checkpoint2[1] - checkpoint1[1])**2)
            print(f"  检查点1到检查点2距离: {checkpoint_dist:.3f}m")
            
            # 检查点与车道的关系
            print(f"\n🛣️ 检查点与车道关系:")
            if current_lane:
                # 检查检查点是否在当前车道上
                lane_y = current_lane.position(0, 0)[1]  # 车道Y坐标
                print(f"  当前车道Y坐标: {lane_y:.3f}")
                print(f"  检查点1 Y偏差: {abs(checkpoint1[1] - lane_y):.3f}m")
                print(f"  检查点2 Y偏差: {abs(checkpoint2[1] - lane_y):.3f}m")
                
                # 检查点是否在车道范围内
                lane_start_x = current_lane.position(0, 0)[0]
                lane_end_x = current_lane.position(current_lane.length, 0)[0]
                
                print(f"  当前车道X范围: {lane_start_x:.1f} ~ {lane_end_x:.1f}")
                print(f"  检查点1在车道范围内: {'是' if lane_start_x <= checkpoint1[0] <= lane_end_x else '否'}")
                print(f"  检查点2在车道范围内: {'是' if lane_start_x <= checkpoint2[0] <= lane_end_x else '否'}")
            
            # 导航信息
            print(f"\n🧭 导航状态信息:")
            print(f"  路径完成度: {navigation.route_completion:.6f}")
            print(f"  已行驶距离: {navigation.travelled_length:.3f}m")
            print(f"  总路径长度: {navigation.total_length:.3f}m")
            
            # 分析问题
            print(f"\n⚠️ 问题分析:")
            
            # 检查是否有检查点在后方
            points_behind = []
            if abs(angle1_deg) > 90:
                points_behind.append("检查点1")
            if abs(angle2_deg) > 90:
                points_behind.append("检查点2")
            
            if points_behind:
                print(f"  ❌ 发现问题: {', '.join(points_behind)} 在主车后方!")
                print(f"     这会导致导航系统认为需要倒车")
                print(f"     PPO会因此接收到错误的导航信号")
            else:
                print(f"  ✅ 检查点位置正常")
            
            # 检查路径完成度
            if navigation.route_completion < 0.1:
                print(f"  ❌ 路径完成度异常低: {navigation.route_completion:.6f}")
                print(f"     可能表示导航计算出现问题")
            
            # 检查已行驶距离
            if navigation.travelled_length < 0:
                print(f"  ❌ 已行驶距离为负: {navigation.travelled_length:.3f}m")
                print(f"     这是导航系统的严重问题")
            
        except Exception as e:
            print(f"❌ 获取检查点失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 尝试获取更多导航内部信息
        print(f"\n" + "=" * 80)
        print("🔍 导航系统内部状态")
        print("=" * 80)
        
        try:
            print(f"  导航模块类型: {type(navigation).__name__}")
            print(f"  当前检查点索引: {getattr(navigation, 'current_checkpoint', 'N/A')}")
            print(f"  下一个检查点索引: {getattr(navigation, 'next_checkpoint', 'N/A')}")
            print(f"  路径节点: {getattr(navigation, 'route', [])}")
            print(f"  目标车道: {getattr(navigation, 'destination_lane', 'N/A')}")
            
            # 检查内部计算状态
            if hasattr(navigation, '_get_info_for_checkpoint'):
                try:
                    info = navigation._get_info_for_checkpoint()
                    print(f"  内部检查点信息: {info}")
                except Exception as e:
                    print(f"  无法获取内部检查点信息: {e}")
            
            # 检查车道定位
            if hasattr(navigation, 'update_localization'):
                print(f"  车道定位方法存在: 是")
            
        except Exception as e:
            print(f"  获取导航内部状态失败: {e}")
        
        env.close()
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_checkpoints() 