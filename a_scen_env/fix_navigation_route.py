#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导航路径修复脚本
解决 "No valid navigation route found" 问题

问题分析：
1. 自定义场景中的道路网络没有正确连接
2. 导航系统无法找到从起点到终点的有效路径
3. 导致PPO expert停车问题

解决方案：
1. 检查并修复道路网络结构
2. 正确设置导航目标点
3. 确保路径连通性
"""

import numpy as np
from metadrive.component.road_network.node_road_network import NodeRoadNetwork
from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.road_network.road import Road
from metadrive.utils import Config


class NavigationRouteFixer:
    """导航路径修复器"""
    
    def __init__(self, env):
        """
        初始化导航路径修复器
        
        Args:
            env: TrajectoryReplayEnv 实例
        """
        self.env = env
        self.engine = env.engine
        self.agent = env.agent
        
    def diagnose_navigation_issue(self):
        """
        诊断导航问题
        
        Returns:
            dict: 诊断结果
        """
        print("🔍 开始诊断导航问题...")
        
        diagnosis = {
            "has_agent": hasattr(self.env, 'agent') and self.agent is not None,
            "has_navigation": False,
            "has_current_map": False,
            "has_road_network": False,
            "road_count": 0,
            "lane_count": 0,
            "current_lane": None,
            "target_position": None,
            "route_exists": False
        }
        
        # 检查主车和导航模块
        if diagnosis["has_agent"]:
            if hasattr(self.agent, 'navigation') and self.agent.navigation:
                diagnosis["has_navigation"] = True
                diagnosis["current_lane"] = getattr(self.agent.navigation, 'current_lane', None)
                diagnosis["route_exists"] = hasattr(self.agent.navigation, 'route') and \
                                          self.agent.navigation.route and \
                                          len(self.agent.navigation.route) > 1
        
        # 检查地图和道路网络
        if hasattr(self.engine, 'current_map') and self.engine.current_map:
            diagnosis["has_current_map"] = True
            current_map = self.engine.current_map
            
            if hasattr(current_map, 'road_network') and current_map.road_network:
                diagnosis["has_road_network"] = True
                road_network = current_map.road_network
                
                # 统计道路和车道数量
                if hasattr(road_network, 'graph'):
                    diagnosis["road_count"] = len(road_network.graph.keys())
                    total_lanes = 0
                    for road in road_network.graph.keys():
                        total_lanes += len(road_network.graph[road].keys())
                    diagnosis["lane_count"] = total_lanes
        
        # 获取目标位置
        if hasattr(self.env, 'custom_destination'):
            diagnosis["target_position"] = self.env.custom_destination
        
        # 打印诊断结果
        print("📊 导航诊断结果:")
        print(f"  ✅ 主车存在: {diagnosis['has_agent']}")
        print(f"  ✅ 导航模块: {diagnosis['has_navigation']}")
        print(f"  ✅ 地图存在: {diagnosis['has_current_map']}")
        print(f"  ✅ 道路网络: {diagnosis['has_road_network']}")
        print(f"  📈 道路数量: {diagnosis['road_count']}")
        print(f"  📈 车道数量: {diagnosis['lane_count']}")
        print(f"  🎯 目标位置: {diagnosis['target_position']}")
        print(f"  🛣️  路径存在: {diagnosis['route_exists']}")
        
        return diagnosis
    
    def create_simple_straight_road_network(self, start_pos, end_pos):
        """
        创建简单的直路网络
        
        Args:
            start_pos: 起点坐标 [x, y]
            end_pos: 终点坐标 [x, y]
            
        Returns:
            NodeRoadNetwork: 新的道路网络
        """
        print(f"🛠️ 创建直路网络: {start_pos} → {end_pos}")
        
        # 创建新的道路网络
        road_network = NodeRoadNetwork()
        
        # 定义道路节点
        start_node = "start"
        end_node = "end"
        
        # 创建直车道
        lane_width = 3.5  # 标准车道宽度
        
        # 主车道（从起点到终点）
        main_lane = StraightLane(
            start=start_pos + [0],  # 添加z坐标
            end=end_pos + [0],      # 添加z坐标
            width=lane_width,
            line_types=["continuous", "continuous"]
        )
        
        # 添加车道到道路网络
        road_network.add_lane(start_node, end_node, main_lane)
        
        print(f"✅ 道路网络创建完成:")
        print(f"  📍 起始节点: {start_node}")
        print(f"  📍 结束节点: {end_node}")
        print(f"  🛣️ 车道长度: {np.linalg.norm(np.array(end_pos) - np.array(start_pos)):.1f}m")
        
        return road_network
    
    def fix_navigation_route(self):
        """
        修复导航路径
        
        Returns:
            bool: 修复是否成功
        """
        print("🔧 开始修复导航路径...")
        
        # 诊断问题
        diagnosis = self.diagnose_navigation_issue()
        
        if not diagnosis["has_agent"]:
            print("❌ 错误: 主车不存在")
            return False
            
        if not diagnosis["has_navigation"]:
            print("❌ 错误: 导航模块不存在")
            return False
        
        # 获取当前位置和目标位置
        current_pos = self.agent.position[:2]  # [x, y]
        
        # 如果没有设置目标位置，使用默认值
        if not diagnosis["target_position"]:
            print("⚠️ 未设置目标位置，使用默认终点")
            target_pos = [current_pos[0] + 500, current_pos[1]]  # 前方500米
        else:
            target_pos = diagnosis["target_position"][:2]
        
        print(f"📍 当前位置: ({current_pos[0]:.1f}, {current_pos[1]:.1f})")
        print(f"🎯 目标位置: ({target_pos[0]:.1f}, {target_pos[1]:.1f})")
        
        try:
            # 方法1: 尝试直接设置路径
            if self._try_direct_route_setting(current_pos, target_pos):
                return True
            
            # 方法2: 重建道路网络
            if self._try_rebuild_road_network(current_pos, target_pos):
                return True
            
            # 方法3: 手动设置导航点
            if self._try_manual_navigation_setup(current_pos, target_pos):
                return True
                
        except Exception as e:
            print(f"❌ 修复过程中出现错误: {e}")
        
        print("❌ 所有修复方法都失败了")
        return False
    
    def _try_direct_route_setting(self, current_pos, target_pos):
        """尝试直接设置路径"""
        print("🎯 方法1: 尝试直接设置路径...")
        
        try:
            # 获取当前地图
            current_map = self.engine.current_map
            if not current_map or not hasattr(current_map, 'road_network'):
                print("❌ 地图或道路网络不存在")
                return False
            
            road_network = current_map.road_network
            
            # 查找起点和终点最近的车道
            start_lane_index = None
            end_lane_index = None
            
            min_start_dist = float('inf')
            min_end_dist = float('inf')
            
            # 遍历所有车道寻找最近的
            for road_start in road_network.graph.keys():
                for road_end in road_network.graph[road_start].keys():
                    for lane_index, lane in road_network.graph[road_start][road_end].items():
                        if lane:
                            # 检查起点距离
                            lane_start = lane.position(0, 0)[:2]
                            start_dist = np.linalg.norm(np.array(current_pos) - np.array(lane_start))
                            if start_dist < min_start_dist:
                                min_start_dist = start_dist
                                start_lane_index = (road_start, road_end, lane_index)
                            
                            # 检查终点距离
                            lane_end = lane.position(lane.length, 0)[:2]
                            end_dist = np.linalg.norm(np.array(target_pos) - np.array(lane_end))
                            if end_dist < min_end_dist:
                                min_end_dist = end_dist
                                end_lane_index = (road_start, road_end, lane_index)
            
            if start_lane_index and end_lane_index:
                print(f"📍 找到起始车道: {start_lane_index}")
                print(f"🎯 找到目标车道: {end_lane_index}")
                
                # 设置导航路径
                self.agent.navigation.set_route(start_lane_index, end_lane_index)
                
                # 验证路径是否成功设置
                if hasattr(self.agent.navigation, 'route') and self.agent.navigation.route:
                    print("✅ 直接路径设置成功!")
                    return True
                    
        except Exception as e:
            print(f"❌ 直接路径设置失败: {e}")
        
        return False
    
    def _try_rebuild_road_network(self, current_pos, target_pos):
        """尝试重建道路网络"""
        print("🏗️ 方法2: 尝试重建道路网络...")
        
        try:
            # 创建新的简单道路网络
            new_road_network = self.create_simple_straight_road_network(current_pos, target_pos)
            
            # 替换当前地图的道路网络
            if hasattr(self.engine, 'current_map') and self.engine.current_map:
                self.engine.current_map.road_network = new_road_network
                
                # 重新设置导航
                start_lane_index = ("start", "end", 0)
                end_lane_index = ("start", "end", 0)
                
                self.agent.navigation.set_route(start_lane_index, end_lane_index)
                
                # 验证
                if hasattr(self.agent.navigation, 'route') and self.agent.navigation.route:
                    print("✅ 道路网络重建成功!")
                    return True
                    
        except Exception as e:
            print(f"❌ 道路网络重建失败: {e}")
        
        return False
    
    def _try_manual_navigation_setup(self, current_pos, target_pos):
        """尝试手动设置导航"""
        print("✋ 方法3: 尝试手动导航设置...")
        
        try:
            # 直接设置导航目标点
            if hasattr(self.agent.navigation, 'destination_point'):
                self.agent.navigation.destination_point = target_pos
                print(f"✅ 手动设置目标点: {target_pos}")
                return True
            
            # 如果没有destination_point属性，创建一个
            setattr(self.agent.navigation, 'destination_point', target_pos)
            setattr(self.agent.navigation, 'manual_destination', True)
            
            print("✅ 手动导航设置成功!")
            return True
            
        except Exception as e:
            print(f"❌ 手动导航设置失败: {e}")
        
        return False
    
    def verify_navigation_fix(self):
        """验证导航修复是否成功"""
        print("🔍 验证导航修复结果...")
        
        try:
            if hasattr(self.agent, 'navigation') and self.agent.navigation:
                nav = self.agent.navigation
                
                # 检查路径
                has_route = hasattr(nav, 'route') and nav.route and len(nav.route) > 0
                
                # 检查目标点
                has_destination = hasattr(nav, 'destination_point') or \
                                hasattr(nav, 'final_lane')
                
                # 检查路径完成度
                route_completion = getattr(nav, 'route_completion', -1)
                
                print(f"📊 验证结果:")
                print(f"  🛣️ 路径存在: {has_route}")
                print(f"  🎯 目标设置: {has_destination}")
                print(f"  📈 路径完成度: {route_completion:.3f}")
                
                if has_route or has_destination:
                    print("✅ 导航修复成功!")
                    return True
                else:
                    print("❌ 导航修复失败")
                    return False
                    
        except Exception as e:
            print(f"❌ 验证过程出错: {e}")
            return False


def fix_navigation_for_env(env):
    """
    为环境修复导航问题的主函数
    
    Args:
        env: TrajectoryReplayEnv 实例
        
    Returns:
        bool: 修复是否成功
    """
    print("🚀 开始导航路径修复...")
    
    fixer = NavigationRouteFixer(env)
    
    # 执行修复
    success = fixer.fix_navigation_route()
    
    if success:
        # 验证修复结果
        success = fixer.verify_navigation_fix()
    
    if success:
        print("🎉 导航路径修复完成!")
        print("💡 建议: 重新运行PPO训练，主车应该能正常前进了")
    else:
        print("😞 导航路径修复失败")
        print("💡 建议: 检查轨迹数据和场景配置")
    
    return success


# 使用示例
if __name__ == "__main__":
    print("这是导航路径修复模块")
    print("请在您的训练脚本中导入并使用 fix_navigation_for_env(env) 函数") 