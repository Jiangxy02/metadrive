#!/usr/bin/env python3
"""
调试脚本：找出正确的车道位置
"""

from metadrive.envs import MetaDriveEnv
import numpy as np

def test_road_position():
    """测试道路和车道的位置"""
    print("="*60)
    print("测试MetaDrive道路的默认位置")
    print("="*60)
    
    # 创建简单环境
    config = {
        "use_render": False,
        "map": "S"*5,  # 5个S段用于测试
        "manual_control": False,
        "horizon": 100,
        "vehicle_config": {
            "show_navi_mark": False,
        }
    }
    
    env = MetaDriveEnv(config)
    obs = env.reset()
    
    # 获取主车的默认位置
    default_pos = env.agent.position
    default_heading = env.agent.heading_theta
    
    print(f"\n默认主车位置:")
    print(f"  Position: ({default_pos[0]:.2f}, {default_pos[1]:.2f})")
    print(f"  Heading: {default_heading:.2f} rad")
    
    # 测试不同Y位置
    print(f"\n测试不同Y坐标的效果:")
    test_y_positions = [0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0]
    
    for y_pos in test_y_positions:
        env.reset()
        # 设置到测试位置
        env.agent.set_position([200.0, y_pos])
        env.agent.set_heading_theta(0.0)
        
        # 运行一步
        obs, reward, done, info = env.step([0.0, 0.0])
        
        out_of_road = info.get("out_of_road", False)
        status = "✅ OK" if not out_of_road else "❌ Out of road"
        
        print(f"  Y={y_pos:5.1f}: {status}")
        
    # 测试不同X位置
    print(f"\n测试不同X坐标的效果 (Y={default_pos[1]:.1f}):")
    test_x_positions = [0, 50, 100, 150, 200, 250, 300]
    
    for x_pos in test_x_positions:
        env.reset()
        # 设置到测试位置
        env.agent.set_position([x_pos, default_pos[1]])
        env.agent.set_heading_theta(0.0)
        
        # 运行一步
        obs, reward, done, info = env.step([0.0, 0.0])
        
        out_of_road = info.get("out_of_road", False)
        status = "✅ OK" if not out_of_road else "❌ Out of road"
        
        print(f"  X={x_pos:5.1f}: {status}")
    
    # 获取道路信息
    print(f"\n道路信息:")
    print(f"  Map config: {env.config['map']}")
    
    # 测试车道宽度
    print(f"\n测试车道范围 (X=200):")
    for y in np.arange(0, 20, 0.5):
        env.reset()
        env.agent.set_position([200.0, y])
        env.agent.set_heading_theta(0.0)
        obs, reward, done, info = env.step([0.0, 0.0])
        
        if not info.get("out_of_road", False):
            print(f"  Y={y:.1f}: 在道路上")
    
    env.close()
    
    return default_pos

def test_with_trajectory():
    """测试使用轨迹数据"""
    from trajectory_replay import TrajectoryReplayEnv, load_trajectory
    
    print("\n" + "="*60)
    print("测试轨迹数据的Y坐标范围")
    print("="*60)
    
    csv_path = "scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    
    # 加载数据看看原始Y坐标范围
    import pandas as pd
    df = pd.read_csv(csv_path)
    
    print(f"\n原始数据Y坐标范围:")
    print(f"  Min Y: {df['position_y'].min():.1f}")
    print(f"  Max Y: {df['position_y'].max():.1f}")
    print(f"  Mean Y: {df['position_y'].mean():.1f}")
    
    # 车辆-1的Y坐标
    vehicle_minus1 = df[df['vehicle_id'] == -1]
    if not vehicle_minus1.empty:
        v1_y_min = vehicle_minus1['position_y'].min()
        v1_y_max = vehicle_minus1['position_y'].max()
        v1_y_mean = vehicle_minus1['position_y'].mean()
        print(f"\n车辆-1的Y坐标:")
        print(f"  Min: {v1_y_min:.1f}, Max: {v1_y_max:.1f}, Mean: {v1_y_mean:.1f}")

if __name__ == "__main__":
    default_pos = test_road_position()
    test_with_trajectory()
    
    print("\n" + "="*60)
    print("建议")
    print("="*60)
    print(f"✅ MetaDrive默认主车位置: ({default_pos[0]:.1f}, {default_pos[1]:.1f})")
    print(f"💡 建议将车辆-1的Y坐标平移到: {default_pos[1]:.1f}")
    print(f"💡 这是MetaDrive道路的默认车道中心") 