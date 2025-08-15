#!/usr/bin/env python3
"""
观测记录测试脚本
用于演示主车观测状态记录功能，分析停车行为
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory

def test_observation_recording():
    """测试观测记录功能"""
    print("🔍 开始观测记录测试")
    print("=" * 60)
    
    # 加载轨迹数据
    csv_path = "scenario_vehicles_9_0_1_2_3_4_5_6_7_8_test3_20250304_150943_row001_filtered.csv"
    traj_data = load_trajectory(
        csv_path=csv_path,
        normalize_position=False,
        max_duration=100,
        use_original_position=False,
        translate_to_origin=True,
        target_fps=50.0,
        use_original_timestamps=True
    )
    
    # 创建环境，启用观测记录
    env = TrajectoryReplayEnv(
        traj_data,
        config=dict(
            use_render=False,  # 关闭渲染以加快测试速度
            manual_control=True,
            enable_background_vehicles=False,  # 只关注主车
            enable_realtime=False,  # 关闭实时模式以加快测试
            target_fps=50.0,
            
            # ===== 观测记录配置 =====
            enable_observation_recording=True,  # 启用观测记录
            recording_session_name="main_car_stop_analysis",  # 会话名称
            recording_output_dir="observation_logs",  # 输出目录
        )
    )
    
    obs = env.reset()
    print(f"✅ 环境初始化完成，主车位置: {env.agent.position}")
    
    # 运行仿真并记录数据
    max_steps = 500  # 运行500步，足够观察停车行为
    
    try:
        for i in range(max_steps):
            # 执行步骤
            action = [0.0, 0.0]  # 默认动作，由PPO专家控制
            obs, reward, done, info = env.step(action)
            
            # 每50步显示一次进度和当前状态
            if i % 50 == 0:
                pos = env.agent.position
                speed = env.agent.speed
                print(f"步骤 {i:3d}: 位置=({pos[0]:6.1f}, {pos[1]:5.1f}), 速度={speed:5.2f} m/s, 模式={info.get('Control Mode', 'unknown')}")
                
                # 显示当前统计信息
                if env.observation_recorder:
                    stats = env.observation_recorder.get_current_stats()
                    if stats:
                        print(f"       统计: 平均速度={stats['average_speed']:.2f} m/s, 停车率={stats['stopped_percentage']:.1f}%")
            
            # 检查是否停车过久（速度低于0.1 m/s持续100步）
            if i > 100 and env.agent.speed < 0.1:
                # 检查过去50步的平均速度
                if env.observation_recorder and len(env.observation_recorder.step_data) >= 50:
                    recent_speeds = [step['speed'] for step in env.observation_recorder.step_data[-50:]]
                    avg_recent_speed = sum(recent_speeds) / len(recent_speeds)
                    if avg_recent_speed < 0.5:
                        print(f"\n⚠️  检测到主车长时间低速/停车 (平均速度: {avg_recent_speed:.2f} m/s)")
                        print(f"   当前位置: ({env.agent.position[0]:.1f}, {env.agent.position[1]:.1f})")
                        print(f"   建议提前结束测试以分析数据")
                        break
            
            if done:
                print(f"\n✅ 环境正常结束于步骤 {i}")
                break
                
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断测试")
    
    finally:
        print(f"\n📊 正在关闭环境并生成分析报告...")
        env.close()  # 这会自动调用观测记录器的finalize_recording()
        
        print(f"✅ 测试完成！")
        print(f"\n📋 生成的文件:")
        if env.observation_recorder:
            print(f"   - CSV数据: {env.observation_recorder.csv_path}")
            print(f"   - JSON数据: {env.observation_recorder.json_path}")
            print(f"   - 分析报告: {env.observation_recorder.analysis_path}")
            print(f"\n💡 建议:")
            print(f"   1. 查看分析报告了解整体情况")
            print(f"   2. 用Excel或Python打开CSV文件进行详细分析")
            print(f"   3. 关注 action_throttle、speed、nav_route_completion 等关键字段")

if __name__ == "__main__":
    test_observation_recording() 