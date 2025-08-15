#!/usr/bin/env python3
"""
认知模块可视化功能测试脚本
运行短期仿真并生成可视化图表
"""

import sys
import os
import numpy as np

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
cognitive_module_dir = os.path.join(parent_dir, 'cognitive_module')
if cognitive_module_dir not in sys.path:
    sys.path.insert(0, cognitive_module_dir)

from trajectory_replay_cognitive import TrajectoryReplayEnvCognitive
from trajectory_loader import load_trajectory

def test_cognitive_visualization():
    """测试认知可视化功能"""
    print("=== 认知模块可视化功能测试 ===\n")
    
    # 测试CSV路径
    csv_path = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/scenario_vehicles_9_0_1_2_3_4_5_6_7_8_test3_20250304_150943_row001_filtered.csv"
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"❌ 轨迹文件不存在: {csv_path}")
        return False
    
    try:
        # 加载轨迹数据
        print("📁 加载轨迹数据...")
        traj_data = load_trajectory(
            csv_path=csv_path,
            normalize_position=False,
            max_duration=30,  # 限制为30秒测试
            use_original_position=False,
            translate_to_origin=True,
            target_fps=50.0,
            use_original_timestamps=True
        )
        
        print(f"✅ 加载了 {len(traj_data)} 辆车的轨迹数据")
        
        # 认知模块配置
        COGNITIVE_CONFIG = {
            'perception': {
                'enable': True,
                'sigma': 0.5,
                'enable_kalman': True,
                'process_noise': 0.1,
                'dt': 0.02
            },
            'cognitive_bias': {
                'enable': False,  # 暂未实现
            },
            'delay': {
                'enable': True,
                'delay_steps': 3,
                'enable_smoothing': True,
                'smoothing_factor': 0.4
            }
        }
        
        # 创建认知环境
        print("🏗️ 初始化认知环境...")
        env = TrajectoryReplayEnvCognitive(
            traj_data,
            config={
                'use_render': False,  # 关闭渲染加快测试
                'manual_control': False,
                'enable_background_vehicles': True,
                'background_vehicle_update_mode': "position",
                'enable_realtime': False,  # 关闭实时模式
                'target_fps': 50.0,
                
                # 启用认知模块
                'enable_cognitive_modules': True,
                'cognitive_config': COGNITIVE_CONFIG,
                
                # 启用可视化（输出目录将自动创建时间戳文件夹）
                'enable_visualization': True
            }
        )
        
        print("✅ 环境初始化完成")
        
        # 重置环境
        obs = env.reset()
        print("🔄 环境已重置")
        
        print(f"📍 主车位置: {env.agent.position}")
        print(f"🎮 控制模式: PPO Expert (默认)")
        
        # 运行仿真
        print("\n🚀 开始仿真测试...")
        max_steps = 200  # 限制步数进行快速测试
        
        for i in range(max_steps):
            # 使用简单的前进动作进行测试
            action = np.array([0.0, 0.3])  # 直行，轻油门
            
            obs, reward, done, info = env.step(action)
            
            # 每50步输出一次状态
            if i % 50 == 0:
                mode = info.get('Control Mode', 'unknown')
                cognitive_active = info.get('cognitive_modules_active', False)
                speed = env.agent.speed
                print(f"  Step {i}: 控制模式={mode}, 认知模块={'激活' if cognitive_active else '未激活'}, 速度={speed:.2f} m/s")
            
            if done:
                print(f"  仿真在第{i}步结束")
                break
        
        print(f"✅ 仿真完成，总计 {min(i+1, max_steps)} 步")
        
        # 关闭环境（这会触发图表生成）
        print("\n📊 生成可视化图表...")
        env.close()
        
        # 检查生成的图表
        if hasattr(env, 'visualization_output_dir') and os.path.exists(env.visualization_output_dir):
            fig_dir = env.visualization_output_dir
            files = [f for f in os.listdir(fig_dir) if f.endswith('.png')]
            print(f"✅ 成功生成 {len(files)} 个图表文件:")
            for file in sorted(files):
                print(f"  - {file}")
            print(f"📁 图表保存位置: {fig_dir}")
        else:
            print("❌ 图表目录不存在")
            
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cognitive_visualization()
    if success:
        print("\n🎉 认知模块可视化功能测试成功!")
        print("📁 查看生成的图表: fig_cog 目录")
    else:
        print("\n💥 测试失败，请检查错误信息")
    
    print("\n=== 测试完成 ===") 