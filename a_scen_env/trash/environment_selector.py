"""
MetaDrive环境选择器 - 展示如何使用不同的环境
"""

import sys
import os
sys.path.append('/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive')

from metadrive.envs import (
    MetaDriveEnv,           # 标准驾驶环境
    SafeMetaDriveEnv,       # 安全强化学习环境（包含事故场景）
    ScenarioEnv,           # 真实场景环境
    MultiAgentMetaDrive,   # 多智能体环境
    MultiAgentIntersectionEnv, # 多智能体十字路口环境
    MultiAgentRoundaboutEnv,   # 多智能体环形路口环境
    MultiAgentTollgateEnv,     # 多智能体收费站环境
    MultiAgentBottleneckEnv,   # 多智能体瓶颈环境
    MultiAgentParkingLotEnv,   # 多智能体停车场环境
)

from metadrive.envs.top_down_env import TopDownMetaDrive  # 俯视图环境
from trajectory_replay import TrajectoryReplayEnv  # 我们的自定义环境
from trajectory_loader import load_trajectory


def create_standard_metadrive_env():
    """创建标准MetaDrive环境"""
    config = {
        "use_render": True,
        "manual_control": True,
        "map": "SSS",  # 3段直道
        "traffic_density": 0.1,
        "vehicle_config": {
            "show_navi_mark": True,
            "show_dest_mark": True,
            "show_line_to_dest": True,
        }
    }
    return MetaDriveEnv(config)


def create_safe_metadrive_env():
    """创建安全强化学习环境（包含事故场景）"""
    config = {
        "use_render": True,
        "manual_control": True,
        "map": "SSS",
        "accident_prob": 0.8,  # 80%概率出现事故
        "traffic_density": 0.2,
        "vehicle_config": {
            "show_navi_mark": True,
            "show_dest_mark": True,
        }
    }
    return SafeMetaDriveEnv(config)


def create_multi_agent_env():
    """创建多智能体环境"""
    config = {
        "use_render": True,
        "manual_control": True,
        "num_agents": 4,  # 4个智能体
        "map": "SSS",
        "traffic_density": 0.1,
        "vehicle_config": {
            "show_navi_mark": True,
        }
    }
    return MultiAgentMetaDrive(config)


def create_intersection_env():
    """创建十字路口多智能体环境"""
    config = {
        "use_render": True,
        "manual_control": True,
        "num_agents": 4,
    }
    return MultiAgentIntersectionEnv(config)


def create_roundabout_env():
    """创建环形路口环境"""
    config = {
        "use_render": True,
        "manual_control": True,
        "num_agents": 6,
    }
    return MultiAgentRoundaboutEnv(config)


def create_top_down_env():
    """创建俯视图环境（用于研究和可视化）"""
    config = {
        "use_render": True,
        "manual_control": True,
        "map": "SSS",
        "traffic_density": 0.1,
    }
    return TopDownMetaDrive(config)


def create_trajectory_replay_env_with_different_base():
    """基于不同基础环境创建轨迹重放环境"""
    
    # 加载轨迹数据
    csv_path = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/scenario_vehicles_9_0_1_2_3_4_5_6_7_8_test3_20250304_150943_row001_filtered.csv"
    
    traj_data = load_trajectory(
        csv_path=csv_path, 
        normalize_position=False,
        max_duration=20,
        use_original_position=False,
        translate_to_origin=True,
        target_fps=50.0,
        use_original_timestamps=True
    )
    
    # 选项1: 使用标准MetaDriveEnv作为基础
    class TrajectoryReplayStandard(TrajectoryReplayEnv):
        def __init__(self, trajectory_dict, config=None):
            super().__init__(trajectory_dict, config)
    
    # 选项2: 使用SafeMetaDriveEnv作为基础  
    class TrajectoryReplaySafe(TrajectoryReplayEnv):
        def __init__(self, trajectory_dict, config=None):
            # 添加安全环境的默认配置
            safe_config = {
                "accident_prob": 0.5,
                "crash_vehicle_done": False,
                "crash_object_done": False,
            }
            if config:
                safe_config.update(config)
            super().__init__(trajectory_dict, safe_config)
    
    return {
        "standard": TrajectoryReplayStandard(traj_data, {"map": "SSS"}),
        "safe": TrajectoryReplaySafe(traj_data, {"map": "SSS"}),
    }


def demonstrate_environment_switching():
    """演示如何在不同环境之间切换"""
    
    environments = {
        "1": ("标准MetaDrive环境", create_standard_metadrive_env),
        "2": ("安全强化学习环境", create_safe_metadrive_env), 
        "3": ("多智能体环境", create_multi_agent_env),
        "4": ("十字路口环境", create_intersection_env),
        "5": ("环形路口环境", create_roundabout_env),
        "6": ("俯视图环境", create_top_down_env),
        "7": ("轨迹重放环境（当前）", lambda: TrajectoryReplayEnv(
            load_trajectory("/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/scenario_vehicles_9_0_1_2_3_4_5_6_7_8_test3_20250304_150943_row001_filtered.csv", 
                          max_duration=10, translate_to_origin=True, use_original_timestamps=True),
            {"map": "SSS", "use_render": False}  # 设置为False避免窗口冲突
        )),
    }
    
    print("=== MetaDrive环境选择器 ===")
    print("可用的环境类型：")
    for key, (name, _) in environments.items():
        print(f"  {key}. {name}")
    
    choice = input("\n请选择环境类型 (1-7): ").strip()
    
    if choice in environments:
        env_name, env_creator = environments[choice]
        print(f"\n正在创建 {env_name}...")
        
        try:
            env = env_creator()
            print(f"✅ {env_name} 创建成功！")
            print(f"   环境类型: {type(env).__name__}")
            
            # 测试环境基本功能
            obs = env.reset()
            print(f"   观察空间: {env.observation_space}")
            print(f"   动作空间: {env.action_space}")
            
            # 运行几步测试
            for i in range(3):
                action = env.action_space.sample()  # 随机动作
                obs, reward, done, info = env.step(action)
                print(f"   Step {i}: reward={reward:.3f}, done={done}")
                if done:
                    break
            
            env.close()
            print(f"✅ {env_name} 测试完成")
            
        except Exception as e:
            print(f"❌ 创建 {env_name} 失败: {e}")
    
    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    # 展示如何选择和切换不同环境
    demonstrate_environment_switching() 