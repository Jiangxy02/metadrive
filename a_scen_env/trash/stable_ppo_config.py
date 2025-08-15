"""
稳定PPO训练配置 - 解决主车控制不稳定问题

针对MetaDrive环境的PPO训练优化配置，解决以下问题：
1. 动作抖动和控制不稳定
2. 训练过程中的奖励波动
3. 网络收敛缓慢

使用方法：
    from stable_ppo_config import get_stable_ppo_config
    config = get_stable_ppo_config()
    model = PPO("MlpPolicy", env, **config)
"""

import numpy as np
from typing import Dict, Any


def get_stable_ppo_config(env_type: str = "visual_training") -> Dict[str, Any]:
    """
    获取稳定的PPO训练配置
    
    Args:
        env_type: 环境类型 ("visual_training", "scenario", "random")
        
    Returns:
        PPO配置字典
    """
    
    # 基础稳定配置
    base_config = {
        # === 学习率配置 ===
        "learning_rate": 2e-4,  # 降低学习率，提高稳定性
        
        # === 训练步数配置 ===
        "n_steps": 2048,        # 增加每次更新的步数，收集更多经验
        "batch_size": 128,      # 增加批量大小，减少梯度噪声
        "n_epochs": 4,          # 减少epoch数，避免过拟合
        
        # === 折扣和GAE ===
        "gamma": 0.995,         # 稍微增加折扣因子，重视长期奖励
        "gae_lambda": 0.98,     # 增加GAE lambda，减少方差
        
        # === PPO裁剪 ===
        "clip_range": 0.1,      # 减小裁剪范围，更保守的策略更新
        "clip_range_vf": 0.1,   # 价值函数裁剪
        
        # === 损失权重 ===
        "ent_coef": 0.005,      # 减小熵系数，减少随机性
        "vf_coef": 0.5,         # 价值函数损失权重
        
        # === 梯度裁剪 ===
        "max_grad_norm": 0.5,   # 限制梯度范数，防止梯度爆炸
        
        # === 网络配置 ===
        "policy_kwargs": {
            "net_arch": [256, 256],           # 增加网络容量
            "activation_fn": "tanh",          # 使用tanh激活函数（更平滑）
            "ortho_init": True,               # 正交初始化
            "use_sde": False,                 # 不使用状态依赖探索
            "log_std_init": -0.5,             # 初始动作标准差
            "full_std": True,                 # 使用完整标准差
            "use_expln": False,               # 不使用指数线性激活
            "squash_output": False,           # 不压缩输出
        },
        
        # === 其他配置 ===
        "verbose": 1,
        "seed": 42,
        "device": "auto",
    }
    
    # 根据环境类型调整配置
    if env_type == "visual_training":
        # 可视化训练：更保守的配置
        base_config.update({
            "learning_rate": 1e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "clip_range": 0.08,     # 更小的裁剪范围
            "ent_coef": 0.001,      # 更小的熵系数
        })
        
    elif env_type == "scenario":
        # 场景训练：平衡配置
        base_config.update({
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 128,
        })
        
    elif env_type == "random":
        # 随机环境：标准配置
        pass  # 使用基础配置
    
    return base_config


def get_action_smoothing_config() -> Dict[str, Any]:
    """
    获取动作平滑配置
    
    Returns:
        动作平滑包装器配置
    """
    return {
        # === 基础平滑配置 ===
        "smoothing_factor": 0.8,    # 平滑因子
        "max_change_rate": 0.25,    # 最大变化率
        
        # === 自适应平滑配置 ===
        "use_adaptive": True,       # 是否使用自适应平滑
        "initial_smoothing": 0.95,  # 初始平滑因子
        "final_smoothing": 0.4,     # 最终平滑因子
        "adaptation_steps": 30000,  # 自适应步数
    }


def get_environment_config() -> Dict[str, Any]:
    """
    获取环境配置，减少控制难度
    
    Returns:
        环境配置字典
    """
    return {
        # === 车辆配置 ===
        "vehicle_config": {
            "show_navi_mark": False,
            "show_dest_mark": False,
            "show_line_to_dest": False,
            "show_line_to_navi_mark": False,
            "show_navigation_arrow": False,
            
            # === 物理参数 ===
            "mass": 1200,               # 车辆质量
            "max_engine_force": 1000,   # 最大引擎力
            "max_brake_force": 1000,    # 最大制动力
            "max_steering": 0.3,        # 最大转向角度
            "friction": 0.8,            # 轮胎摩擦力
        },
        
        # === 环境参数 ===
        "physics_world_step_size": 0.02,  # 物理步长 (50Hz)
        "decision_repeat": 1,             # 决策重复次数
        "render_mode": "human",
        "use_render": True,
        
        # === 奖励配置 ===
        "crash_penalty": -10.0,          # 碰撞惩罚
        "out_of_road_penalty": -5.0,     # 出界惩罚
        "speed_reward": 0.1,             # 速度奖励
        "lane_center_reward": 0.05,      # 车道中心奖励
    }


def create_stable_training_env(base_env_class, env_config=None, use_action_smoothing=True):
    """
    创建稳定的训练环境
    
    Args:
        base_env_class: 基础环境类
        env_config: 环境配置
        use_action_smoothing: 是否使用动作平滑
        
    Returns:
        配置好的训练环境
    """
    from metadrive.a_scen_env.trash.action_smoother import ActionSmootherWrapper, AdaptiveActionSmootherWrapper
    
    # 创建基础环境
    env_config = env_config or get_environment_config()
    env = base_env_class(env_config)
    
    # 添加动作平滑
    if use_action_smoothing:
        smoothing_config = get_action_smoothing_config()
        
        if smoothing_config["use_adaptive"]:
            env = AdaptiveActionSmootherWrapper(
                env,
                initial_smoothing=smoothing_config["initial_smoothing"],
                final_smoothing=smoothing_config["final_smoothing"],
                adaptation_steps=smoothing_config["adaptation_steps"],
                max_change_rate=smoothing_config["max_change_rate"]
            )
        else:
            env = ActionSmootherWrapper(
                env,
                smoothing_factor=smoothing_config["smoothing_factor"],
                max_change_rate=smoothing_config["max_change_rate"]
            )
    
    print("🚗 稳定训练环境已创建:")
    print(f"   动作平滑: {'开启' if use_action_smoothing else '关闭'}")
    print(f"   动作空间: {env.action_space}")
    print(f"   观测空间: {env.observation_space}")
    
    return env


# === 使用示例 ===
if __name__ == "__main__":
    # 示例：如何使用稳定配置
    print("🔧 稳定PPO配置示例:")
    
    # 获取PPO配置
    ppo_config = get_stable_ppo_config("visual_training")
    print(f"PPO配置: {ppo_config}")
    
    # 获取动作平滑配置
    smooth_config = get_action_smoothing_config()
    print(f"动作平滑配置: {smooth_config}")
    
    # 获取环境配置
    env_config = get_environment_config()
    print(f"环境配置: {env_config}")
    
    print("\n使用方法:")
    print("from stable_ppo_config import get_stable_ppo_config, create_stable_training_env")
    print("ppo_config = get_stable_ppo_config('visual_training')")
    print("env = create_stable_training_env(YourEnvClass)")
    print("model = PPO('MlpPolicy', env, **ppo_config)") 