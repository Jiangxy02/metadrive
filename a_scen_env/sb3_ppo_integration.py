#!/usr/bin/env python3
"""
Stable Baselines3 PPO集成方案
提供TrajectoryReplayEnv与SB3的接口集成
"""

import sys
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Union, Tuple

# 导入现有模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trajectory_loader import load_trajectory
from trajectory_replay import TrajectoryReplayEnv

# SB3相关导入
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    SB3_AVAILABLE = True
except ImportError:
    print("⚠️ Stable Baselines3 not found. Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False


class SB3TrajectoryReplayWrapper(gym.Wrapper):
    """
    Stable Baselines3兼容的轨迹重放环境包装器
    
    功能：
    1. 确保与SB3的gym.Env接口完全兼容
    2. 处理observation_space和action_space
    3. 统一step返回格式（5元组：obs, reward, terminated, truncated, info）
    4. 提供合理的默认奖励设计
    """
    
    def __init__(self, 
                 csv_path: str,
                 config: Optional[Dict[str, Any]] = None,
                 reward_config: Optional[Dict[str, Any]] = None,
                 max_duration: float = 30.0,
                 use_original_timestamps: bool = True):
        """
        初始化SB3兼容的环境
        
        Args:
            csv_path: CSV轨迹文件路径
            config: TrajectoryReplayEnv配置
            reward_config: 奖励函数配置
            max_duration: 最大仿真时长（秒）
            use_original_timestamps: 是否使用原始时间戳
        """
        
        # 加载轨迹数据
        traj_data = load_trajectory(
            csv_path=csv_path,
            max_duration=max_duration,
            use_original_timestamps=use_original_timestamps
        )
        
        # 默认配置
        default_config = {
            "use_render": False,  # SB3训练通常不需要渲染
            "manual_control": False,  # 禁用手动控制
            "enable_realtime": False,  # 禁用实时模式，加速训练
            "background_vehicle_update_mode": "position",  # 使用位置更新模式
            "end_on_crash": True,  # 碰撞时结束
            "end_on_out_of_road": True,  # 出路时结束
            "end_on_arrive_dest": True,  # 到达终点时结束
            "end_on_horizon": True,  # 超时时结束
        }
        
        if config:
            default_config.update(config)
            
        # 创建基础环境
        base_env = TrajectoryReplayEnv(traj_data, config=default_config)
        
        super().__init__(base_env)
        
        # 奖励配置
        self.reward_config = reward_config or self._get_default_reward_config()
        
        # 记录轨迹信息
        self.main_trajectory = base_env.main_vehicle_trajectory
        self.csv_path = csv_path
        self.max_duration = max_duration
        
        # 确保action_space和observation_space兼容SB3
        self._setup_spaces()
        
        print(f"🎯 SB3环境初始化完成:")
        print(f"  观测空间: {self.observation_space}")
        print(f"  动作空间: {self.action_space}")
        print(f"  主车轨迹点数: {len(self.main_trajectory) if self.main_trajectory else 0}")
        
    def _get_default_reward_config(self) -> Dict[str, Any]:
        """默认奖励配置"""
        return {
            "speed_reward_weight": 0.1,      # 速度奖励权重
            "progress_reward_weight": 1.0,   # 进度奖励权重
            "lane_keeping_weight": 0.5,      # 车道保持权重
            "crash_penalty": -10.0,          # 碰撞惩罚
            "out_road_penalty": -5.0,        # 出路惩罚
            "timeout_penalty": -1.0,         # 超时惩罚
            "completion_bonus": 10.0,        # 完成奖励
        }
    
    def _setup_spaces(self):
        """设置观测和动作空间，确保SB3兼容性"""
        
        # 从基础环境获取空间定义
        base_obs_space = self.env.observation_space
        base_action_space = self.env.action_space
        
        # 确保是gymnasium格式
        if hasattr(base_obs_space, 'shape'):
            self.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=base_obs_space.shape, 
                dtype=np.float32
            )
        else:
            # 默认观测空间（MetaDrive通常是49维）
            self.observation_space = spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(49,), 
                dtype=np.float32
            )
            
        if hasattr(base_action_space, 'shape'):
            self.action_space = spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=base_action_space.shape, 
                dtype=np.float32
            )
        else:
            # 默认动作空间（转向，油门/刹车）
            self.action_space = spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(2,), 
                dtype=np.float32
            )
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境，返回SB3兼容格式"""
        
        if seed is not None:
            np.random.seed(seed)
            
        obs = self.env.reset()
        
        # 确保obs是numpy数组
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
            
        info = {
            "episode_step": 0,
            "simulation_time": 0.0,
            "csv_path": self.csv_path
        }
        
        return obs, info
    
    def step(self, action):
        """执行动作，返回SB3兼容的5元组格式"""
        
        # 确保action是正确格式
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
            
        # 调用基础环境step
        obs, base_reward, done, info = self.env.step(action)
        
        # 确保obs是numpy数组
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
            
        # 计算自定义奖励
        reward = self._compute_reward(obs, action, done, info)
        
        # SB3需要分离terminated和truncated
        terminated = done and (
            info.get("crash_flag", False) or 
            info.get("out_of_road_flag", False) or
            info.get("arrive_dest_flag", False)
        )
        truncated = done and info.get("horizon_reached_flag", False)
        
        # 添加额外信息
        info.update({
            "base_reward": base_reward,
            "custom_reward": reward,
            "terminated": terminated,
            "truncated": truncated
        })
        
        return obs, reward, terminated, truncated, info
    
    def _compute_reward(self, obs, action, done, info) -> float:
        """计算自定义奖励函数"""
        
        reward = 0.0
        config = self.reward_config
        
        # 基础速度奖励
        if hasattr(self.env.agent, 'speed'):
            speed = self.env.agent.speed
            # 鼓励保持合理速度（15-25 m/s）
            target_speed = 20.0
            speed_reward = -abs(speed - target_speed) / target_speed
            reward += config["speed_reward_weight"] * speed_reward
        
        # 进度奖励（基于前进距离）
        if hasattr(self.env.agent, 'position'):
            # 简单的前进奖励
            progress_reward = 0.1  # 每步基础奖励
            reward += config["progress_reward_weight"] * progress_reward
        
        # 结束状态奖励/惩罚
        if done:
            if info.get("crash_flag", False):
                reward += config["crash_penalty"]
            elif info.get("out_of_road_flag", False):
                reward += config["out_road_penalty"]
            elif info.get("arrive_dest_flag", False):
                reward += config["completion_bonus"]
            elif info.get("horizon_reached_flag", False):
                reward += config["timeout_penalty"]
        
        return reward


class TrajectoryReplayCallback(BaseCallback):
    """
    轨迹重放训练的SB3回调函数
    """
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:
        """每步调用"""
        
        if self.n_calls % self.log_freq == 0:
            # 记录训练统计
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                if self.verbose >= 1:
                    print(f"Step {self.n_calls}: Episode reward: {ep_info.get('r', 0):.2f}")
                    
        return True
    
    def _on_rollout_end(self) -> None:
        """回合结束时调用"""
        pass


def create_sb3_env(csv_path: str, 
                   config: Optional[Dict[str, Any]] = None,
                   reward_config: Optional[Dict[str, Any]] = None,
                   max_duration: float = 30.0,
                   use_original_timestamps: bool = True):
    """
    创建SB3兼容的环境实例
    
    Args:
        csv_path: CSV轨迹文件路径
        config: 环境配置
        reward_config: 奖励配置
        max_duration: 最大仿真时长
        use_original_timestamps: 是否使用原始时间戳
        
    Returns:
        SB3兼容的环境实例
    """
    
    def _make_env():
        return SB3TrajectoryReplayWrapper(
            csv_path=csv_path,
            config=config,
            reward_config=reward_config,
            max_duration=max_duration,
            use_original_timestamps=use_original_timestamps
        )
    
    return _make_env


def train_sb3_ppo(csv_path: str,
                  total_timesteps: int = 100000,
                  config: Optional[Dict[str, Any]] = None,
                  model_save_path: str = "sb3_ppo_model",
                  log_dir: str = "./sb3_logs/"):
    """
    使用SB3训练PPO模型
    
    Args:
        csv_path: CSV轨迹文件路径
        total_timesteps: 总训练步数
        config: 环境配置
        model_save_path: 模型保存路径
        log_dir: 日志目录
        
    Returns:
        训练好的PPO模型
    """
    
    if not SB3_AVAILABLE:
        raise ImportError("Stable Baselines3 not available. Install with: pip install stable-baselines3")
    
    print(f"🚀 开始SB3 PPO训练:")
    print(f"  CSV文件: {csv_path}")
    print(f"  训练步数: {total_timesteps}")
    print(f"  保存路径: {model_save_path}")
    
    # 创建环境
    env = create_sb3_env(csv_path, config)()
    
    # 设置日志
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # 创建PPO模型
    model = PPO(
        "MlpPolicy",  # 多层感知机策略
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=log_dir
    )
    
    # 设置日志
    model.set_logger(new_logger)
    
    # 创建回调
    callback = TrajectoryReplayCallback(log_freq=1000, verbose=1)
    
    # 训练模型
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # 保存模型
    model.save(model_save_path)
    print(f"✅ 模型已保存: {model_save_path}")
    
    return model


def load_and_test_sb3_model(model_path: str,
                            csv_path: str,
                            num_episodes: int = 5,
                            config: Optional[Dict[str, Any]] = None,
                            render: bool = True):
    """
    加载并测试SB3模型
    
    Args:
        model_path: 模型文件路径
        csv_path: 测试用CSV文件路径
        num_episodes: 测试回合数
        config: 环境配置
        render: 是否渲染
        
    Returns:
        测试结果统计
    """
    
    if not SB3_AVAILABLE:
        raise ImportError("Stable Baselines3 not available")
    
    print(f"🧪 测试SB3模型: {model_path}")
    
    # 测试配置
    test_config = config or {}
    if render:
        test_config["use_render"] = True
    
    # 创建环境
    env = create_sb3_env(csv_path, test_config)()
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 测试统计
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    # 计算统计
    stats = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "episodes": num_episodes
    }
    
    print(f"\n📊 测试结果:")
    print(f"  平均奖励: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    print(f"  平均长度: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    
    env.close()
    return stats


if __name__ == "__main__":
    # 示例用法
    
    # CSV文件路径
    csv_path = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/scenario_vehicles_9_0_1_2_3_4_5_6_7_8_test3_20250304_150943_row001_filtered.csv"
    
    # 检查SB3可用性
    if not SB3_AVAILABLE:
        print("❌ Stable Baselines3 未安装")
        print("安装命令: pip install stable-baselines3")
        exit(1)
    
    print("🎯 SB3 PPO集成示例")
    
    # 创建测试环境
    test_config = {
        "use_render": False,
        "enable_realtime": False,
        "max_duration": 10.0
    }
    
    env = create_sb3_env(csv_path, config=test_config)()
    
    print(f"环境测试:")
    obs, info = env.reset()
    print(f"  初始观测形状: {obs.shape}")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    
    # 测试一步
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  步进测试: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
    
    env.close()
    
    print("\n✅ SB3集成测试完成!")
    print("\n🚀 训练命令示例:")
    print("python sb3_ppo_integration.py")
    print("\n📝 编程接口:")
    print("model = train_sb3_ppo(csv_path, total_timesteps=50000)")
    print("stats = load_and_test_sb3_model('sb3_ppo_model', csv_path)") 