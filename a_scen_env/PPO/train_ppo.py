"""
PPO训练脚本 - 基于TrajectoryReplayEnv环境

功能：
1. 使用Stable-Baselines3训练PPO模型
2. 支持多个CSV轨迹文件的随机选择训练
3. 保存训练好的模型和训练日志
4. 与trajectory_replay.py完全兼容

作者：PPO训练系统
日期：2025-01-03
"""

import os
import sys
import glob
import random
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory


class TrajectoryReplayWrapper(gym.Env):
    """
    Gymnasium包装器，使TrajectoryReplayEnv与Stable-Baselines3兼容
    
    功能：
    - 将MetaDrive环境包装为标准Gymnasium环境
    - 处理观察空间和动作空间的转换
    - 支持多个轨迹文件的随机选择
    - 禁用渲染以提高训练速度
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化环境包装器
        
        Args:
            config: 环境配置字典
        """
        super().__init__()
        
        # 默认配置
        default_config = {
            "csv_paths": None,  # CSV文件路径列表
            "use_render": False,  # 训练时禁用渲染
            "manual_control": False,  # 训练时禁用手动控制
            "horizon": 2000,  # 每个episode的最大步数
            "end_on_crash": True,  # 碰撞时结束
            "end_on_out_of_road": True,  # 出界时结束
            "end_on_arrive_dest": False,  # 到达终点时不结束（继续训练）
            "end_on_horizon": True,  # 达到最大步数时结束
            "background_vehicle_update_mode": "position",  # 背景车更新模式
            "enable_realtime": False,  # 训练时禁用实时模式
            "target_fps": 50.0,
            "map": "S" * 30,  # 更长的直道用于训练
            "traffic_density": 0.0,  # 禁用自动交通
            "normalize_position": False,
            "max_duration": 100,
            "use_original_position": False,
            "translate_to_origin": True,
            "use_original_timestamps": True,
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.csv_paths = self._get_csv_paths()
        
        # 创建环境实例（用第一个CSV初始化）
        self._create_env()
        
        # 定义观察空间和动作空间
        # 使用MetaDrive的默认观察空间
        obs = self.env.reset()
        # 处理可能的返回格式差异
        if isinstance(obs, tuple):
            obs = obs[0]  # 如果返回的是元组，取第一个元素
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32
        )
        
        # 动作空间：[转向, 油门/刹车]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
    def _get_csv_paths(self):
        """获取CSV文件路径列表"""
        csv_paths = self.config.get("csv_paths")
        
        if csv_paths is None:
            # 默认使用a_scen_env目录下的所有CSV文件
            csv_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_pattern = os.path.join(csv_dir, "scenario_vehicles_*.csv")
            csv_paths = glob.glob(csv_pattern)
            
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
            
        if not csv_paths:
            raise ValueError("No CSV files found for training")
            
        print(f"Found {len(csv_paths)} CSV files for training")
        return csv_paths
    
    def _create_env(self, csv_path=None):
        """
        创建或重新创建环境实例
        
        Args:
            csv_path: 指定的CSV路径，如果为None则随机选择
        """
        # 如果已有环境，先关闭
        if hasattr(self, 'env'):
            try:
                self.env.close()
            except:
                pass
        
        # 选择CSV文件
        if csv_path is None:
            csv_path = random.choice(self.csv_paths)
        
        # 加载轨迹数据
        traj_data = load_trajectory(
            csv_path=csv_path,
            normalize_position=self.config.get("normalize_position", False),
            max_duration=self.config.get("max_duration", 100),
            use_original_position=self.config.get("use_original_position", False),
            translate_to_origin=self.config.get("translate_to_origin", True),
            target_fps=self.config.get("target_fps", 50.0),
            use_original_timestamps=self.config.get("use_original_timestamps", True)
        )
        
        # 创建环境（训练时禁用PPO专家）
        env_config = {k: v for k, v in self.config.items() 
                     if k not in ["csv_paths", "normalize_position", "max_duration",
                                 "use_original_position", "translate_to_origin", 
                                 "use_original_timestamps"]}
        
        # 训练时强制禁用PPO专家，避免循环依赖
        env_config.update({
            "use_render": False,
            "manual_control": False,
            "disable_ppo_expert": True,  # 新增标志
        })
        
        self.env = TrajectoryReplayEnv(traj_data, config=env_config)
        self.current_csv = csv_path
        
    def reset(self, seed=None, options=None):
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            tuple: (观察值, 信息字典)
        """
        # 随机选择新的CSV文件（可选）
        if random.random() < 0.3:  # 30%概率切换场景
            self._create_env()
        
        obs = self.env.reset()
        
        # 处理不同的返回格式
        if isinstance(obs, tuple):
            obs, _ = obs  # 如果已经是元组，解包
        
        info = {"csv_file": self.current_csv}
        
        return obs, info
    
    def step(self, action):
        """
        执行一步动作
        
        Args:
            action: 动作数组 [转向, 油门/刹车]
            
        Returns:
            tuple: (观察值, 奖励, 终止标志, 截断标志, 信息字典)
        """
        obs, reward, done, info = self.env.step(action)
        
        # 转换为新的Gymnasium接口
        terminated = done  # 因为环境失败而结束
        truncated = False  # 因为时间限制而结束
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """渲染环境（训练时通常禁用）"""
        if self.config.get("use_render", False):
            return self.env.render()
        return None
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'env'):
            self.env.close()


def make_env(rank: int, seed: int = 0, config: Dict = None):
    """
    创建环境的工厂函数（用于并行训练）
    
    Args:
        rank: 环境编号
        seed: 随机种子
        config: 环境配置
        
    Returns:
        callable: 创建环境的函数
    """
    def _init():
        env = TrajectoryReplayWrapper(config)
        env.reset(seed=seed + rank)
        return env
    
    set_random_seed(seed)
    return _init


def train_ppo(
    total_timesteps: int = 1000000,
    n_envs: int = 4,
    save_dir: str = None,
    config: Dict = None
):
    """
    训练PPO模型
    
    Args:
        total_timesteps: 总训练步数
        n_envs: 并行环境数量
        save_dir: 模型保存目录
        config: 环境配置
    """
    # 创建保存目录
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "models")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(save_dir, f"ppo_trajectory_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建日志目录
    log_dir = os.path.join(model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"Training PPO model")
    print(f"Model directory: {model_dir}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Number of environments: {n_envs}")
    
    # 创建向量化环境
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, seed=0, config=config) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, seed=0, config=config)])
    
    env = VecMonitor(env, log_dir)
    
    # 创建评估环境
    eval_env = DummyVecEnv([make_env(99, seed=42, config=config)])
    
    # PPO超参数
    ppo_params = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "tensorboard_log": log_dir,
        "policy": "MlpPolicy",  # 使用MLP策略网络
        "verbose": 1,
    }
    
    # 创建PPO模型
    model = PPO(env=env, **ppo_params)
    
    # 创建回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # 每50000步保存一次
        save_path=model_dir,
        name_prefix="ppo_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=os.path.join(model_dir, "eval_logs"),
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )
    
    callbacks = CallbackList([checkpoint_callback, eval_callback])
    
    # 开始训练
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"\nTraining completed! Final model saved to: {final_model_path}")
    
    # 关闭环境
    env.close()
    eval_env.close()
    
    return model, model_dir


if __name__ == "__main__":
    # 训练配置
    training_config = {
        "horizon": 2000,  # 每个episode的最大步数
        "end_on_crash": True,
        "end_on_out_of_road": True,
        "end_on_arrive_dest": False,
        "end_on_horizon": True,
        "background_vehicle_update_mode": "position",
        "map": "S" * 30,  # 长直道
    }
    
    # 开始训练
    model, model_dir = train_ppo(
        total_timesteps=500000,  # 训练50万步
        n_envs=4,  # 使用4个并行环境
        config=training_config
    )
    
    print(f"\nModel saved to: {model_dir}")
    print("You can now use the trained model in trajectory_replay.py!") 