#!/usr/bin/env python3
"""
最小化PPO训练脚本

完全避免背景车和复杂轨迹功能，专注于验证训练流程
"""

import os
import sys
import numpy as np
from datetime import datetime

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from metadrive.envs import MetaDriveEnv


class MinimalTrainingWrapper(gym.Env):
    """
    最小化的Gymnasium包装器，使用标准MetaDriveEnv
    """
    
    def __init__(self):
        super().__init__()
        
        # 环境配置
        env_config = {
            "use_render": False,
            "manual_control": False,
            "horizon": 1000,
            "map": "S" * 20,  # 简单的直道
            "traffic_density": 0.0,  # 无背景交通
            "start_seed": 0,
        }
        
        print("Creating minimal MetaDrive environment...")
        self.env = MetaDriveEnv(config=env_config)
        
        # 获取观察空间
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
            
        # 定义空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs.shape,
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        print(f"✅ Environment created successfully!")
        print(f"Observation space: {self.observation_space.shape}")
        print(f"Action space: {self.action_space}")
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        return obs, {}
    
    def step(self, action):
        """执行动作"""
        result = self.env.step(action)
        
        # 处理不同的返回值数量
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        elif len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            raise ValueError(f"Unexpected number of return values: {len(result)}")
        
        # 处理返回格式
        if isinstance(obs, tuple):
            obs = obs[0]
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """关闭环境"""
        self.env.close()


def minimal_train():
    """最小化训练函数"""
    print("=" * 60)
    print("Minimal PPO Training (No Background Vehicles)")
    print("=" * 60)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(os.path.dirname(__file__), "models", f"minimal_ppo_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Model will be saved to: {model_dir}")
    
    # 创建环境
    print("\nCreating environment...")
    env = MinimalTrainingWrapper()
    env = DummyVecEnv([lambda: env])
    
    # PPO参数（更保守的设置）
    ppo_params = {
        "learning_rate": 3e-4,
        "n_steps": 512,   # 更小的步数
        "batch_size": 32,  # 更小的批次
        "n_epochs": 3,     # 更少的epochs
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "policy": "MlpPolicy",
        "verbose": 1,
        "device": "cpu",  # 强制使用CPU
    }
    
    print("\nCreating PPO model...")
    model = PPO(env=env, **ppo_params)
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,  # 更频繁的保存
        save_path=model_dir,
        name_prefix="minimal_ppo",
    )
    
    # 开始训练
    total_timesteps = 10000  # 更少的步数用于快速验证
    print(f"\nStarting training for {total_timesteps} timesteps...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
        
        # 保存最终模型
        final_model_path = os.path.join(model_dir, "final_model")
        model.save(final_model_path)
        
        print(f"\n✅ Training completed successfully!")
        print(f"Model saved to: {final_model_path}")
        
        return model, model_dir
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None
    finally:
        env.close()


if __name__ == "__main__":
    model, model_dir = minimal_train()
    
    if model is not None:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your minimal PPO model is ready!")
        print("=" * 60)
        print(f"Model location: {model_dir}")
        print("\nThis proves the PPO training system works!")
        print("Now you can:")
        print("1. Use this model in trajectory_replay.py")
        print("2. Extend to more complex scenarios")
    else:
        print("\n" + "=" * 60)
        print("❌ Training failed. Please check the errors above.")
        print("=" * 60) 