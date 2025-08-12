#!/usr/bin/env python3
"""
简化的PPO训练脚本

解决多进程和环境初始化问题的简化版本
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

from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory


class SimpleTrajectoryWrapper(gym.Env):
    """
    简化的Gymnasium包装器
    避免复杂的环境创建和多进程问题
    """
    
    def __init__(self, csv_path=None):
        super().__init__()
        
        # 使用固定的CSV文件
        if csv_path is None:
            csv_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(csv_dir, "scenario_vehicles_9_0_1_2_3_4_5_6_7_8_test3_20250304_150943_row001_filtered.csv")
        
        print(f"Loading trajectory from: {os.path.basename(csv_path)}")
        
        # 加载轨迹数据
        traj_data = load_trajectory(
            csv_path=csv_path,
            normalize_position=False,
            max_duration=50,  # 缩短到50秒
            use_original_position=False,
            translate_to_origin=True,
            target_fps=20.0,  # 降低频率
            use_original_timestamps=False  # 使用重采样
        )
        
        # 环境配置
        env_config = {
            "use_render": False,
            "manual_control": False,
            "horizon": 1000,  # 减少步数
            "end_on_crash": True,
            "end_on_out_of_road": True,
            "end_on_arrive_dest": False,
            "end_on_horizon": True,
            "background_vehicle_update_mode": "position",
            "enable_realtime": False,
            "target_fps": 20.0,
            "map": "S" * 20,  # 缩短地图
            "traffic_density": 0.0,
            "disable_ppo_expert": True,  # 禁用PPO专家
        }
        
        # 创建环境
        self.env = TrajectoryReplayEnv(traj_data, config=env_config)
        
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
        
        print(f"Environment created successfully!")
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
        obs, reward, done, info = self.env.step(action)
        
        # 处理返回格式
        if isinstance(obs, tuple):
            obs = obs[0]
        
        return obs, reward, done, False, info
    
    def close(self):
        """关闭环境"""
        self.env.close()


def simple_train():
    """简化的训练函数"""
    print("=" * 60)
    print("Simple PPO Training")
    print("=" * 60)
    
    # 创建保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(os.path.dirname(__file__), "models", f"simple_ppo_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Model will be saved to: {model_dir}")
    
    # 创建环境
    print("\nCreating environment...")
    env = SimpleTrajectoryWrapper()
    env = DummyVecEnv([lambda: env])
    
    # PPO参数
    ppo_params = {
        "learning_rate": 3e-4,
        "n_steps": 1024,  # 减少步数
        "batch_size": 64,
        "n_epochs": 4,  # 减少epochs
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "policy": "MlpPolicy",
        "verbose": 1,
    }
    
    print("\nCreating PPO model...")
    model = PPO(env=env, **ppo_params)
    
    # 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=model_dir,
        name_prefix="simple_ppo",
    )
    
    # 开始训练
    total_timesteps = 20000  # 较少的步数用于测试
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
    model, model_dir = simple_train()
    
    if model is not None:
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! Your PPO model is ready!")
        print("=" * 60)
        print(f"Model location: {model_dir}")
        print("\nNext steps:")
        print("1. Test the model: python evaluate_ppo.py")
        print("2. Use in trajectory_replay.py (automatic)")
    else:
        print("\n" + "=" * 60)
        print("❌ Training failed. Please check the errors above.")
        print("=" * 60) 