"""
PPO专家策略接口

功能：
1. 加载训练好的PPO模型
2. 提供与trajectory_replay.py兼容的expert函数接口
3. 支持模型的动态加载和切换

作者：PPO训练系统
日期：2025-01-03
"""

import os
import sys
import numpy as np
from typing import Optional
from stable_baselines3 import PPO

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PPOExpert:
    """
    PPO专家策略类
    
    单例模式，确保整个程序只加载一次模型
    """
    _instance = None
    _model = None
    _model_path = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PPOExpert, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: str = None):
        """
        加载PPO模型
        
        Args:
            model_path: 模型文件路径，如果为None则尝试加载最新模型
        """
        if model_path is None:
            model_path = self.find_latest_model()
            if model_path is None:
                raise FileNotFoundError("No trained PPO model found. Please train a model first.")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # 如果是新模型路径，重新加载
        if model_path != self._model_path:
            print(f"Loading PPO model from: {model_path}")
            self._model = PPO.load(model_path)
            self._model_path = model_path
            print("PPO model loaded successfully!")
        
        return self._model
    
    def find_latest_model(self) -> Optional[str]:
        """
        查找最新的训练模型
        
        Returns:
            str: 最新模型的路径，如果没有找到则返回None
        """
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        
        if not os.path.exists(models_dir):
            return None
        
        # 查找所有模型目录
        model_dirs = [d for d in os.listdir(models_dir) 
                      if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("ppo_trajectory_")]
        
        if not model_dirs:
            return None
        
        # 按时间戳排序，获取最新的
        model_dirs.sort()
        latest_dir = model_dirs[-1]
        
        # 优先使用best_model，如果没有则使用final_model
        best_model_path = os.path.join(models_dir, latest_dir, "best_model", "best_model.zip")
        final_model_path = os.path.join(models_dir, latest_dir, "final_model.zip")
        
        if os.path.exists(best_model_path):
            return best_model_path
        elif os.path.exists(final_model_path):
            return final_model_path
        else:
            return None
    
    def get_action(self, observation):
        """
        根据观察值获取动作
        
        Args:
            observation: 环境观察值
            
        Returns:
            numpy.ndarray: 动作数组 [转向, 油门/刹车]
        """
        if self._model is None:
            self.load_model()
        
        # 使用模型预测动作（确定性策略）
        action, _states = self._model.predict(observation, deterministic=True)
        
        return action
    
    def set_model_path(self, model_path: str):
        """
        设置模型路径（用于指定特定模型）
        
        Args:
            model_path: 模型文件路径
        """
        self.load_model(model_path)


# 全局PPO专家实例
_ppo_expert = PPOExpert()


def expert(agent):
    """
    PPO专家策略函数
    
    与metadrive.examples.expert兼容的接口
    用于trajectory_replay.py中的控制模式管理器
    
    Args:
        agent: MetaDrive车辆实例
        
    Returns:
        numpy.ndarray: 动作数组 [转向, 油门/刹车]
    """
    try:
        # 获取观察值
        # agent.get_state() 返回车辆的观察向量
        observation = agent.get_state()
        
        # 使用PPO模型获取动作
        action = _ppo_expert.get_action(observation)
        
        return action
    except (FileNotFoundError, ValueError, Exception) as e:
        # 如果没有训练好的模型，fallback到MetaDrive默认expert
        try:
            from metadrive.examples import expert as metadrive_expert
            return metadrive_expert(agent)
        except Exception as e2:
            # 最后的fallback：返回默认动作
            print(f"All expert methods failed, using default action: {e}, {e2}")
            return [0.0, 0.0]


def set_expert_model(model_path: str):
    """
    设置专家模型路径
    
    Args:
        model_path: 模型文件路径
    """
    _ppo_expert.set_model_path(model_path)


def get_expert_model_path() -> Optional[str]:
    """
    获取当前专家模型路径
    
    Returns:
        str: 当前模型路径，如果没有加载模型则返回None
    """
    return _ppo_expert._model_path


# 导出接口
__all__ = ['expert', 'set_expert_model', 'get_expert_model_path'] 