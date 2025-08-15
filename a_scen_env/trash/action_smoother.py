import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ActionSmootherWrapper(gym.Wrapper):
    """
    动作平滑包装器 - 解决PPO训练中的控制抖动问题
    
    功能：
    1. 使用指数移动平均平滑动作
    2. 限制动作变化率
    3. 提供可配置的平滑参数
    """
    
    def __init__(self, env, smoothing_factor=0.8, max_change_rate=0.3):
        """
        初始化动作平滑包装器
        
        Args:
            env: 原始环境
            smoothing_factor: 平滑因子 [0, 1]，越大越平滑，0.8是推荐值
            max_change_rate: 最大动作变化率 [0, 1]，限制单步最大变化
        """
        super().__init__(env)
        
        self.smoothing_factor = smoothing_factor
        self.max_change_rate = max_change_rate
        
        # 初始化历史动作
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.step_count = 0
        
        print(f"🔧 ActionSmootherWrapper 已启用:")
        print(f"   平滑因子: {smoothing_factor}")
        print(f"   最大变化率: {max_change_rate}")
    
    def reset(self, **kwargs):
        """重置环境时清空历史动作"""
        obs, info = self.env.reset(**kwargs)
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.step_count = 0
        return obs, info
    
    def step(self, action):
        """应用动作平滑后执行步骤"""
        # 确保动作在有效范围内
        action = np.clip(action, -1.0, 1.0)
        
        # 第一步直接使用原始动作
        if self.step_count == 0:
            smoothed_action = action.copy()
        else:
            # 指数移动平均平滑
            smoothed_action = (
                self.smoothing_factor * self.last_action + 
                (1 - self.smoothing_factor) * action
            )
            
            # 限制变化率
            action_diff = smoothed_action - self.last_action
            max_change = self.max_change_rate
            
            # 对每个维度限制变化量
            for i in range(len(action_diff)):
                if abs(action_diff[i]) > max_change:
                    action_diff[i] = np.sign(action_diff[i]) * max_change
            
            smoothed_action = self.last_action + action_diff
            
            # 再次确保在有效范围内
            smoothed_action = np.clip(smoothed_action, -1.0, 1.0)
        
        # 更新历史
        self.last_action = smoothed_action.copy()
        self.step_count += 1
        
        # 执行平滑后的动作
        obs, reward, terminated, truncated, info = self.env.step(smoothed_action)
        
        # 在info中添加动作信息
        info['raw_action'] = action
        info['smoothed_action'] = smoothed_action
        info['action_change'] = np.linalg.norm(smoothed_action - 
                                             (action if self.step_count == 1 else self.last_action))
        
        return obs, reward, terminated, truncated, info


class AdaptiveActionSmootherWrapper(gym.Wrapper):
    """
    自适应动作平滑包装器 - 根据训练进度调整平滑强度
    
    训练初期使用强平滑，后期逐渐减弱平滑，让模型学会精确控制
    """
    
    def __init__(self, env, initial_smoothing=0.9, final_smoothing=0.3, 
                 adaptation_steps=50000, max_change_rate=0.2):
        """
        初始化自适应动作平滑包装器
        
        Args:
            env: 原始环境
            initial_smoothing: 初始平滑因子
            final_smoothing: 最终平滑因子
            adaptation_steps: 自适应步数
            max_change_rate: 最大动作变化率
        """
        super().__init__(env)
        
        self.initial_smoothing = initial_smoothing
        self.final_smoothing = final_smoothing
        self.adaptation_steps = adaptation_steps
        self.max_change_rate = max_change_rate
        
        # 状态变量
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.global_step = 0
        self.episode_step = 0
        
        print(f"🔧 AdaptiveActionSmootherWrapper 已启用:")
        print(f"   初始平滑因子: {initial_smoothing}")
        print(f"   最终平滑因子: {final_smoothing}")
        print(f"   自适应步数: {adaptation_steps}")
    
    def _get_current_smoothing_factor(self):
        """根据训练进度计算当前平滑因子"""
        if self.global_step >= self.adaptation_steps:
            return self.final_smoothing
        
        # 线性衰减
        progress = self.global_step / self.adaptation_steps
        return self.initial_smoothing - (self.initial_smoothing - self.final_smoothing) * progress
    
    def reset(self, **kwargs):
        """重置环境"""
        obs, info = self.env.reset(**kwargs)
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.episode_step = 0
        return obs, info
    
    def step(self, action):
        """执行自适应平滑的步骤"""
        # 更新步数
        self.global_step += 1
        self.episode_step += 1
        
        # 获取当前平滑因子
        current_smoothing = self._get_current_smoothing_factor()
        
        # 确保动作在有效范围内
        action = np.clip(action, -1.0, 1.0)
        
        # 应用平滑
        if self.episode_step == 1:
            smoothed_action = action.copy()
        else:
            # 指数移动平均
            smoothed_action = (
                current_smoothing * self.last_action + 
                (1 - current_smoothing) * action
            )
            
            # 限制变化率
            action_diff = smoothed_action - self.last_action
            max_change = self.max_change_rate
            
            for i in range(len(action_diff)):
                if abs(action_diff[i]) > max_change:
                    action_diff[i] = np.sign(action_diff[i]) * max_change
            
            smoothed_action = self.last_action + action_diff
            smoothed_action = np.clip(smoothed_action, -1.0, 1.0)
        
        # 更新历史
        self.last_action = smoothed_action.copy()
        
        # 执行动作
        obs, reward, terminated, truncated, info = self.env.step(smoothed_action)
        
        # 添加调试信息
        info.update({
            'raw_action': action,
            'smoothed_action': smoothed_action,
            'current_smoothing_factor': current_smoothing,
            'global_step': self.global_step,
            'adaptation_progress': min(1.0, self.global_step / self.adaptation_steps)
        })
        
        return obs, reward, terminated, truncated, info 