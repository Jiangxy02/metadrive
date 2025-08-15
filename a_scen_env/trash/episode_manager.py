"""
Episode管理器 - 解决训练卡死问题

问题：
- PPO训练卡在同一个step不动
- 环境无法正确结束episode
- 缺少强制重置机制

解决方案：
- 添加多种结束条件检查
- 强制episode超时重置
- 确保done状态正确传递
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import time


class EpisodeManager:
    """
    Episode生命周期管理器
    
    功能：
    1. 监控episode状态
    2. 强制超时重置
    3. 确保done条件正确触发
    4. 提供调试信息
    """
    
    def __init__(self, 
                 max_episode_steps: int = 1000,
                 max_episode_time: float = 300.0,  # 5分钟超时
                 force_reset_threshold: int = 1500,  # 强制重置阈值
                 stale_detection_steps: int = 50):   # 停滞检测步数
        
        self.max_episode_steps = max_episode_steps
        self.max_episode_time = max_episode_time
        self.force_reset_threshold = force_reset_threshold
        self.stale_detection_steps = stale_detection_steps
        
        # 状态跟踪
        self.reset_episode_state()
        
        # 统计信息
        self.total_episodes = 0
        self.forced_resets = 0
        self.timeout_resets = 0
        self.stale_resets = 0
        
        print(f"🎮 EpisodeManager 初始化:")
        print(f"   最大步数: {max_episode_steps}")
        print(f"   超时时间: {max_episode_time}s")
        print(f"   强制重置阈值: {force_reset_threshold}")
    
    def reset_episode_state(self):
        """重置episode状态"""
        self.episode_start_time = time.time()
        self.episode_step_count = 0
        self.last_position = None
        self.position_history = []
        self.stuck_counter = 0
        self.last_reward = 0.0
        self.cumulative_reward = 0.0
        
    def should_end_episode(self, 
                          obs: np.ndarray,
                          reward: float,
                          done: bool,
                          info: Dict[str, Any],
                          agent_position: Optional[Tuple[float, float]] = None) -> Tuple[bool, str]:
        """
        检查是否应该结束episode
        
        Returns:
            (should_end, reason)
        """
        self.episode_step_count += 1
        self.cumulative_reward += reward
        
        # 1. 原始done状态
        if done:
            return True, "original_done"
        
        # 2. 步数超限
        if self.episode_step_count >= self.max_episode_steps:
            return True, "max_steps"
        
        # 3. 时间超时
        episode_time = time.time() - self.episode_start_time
        if episode_time > self.max_episode_time:
            self.timeout_resets += 1
            return True, f"timeout_{episode_time:.1f}s"
        
        # 4. 强制重置（防止死锁）
        if self.episode_step_count >= self.force_reset_threshold:
            self.forced_resets += 1
            return True, f"force_reset_{self.episode_step_count}"
        
        # 5. 停滞检测（位置不变）
        if agent_position is not None:
            self.position_history.append(agent_position)
            if len(self.position_history) > self.stale_detection_steps:
                self.position_history.pop(0)
                
                # 检查是否在原地不动
                if len(self.position_history) == self.stale_detection_steps:
                    positions = np.array(self.position_history)
                    position_variance = np.var(positions, axis=0)
                    
                    # 如果位置方差很小，说明在原地打转
                    if np.all(position_variance < 0.1):  # 方差阈值
                        self.stale_resets += 1
                        return True, f"stale_position_var_{position_variance.max():.4f}"
        
        # 6. 检查reward异常
        if np.isnan(reward) or np.isinf(reward):
            return True, f"invalid_reward_{reward}"
        
        # 7. 检查obs异常
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            return True, "invalid_observation"
        
        return False, "continue"
    
    def on_episode_end(self, reason: str):
        """episode结束时调用"""
        self.total_episodes += 1
        episode_time = time.time() - self.episode_start_time
        
        # 输出调试信息
        if self.total_episodes % 10 == 0 or reason.startswith(("timeout", "force", "stale")):
            print(f"📊 Episode {self.total_episodes} 结束: {reason}")
            print(f"   步数: {self.episode_step_count}")
            print(f"   时间: {episode_time:.1f}s")
            print(f"   累积奖励: {self.cumulative_reward:.2f}")
            if reason.startswith(("timeout", "force", "stale")):
                print(f"   ⚠️ 异常结束原因: {reason}")
        
        # 重置状态
        self.reset_episode_state()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_episodes": self.total_episodes,
            "forced_resets": self.forced_resets,
            "timeout_resets": self.timeout_resets,
            "stale_resets": self.stale_resets,
            "current_episode_steps": self.episode_step_count,
            "current_episode_time": time.time() - self.episode_start_time,
        }


class EpisodeManagedEnv:
    """
    支持Episode管理的环境混入类
    """
    
    def __init__(self, *args, **kwargs):
        # 提取episode管理器配置
        episode_config = {
            "max_episode_steps": kwargs.pop("max_episode_steps", 1000),
            "max_episode_time": kwargs.pop("max_episode_time", 300.0),
            "force_reset_threshold": kwargs.pop("force_reset_threshold", 1500),
            "stale_detection_steps": kwargs.pop("stale_detection_steps", 50),
        }
        
        # 初始化父类
        super().__init__(*args, **kwargs)
        
        # 创建episode管理器
        self.episode_manager = EpisodeManager(**episode_config)
        
        print("🎮 EpisodeManagedEnv 已启用")
    
    def reset(self, **kwargs):
        """重置环境时重置episode管理器"""
        # 重置episode管理器
        self.episode_manager.reset_episode_state()
        
        # 调用父类reset
        result = super().reset(**kwargs)
        
        return result
    
    def step(self, action):
        """步骤执行时检查episode结束条件"""
        # 执行原始步骤
        result = super().step(action)
        
        # 解析结果
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = truncated = done
        else:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        
        # 获取agent位置（如果可用）
        agent_position = None
        if hasattr(self, 'agent') and hasattr(self.agent, 'position'):
            agent_position = tuple(self.agent.position[:2])
        
        # 检查是否应该结束episode
        should_end, reason = self.episode_manager.should_end_episode(
            obs, reward, done, info, agent_position
        )
        
        # 如果需要强制结束
        if should_end and not done:
            print(f"🔄 强制结束episode: {reason}")
            done = terminated = truncated = True
            info["forced_termination"] = True
            info["termination_reason"] = reason
        
        # 如果episode结束，通知管理器
        if done:
            self.episode_manager.on_episode_end(reason if should_end else "natural")
        
        # 添加episode统计到info
        info.update(self.episode_manager.get_statistics())
        
        # 返回结果
        if len(result) == 4:
            return obs, reward, done, info
        else:
            return obs, reward, terminated, truncated, info


def add_episode_management(env_class):
    """
    装饰器：为任何环境类添加episode管理功能
    
    用法:
        @add_episode_management
        class MyEnv(SomeBaseEnv):
            pass
    """
    
    class ManagedEnv(EpisodeManagedEnv, env_class):
        """带episode管理的环境"""
        pass
    
    # 保持原有类名
    ManagedEnv.__name__ = f"Managed{env_class.__name__}"
    ManagedEnv.__qualname__ = f"Managed{env_class.__qualname__}"
    
    return ManagedEnv


# === 使用示例 ===
if __name__ == "__main__":
    print("Episode管理器测试:")
    
    manager = EpisodeManager()
    
    # 模拟episode
    for step in range(100):
        obs = np.random.randn(10)
        reward = np.random.randn()
        done = False
        info = {}
        position = (step * 0.1, 0)  # 缓慢移动
        
        should_end, reason = manager.should_end_episode(
            obs, reward, done, info, position
        )
        
        if should_end:
            print(f"Episode在步骤{step}结束: {reason}")
            manager.on_episode_end(reason)
            break
    
    print("\n统计信息:")
    stats = manager.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}") 