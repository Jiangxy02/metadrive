#!/usr/bin/env python3
"""
可视化PPO训练监控器
在训练过程中实时显示场景和主车控制情况
"""

import sys
import os
import time
import threading
import queue
from typing import Dict, Any, Optional, List
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from random_scenario_generator import RandomScenarioGenerator, MetaDriveRandomEnv
from metadrive.a_scen_env.trash.render_text_fixer import fix_render_text
from metadrive.a_scen_env.trash.episode_manager import EpisodeManagedEnv

# SB3和其他导入
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    import gymnasium as gym
    SB3_AVAILABLE = True
except ImportError:
    print("❌ Stable Baselines3 未安装")
    SB3_AVAILABLE = False
    # 创建虚拟BaseCallback用于类定义
    class BaseCallback:
        def __init__(self, verbose=0):
            self.verbose = verbose
            self.n_calls = 0
            self.logger = None
        def _on_step(self):
            return True

# 可视化相关导入
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Rectangle, Circle
    from collections import deque
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("❌ Matplotlib 未安装，无法显示训练曲线")
    MATPLOTLIB_AVAILABLE = False


class VisualTrainingEnv(EpisodeManagedEnv, MetaDriveRandomEnv):
    """
    可视化训练环境
    在训练过程中启用渲染和记录
    """
    
    def __init__(self, 
                 scenario_type: str = "random",
                 num_scenarios: int = 1000,
                 seed: Optional[int] = None,
                 visual_config: Optional[Dict[str, Any]] = None,
                 **env_kwargs):
        """
        初始化可视化训练环境
        
        Args:
            visual_config: 可视化配置
                - render_mode: 渲染模式 ("human", "rgb_array")
                - window_size: 窗口大小
                - record_video: 是否录制视频
                - video_length: 视频长度（秒）
        """
        
        # 可视化配置
        self.visual_config = visual_config or {
            "render_mode": "human",
            "window_size": [1200, 800],
            "record_video": False,
            "video_length": 30,
            "show_sensors": True,
            "show_trajectory": True,
        }
        
        # 强制启用渲染
        window_size = self.visual_config.get("window_size", [1200, 800])
        if isinstance(window_size, list):
            window_size = tuple(window_size)  # 转换为tuple格式
            
        env_kwargs.update({
            "use_render": True,
            "window_size": window_size,
            "manual_control": False,  # 关闭手动控制，使用PPO
        })
        
        # 提取MetaDriveRandomEnv需要的参数
        generator = RandomScenarioGenerator(seed=seed)
        base_config = {
            "num_scenarios": num_scenarios,
            "start_seed": seed or 0,
        }
        base_config.update(env_kwargs)
        
        super().__init__(
            generator=generator,
            scenario_type=scenario_type,
            base_config=base_config
        )
        
        # 训练数据记录
        self.episode_count = 0
        self.step_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # 性能统计
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.action_history = deque(maxlen=1000)
        self.speed_history = deque(maxlen=1000)
        
        # 可视化状态
        self.render_enabled = True
        self.last_action = np.array([0.0, 0.0])
        self.predicted_action = np.array([0.0, 0.0])
        
        # 奖励函数配置和追踪（与MetaDriveRandomWrapper保持一致）
        self.reward_config = self._get_reward_config()
        self._last_position = None
        self._total_distance = 0.0
        self._episode_start_pos = None
        
    def reset(self, **kwargs):
        """重置环境并记录统计信息"""
        
        # 记录上一个episode的统计
        if self.episode_count > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
        
        # 重置计数器
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.episode_count += 1
        
        # 调用父类reset
        result = super().reset(**kwargs)
        
        # 重置进度跟踪变量
        if hasattr(self, 'agent') and hasattr(self.agent, 'position'):
            self._episode_start_pos = np.array(self.agent.position[:2])
            self._last_position = self._episode_start_pos.copy()
        else:
            self._episode_start_pos = np.array([0.0, 0.0])
            self._last_position = self._episode_start_pos.copy()
        
        self._total_distance = 0.0
        
        # 首次渲染
        if self.render_enabled:
            self.render_training_info()
        
        return result
    
    def step(self, action):
        """执行步骤并记录信息"""
        
        # 记录预测动作
        self.predicted_action = np.array(action)
        
        # 执行步骤
        result = super().step(action)
        
        # 解析结果
        if len(result) == 4:
            obs, original_reward, done, info = result
            terminated = truncated = done
        else:
            obs, original_reward, terminated, truncated, info = result
            done = terminated or truncated
        
        # 计算改进的奖励
        improved_reward = self._compute_improved_reward(obs, action, done, info, original_reward)
        
        # 更新统计
        self.current_episode_reward += improved_reward
        self.current_episode_length += 1
        self.step_count += 1
        
        # 记录历史数据
        self.action_history.append(action.copy())
        if hasattr(self, 'agent') and hasattr(self.agent, 'speed'):
            self.speed_history.append(self.agent.speed)
        else:
            self.speed_history.append(0.0)
        
        # 渲染训练信息
        if self.render_enabled:
            self.render_training_info()
        
        # 返回修改后的结果（使用改进的奖励）
        if len(result) == 4:
            return obs, improved_reward, done, info
        else:
            return obs, improved_reward, terminated, truncated, info
    
    def render_training_info(self):
        """渲染训练信息到环境窗口"""
        
        try:
            # 基础环境渲染
            training_text = self._get_training_text()
            # 🔧 修复渲染文本，避免dtype错误
            safe_text = fix_render_text(training_text)
            # 直接调用父类的render方法
            super().render(text=safe_text)
            
        except Exception as e:
            print(f"渲染错误: {e}")
            # 尝试无文本渲染
            try:
                super().render(text={})
            except:
                super().render()
    
    def _get_training_text(self) -> Dict[str, str]:
        """获取训练信息文本"""
        
        # 基础信息
        training_info = {
            "🎯 Training Info": "=" * 20,
            "Episode": f"{self.episode_count}",
            "Step": f"{self.step_count}",
            "Episode Reward": f"{self.current_episode_reward:.2f}",
            "Episode Length": f"{self.current_episode_length}",
        }
        
        # 性能统计
        if self.episode_rewards:
            mean_reward = np.mean(list(self.episode_rewards))
            mean_length = np.mean(list(self.episode_lengths))
            training_info.update({
                "Mean Reward (100)": f"{mean_reward:.2f}",
                "Mean Length (100)": f"{mean_length:.1f}",
            })
        
        # 当前动作信息
        training_info.update({
            "🎮 Action Info": "=" * 15,
            "Predicted Action": f"[{self.predicted_action[0]:.2f}, {self.predicted_action[1]:.2f}]",
            "Steering": f"{self.predicted_action[0]:.2f}",
            "Throttle": f"{self.predicted_action[1]:.2f}",
        })
        
        # 车辆状态
        if hasattr(self, 'agent'):
            agent = self.agent
            training_info.update({
                "🚗 Vehicle Info": "=" * 15,
                "Speed": f"{getattr(agent, 'speed', 0):.1f} m/s",
                "Position": f"({getattr(agent, 'position', [0, 0])[0]:.1f}, {getattr(agent, 'position', [0, 0])[1]:.1f})",
                "Heading": f"{getattr(agent, 'heading_theta', 0):.2f} rad",
            })
        
        # 场景信息
        if hasattr(self.env, 'config'):
            config = self.env.config
            training_info.update({
                "🗺️ Scenario Info": "=" * 15,
                "Map Blocks": f"{config.get('map', 'N/A')}",
                "Traffic Density": f"{config.get('traffic_density', 0):.2f}",
                "Accident Prob": f"{config.get('accident_prob', 0):.2f}",
            })
        
        return training_info
    
    def toggle_render(self):
        """切换渲染状态"""
        self.render_enabled = not self.render_enabled
        print(f"渲染状态: {'开启' if self.render_enabled else '关闭'}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        return {
            "episode_count": self.episode_count,
            "step_count": self.step_count,
            "episode_rewards": list(self.episode_rewards),
            "episode_lengths": list(self.episode_lengths),
            "action_history": list(self.action_history),
            "speed_history": list(self.speed_history),
        }
    
    def _get_reward_config(self) -> Dict[str, Any]:
        """获取改进的奖励配置"""
        return {
            # 前进奖励 - 核心驱动力
            "forward_reward_weight": 5.0,      # 前进距离奖励权重（最重要）
            "speed_reward_weight": 1.0,        # 合理速度奖励权重
            
            # 方向性奖励
            "heading_reward_weight": 2.0,      # 朝向正确方向奖励
            "lane_center_weight": 0.5,         # 车道中心保持奖励
            
            # 惩罚项
            "crash_penalty": -20.0,            # 碰撞严重惩罚
            "out_road_penalty": -10.0,         # 出路惩罚
            "backward_penalty": -2.0,          # 倒退惩罚
            "stop_penalty": -0.5,              # 停车惩罚
            
            # 完成奖励
            "completion_bonus": 50.0,          # 到达终点大奖励
            "distance_bonus_threshold": 100.0, # 距离奖励阈值
            
            # 时间惩罚（轻微）
            "time_penalty": -0.02,             # 减少每步惩罚
        }
    
    def _compute_improved_reward(self, obs, action, done, info, original_reward) -> float:
        """
        计算改进的奖励函数 - 与MetaDriveRandomWrapper保持一致
        """
        reward = 0.0
        config = self.reward_config
        
        # 时间惩罚（轻微）
        reward += config["time_penalty"]
        
        try:
            if hasattr(self, 'agent'):
                agent = self.agent
                
                # 1. 前进距离奖励（核心驱动力）
                if hasattr(agent, 'position') and self._last_position is not None:
                    current_pos = np.array(agent.position[:2])
                    
                    # 计算相对于起点的总距离
                    total_distance_from_start = np.linalg.norm(current_pos - self._episode_start_pos)
                    
                    # 判断是否在前进（基于与起点的距离增加）
                    distance_increase = total_distance_from_start - self._total_distance
                    
                    if distance_increase > 0:
                        # 前进奖励 - 线性增长
                        forward_reward = distance_increase * config["forward_reward_weight"]
                        reward += forward_reward
                        self._total_distance = total_distance_from_start
                    elif distance_increase < -0.5:  # 明显后退
                        # 倒退惩罚
                        reward += config["backward_penalty"]
                    
                    # 更新位置
                    self._last_position = current_pos
                
                # 2. 速度奖励（鼓励保持合理的前进速度）
                if hasattr(agent, 'speed'):
                    speed = agent.speed
                    
                    if speed < 1.0:  # 停车惩罚
                        reward += config["stop_penalty"]
                    elif 5.0 <= speed <= 25.0:  # 理想速度范围
                        speed_reward = min(speed / 25.0, 1.0)  # 归一化到[0,1]
                        reward += config["speed_reward_weight"] * speed_reward
                    elif speed > 35.0:  # 过快惩罚
                        reward += config["speed_reward_weight"] * (-0.5)
                
                # 3. 方向奖励（朝向道路前方）
                if hasattr(agent, 'heading_theta'):
                    heading = agent.heading_theta
                    # 奖励朝向正确方向（-π/4 到 π/4）
                    heading_penalty = abs(heading) / (np.pi / 2)  # 归一化
                    if heading_penalty < 0.5:  # 方向偏差不大
                        reward += config["heading_reward_weight"] * (1 - heading_penalty)
                
        except Exception as e:
            # 如果获取信息失败，不影响基础流程
            pass
        
        # 5. 结束状态奖励/惩罚
        if done:
            if info.get("crash", False) or info.get("crash_vehicle", False):
                reward += config["crash_penalty"]
            elif info.get("out_of_road", False):
                reward += config["out_road_penalty"]
            elif info.get("arrive_dest", False):
                # 到达目标的大奖励
                completion_reward = config["completion_bonus"]
                # 额外距离奖励
                if self._total_distance > config["distance_bonus_threshold"]:
                    completion_reward += self._total_distance * 0.1
                reward += completion_reward
        
        return reward


class VisualTrainingCallback(BaseCallback):
    """
    可视化训练回调
    集成训练监控和可视化
    """
    
    def __init__(self, 
                 log_freq: int = 1000,
                 plot_freq: int = 5000,
                 save_freq: int = 10000,
                 verbose: int = 1,
                 training_env=None):
        super().__init__(verbose)
        
        self.log_freq = log_freq
        self.plot_freq = plot_freq
        self.save_freq = save_freq
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_steps = []
        self.loss_history = []
        
        # 训练环境引用
        self._training_env = training_env
        
        # 实时绘图设置
        if MATPLOTLIB_AVAILABLE:
            self.setup_realtime_plots()
        
    def setup_realtime_plots(self):
        """设置实时绘图"""
        try:
            plt.ion()  # 交互模式
            
            # 设置中文字体（如果可用）
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                pass  # 忽略字体设置失败
            
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle("PPO Training Monitor", fontsize=16)  # 使用英文避免字体问题
            
            # 子图标题（使用英文）
            self.axes[0, 0].set_title("Episode Rewards")
            self.axes[0, 1].set_title("Episode Lengths")
            self.axes[1, 0].set_title("Action Distribution")
            self.axes[1, 1].set_title("Speed Distribution")
            
            self.fig.tight_layout()
            self.plot_initialized = True
            
        except Exception as e:
            print(f"绘图初始化失败: {e}")
            self.plot_initialized = False
    
    def _on_step(self) -> bool:
        """每步调用"""
        
        # 收集训练数据
        if self._training_env:
            if hasattr(self._training_env, 'envs'):
                # 向量化环境
                for i, env in enumerate(self._training_env.envs):
                    if hasattr(env, 'get_training_stats'):
                        stats = env.get_training_stats()
                        self._update_stats(stats)
            else:
                # 单一环境
                if hasattr(self._training_env, 'get_training_stats'):
                    stats = self._training_env.get_training_stats()
                    self._update_stats(stats)
        
        # 定期日志
        if self.n_calls % self.log_freq == 0:
            self._log_training_progress()
        
        # 定期绘图
        if self.plot_initialized and self.n_calls % self.plot_freq == 0:
            self._update_plots()
        
        # 定期保存
        if self.n_calls % self.save_freq == 0:
            self._save_training_snapshot()
        
        return True
    
    def _update_stats(self, stats: Dict[str, Any]):
        """更新训练统计"""
        if stats['episode_rewards']:
            self.episode_rewards.extend(stats['episode_rewards'])
        if stats['episode_lengths']:
            self.episode_lengths.extend(stats['episode_lengths'])
    
    def _log_training_progress(self):
        """记录训练进度"""
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-10:]
            recent_lengths = self.episode_lengths[-10:]
            
            mean_reward = np.mean(recent_rewards)
            mean_length = np.mean(recent_lengths)
            
            if self.verbose >= 1:
                print(f"\n🎯 Step {self.n_calls}:")
                print(f"  Recent 10 episodes - Reward: {mean_reward:.2f}, Length: {mean_length:.1f}")
                
                if len(self.episode_rewards) >= 100:
                    last_100_reward = np.mean(self.episode_rewards[-100:])
                    print(f"  Last 100 episodes - Mean Reward: {last_100_reward:.2f}")
            
            # 记录到tensorboard
            self.logger.record("train/mean_episode_reward", mean_reward)
            self.logger.record("train/mean_episode_length", mean_length)
            self.logger.record("train/total_episodes", len(self.episode_rewards))
    
    def _update_plots(self):
        """更新实时绘图"""
        if not self.plot_initialized:
            return
            
        try:
            # 清除所有子图
            for ax in self.axes.flat:
                ax.clear()
            
            # 绘制奖励曲线
            if self.episode_rewards:
                self.axes[0, 0].plot(self.episode_rewards, 'b-', alpha=0.7)
                self.axes[0, 0].set_title(f"Episode奖励 (最新: {self.episode_rewards[-1]:.2f})")
                self.axes[0, 0].grid(True)
            
            # 绘制长度曲线
            if self.episode_lengths:
                self.axes[0, 1].plot(self.episode_lengths, 'g-', alpha=0.7)
                self.axes[0, 1].set_title(f"Episode长度 (最新: {self.episode_lengths[-1]})")
                self.axes[0, 1].grid(True)
            
            # 绘制动作分布（如果有环境访问权限）
            if self._training_env and hasattr(self._training_env, 'get_training_stats'):
                stats = self._training_env.get_training_stats()
                if stats['action_history']:
                    actions = np.array(stats['action_history'])
                    self.axes[1, 0].hist2d(actions[:, 0], actions[:, 1], bins=20, alpha=0.7)
                    self.axes[1, 0].set_title("Action Distribution")
                    self.axes[1, 0].set_xlabel("Steering")
                    self.axes[1, 0].set_ylabel("Throttle")
            
            # 绘制速度分布
            if self._training_env and hasattr(self._training_env, 'get_training_stats'):
                stats = self._training_env.get_training_stats()
                if stats['speed_history']:
                    self.axes[1, 1].hist(stats['speed_history'], bins=30, alpha=0.7, color='orange')
                    self.axes[1, 1].set_title("Speed Distribution")
                    self.axes[1, 1].set_xlabel("Speed (m/s)")
                    self.axes[1, 1].set_ylabel("Frequency")
            
            plt.tight_layout()
            plt.pause(0.01)  # 短暂暂停以更新显示
            
        except Exception as e:
            print(f"绘图更新失败: {e}")
    
    def _save_training_snapshot(self):
        """保存训练快照"""
        try:
            snapshot = {
                "step": self.n_calls,
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "timestamp": time.time(),
            }
            
            # 保存到日志目录
            if hasattr(self.logger, 'dir'):
                snapshot_path = os.path.join(self.logger.dir, f"training_snapshot_{self.n_calls}.json")
                import json
                with open(snapshot_path, 'w') as f:
                    json.dump(snapshot, f, indent=2)
                
                if self.verbose >= 1:
                    print(f"💾 训练快照已保存: {snapshot_path}")
                    
        except Exception as e:
            print(f"保存快照失败: {e}")


def create_visual_training_environment(scenario_type: str = "random",
                                     num_scenarios: int = 100,
                                     seed: Optional[int] = None,
                                     visual_config: Optional[Dict[str, Any]] = None,
                                     **config_kwargs):
    """
    创建可视化训练环境
    
    Args:
        scenario_type: 场景类型
        num_scenarios: 场景数量  
        seed: 随机种子
        visual_config: 可视化配置
        **config_kwargs: 额外配置
        
    Returns:
        可视化训练环境
    """
    
    def _make_env():
        return VisualTrainingEnv(
            scenario_type=scenario_type,
            num_scenarios=num_scenarios,
            seed=seed,
            visual_config=visual_config,
            **config_kwargs
        )
    
    return _make_env


def train_ppo_with_visualization(args):
    """
    开始可视化PPO训练
    """
    
    if not SB3_AVAILABLE:
        raise ImportError("Stable Baselines3 not available")
    
    print(f"🎬 开始可视化PPO训练")
    print(f"  场景类型: {args.scenario_type}")
    print(f"  可视化: 开启")
    print(f"  训练步数: {args.total_timesteps}")
    
    # 可视化配置
    window_size = getattr(args, 'window_size', [1200, 800])
    if isinstance(window_size, list):
        window_size = tuple(window_size)
        
    visual_config = {
        "render_mode": "human",
        "window_size": window_size,
        "show_sensors": getattr(args, 'show_sensors', True),
        "show_trajectory": getattr(args, 'show_trajectory', True),
    }
    
    # 创建可视化环境
    env_config = {
        "scenario_type": args.scenario_type,
        "num_scenarios": getattr(args, 'num_scenarios', 100),
        "seed": getattr(args, 'seed', None),
        "visual_config": visual_config,
    }
    
    # 单环境训练（可视化模式下不使用并行）
    env = create_visual_training_environment(**env_config)()
    
    # 设置日志
    log_dir = getattr(args, 'log_dir', './logs/ppo_visual')
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
    
    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=getattr(args, 'learning_rate', 3e-4),
        n_steps=getattr(args, 'n_steps', 1024),  # 较小的步数，便于观察
        batch_size=getattr(args, 'batch_size', 64),
        n_epochs=getattr(args, 'n_epochs', 10),
        gamma=getattr(args, 'gamma', 0.99),
        gae_lambda=getattr(args, 'gae_lambda', 0.95),
        clip_range=getattr(args, 'clip_range', 0.2),
        tensorboard_log=log_dir,
        device=getattr(args, 'device', 'auto'),
        seed=getattr(args, 'seed', None)
    )
    
    model.set_logger(new_logger)
    
    # 创建可视化回调
    callback = VisualTrainingCallback(
        log_freq=getattr(args, 'log_freq', 500),
        plot_freq=getattr(args, 'plot_freq', 2000),
        verbose=1,
        training_env=env
    )
    
    print(f"\n🎯 开始可视化训练...")
    print(f"💡 提示: 训练过程中可以观看MetaDrive窗口和实时训练曲线")
    
    # 开始训练
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # 保存模型
        model_path = os.path.join(log_dir, "ppo_visual_model")
        model.save(model_path)
        print(f"✅ 模型已保存: {model_path}")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断训练")
        model_path = os.path.join(log_dir, "ppo_visual_interrupted")
        model.save(model_path)
        print(f"💾 中断模型已保存: {model_path}")
    
    finally:
        env.close()
        if MATPLOTLIB_AVAILABLE:
            plt.ioff()  # 关闭交互模式
    
    return model


if __name__ == "__main__":
    print("🎬 可视化PPO训练启动器")
    
    # 简单配置类
    class Args:
        def __init__(self):
            self.scenario_type = "random"
            self.total_timesteps = 50000
            self.learning_rate = 3e-4
            self.n_steps = 1024
            self.batch_size = 64
            self.seed = 42
            self.log_freq = 500
            self.plot_freq = 2000
            self.window_size = (1200, 800)
    
    args = Args()
    
    try:
        model = train_ppo_with_visualization(args)
        print("\n🎉 可视化训练完成!")
        
    except KeyboardInterrupt:
        print("\n👋 用户退出")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc() 