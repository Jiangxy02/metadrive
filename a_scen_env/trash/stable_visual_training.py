#!/usr/bin/env python3
"""
稳定可视化PPO训练脚本 - 解决主车控制不稳定问题

主要改进：
1. 集成动作平滑包装器，解决控制抖动
2. 使用优化的PPO参数配置
3. 增强的训练监控和调试信息
4. 自适应训练策略

使用方法：
    python stable_visual_training.py --scenario-type random --total-timesteps 100000
"""

import argparse
import sys
import os
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入依赖
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️ Stable Baselines3 未安装")

# 导入本地模块
from visual_training_monitor import VisualTrainingEnv, VisualTrainingCallback
from metadrive.a_scen_env.trash.action_smoother import ActionSmootherWrapper, AdaptiveActionSmootherWrapper
from stable_ppo_config import get_stable_ppo_config, get_action_smoothing_config
from metadrive.a_scen_env.trash.render_text_fixer import fix_render_text
from metadrive.a_scen_env.trash.episode_manager import add_episode_management


class StableTrainingCallback(BaseCallback):
    """
    稳定训练回调 - 监控训练稳定性
    """
    
    def __init__(self, log_freq=1000, action_log_freq=100, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.action_log_freq = action_log_freq
        self.action_history = []
        self.reward_history = []
        
    def _on_step(self) -> bool:
        # 记录动作和奖励
        if len(self.locals.get('infos', [])) > 0:
            info = self.locals['infos'][0]
            
            # 记录动作信息
            if 'smoothed_action' in info:
                self.action_history.append({
                    'step': self.num_timesteps,
                    'raw_action': info.get('raw_action', [0, 0]),
                    'smoothed_action': info['smoothed_action'],
                    'action_change': info.get('action_change', 0),
                    'smoothing_factor': info.get('current_smoothing_factor', 0.8)
                })
            
            # 记录奖励
            if 'rewards' in self.locals:
                self.reward_history.append({
                    'step': self.num_timesteps,
                    'reward': self.locals['rewards'][0]
                })
        
        # 定期输出统计信息
        if self.num_timesteps % self.log_freq == 0:
            self._log_training_stats()
            
        if self.num_timesteps % self.action_log_freq == 0:
            self._log_action_stats()
            
        return True
    
    def _log_training_stats(self):
        """输出训练统计信息"""
        if len(self.reward_history) > 100:
            recent_rewards = [r['reward'] for r in self.reward_history[-100:]]
            avg_reward = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)
            
            print(f"\n📊 训练统计 (Step {self.num_timesteps}):")
            print(f"   近100步平均奖励: {avg_reward:.3f} ± {reward_std:.3f}")
    
    def _log_action_stats(self):
        """输出动作统计信息"""
        if len(self.action_history) > 10:
            recent_actions = self.action_history[-10:]
            
            # 计算动作变化
            action_changes = [a['action_change'] for a in recent_actions]
            avg_change = np.mean(action_changes)
            
            # 计算动作范围
            raw_actions = np.array([a['raw_action'] for a in recent_actions])
            smoothed_actions = np.array([a['smoothed_action'] for a in recent_actions])
            
            raw_std = np.std(raw_actions, axis=0)
            smooth_std = np.std(smoothed_actions, axis=0)
            
            # 获取当前平滑因子
            current_smoothing = recent_actions[-1].get('smoothing_factor', 0.8)
            
            print(f"🎮 动作统计 (Step {self.num_timesteps}):")
            print(f"   平均动作变化: {avg_change:.4f}")
            print(f"   原始动作标准差: [{raw_std[0]:.3f}, {raw_std[1]:.3f}]")
            print(f"   平滑动作标准差: [{smooth_std[0]:.3f}, {smooth_std[1]:.3f}]")
            print(f"   当前平滑因子: {current_smoothing:.3f}")


def create_stable_visual_env(scenario_type="random", num_scenarios=100, seed=None, 
                           use_action_smoothing=True, smoothing_config=None):
    """
    创建稳定的可视化训练环境
    """
    
    def _make_env():
        # 创建基础环境（已经包含episode管理）
        env = VisualTrainingEnv(
            scenario_type=scenario_type,
            num_scenarios=num_scenarios,
            seed=seed,
            visual_config={
                "render_mode": "human",
                "window_size": (1200, 800),
                "show_sensors": True,
                "show_trajectory": True,
            },
            # Episode管理配置
            max_episode_steps=1000,      # 每个episode最大步数
            max_episode_time=180.0,      # 每个episode最大时间（3分钟）
            force_reset_threshold=1200,  # 强制重置阈值
            stale_detection_steps=30     # 停滞检测步数
        )
        
        # 添加动作平滑
        if use_action_smoothing:
            config = smoothing_config or get_action_smoothing_config()
            
            if config["use_adaptive"]:
                env = AdaptiveActionSmootherWrapper(
                    env,
                    initial_smoothing=config["initial_smoothing"],
                    final_smoothing=config["final_smoothing"],
                    adaptation_steps=config["adaptation_steps"],
                    max_change_rate=config["max_change_rate"]
                )
                print(f"🔧 自适应动作平滑已启用")
            else:
                env = ActionSmootherWrapper(
                    env,
                    smoothing_factor=config["smoothing_factor"],
                    max_change_rate=config["max_change_rate"]
                )
                print(f"🔧 基础动作平滑已启用")
        
        return env
    
    return _make_env


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="稳定可视化PPO训练")
    
    # 基础配置
    parser.add_argument("--scenario-type", type=str, default="random",
                        choices=["random", "curriculum", "safe"],
                        help="场景类型 (default: random)")
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="总训练步数 (default: 100000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    
    # 稳定性配置
    parser.add_argument("--use-action-smoothing", action="store_true", default=True,
                        help="启用动作平滑 (default: True)")
    parser.add_argument("--smoothing-factor", type=float, default=0.8,
                        help="动作平滑因子 (default: 0.8)")
    parser.add_argument("--max-change-rate", type=float, default=0.25,
                        help="最大动作变化率 (default: 0.25)")
    parser.add_argument("--use-adaptive-smoothing", action="store_true", default=True,
                        help="使用自适应平滑 (default: True)")
    
    # PPO配置
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="学习率 (default: 1e-4)")
    parser.add_argument("--n-steps", type=int, default=1024,
                        help="每次更新步数 (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批量大小 (default: 64)")
    parser.add_argument("--clip-range", type=float, default=0.08,
                        help="PPO裁剪范围 (default: 0.08)")
    
    # 输出配置
    parser.add_argument("--log-dir", type=str, default="./logs/stable_ppo_visual",
                        help="日志目录 (default: ./logs/stable_ppo_visual)")
    parser.add_argument("--log-freq", type=int, default=1000,
                        help="日志频率 (default: 1000)")
    
    args = parser.parse_args()
    
    if not SB3_AVAILABLE:
        print("❌ 需要安装 Stable Baselines3")
        print("安装命令: pip install stable-baselines3")
        return
    
    print("🚗 启动稳定可视化PPO训练")
    print("=" * 60)
    print(f"场景类型: {args.scenario_type}")
    print(f"训练步数: {args.total_timesteps:,}")
    print(f"动作平滑: {'开启' if args.use_action_smoothing else '关闭'}")
    print(f"自适应平滑: {'开启' if args.use_adaptive_smoothing else '关闭'}")
    print(f"学习率: {args.learning_rate}")
    print(f"裁剪范围: {args.clip_range}")
    print("=" * 60)
    
    try:
        # 配置动作平滑
        smoothing_config = {
            "smoothing_factor": args.smoothing_factor,
            "max_change_rate": args.max_change_rate,
            "use_adaptive": args.use_adaptive_smoothing,
            "initial_smoothing": 0.95,
            "final_smoothing": 0.4,
            "adaptation_steps": args.total_timesteps // 3,  # 前1/3训练时间自适应
        }
        
        # 创建环境
        env = create_stable_visual_env(
            scenario_type=args.scenario_type,
            num_scenarios=100,
            seed=args.seed,
            use_action_smoothing=args.use_action_smoothing,
            smoothing_config=smoothing_config
        )()
        
        # 设置日志
        os.makedirs(args.log_dir, exist_ok=True)
        logger = configure(args.log_dir, ["stdout", "csv", "tensorboard"])
        
        # 获取稳定的PPO配置
        ppo_config = get_stable_ppo_config("visual_training")
        
        # 用命令行参数覆盖配置
        ppo_config.update({
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "clip_range": args.clip_range,
            "tensorboard_log": args.log_dir,
            "seed": args.seed,
        })
        
        # 创建PPO模型
        print(f"\n🤖 创建PPO模型...")
        model = PPO("MlpPolicy", env, **ppo_config)
        model.set_logger(logger)
        
        # 创建回调
        stable_callback = StableTrainingCallback(
            log_freq=args.log_freq,
            action_log_freq=100,
            verbose=1
        )
        
        visual_callback = VisualTrainingCallback(
            log_freq=args.log_freq,
            plot_freq=2000,
            verbose=1,
            training_env=env
        )
        
        print(f"\n🎯 开始稳定训练...")
        print(f"💡 观察提示:")
        print(f"   - 训练初期动作会比较平滑，随训练进行逐渐精确")
        print(f"   - 注意观察动作变化统计，确认平滑效果")
        print(f"   - MetaDrive窗口显示实时驾驶，Matplotlib显示训练曲线")
        
        # 开始训练
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[stable_callback, visual_callback],
            progress_bar=True
        )
        
        # 保存模型
        model_path = os.path.join(args.log_dir, "stable_ppo_model")
        model.save(model_path)
        print(f"\n✅ 稳定模型已保存: {model_path}")
        
        # 输出训练摘要
        print(f"\n📈 训练完成摘要:")
        if len(stable_callback.action_history) > 0:
            final_smoothing = stable_callback.action_history[-1].get('smoothing_factor', 0.8)
            print(f"   最终平滑因子: {final_smoothing:.3f}")
        
        if len(stable_callback.reward_history) > 100:
            final_rewards = [r['reward'] for r in stable_callback.reward_history[-100:]]
            print(f"   最终100步平均奖励: {np.mean(final_rewards):.3f}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断训练")
        # 保存中断的模型
        if 'model' in locals():
            interrupted_path = os.path.join(args.log_dir, "stable_ppo_interrupted")
            model.save(interrupted_path)
            print(f"💾 中断模型已保存: {interrupted_path}")
            
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if 'env' in locals():
            env.close()
            print("🔒 环境已关闭")


if __name__ == "__main__":
    main() 