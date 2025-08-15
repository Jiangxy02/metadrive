#!/usr/bin/env python3
"""
修复版PPO训练脚本 - 解决所有已知问题

集成修复：
1. 渲染错误修复
2. 动作平滑
3. Episode管理（防止卡死）
4. 稳定的PPO配置
5. 详细的调试信息

使用方法：
    python fixed_training.py --total-timesteps 50000
"""

import argparse
import sys
import os
import numpy as np
import time

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入依赖
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️ Stable Baselines3 未安装")

# 导入本地模块
from visual_training_monitor import VisualTrainingEnv
from metadrive.a_scen_env.trash.action_smoother import AdaptiveActionSmootherWrapper
from stable_ppo_config import get_stable_ppo_config, get_action_smoothing_config
from metadrive.a_scen_env.trash.episode_manager import EpisodeManager


class FixedTrainingCallback(BaseCallback):
    """
    修复版训练回调 - 监控所有可能的问题
    """
    
    def __init__(self, log_freq=500, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_count = 0
        self.last_episode_count = 0
        self.stuck_detection_threshold = 100  # 连续100步没有episode变化
        self.stuck_counter = 0
        self.last_reset_time = time.time()
        
    def _on_step(self) -> bool:
        # 检查是否有新episode
        if hasattr(self.training_env, 'episode_manager'):
            current_episodes = self.training_env.episode_manager.total_episodes
            if current_episodes > self.last_episode_count:
                self.stuck_counter = 0
                self.last_episode_count = current_episodes
                self.last_reset_time = time.time()
            else:
                self.stuck_counter += 1
        
        # 定期输出状态
        if self.num_timesteps % self.log_freq == 0:
            self._log_status()
        
        # 检查是否卡死
        if self.stuck_counter > self.stuck_detection_threshold:
            print(f"🚨 检测到训练卡死！连续{self.stuck_counter}步没有episode变化")
            print(f"   当前step: {self.num_timesteps}")
            print(f"   上次重置时间: {time.time() - self.last_reset_time:.1f}秒前")
            
            # 尝试强制重置环境
            if hasattr(self.training_env, 'reset'):
                print("   尝试强制重置环境...")
                try:
                    self.training_env.reset()
                    self.stuck_counter = 0
                    self.last_reset_time = time.time()
                    print("   ✅ 环境重置成功")
                except Exception as e:
                    print(f"   ❌ 环境重置失败: {e}")
                    # 如果重置失败，可能需要停止训练
                    return False
        
        return True
    
    def _log_status(self):
        """输出状态信息"""
        print(f"\n🔍 训练状态检查 (Step {self.num_timesteps}):")
        print(f"   Episode数量: {self.last_episode_count}")
        print(f"   卡死计数器: {self.stuck_counter}/{self.stuck_detection_threshold}")
        print(f"   距离上次重置: {time.time() - self.last_reset_time:.1f}秒")
        
        # 显示环境统计
        if hasattr(self.training_env, 'episode_manager'):
            stats = self.training_env.episode_manager.get_statistics()
            print(f"   强制重置次数: {stats.get('forced_resets', 0)}")
            print(f"   超时重置次数: {stats.get('timeout_resets', 0)}")
            print(f"   停滞重置次数: {stats.get('stale_resets', 0)}")


def create_fixed_env(scenario_type="random", num_scenarios=100, seed=None):
    """
    创建修复版训练环境
    """
    
    def _make_env():
        # 创建基础环境
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
            # 🔧 关键修复：Episode管理配置
            max_episode_steps=800,       # 减少最大步数，更频繁重置
            max_episode_time=120.0,      # 2分钟超时
            force_reset_threshold=1000,  # 强制重置阈值
            stale_detection_steps=20     # 停滞检测步数
        )
        
        # 🔧 关键修复：添加动作平滑
        env = AdaptiveActionSmootherWrapper(
            env,
            initial_smoothing=0.9,   # 初期强平滑
            final_smoothing=0.3,     # 后期精确控制
            adaptation_steps=20000,  # 前20k步自适应
            max_change_rate=0.2      # 限制动作变化率
        )
        
        print("🛠️ 修复版环境已创建:")
        print("   ✅ Episode管理器")
        print("   ✅ 动作平滑器")
        print("   ✅ 渲染文本修复")
        
        return env
    
    return _make_env


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="修复版PPO训练")
    
    parser.add_argument("--total-timesteps", type=int, default=50000,
                        help="总训练步数 (default: 50000)")
    parser.add_argument("--scenario-type", type=str, default="random",
                        choices=["random", "curriculum", "safe"],
                        help="场景类型 (default: random)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    parser.add_argument("--log-dir", type=str, default="./logs/fixed_training",
                        help="日志目录 (default: ./logs/fixed_training)")
    parser.add_argument("--log-freq", type=int, default=500,
                        help="日志频率 (default: 500)")
    
    args = parser.parse_args()
    
    if not SB3_AVAILABLE:
        print("❌ 需要安装 Stable Baselines3")
        print("安装命令: pip install stable-baselines3")
        return
    
    print("🛠️ 启动修复版PPO训练")
    print("=" * 60)
    print(f"训练步数: {args.total_timesteps:,}")
    print(f"场景类型: {args.scenario_type}")
    print(f"随机种子: {args.seed}")
    print(f"日志目录: {args.log_dir}")
    print("=" * 60)
    print("🔧 已集成修复:")
    print("   - Episode管理器（防止卡死）")
    print("   - 动作平滑器（稳定控制）")
    print("   - 渲染文本修复（无dtype错误）")
    print("   - 稳定PPO配置（优化参数）")
    print("   - 强化监控回调（实时检测）")
    print("=" * 60)
    
    try:
        # 创建环境
        print("🏗️ 创建训练环境...")
        env = create_fixed_env(
            scenario_type=args.scenario_type,
            num_scenarios=100,
            seed=args.seed
        )()
        
        # 设置日志
        os.makedirs(args.log_dir, exist_ok=True)
        logger = configure(args.log_dir, ["stdout", "csv", "tensorboard"])
        
        # 获取稳定的PPO配置
        print("⚙️ 配置PPO参数...")
        ppo_config = get_stable_ppo_config("visual_training")
        ppo_config.update({
            "tensorboard_log": args.log_dir,
            "seed": args.seed,
            # 🔧 关键修复：更保守的配置
            "learning_rate": 5e-5,    # 更低的学习率
            "n_steps": 512,           # 更小的步数
            "batch_size": 32,         # 更小的批次
            "clip_range": 0.05,       # 更小的裁剪范围
            "ent_coef": 0.0001,       # 更小的熵系数
        })
        
        # 创建PPO模型
        print("🤖 创建PPO模型...")
        model = PPO("MlpPolicy", env, **ppo_config)
        model.set_logger(logger)
        
        # 创建修复版回调
        callback = FixedTrainingCallback(
            log_freq=args.log_freq,
            verbose=1
        )
        
        print(f"\n🎯 开始修复版训练...")
        print(f"💡 监控提示:")
        print(f"   - 每{args.log_freq}步输出状态检查")
        print(f"   - 自动检测卡死并强制重置")
        print(f"   - 动作平滑逐步适应")
        print(f"   - Episode自动超时管理")
        
        # 记录开始时间
        start_time = time.time()
        
        # 开始训练
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # 训练完成
        training_time = time.time() - start_time
        print(f"\n🎉 训练完成!")
        print(f"   耗时: {training_time/60:.1f}分钟")
        print(f"   平均速度: {args.total_timesteps/training_time:.1f} steps/s")
        
        # 保存模型
        model_path = os.path.join(args.log_dir, "fixed_ppo_model")
        model.save(model_path)
        print(f"✅ 模型已保存: {model_path}")
        
        # 输出最终统计
        if hasattr(env, 'episode_manager'):
            stats = env.episode_manager.get_statistics()
            print(f"\n📊 最终统计:")
            print(f"   总Episode数: {stats['total_episodes']}")
            print(f"   强制重置: {stats['forced_resets']}")
            print(f"   超时重置: {stats['timeout_resets']}")
            print(f"   停滞重置: {stats['stale_resets']}")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断训练")
        if 'model' in locals():
            interrupted_path = os.path.join(args.log_dir, "fixed_ppo_interrupted")
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