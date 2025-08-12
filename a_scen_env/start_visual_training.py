#!/usr/bin/env python3
"""
简化版可视化PPO训练启动器
一键启动可视化训练
"""

import argparse
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visual_training_monitor import train_ppo_with_visualization


def main():
    parser = argparse.ArgumentParser(description="可视化PPO训练")
    
    # 基础配置
    parser.add_argument("--scenario-type", type=str, default="random",
                        choices=["random", "curriculum", "safe"],
                        help="场景类型 (default: random)")
    parser.add_argument("--total-timesteps", type=int, default=50000,
                        help="总训练步数 (default: 50000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (default: 42)")
    
    # 训练参数
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="学习率 (default: 3e-4)")
    parser.add_argument("--n-steps", type=int, default=1024,
                        help="每次更新步数 (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批量大小 (default: 64)")
    
    # 可视化配置
    parser.add_argument("--window-size", type=int, nargs=2, default=[1200, 800],
                        help="窗口大小 [宽度 高度] (default: 1200 800)")
    parser.add_argument("--log-freq", type=int, default=500,
                        help="日志频率 (default: 500)")
    parser.add_argument("--plot-freq", type=int, default=2000,
                        help="绘图更新频率 (default: 2000)")
    
    # 输出配置
    parser.add_argument("--log-dir", type=str, default="./logs/ppo_visual",
                        help="日志目录 (default: ./logs/ppo_visual)")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备 (default: auto)")
    
    args = parser.parse_args()
    
    print("🎬 启动可视化PPO训练")
    print("=" * 50)
    print(f"场景类型: {args.scenario_type}")
    print(f"训练步数: {args.total_timesteps:,}")
    print(f"学习率: {args.learning_rate}")
    print(f"窗口大小: {args.window_size[0]}x{args.window_size[1]}")
    print(f"日志目录: {args.log_dir}")
    print("=" * 50)
    print("💡 提示:")
    print("  - MetaDrive窗口将显示实时场景和主车控制")
    print("  - Matplotlib窗口将显示训练曲线")
    print("  - 按Ctrl+C可以安全停止训练")
    print("=" * 50)
    
    try:
        # 开始训练
        model = train_ppo_with_visualization(args)
        print("\n🎉 训练完成!")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断训练")
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 