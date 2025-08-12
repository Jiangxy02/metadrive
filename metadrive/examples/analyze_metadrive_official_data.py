#!/usr/bin/env python3
"""
MetaDrive官方环境观测数据深度分析脚本

对观测记录器生成的官方MetaDrive环境数据进行深入分析，
比较官方环境与自定义trajectory replay环境的行为差异。

功能：
1. 分析MetaDrive官方环境中PPO专家的行为表现
2. 生成详细的统计分析和可视化图表
3. 对比分析不同环境配置下的车辆行为差异
4. 识别PPO专家在标准环境vs自定义环境中的行为模式

作者：认知建模项目组
日期：2025-01-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import os
import sys
import argparse
from datetime import datetime

# 中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def analyze_metadrive_official_data(csv_path):
    """
    分析MetaDrive官方环境的观测数据
    
    Args:
        csv_path: CSV文件路径
    """
    print(f"📊 分析MetaDrive官方环境观测数据")
    print(f"数据文件：{csv_path}")
    
    # 读取数据
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功读取数据，共 {len(df)} 行记录")
    except Exception as e:
        print(f"❌ 读取数据失败：{e}")
        return
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(csv_path), "official_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 数据基本信息
    print(f"\n📋 数据基本信息：")
    print(f"  - 总步数：{len(df)}")
    print(f"  - 仿真时间：{df['simulation_time'].max():.1f}秒")
    print(f"  - 数据完整性：{(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
    
    # 2. 控制模式分析
    print(f"\n🎮 控制模式分析：")
    if 'action_source' in df.columns:
        action_source_counts = df['action_source'].value_counts()
        print(f"  - 动作来源分布：")
        for source, count in action_source_counts.items():
            print(f"    * {source}: {count} 步 ({count/len(df)*100:.1f}%)")
    
    if 'expert_takeover' in df.columns:
        expert_counts = df['expert_takeover'].value_counts()
        print(f"  - 专家接管状态：")
        for status, count in expert_counts.items():
            print(f"    * {status}: {count} 步 ({count/len(df)*100:.1f}%)")
    
    # 3. 运动行为分析
    print(f"\n🚗 运动行为分析：")
    speed_stats = df['speed'].describe()
    print(f"  - 速度统计：")
    print(f"    * 平均速度：{speed_stats['mean']:.2f} m/s")
    print(f"    * 最大速度：{speed_stats['max']:.2f} m/s")
    print(f"    * 最小速度：{speed_stats['min']:.2f} m/s")
    print(f"    * 速度标准差：{speed_stats['std']:.2f} m/s")
    
    # 停车行为分析
    stopped_threshold = 0.1  # m/s
    stopped_steps = (df['speed'] < stopped_threshold).sum()
    print(f"  - 停车行为：")
    print(f"    * 停车步数：{stopped_steps} ({stopped_steps/len(df)*100:.1f}%)")
    
    if stopped_steps > 0:
        first_stop = df[df['speed'] < stopped_threshold].index[0]
        print(f"    * 首次停车：第{first_stop}步")
    
    # 4. 动作分析
    print(f"\n🎯 动作分析：")
    if 'action_throttle' in df.columns:
        throttle_stats = df['action_throttle'].describe()
        print(f"  - 油门/刹车统计：")
        print(f"    * 平均值：{throttle_stats['mean']:.3f}")
        print(f"    * 最大值：{throttle_stats['max']:.3f}")
        print(f"    * 最小值：{throttle_stats['min']:.3f}")
        
        # 分析油门vs刹车比例
        positive_throttle = (df['action_throttle'] > 0).sum()
        negative_throttle = (df['action_throttle'] < 0).sum()
        zero_throttle = (df['action_throttle'] == 0).sum()
        
        print(f"  - 动作分布：")
        print(f"    * 加速 (>0)：{positive_throttle} 步 ({positive_throttle/len(df)*100:.1f}%)")
        print(f"    * 刹车 (<0)：{negative_throttle} 步 ({negative_throttle/len(df)*100:.1f}%)")
        print(f"    * 空档 (=0)：{zero_throttle} 步 ({zero_throttle/len(df)*100:.1f}%)")
    
    if 'action_steering' in df.columns:
        steering_stats = df['action_steering'].describe()
        print(f"  - 转向统计：")
        print(f"    * 平均转向：{steering_stats['mean']:.3f}")
        print(f"    * 转向幅度：{steering_stats['std']:.3f}")
    
    # 5. 导航分析
    print(f"\n🧭 导航分析：")
    if 'nav_route_completion' in df.columns:
        route_progress = df['nav_route_completion'].describe()
        print(f"  - 路径完成度：")
        print(f"    * 初始：{df['nav_route_completion'].iloc[0]:.3f}")
        print(f"    * 最终：{df['nav_route_completion'].iloc[-1]:.3f}")
        print(f"    * 最大：{route_progress['max']:.3f}")
        print(f"    * 进度变化：{df['nav_route_completion'].iloc[-1] - df['nav_route_completion'].iloc[0]:.3f}")
    
    if 'nav_distance_to_dest' in df.columns:
        distance_stats = df['nav_distance_to_dest'].describe()
        print(f"  - 到目标距离：")
        print(f"    * 初始距离：{df['nav_distance_to_dest'].iloc[0]:.1f}m")
        if not pd.isna(df['nav_distance_to_dest'].iloc[-1]):
            print(f"    * 最终距离：{df['nav_distance_to_dest'].iloc[-1]:.1f}m")
        print(f"    * 最小距离：{distance_stats['min']:.1f}m")
    
    # 6. 奖励分析
    print(f"\n🏆 奖励分析：")
    if 'reward' in df.columns:
        reward_stats = df['reward'].describe()
        total_reward = df['reward'].sum()
        print(f"  - 奖励统计：")
        print(f"    * 总奖励：{total_reward:.2f}")
        print(f"    * 平均奖励：{reward_stats['mean']:.3f}")
        print(f"    * 最大奖励：{reward_stats['max']:.3f}")
        print(f"    * 最小奖励：{reward_stats['min']:.3f}")
    
    # 7. 生成可视化图表
    print(f"\n📈 生成可视化图表...")
    
    # 创建图表
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('MetaDrive官方环境 - 车辆行为分析', fontsize=16, fontweight='bold')
    
    # 7.1 速度时间序列
    axes[0, 0].plot(df['step'], df['speed'], 'b-', alpha=0.7, linewidth=1)
    axes[0, 0].axhline(y=stopped_threshold, color='r', linestyle='--', alpha=0.5, label=f'停车阈值 ({stopped_threshold} m/s)')
    axes[0, 0].set_title('Speed Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Speed (m/s)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 7.2 位置轨迹
    axes[0, 1].plot(df['pos_x'], df['pos_y'], 'g-', alpha=0.7, linewidth=2)
    axes[0, 1].scatter(df['pos_x'].iloc[0], df['pos_y'].iloc[0], color='green', s=100, marker='o', label='Start')
    axes[0, 1].scatter(df['pos_x'].iloc[-1], df['pos_y'].iloc[-1], color='red', s=100, marker='s', label='End')
    axes[0, 1].set_title('Vehicle Trajectory')
    axes[0, 1].set_xlabel('X Position (m)')
    axes[0, 1].set_ylabel('Y Position (m)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')
    
    # 7.3 动作分析
    if 'action_throttle' in df.columns and 'action_steering' in df.columns:
        axes[1, 0].plot(df['step'], df['action_throttle'], 'r-', alpha=0.7, label='Throttle')
        axes[1, 0].plot(df['step'], df['action_steering'], 'b-', alpha=0.7, label='Steering')
        axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[1, 0].set_title('Actions Over Time')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Action Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 7.4 导航进度
    if 'nav_route_completion' in df.columns:
        axes[1, 1].plot(df['step'], df['nav_route_completion'], 'm-', alpha=0.7, linewidth=2)
        axes[1, 1].set_title('Navigation Route Completion')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Route Completion')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
    
    # 7.5 奖励累积
    if 'reward' in df.columns:
        cumulative_reward = df['reward'].cumsum()
        axes[2, 0].plot(df['step'], cumulative_reward, 'orange', linewidth=2)
        axes[2, 0].set_title('Cumulative Reward')
        axes[2, 0].set_xlabel('Step')
        axes[2, 0].set_ylabel('Cumulative Reward')
        axes[2, 0].grid(True, alpha=0.3)
    
    # 7.6 速度vs油门相关性
    if 'action_throttle' in df.columns:
        scatter = axes[2, 1].scatter(df['action_throttle'], df['speed'], 
                                   c=df['step'], cmap='viridis', alpha=0.6, s=20)
        axes[2, 1].set_title('Speed vs Throttle Correlation')
        axes[2, 1].set_xlabel('Throttle Action')
        axes[2, 1].set_ylabel('Speed (m/s)')
        axes[2, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[2, 1], label='Step')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'metadrive_official_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 主要分析图表已保存：{plot_path}")
    
    # 8. 动作分布分析
    if 'action_throttle' in df.columns and 'action_steering' in df.columns:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('MetaDrive官方环境 - 动作分布分析', fontsize=14, fontweight='bold')
        
        # 油门分布直方图
        axes[0, 0].hist(df['action_throttle'], bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Throttle Action Distribution')
        axes[0, 0].set_xlabel('Throttle Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 转向分布直方图
        axes[0, 1].hist(df['action_steering'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Steering Action Distribution')
        axes[0, 1].set_xlabel('Steering Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 动作热力图
        action_heatmap_data = np.histogram2d(df['action_steering'], df['action_throttle'], bins=30)
        im = axes[1, 0].imshow(action_heatmap_data[0], origin='lower', cmap='YlOrRd', aspect='auto')
        axes[1, 0].set_title('Action Heatmap (Steering vs Throttle)')
        axes[1, 0].set_xlabel('Throttle Bins')
        axes[1, 0].set_ylabel('Steering Bins')
        plt.colorbar(im, ax=axes[1, 0])
        
        # 时间序列动作
        axes[1, 1].plot(df['step'], df['action_throttle'], 'r-', alpha=0.5, label='Throttle')
        axes[1, 1].plot(df['step'], df['action_steering'], 'b-', alpha=0.5, label='Steering')
        axes[1, 1].set_title('Actions Time Series')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Action Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        action_plot_path = os.path.join(output_dir, 'action_distribution_analysis.png')
        plt.savefig(action_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 动作分布图表已保存：{action_plot_path}")
    
    # 9. 生成文本报告
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("MetaDrive官方环境观测数据分析报告\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据文件：{csv_path}\n")
        f.write(f"分析步数：{len(df)}\n")
        f.write(f"仿真时间：{df['simulation_time'].max():.1f}秒\n\n")
        
        f.write("🎮 控制模式分析\n")
        f.write("-" * 40 + "\n")
        if 'action_source' in df.columns:
            for source, count in df['action_source'].value_counts().items():
                f.write(f"  {source}: {count} 步 ({count/len(df)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("🚗 运动行为分析\n")
        f.write("-" * 40 + "\n")
        f.write(f"  平均速度: {df['speed'].mean():.2f} m/s\n")
        f.write(f"  最大速度: {df['speed'].max():.2f} m/s\n")
        f.write(f"  停车比例: {(df['speed'] < 0.1).sum() / len(df) * 100:.1f}%\n")
        if 'action_throttle' in df.columns:
            f.write(f"  平均油门: {df['action_throttle'].mean():.3f}\n")
            f.write(f"  刹车比例: {(df['action_throttle'] < 0).sum() / len(df) * 100:.1f}%\n")
        f.write("\n")
        
        f.write("🧭 导航性能\n")
        f.write("-" * 40 + "\n")
        if 'nav_route_completion' in df.columns:
            f.write(f"  路径完成度变化: {df['nav_route_completion'].iloc[-1] - df['nav_route_completion'].iloc[0]:.3f}\n")
        if 'nav_distance_to_dest' in df.columns and not pd.isna(df['nav_distance_to_dest'].iloc[0]):
            f.write(f"  初始距离目标: {df['nav_distance_to_dest'].iloc[0]:.1f}m\n")
            if not pd.isna(df['nav_distance_to_dest'].iloc[-1]):
                f.write(f"  最终距离目标: {df['nav_distance_to_dest'].iloc[-1]:.1f}m\n")
        f.write("\n")
        
        f.write("🏆 奖励表现\n")
        f.write("-" * 40 + "\n")
        if 'reward' in df.columns:
            f.write(f"  总奖励: {df['reward'].sum():.2f}\n")
            f.write(f"  平均奖励: {df['reward'].mean():.3f}\n")
        f.write("\n")
        
        # 问题诊断
        f.write("❗ 问题诊断\n")
        f.write("-" * 40 + "\n")
        issues = []
        
        if (df['speed'] < 0.1).sum() / len(df) > 0.5:
            issues.append("车辆长时间停车(>50%时间)")
        
        if 'action_throttle' in df.columns and (df['action_throttle'] < 0).sum() / len(df) > 0.3:
            issues.append("频繁刹车行为(>30%时间)")
        
        if 'nav_route_completion' in df.columns:
            progress_change = df['nav_route_completion'].iloc[-1] - df['nav_route_completion'].iloc[0]
            if abs(progress_change) < 0.1:
                issues.append("导航进度停滞")
        
        if 'reward' in df.columns and df['reward'].mean() < 0:
            issues.append("平均奖励为负")
        
        if issues:
            for issue in issues:
                f.write(f"  ⚠️  {issue}\n")
        else:
            f.write("  ✅ 未检测到明显问题\n")
    
    print(f"✅ 分析报告已保存：{report_path}")
    print(f"\n📊 分析完成！输出目录：{output_dir}")


def main():
    parser = argparse.ArgumentParser(description="分析MetaDrive官方环境观测数据")
    parser.add_argument("csv_path", nargs='?', help="CSV数据文件路径")
    parser.add_argument("--auto_find", action="store_true", help="自动查找最新的CSV文件")
    args = parser.parse_args()
    
    if args.auto_find or not args.csv_path:
        # 自动查找最新的CSV文件
        search_dirs = [
            "metadrive_official_logs",
            "../metadrive_official_logs",
            ".",
        ]
        
        csv_files = []
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.endswith('_observations.csv'):
                        csv_files.append(os.path.join(search_dir, file))
        
        if csv_files:
            # 选择最新的文件
            latest_file = max(csv_files, key=os.path.getmtime)
            print(f"🔍 自动找到最新的CSV文件：{latest_file}")
            args.csv_path = latest_file
        else:
            print("❌ 未找到CSV数据文件")
            print("请确保运行了 drive_in_single_agent_env_with_recorder.py 生成数据")
            return
    
    if not os.path.exists(args.csv_path):
        print(f"❌ 文件不存在：{args.csv_path}")
        return
    
    analyze_metadrive_official_data(args.csv_path)


if __name__ == "__main__":
    main() 