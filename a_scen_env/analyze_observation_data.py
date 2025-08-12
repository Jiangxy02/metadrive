#!/usr/bin/env python3
"""
观测数据深度分析脚本
对观测记录器生成的数据进行深入分析，找出主车停车行为的根本原因
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import os

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def analyze_observation_data(csv_path):
    """
    深度分析观测数据
    
    Args:
        csv_path (str): CSV数据文件路径
    """
    print("🔍 开始深度分析观测数据")
    print("=" * 80)
    
    # 读取数据
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 成功读取数据: {len(df)} 行, {len(df.columns)} 列")
        print(f"时间范围: {df['simulation_time'].min():.2f} - {df['simulation_time'].max():.2f} 秒")
    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return
    
    # 创建分析输出目录
    output_dir = os.path.dirname(csv_path)
    analysis_dir = os.path.join(output_dir, "detailed_analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    # ===== 1. 关键指标时间序列分析 =====
    print(f"\n📊 1. 关键指标时间序列分析")
    print("-" * 50)
    
    # 绘制关键指标变化图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('主车关键指标时间序列分析', fontsize=16)
    
    # 速度变化
    axes[0, 0].plot(df['simulation_time'], df['speed'], 'b-', linewidth=2)
    axes[0, 0].set_title('速度变化')
    axes[0, 0].set_xlabel('仿真时间 (秒)')
    axes[0, 0].set_ylabel('速度 (m/s)')
    axes[0, 0].grid(True)
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='低速阈值')
    axes[0, 0].axhline(y=0.1, color='r', linestyle='-', alpha=0.7, label='停车阈值')
    axes[0, 0].legend()
    
    # 位置变化
    axes[0, 1].plot(df['simulation_time'], df['pos_x'], 'g-', linewidth=2, label='X位置')
    axes[0, 1].plot(df['simulation_time'], df['pos_y'], 'r-', linewidth=2, label='Y位置')
    axes[0, 1].set_title('位置变化')
    axes[0, 1].set_xlabel('仿真时间 (秒)')
    axes[0, 1].set_ylabel('位置 (m)')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # PPO动作
    axes[1, 0].plot(df['simulation_time'], df['action_steering'], 'purple', linewidth=2, label='转向')
    axes[1, 0].plot(df['simulation_time'], df['action_throttle'], 'orange', linewidth=2, label='油门')
    axes[1, 0].set_title('PPO专家动作')
    axes[1, 0].set_xlabel('仿真时间 (秒)')
    axes[1, 0].set_ylabel('动作值')
    axes[1, 0].grid(True)
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].legend()
    
    # 导航状态
    axes[1, 1].plot(df['simulation_time'], df['nav_route_completion'], 'brown', linewidth=2, label='路径完成度')
    if 'distance_to_custom_dest' in df.columns:
        axes[1, 1].plot(df['simulation_time'], df['distance_to_custom_dest']/100, 'cyan', linewidth=2, label='目标距离/100')
    axes[1, 1].set_title('导航状态')
    axes[1, 1].set_xlabel('仿真时间 (秒)')
    axes[1, 1].set_ylabel('完成度 / 距离比例')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, '01_key_metrics_timeline.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== 2. 停车行为详细分析 =====
    print(f"\n🚗 2. 停车行为详细分析")
    print("-" * 50)
    
    # 找出停车阶段
    stopped_mask = df['speed'] < 0.1
    low_speed_mask = df['speed'] < 0.5
    
    print(f"总步数: {len(df)}")
    print(f"停车步数 (<0.1 m/s): {stopped_mask.sum()} ({stopped_mask.mean()*100:.1f}%)")
    print(f"低速步数 (<0.5 m/s): {low_speed_mask.sum()} ({low_speed_mask.mean()*100:.1f}%)")
    
    if stopped_mask.any():
        first_stop_idx = df[stopped_mask].index[0]
        first_stop_time = df.loc[first_stop_idx, 'simulation_time']
        first_stop_pos = (df.loc[first_stop_idx, 'pos_x'], df.loc[first_stop_idx, 'pos_y'])
        
        print(f"首次停车时间: {first_stop_time:.2f} 秒 (步骤 {first_stop_idx})")
        print(f"首次停车位置: ({first_stop_pos[0]:.1f}, {first_stop_pos[1]:.1f})")
        
        # 分析停车前的状态
        print(f"\n停车前状态分析:")
        if first_stop_idx > 5:
            before_stop = df.iloc[max(0, first_stop_idx-5):first_stop_idx]
            print(f"停车前5步平均速度: {before_stop['speed'].mean():.2f} m/s")
            print(f"停车前5步平均油门: {before_stop['action_throttle'].mean():.3f}")
            print(f"停车前5步平均转向: {before_stop['action_steering'].mean():.3f}")
            print(f"停车前5步路径完成度变化: {before_stop['nav_route_completion'].iloc[0]:.3f} → {before_stop['nav_route_completion'].iloc[-1]:.3f}")
    
    # ===== 3. PPO专家行为分析 =====
    print(f"\n🤖 3. PPO专家行为分析")
    print("-" * 50)
    
    # 动作统计
    print(f"转向动作统计:")
    print(f"  范围: {df['action_steering'].min():.3f} ~ {df['action_steering'].max():.3f}")
    print(f"  平均: {df['action_steering'].mean():.3f}")
    print(f"  标准差: {df['action_steering'].std():.3f}")
    
    print(f"\n油门动作统计:")
    print(f"  范围: {df['action_throttle'].min():.3f} ~ {df['action_throttle'].max():.3f}")
    print(f"  平均: {df['action_throttle'].mean():.3f}")
    print(f"  标准差: {df['action_throttle'].std():.3f}")
    
    # 负油门分析
    negative_throttle = df[df['action_throttle'] < 0]
    print(f"\n负油门 (刹车) 分析:")
    print(f"  负油门步数: {len(negative_throttle)} ({len(negative_throttle)/len(df)*100:.1f}%)")
    if len(negative_throttle) > 0:
        print(f"  平均刹车强度: {negative_throttle['action_throttle'].mean():.3f}")
        print(f"  最大刹车强度: {negative_throttle['action_throttle'].min():.3f}")
    
    # 绘制动作分布图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(df['action_steering'], bins=30, alpha=0.7, color='purple')
    axes[0].set_title('转向动作分布')
    axes[0].set_xlabel('转向值')
    axes[0].set_ylabel('频次')
    axes[0].axvline(x=0, color='r', linestyle='--', alpha=0.7)
    axes[0].grid(True)
    
    axes[1].hist(df['action_throttle'], bins=30, alpha=0.7, color='orange')
    axes[1].set_title('油门动作分布')
    axes[1].set_xlabel('油门值')
    axes[1].set_ylabel('频次')
    axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.7, label='刹车/加速界限')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, '02_action_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== 4. 导航系统分析 =====
    print(f"\n🧭 4. 导航系统分析")
    print("-" * 50)
    
    # 路径完成度分析
    route_completion = df['nav_route_completion'].dropna()
    if len(route_completion) > 0:
        print(f"路径完成度统计:")
        print(f"  范围: {route_completion.min():.3f} ~ {route_completion.max():.3f}")
        print(f"  变化量: {route_completion.max() - route_completion.min():.3f}")
        print(f"  标准差: {route_completion.std():.3f}")
        
        # 检查是否卡住
        if route_completion.std() < 0.01:
            print(f"  ⚠️  路径完成度几乎不变，可能卡住！")
        
        # 分析完成度变化率
        route_diff = route_completion.diff().dropna()
        positive_progress = route_diff[route_diff > 0]
        print(f"  正向进展步数: {len(positive_progress)} ({len(positive_progress)/len(route_diff)*100:.1f}%)")
        if len(positive_progress) > 0:
            print(f"  平均进展速度: {positive_progress.mean():.6f}/步")
    
    # 目标距离分析
    if 'distance_to_custom_dest' in df.columns:
        dist_to_dest = df['distance_to_custom_dest'].dropna()
        if len(dist_to_dest) > 0:
            print(f"\n目标距离统计:")
            print(f"  初始距离: {dist_to_dest.iloc[0]:.1f} m")
            print(f"  最终距离: {dist_to_dest.iloc[-1]:.1f} m")
            print(f"  距离减少: {dist_to_dest.iloc[0] - dist_to_dest.iloc[-1]:.1f} m")
            print(f"  减少比例: {(dist_to_dest.iloc[0] - dist_to_dest.iloc[-1])/dist_to_dest.iloc[0]*100:.1f}%")
    
    # ===== 5. 观测状态分析 =====
    print(f"\n👁️  5. 观测状态分析")
    print("-" * 50)
    
    # 观测向量统计
    obs_cols = [col for col in df.columns if col.startswith('obs_')]
    if obs_cols:
        print(f"观测向量特征数: {len(obs_cols)}")
        
        # 关键观测值分析
        if 'obs_0_speed_related' in df.columns:
            print(f"观测值[0] (速度相关): {df['obs_0_speed_related'].mean():.3f} ± {df['obs_0_speed_related'].std():.3f}")
        if 'obs_1_steering_related' in df.columns:
            print(f"观测值[1] (转向相关): {df['obs_1_steering_related'].mean():.3f} ± {df['obs_1_steering_related'].std():.3f}")
        
        # 观测向量统计指标
        if 'obs_mean' in df.columns:
            print(f"观测向量平均值: {df['obs_mean'].mean():.3f} ± {df['obs_mean'].std():.3f}")
        if 'obs_std' in df.columns:
            print(f"观测向量标准差: {df['obs_std'].mean():.3f} ± {df['obs_std'].std():.3f}")
    
    # ===== 6. 相关性分析 =====
    print(f"\n🔗 6. 关键变量相关性分析")
    print("-" * 50)
    
    # 选择关键变量进行相关性分析
    key_vars = ['speed', 'action_steering', 'action_throttle', 'nav_route_completion', 'reward']
    if 'distance_to_custom_dest' in df.columns:
        key_vars.append('distance_to_custom_dest')
    
    correlation_matrix = df[key_vars].corr()
    print("关键变量相关性矩阵:")
    print(correlation_matrix.round(3))
    
    # 绘制相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f', cbar_kws={'label': '相关系数'})
    plt.title('关键变量相关性热力图')
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_dir, '03_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ===== 7. 问题诊断总结 =====
    print(f"\n⚠️  7. 问题诊断总结")
    print("-" * 50)
    
    issues = []
    
    # 检查停车问题
    if stopped_mask.mean() > 0.3:
        issues.append(f"❌ 停车时间过长: {stopped_mask.mean()*100:.1f}%的时间在停车")
    
    # 检查负油门问题
    if negative_throttle_ratio := (df['action_throttle'] < 0).mean():
        if negative_throttle_ratio > 0.5:
            issues.append(f"❌ 过多刹车行为: {negative_throttle_ratio*100:.1f}%的时间在刹车")
    
    # 检查导航问题
    if len(route_completion) > 0 and route_completion.std() < 0.01:
        issues.append(f"❌ 导航进度卡住: 路径完成度标准差仅{route_completion.std():.6f}")
    
    # 检查进展问题
    total_distance = np.sqrt((df['pos_x'].iloc[-1] - df['pos_x'].iloc[0])**2 + 
                           (df['pos_y'].iloc[-1] - df['pos_y'].iloc[0])**2)
    if total_distance < 50:
        issues.append(f"❌ 前进距离过短: 总位移仅{total_distance:.1f}米")
    
    # 检查油门-速度不匹配问题
    speed_throttle_corr = df['speed'].corr(df['action_throttle'])
    if speed_throttle_corr < 0:  # 负相关表示油门越大速度越小，异常
        issues.append(f"❌ 油门-速度负相关: 相关系数{speed_throttle_corr:.3f}，可能存在控制问题")
    
    if issues:
        print("检测到的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ 未检测到明显问题")
    
    # ===== 8. 建议措施 =====
    print(f"\n💡 8. 建议措施")
    print("-" * 50)
    
    suggestions = []
    
    if negative_throttle_ratio > 0.5:
        suggestions.append("🔧 PPO专家输出过多负油门，建议检查训练数据或重新训练")
    
    if len(route_completion) > 0 and route_completion.std() < 0.01:
        suggestions.append("🔧 导航路径生成失败，建议检查目标点设置和路径规划算法")
    
    if total_distance < 50:
        suggestions.append("🔧 车辆前进能力不足，建议检查奖励函数设计")
    
    suggestions.append("🔧 建议分析观测向量是否包含足够的前进激励信息")
    suggestions.append("🔧 建议检查PPO模型是否在类似场景下训练过")
    
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")
    
    print(f"\n📊 详细分析图表已保存到: {analysis_dir}")
    print(f"   - 01_key_metrics_timeline.png: 关键指标时间序列")
    print(f"   - 02_action_distributions.png: 动作分布分析")
    print(f"   - 03_correlation_heatmap.png: 变量相关性分析")
    
    return df, analysis_dir

def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # 默认路径
        csv_path = "observation_logs/main_car_stop_analysis_observations.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        print(f"请提供正确的CSV文件路径")
        return
    
    try:
        df, analysis_dir = analyze_observation_data(csv_path)
        print(f"\n✅ 分析完成！")
    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 