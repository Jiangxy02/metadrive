"""
PPO模型评估脚本

功能：
1. 加载训练好的PPO模型
2. 在测试环境中评估模型性能
3. 支持可视化渲染
4. 记录评估指标

作者：PPO训练系统
日期：2025-01-03
"""

import os
import sys
import numpy as np
from typing import Dict, Optional

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory
from train_ppo import TrajectoryReplayWrapper


def evaluate_model(
    model_path: str,
    csv_path: str = None,
    n_episodes: int = 10,
    render: bool = True,
    verbose: bool = True
) -> Dict:
    """
    评估训练好的PPO模型
    
    Args:
        model_path: 模型文件路径
        csv_path: CSV轨迹文件路径
        n_episodes: 评估的episode数量
        render: 是否渲染可视化
        verbose: 是否打印详细信息
        
    Returns:
        dict: 评估结果统计
    """
    # 如果没有指定CSV文件，使用默认的
    if csv_path is None:
        csv_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(csv_dir, "scenario_vehicles_9_0_1_2_3_4_5_6_7_8_test3_20250304_150943_row001_filtered.csv")
    
    # 加载模型
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # 创建评估环境配置
    eval_config = {
        "csv_paths": [csv_path],
        "use_render": render,
        "manual_control": False,  # 禁用手动控制
        "horizon": 2000,
        "end_on_crash": True,
        "end_on_out_of_road": True,
        "end_on_arrive_dest": False,
        "end_on_horizon": True,
        "background_vehicle_update_mode": "position",
        "enable_realtime": render,  # 渲染时启用实时模式
        "target_fps": 50.0 if render else 0,
        "map": "S" * 30,
        "traffic_density": 0.0,
    }
    
    # 创建环境
    env = TrajectoryReplayWrapper(eval_config)
    
    # 评估统计
    episode_rewards = []
    episode_lengths = []
    episode_crashes = 0
    episode_out_of_road = 0
    episode_success = 0
    
    print(f"\nStarting evaluation for {n_episodes} episodes...")
    print("=" * 50)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # 使用模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            
            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # 渲染
            if render:
                env.render()
            
            if verbose and episode_length % 100 == 0:
                print(f"Episode {episode + 1}, Step {episode_length}: Reward = {episode_reward:.2f}")
        
        # 记录episode结果
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 统计结束原因
        if info.get("crash_flag", False):
            episode_crashes += 1
        elif info.get("out_of_road_flag", False):
            episode_out_of_road += 1
        else:
            episode_success += 1
        
        print(f"Episode {episode + 1}/{n_episodes} completed:")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Episode Length: {episode_length}")
        print(f"  End Reason: ", end="")
        if info.get("crash_flag", False):
            print("Crash")
        elif info.get("out_of_road_flag", False):
            print("Out of Road")
        elif info.get("horizon_reached_flag", False):
            print("Max Steps Reached")
        else:
            print("Success")
        print("-" * 30)
    
    # 关闭环境
    env.close()
    
    # 计算统计结果
    results = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "crash_rate": episode_crashes / n_episodes,
        "out_of_road_rate": episode_out_of_road / n_episodes,
        "success_rate": episode_success / n_episodes,
    }
    
    # 打印统计结果
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS:")
    print("=" * 50)
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Max/Min Reward: {results['max_reward']:.2f} / {results['min_reward']:.2f}")
    print(f"Mean Episode Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Success Rate: {results['success_rate']*100:.1f}%")
    print(f"Crash Rate: {results['crash_rate']*100:.1f}%")
    print(f"Out of Road Rate: {results['out_of_road_rate']*100:.1f}%")
    print("=" * 50)
    
    return results


def find_latest_model(models_dir: str = None) -> Optional[str]:
    """
    查找最新的训练模型
    
    Args:
        models_dir: 模型目录
        
    Returns:
        str: 最新模型的路径，如果没有找到则返回None
    """
    if models_dir is None:
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained PPO model")
    parser.add_argument("--model", type=str, default=None, 
                       help="Path to the model file. If not specified, use the latest model.")
    parser.add_argument("--csv", type=str, default=None,
                       help="Path to the CSV trajectory file")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # 查找模型路径
    model_path = args.model
    if model_path is None:
        model_path = find_latest_model()
        if model_path is None:
            print("Error: No trained model found. Please train a model first or specify a model path.")
            sys.exit(1)
        print(f"Using latest model: {model_path}")
    
    # 评估模型
    results = evaluate_model(
        model_path=model_path,
        csv_path=args.csv,
        n_episodes=args.episodes,
        render=not args.no_render,
        verbose=not args.quiet
    )
    
    # 保存评估结果
    import json
    results_file = model_path.replace(".zip", "_eval_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation results saved to: {results_file}") 