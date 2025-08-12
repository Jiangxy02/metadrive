#!/usr/bin/env python3
"""
测试训练好的PPO模型
可以选择有渲染或无渲染模式
"""

import os
import argparse
from pathlib import Path
import numpy as np

# 避免Qt问题的环境变量
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    from stable_baselines3 import PPO
    SB3_AVAILABLE = True
except ImportError:
    print("❌ Stable Baselines3 不可用")
    SB3_AVAILABLE = False

try:
    from random_scenario_generator import MetaDriveRandomEnv, RandomScenarioGenerator
    METADRIVE_AVAILABLE = True
except ImportError:
    print("❌ MetaDrive 不可用")
    METADRIVE_AVAILABLE = False

def create_test_env(use_render=False):
    """创建测试环境"""
    generator = RandomScenarioGenerator(seed=123)  # 使用不同的种子测试泛化能力
    
    base_config = {
        "use_render": use_render,
        "manual_control": False,
        "horizon": 1000,  # 更长的测试时间
        "map_region_size": 1024,
        "traffic_density": 0.2,
        "num_scenarios": 50,
        "start_seed": 123,
    }
    
    if use_render:
        base_config.update({
            "window_size": (1200, 800),
            "render_mode": "human"
        })
    
    env = MetaDriveRandomEnv(
        generator=generator,
        scenario_type="random", 
        base_config=base_config
    )
    
    return env

def test_model(model_path, num_episodes=5, use_render=False):
    """测试训练好的模型"""
    if not SB3_AVAILABLE or not METADRIVE_AVAILABLE:
        print("❌ 缺少必要的依赖")
        return
    
    # 检查模型文件是否存在
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    print(f"🔍 加载模型: {model_path}")
    try:
        model = PPO.load(model_path)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    print(f"🎮 创建测试环境 (渲染: {'开启' if use_render else '关闭'})")
    env = create_test_env(use_render=use_render)
    
    print(f"🚗 开始测试 {num_episodes} 个episodes...")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # 使用训练好的模型预测动作
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if use_render:
                env.render()
            
            # 每100步显示一次进度
            if episode_length % 100 == 0:
                speed = env.agent.speed if hasattr(env, 'agent') else 0
                print(f"  步数: {episode_length}, 累积奖励: {episode_reward:.2f}, 速度: {speed:.1f}m/s")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1} 完成:")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  总步数: {episode_length}")
        print(f"  结束原因: {info.get('termination_info', '未知')}")
    
    # 统计结果
    print(f"\n📊 测试结果统计 ({num_episodes} episodes):")
    print(f"  平均奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  平均长度: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  最佳奖励: {np.max(episode_rewards):.2f}")
    print(f"  最长episode: {np.max(episode_lengths)} 步")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description="测试训练好的PPO模型")
    
    parser.add_argument("--model-path", type=str, 
                        default="./logs/simple_ppo/final_model",
                        help="模型文件路径")
    parser.add_argument("--episodes", type=int, default=5,
                        help="测试episodes数量")
    parser.add_argument("--render", action="store_true",
                        help="是否开启渲染")
    
    args = parser.parse_args()
    
    print("🧪 PPO模型测试")
    print("=" * 50)
    print(f"模型路径: {args.model_path}")
    print(f"测试episodes: {args.episodes}")
    print(f"渲染模式: {'开启' if args.render else '关闭'}")
    print("=" * 50)
    
    try:
        test_model(args.model_path, args.episodes, args.render)
    except KeyboardInterrupt:
        print("\n🛑 用户中断测试")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main() 