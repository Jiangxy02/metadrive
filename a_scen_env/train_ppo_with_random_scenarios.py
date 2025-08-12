#!/usr/bin/env python3
"""
基于MetaDrive随机场景的SB3 PPO训练脚本
"""

import sys
import os
import argparse
import numpy as np
from typing import Dict, Any, Optional

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from random_scenario_generator import RandomScenarioGenerator, create_random_training_environment
from sb3_ppo_integration import SB3TrajectoryReplayWrapper

# SB3相关导入
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import set_random_seed
    import gymnasium as gym
    from gymnasium import spaces
    SB3_AVAILABLE = True
except ImportError:
    print("❌ Stable Baselines3 未安装。安装命令: pip install stable-baselines3")
    SB3_AVAILABLE = False


class MetaDriveRandomWrapper(gym.Wrapper):
    """
    将MetaDrive随机环境包装为SB3兼容格式
    """
    
    def __init__(self, 
                 scenario_type: str = "curriculum",
                 num_scenarios: int = 1000,
                 seed: Optional[int] = None,
                 reward_config: Optional[Dict[str, Any]] = None,
                 **env_kwargs):
        """
        初始化包装器
        
        Args:
            scenario_type: 场景类型
            num_scenarios: 场景总数
            seed: 随机种子
            reward_config: 奖励配置
            **env_kwargs: 环境额外参数
        """
        
        # 保存创建参数（用于重新创建环境）
        self._scenario_type = scenario_type
        self._num_scenarios = num_scenarios
        self._seed = seed
        self._env_kwargs = env_kwargs
        
        # 创建随机环境
        base_env = self._create_base_env()
        
        super().__init__(base_env)
        
        # 奖励配置
        self.reward_config = reward_config or self._get_default_reward_config()
        
        # 进度跟踪变量
        self._last_position = None
        self._total_distance = 0.0
        self._episode_start_pos = None
        
        # 设置标准化的动作和观测空间
        self._setup_spaces()
    
    def _create_base_env(self):
        """创建基础环境"""
        return create_random_training_environment(
            scenario_type=self._scenario_type,
            num_scenarios=self._num_scenarios,
            seed=self._seed,
            **self._env_kwargs
        )
        
    def _get_default_reward_config(self) -> Dict[str, Any]:
        """默认奖励配置 - 优化版本，鼓励车辆向前移动"""
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
    
    def _setup_spaces(self):
        """设置标准化的观测和动作空间"""
        
        # 使用基础环境的实际空间，而不是硬编码
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        print(f"🔧 环境空间信息:")
        print(f"  观测空间: {self.observation_space}")
        print(f"  动作空间: {self.action_space}")
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境"""
        
        if seed is not None:
            np.random.seed(seed)
        
        # 调用基础环境的reset
        # MetaDrive的reset不支持seed和options参数，所以直接调用
        result = self.env.reset()
        
        # 处理返回格式 - MetaDrive返回(obs, info)元组
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        # 确保obs格式正确
        obs = np.array(obs, dtype=np.float32)
        if obs.shape[0] != self.observation_space.shape[0]:
            # 如果维度不匹配，填充或截断
            target_dim = self.observation_space.shape[0]
            if len(obs) < target_dim:
                # 填充0
                padded_obs = np.zeros(target_dim, dtype=np.float32)
                padded_obs[:len(obs)] = obs
                obs = padded_obs
            else:
                # 截断
                obs = obs[:target_dim]
        
        # 重置进度跟踪变量
        if hasattr(self.env, 'agent') and hasattr(self.env.agent, 'position'):
            self._episode_start_pos = np.array(self.env.agent.position[:2])  # 只取x,y坐标
            self._last_position = self._episode_start_pos.copy()
        else:
            self._episode_start_pos = np.array([0.0, 0.0])
            self._last_position = self._episode_start_pos.copy()
        
        self._total_distance = 0.0
        
        return obs, info
    
    def step(self, action):
        """执行动作"""
        
        # 确保action格式正确
        action = np.array(action, dtype=np.float32)
        
        # 调用基础环境
        obs, reward, done, info = self.env.step(action)
        
        # 处理观测格式
        obs = np.array(obs, dtype=np.float32)
        if obs.shape[0] != self.observation_space.shape[0]:
            target_dim = self.observation_space.shape[0]
            if len(obs) < target_dim:
                padded_obs = np.zeros(target_dim, dtype=np.float32)
                padded_obs[:len(obs)] = obs
                obs = padded_obs
            else:
                obs = obs[:target_dim]
        
        # 计算自定义奖励
        custom_reward = self._compute_reward(obs, action, done, info)
        
        # SB3格式：分离terminated和truncated
        terminated = done and not info.get("timeout", False)
        truncated = done and info.get("timeout", False)
        
        # 更新info
        info.update({
            "original_reward": reward,
            "custom_reward": custom_reward,
            "terminated": terminated,
            "truncated": truncated
        })
        
        return obs, custom_reward, terminated, truncated, info
    
    def _compute_reward(self, obs, action, done, info) -> float:
        """
        计算改进的自定义奖励函数 - 强调前进和到达目标
        """
        
        reward = 0.0
        config = self.reward_config
        
        # 时间惩罚（轻微）
        reward += config["time_penalty"]
        
        try:
            if hasattr(self.env, 'agent'):
                agent = self.env.agent
                
                # 1. 前进距离奖励（核心驱动力）
                if hasattr(agent, 'position') and self._last_position is not None:
                    current_pos = np.array(agent.position[:2])
                    
                    # 计算这一步的前进距离
                    step_distance = np.linalg.norm(current_pos - self._last_position)
                    
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
                    # 这里可以根据道路方向计算，暂时简化
                    # 假设道路大致沿x轴正方向
                    heading = agent.heading_theta
                    # 奖励朝向正确方向（-π/4 到 π/4）
                    heading_penalty = abs(heading) / (np.pi / 2)  # 归一化
                    if heading_penalty < 0.5:  # 方向偏差不大
                        reward += config["heading_reward_weight"] * (1 - heading_penalty)
                
                # 4. 车道中心奖励（如果有相关信息）
                # 这里需要根据MetaDrive的观测信息来实现
                # 暂时简化处理
                
        except Exception as e:
            # 如果获取信息失败，不影响基础流程
            pass
        
        # 5. 结束状态奖励/惩罚
        if done:
            if info.get("crash", False) or info.get("crash_vehicle", False):
                reward += config["crash_penalty"]
                print(f"🚗💥 Crash penalty: {config['crash_penalty']}")
            elif info.get("out_of_road", False):
                reward += config["out_road_penalty"]
                print(f"🛣️❌ Out of road penalty: {config['out_road_penalty']}")
            elif info.get("arrive_dest", False):
                # 到达目标的大奖励
                completion_reward = config["completion_bonus"]
                # 额外距离奖励
                if self._total_distance > config["distance_bonus_threshold"]:
                    completion_reward += self._total_distance * 0.1
                reward += completion_reward
                print(f"🎯✅ Completion bonus: {completion_reward}")
        
        return reward


class RandomScenarioCallback(BaseCallback):
    """
    随机场景训练回调函数
    """
    
    def __init__(self, 
                 log_freq: int = 1000,
                 eval_freq: int = 10000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        """每步调用"""
        
        # 收集episode统计
        if "episode" in self.locals["infos"][0]:
            ep_info = self.locals["infos"][0]["episode"]
            self.episode_rewards.append(ep_info["r"])
            self.episode_lengths.append(ep_info["l"])
        
        # 定期日志
        if self.n_calls % self.log_freq == 0:
            if self.episode_rewards:
                mean_reward = np.mean(self.episode_rewards[-10:])  # 最近10个episode
                mean_length = np.mean(self.episode_lengths[-10:])
                
                if self.verbose >= 1:
                    print(f"Step {self.n_calls}: Mean reward: {mean_reward:.2f}, Mean length: {mean_length:.1f}")
                
                # 记录到tensorboard
                self.logger.record("train/mean_episode_reward", mean_reward)
                self.logger.record("train/mean_episode_length", mean_length)
        
        return True


def create_training_environment(config: Dict[str, Any]):
    """创建训练环境"""
    
    def _make_env():
        return MetaDriveRandomWrapper(**config)
    
    return _make_env


def train_ppo_random_scenarios(args):
    """训练PPO模型在随机场景上"""
    
    if not SB3_AVAILABLE:
        raise ImportError("Stable Baselines3 not available")
    
    print(f"🚀 开始PPO随机场景训练")
    print(f"  场景类型: {args.scenario_type}")
    print(f"  场景数量: {args.num_scenarios}")
    print(f"  训练步数: {args.total_timesteps}")
    print(f"  并行环境: {args.n_envs}")
    
    # 设置随机种子
    if args.seed is not None:
        set_random_seed(args.seed)
    
    # 创建环境配置
    env_config = {
        "scenario_type": args.scenario_type,
        "num_scenarios": args.num_scenarios,
        "seed": args.seed,
    }
    
    # 创建向量化环境
    if args.n_envs > 1:
        env = make_vec_env(
            create_training_environment(env_config),
            n_envs=args.n_envs,
            seed=args.seed,
            vec_env_cls=DummyVecEnv  # 或SubprocVecEnv用于真正的并行
        )
    else:
        env = create_training_environment(env_config)()
    
    # 设置日志
    os.makedirs(args.log_dir, exist_ok=True)
    new_logger = configure(args.log_dir, ["stdout", "csv", "tensorboard"])
    
    # 创建PPO模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        tensorboard_log=args.log_dir,
        device=args.device,
        seed=args.seed
    )
    
    # 设置日志
    model.set_logger(new_logger)
    
    # 创建回调
    callback = RandomScenarioCallback(
        log_freq=args.log_freq,
        verbose=1
    )
    
    # 开始训练
    print(f"\n🎯 开始训练...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # 保存模型
    model_path = os.path.join(args.log_dir, "ppo_random_scenarios")
    model.save(model_path)
    print(f"✅ 模型已保存: {model_path}")
    
    # 清理环境
    env.close()
    
    return model


def evaluate_model(model_path: str, args):
    """评估训练好的模型"""
    
    print(f"\n🧪 评估模型: {model_path}")
    
    # 创建评估环境
    eval_config = {
        "scenario_type": args.scenario_type,
        "num_scenarios": 20,  # 评估用少量场景
        "seed": args.seed + 1000 if args.seed else None,  # 不同的种子
    }
    
    env = create_training_environment(eval_config)()
    
    # 加载模型
    model = PPO.load(model_path)
    
    # 评估统计
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(args.eval_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                # 检查是否成功完成
                if info.get("arrive_dest", False):
                    success_count += 1
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode % 5 == 0:
            print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}")
    
    # 计算统计
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    success_rate = success_count / args.eval_episodes
    
    print(f"\n📊 评估结果:")
    print(f"  平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  平均长度: {mean_length:.1f}")
    print(f"  成功率: {success_rate:.2%}")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="PPO随机场景训练")
    
    # 场景配置
    parser.add_argument("--scenario-type", type=str, default="random",
                        choices=["random", "curriculum", "safe"],
                        help="场景类型")
    parser.add_argument("--num-scenarios", type=int, default=1000,
                        help="场景总数")
    
    # 训练配置
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="总训练步数")
    parser.add_argument("--n-envs", type=int, default=4,
                        help="并行环境数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # PPO参数
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="学习率")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="每次更新的步数")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批量大小")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO更新轮数")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="折扣因子")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip-range", type=float, default=0.2,
                        help="PPO裁剪范围")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="熵系数")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="价值函数系数")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="梯度裁剪")
    
    # 输出配置
    parser.add_argument("--log-dir", type=str, default="./logs/ppo_random",
                        help="日志目录")
    parser.add_argument("--log-freq", type=int, default=1000,
                        help="日志频率")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备")
    
    # 评估配置
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="评估回合数")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估模式")
    parser.add_argument("--model-path", type=str,
                        help="评估模型路径")
    
    args = parser.parse_args()
    
    if args.eval_only:
        if not args.model_path:
            print("❌ 评估模式需要指定 --model-path")
            return
        evaluate_model(args.model_path, args)
    else:
        # 训练模式
        model = train_ppo_random_scenarios(args)
        
        # 训练后评估
        model_path = os.path.join(args.log_dir, "ppo_random_scenarios")
        evaluate_model(model_path, args)


if __name__ == "__main__":
    main() 