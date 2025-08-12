#!/usr/bin/env python3
"""
MetaDrive随机场景生成器
提供丰富的随机化场景配置用于PPO训练
"""

import sys
import os
import numpy as np
import random
from typing import Dict, Any, List, Optional, Union, Tuple

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# MetaDrive导入
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod

class RandomScenarioGenerator:
    """
    MetaDrive随机场景生成器
    
    提供多种随机化维度：
    1. 地图结构：长度、复杂度、车道数、车道宽度
    2. 交通环境：车流密度、事故概率、交通模式
    3. 环境条件：天气、光照、地形
    4. 任务难度：起始位置、目标距离、时间限制
    """
    
    def __init__(self, seed: Optional[int] = None):
        """
        初始化随机场景生成器
        
        Args:
            seed: 随机种子，用于可重现的场景生成
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.seed = seed
        
    def generate_basic_scenarios(self, num_scenarios: int = 100) -> List[Dict[str, Any]]:
        """
        生成基础随机场景配置列表
        
        Args:
            num_scenarios: 场景数量
            
        Returns:
            场景配置列表
        """
        scenarios = []
        
        for i in range(num_scenarios):
            config = self._generate_single_scenario(i)
            scenarios.append(config)
            
        return scenarios
    
    def _generate_single_scenario(self, scenario_id: int) -> Dict[str, Any]:
        """生成单个随机场景配置"""
        
        # 基础配置
        base_config = {
            "start_seed": scenario_id,
            "num_scenarios": 1,
            "use_render": False,
            "manual_control": False,
            "horizon": random.randint(800, 1200),  # 随机化episode长度
        }
        
        # 地图配置
        map_config = self._generate_random_map_config()
        base_config.update(map_config)
        
        # 交通配置
        traffic_config = self._generate_random_traffic_config()
        base_config.update(traffic_config)
        
        # 随机化车辆配置
        vehicle_config = self._generate_random_vehicle_config()
        base_config.update(vehicle_config)
        
        # 环境随机化
        env_config = self._generate_random_environment_config()
        base_config.update(env_config)
        
        return base_config
    
    def _generate_random_map_config(self) -> Dict[str, Any]:
        """生成随机地图配置"""
        
        # 地图复杂度：简单(1-3)、中等(4-7)、复杂(8-15)
        complexity_level = random.choice(["simple", "medium", "complex"])
        
        if complexity_level == "simple":
            map_blocks = random.randint(1, 3)
        elif complexity_level == "medium":
            map_blocks = random.randint(4, 7)
        else:  # complex
            map_blocks = random.randint(8, 15)
        
        # 车道配置
        lane_num = random.randint(2, 4)
        lane_width = round(random.uniform(3.0, 4.5), 1)
        
        # 随机化车道参数
        random_lane_width = random.choice([True, False])
        random_lane_num = random.choice([True, False])
        
        map_config = {
            "map": map_blocks,
            "random_lane_width": random_lane_width,
            "random_lane_num": random_lane_num,
            "map_config": {
                BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
                BaseMap.GENERATE_CONFIG: map_blocks,
                BaseMap.LANE_WIDTH: lane_width,
                BaseMap.LANE_NUM: lane_num,
                "exit_length": random.randint(30, 80),
            }
        }
        
        return map_config
    
    def _generate_random_traffic_config(self) -> Dict[str, Any]:
        """生成随机交通配置"""
        
        # 交通密度：稀疏、正常、密集
        traffic_density_level = random.choice(["sparse", "normal", "dense"])
        
        if traffic_density_level == "sparse":
            traffic_density = round(random.uniform(0.0, 0.1), 2)
        elif traffic_density_level == "normal":
            traffic_density = round(random.uniform(0.1, 0.3), 2)
        else:  # dense
            traffic_density = round(random.uniform(0.3, 0.6), 2)
        
        # 事故概率
        accident_prob = round(random.uniform(0.0, 0.3), 2)
        
        # 交通模式 (使用MetaDrive支持的模式)
        from metadrive.manager.traffic_manager import TrafficMode
        traffic_mode = random.choice([TrafficMode.Trigger, TrafficMode.Hybrid])
        
        # 随机交通
        random_traffic = random.choice([True, False])
        
        traffic_config = {
            "traffic_density": traffic_density,
            "accident_prob": accident_prob,
            "traffic_mode": traffic_mode,
            "random_traffic": random_traffic,
            "need_inverse_traffic": random.choice([True, False]),
        }
        
        return traffic_config
    
    def _generate_random_vehicle_config(self) -> Dict[str, Any]:
        """生成随机车辆配置"""
        
        # 随机生成位置
        random_spawn = random.choice([True, False])
        
        # 随机车辆模型
        random_agent_model = random.choice([True, False])
        
        vehicle_config = {
            "random_spawn_lane_index": random_spawn,
            "random_agent_model": random_agent_model,
            "vehicle_config": {
                "show_navi_mark": random.choice([True, False]),
                "show_dest_mark": random.choice([True, False]),
                "show_line_to_dest": random.choice([True, False]),
                # 传感器配置随机化
                "lidar": {
                    "num_lasers": random.choice([32, 64, 120]),
                    "distance": random.randint(30, 80),
                },
                "side_detector": {
                    "num_lasers": random.randint(8, 16),
                    "distance": random.randint(20, 50),
                }
            }
        }
        
        return vehicle_config
    
    def _generate_random_environment_config(self) -> Dict[str, Any]:
        """生成随机环境配置"""
        
        # 终止条件随机化（使用MetaDrive支持的配置键）
        env_config = {
            "crash_vehicle_done": random.choice([True, False]),
            "crash_object_done": random.choice([True, False]),
            "out_of_road_done": random.choice([True, False]),
            
            # AI保护级别
            "use_AI_protector": random.choice([True, False]),
            "save_level": round(random.uniform(0.0, 1.0), 1),
        }
        
        return env_config
    
    def generate_curriculum_scenarios(self, 
                                    total_scenarios: int = 1000,
                                    difficulty_levels: int = 5) -> List[List[Dict[str, Any]]]:
        """
        生成课程学习场景序列
        
        Args:
            total_scenarios: 总场景数
            difficulty_levels: 难度级别数
            
        Returns:
            按难度分组的场景列表
        """
        scenarios_per_level = total_scenarios // difficulty_levels
        curriculum = []
        
        for level in range(difficulty_levels):
            level_scenarios = []
            
            for i in range(scenarios_per_level):
                scenario_id = level * scenarios_per_level + i
                config = self._generate_curriculum_scenario(scenario_id, level, difficulty_levels)
                level_scenarios.append(config)
            
            curriculum.append(level_scenarios)
            
        return curriculum
    
    def _generate_curriculum_scenario(self, 
                                    scenario_id: int, 
                                    level: int, 
                                    max_levels: int) -> Dict[str, Any]:
        """生成课程学习的单个场景"""
        
        # 基础配置
        config = {
            "start_seed": scenario_id,
            "num_scenarios": 1,
            "use_render": False,
            "manual_control": False,
        }
        
        # 难度进度 (0.0 到 1.0)
        difficulty = level / (max_levels - 1)
        
        # 基于难度调整参数
        
        # 地图复杂度
        min_blocks = 1 + int(difficulty * 5)  # 1-6
        max_blocks = 3 + int(difficulty * 12)  # 3-15
        map_blocks = random.randint(min_blocks, max_blocks)
        
        # 交通密度
        min_density = difficulty * 0.1  # 0.0-0.1
        max_density = difficulty * 0.5  # 0.0-0.5
        traffic_density = round(random.uniform(min_density, max_density), 2)
        
        # 事故概率
        accident_prob = round(difficulty * 0.4, 2)  # 0.0-0.4
        
        # Episode长度（随难度增加）
        min_horizon = 500 + int(difficulty * 300)  # 500-800
        max_horizon = 800 + int(difficulty * 400)  # 800-1200
        horizon = random.randint(min_horizon, max_horizon)
        
        config.update({
            "map": map_blocks,
            "traffic_density": traffic_density,
            "accident_prob": accident_prob,
            "horizon": horizon,
            "random_lane_width": difficulty > 0.5,
            "random_lane_num": difficulty > 0.3,
            "random_traffic": difficulty > 0.4,
            # "curriculum_level": level,  # 仅ScenarioEnv支持
            "difficulty": difficulty,
        })
        
        return config
    
    def create_training_env_config(self, 
                                 scenario_type: str = "random",
                                 num_scenarios: int = 100,
                                 **kwargs) -> Dict[str, Any]:
        """
        创建训练环境配置
        
        Args:
            scenario_type: 场景类型 ("random", "curriculum", "safe")
            num_scenarios: 场景数量
            **kwargs: 额外配置参数
            
        Returns:
            训练环境配置
        """
        
        if scenario_type == "random":
            return self._create_random_training_config(num_scenarios, **kwargs)
        elif scenario_type == "curriculum":
            return self._create_curriculum_training_config(num_scenarios, **kwargs)
        elif scenario_type == "safe":
            return self._create_safe_training_config(num_scenarios, **kwargs)
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    def _create_random_training_config(self, num_scenarios: int, **kwargs) -> Dict[str, Any]:
        """创建随机训练配置"""
        
        base_config = {
            "num_scenarios": num_scenarios,
            "start_seed": kwargs.get("start_seed", 0),
            "use_render": False,
            "manual_control": False,
            "horizon": 1000,
            
            # 随机化设置
            "random_lane_width": True,
            "random_lane_num": True,
            "random_traffic": True,
            "random_spawn_lane_index": True,
            "random_agent_model": True,
            
            # 中等难度设置
            "map": random.randint(3, 8),
            "traffic_density": 0.2,
            "accident_prob": 0.1,
            
            # 训练友好设置
            "crash_vehicle_done": True,
            "crash_object_done": True,
            "out_of_road_done": True,
        }
        
        base_config.update(kwargs)
        return base_config
    
    def _create_curriculum_training_config(self, num_scenarios: int, **kwargs) -> Dict[str, Any]:
        """创建课程学习训练配置"""
        
        base_config = {
            "num_scenarios": num_scenarios,
            "start_seed": kwargs.get("start_seed", 0),
            "use_render": False,
            "manual_control": False,
            
            # 课程学习相关配置（这些键不被MetaDriveEnv支持，注释掉）
            # "curriculum_level": kwargs.get("curriculum_level", 1),  # 仅ScenarioEnv支持
            # "target_success_rate": kwargs.get("target_success_rate", 0.8),  # 仅ScenarioEnv支持
            
            # 逐渐增加难度
            "map": 2,  # 从简单开始
            "traffic_density": 0.05,  # 从低密度开始
            "accident_prob": 0.0,  # 从无事故开始
            "horizon": 800,
            
            # 训练设置
            "crash_vehicle_done": True,
            "out_of_road_done": True,
        }
        
        base_config.update(kwargs)
        return base_config
    
    def _create_safe_training_config(self, num_scenarios: int, **kwargs) -> Dict[str, Any]:
        """创建安全驾驶训练配置"""
        
        base_config = {
            "num_scenarios": num_scenarios,
            "start_seed": kwargs.get("start_seed", 0),
            "use_render": False,
            "manual_control": False,
            
            # SafeMetaDrive特色设置
            "accident_prob": 0.8,  # 高事故概率
            "traffic_density": 0.3,  # 中高交通密度
            "crash_vehicle_done": False,  # 不因碰撞立即结束
            "crash_object_done": False,
            "cost_to_reward": False,  # 使用成本函数
            "horizon": 1000,
            
            # 安全相关
            "use_AI_protector": True,
            "save_level": 0.8,
        }
        
        base_config.update(kwargs)
        return base_config


class MetaDriveRandomEnv(MetaDriveEnv):
    """
    基于MetaDrive的随机场景环境
    每次reset时生成新的随机场景
    """
    
    def __init__(self, 
                 generator: RandomScenarioGenerator,
                 scenario_type: str = "random",
                 base_config: Optional[Dict[str, Any]] = None):
        """
        初始化随机环境
        
        Args:
            generator: 场景生成器
            scenario_type: 场景类型
            base_config: 基础配置
        """
        
        self.generator = generator
        self.scenario_type = scenario_type
        self.episode_count = 0
        self.base_config = base_config or {}
        
        # 初始配置
        initial_config = self.generator.create_training_env_config(
            scenario_type=scenario_type,
            **self.base_config
        )
        
        super().__init__(initial_config)
        
    def reset(self, **kwargs):
        """重置环境，生成新的随机场景"""
        
        # 生成新场景配置
        if self.scenario_type == "curriculum":
            # 课程学习：根据训练进度调整难度
            difficulty_level = min(self.episode_count // 100, 4)  # 每100个episode增加难度
            new_config = self.generator._generate_curriculum_scenario(
                self.episode_count, 
                difficulty_level, 
                5
            )
        else:
            # 随机场景
            new_config = self.generator._generate_single_scenario(self.episode_count)
        
        # 更新环境配置，但保护重要的基础配置
        protected_keys = ["num_agents", "num_scenarios", "start_seed"]
        protected_config = {key: self.config.get(key) for key in protected_keys if key in self.config}
        
        # 先更新新配置
        combined_config = new_config.copy()
        combined_config.update(self.base_config)
        
        # 然后恢复受保护的配置
        combined_config.update(protected_config)
        
        self.config.update(combined_config)
        
        self.episode_count += 1
        
        # 调用父类reset
        try:
            # 尝试新版gymnasium接口
            return super().reset(**kwargs)
        except TypeError:
            # 兼容旧版gym接口
            obs = super().reset()
            if isinstance(obs, tuple):
                return obs
            else:
                return obs, {}


def create_random_training_environment(scenario_type: str = "random",
                                     num_scenarios: int = 1000,
                                     seed: Optional[int] = None,
                                     **config_kwargs) -> MetaDriveRandomEnv:
    """
    创建用于PPO训练的随机场景环境
    
    Args:
        scenario_type: 场景类型 ("random", "curriculum", "safe")
        num_scenarios: 场景总数
        seed: 随机种子
        **config_kwargs: 额外配置
        
    Returns:
        随机场景环境
    """
    
    # 创建场景生成器
    generator = RandomScenarioGenerator(seed=seed)
    
    # 创建基础配置
    base_config = {
        "num_scenarios": num_scenarios,
        "start_seed": seed or 0,
        "use_render": False,  # 确保有默认渲染设置
        "manual_control": False,  # 确保不是手动控制
        "num_agents": 1,  # 明确指定agent数量
    }
    base_config.update(config_kwargs)
    
    # 创建环境
    env = MetaDriveRandomEnv(
        generator=generator,
        scenario_type=scenario_type,
        base_config=base_config
    )
    
    return env


if __name__ == "__main__":
    print("🎯 MetaDrive随机场景生成器测试")
    
    # 创建场景生成器
    generator = RandomScenarioGenerator(seed=42)
    
    print("\n1. 📋 生成基础随机场景")
    scenarios = generator.generate_basic_scenarios(5)
    for i, scenario in enumerate(scenarios):
        print(f"  场景{i+1}: 地图={scenario['map']}, 交通密度={scenario['traffic_density']}, 事故率={scenario['accident_prob']}")
    
    print("\n2. 📚 生成课程学习场景")
    curriculum = generator.generate_curriculum_scenarios(20, 4)
    for level, level_scenarios in enumerate(curriculum):
        print(f"  难度{level+1}: {len(level_scenarios)}个场景")
        sample = level_scenarios[0]
        print(f"    示例 - 难度={sample['difficulty']:.2f}, 地图={sample['map']}, 交通={sample['traffic_density']}")
    
    print("\n3. 🚗 创建训练环境配置")
    configs = ["random", "curriculum", "safe"]
    for config_type in configs:
        config = generator.create_training_env_config(config_type, num_scenarios=100)
        print(f"  {config_type}: 场景数={config['num_scenarios']}, 地图={config.get('map', 'N/A')}")
    
    print("\n4. 🧪 测试随机环境")
    try:
        env = create_random_training_environment("random", num_scenarios=10, seed=42)
        obs, info = env.reset()
        print(f"  环境创建成功: 观测维度={obs.shape if hasattr(obs, 'shape') else len(obs)}")
        
        # 测试几步
        for step in range(3):
            action = env.action_space.sample()
            result = env.step(action)
            
            # 处理不同的返回格式
            if len(result) == 4:
                obs, reward, done, info = result
                terminated = truncated = done
            else:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
                
            print(f"    步骤{step+1}: reward={reward:.3f}, done={done}")
            if done:
                obs_result = env.reset()
                if isinstance(obs_result, tuple):
                    obs, info = obs_result
                else:
                    obs = obs_result
                print(f"    环境重置，新场景#{env.episode_count}")
        
        env.close()
        print("  ✅ 环境测试完成")
        
    except Exception as e:
        print(f"  ❌ 环境测试失败: {e}")
    
    print("\n🎉 随机场景生成器测试完成!")
    print("\n📝 使用示例:")
    print("generator = RandomScenarioGenerator(seed=42)")
    print("env = create_random_training_environment('curriculum', num_scenarios=1000)")
    print("# 然后就可以用于SB3训练了！") 