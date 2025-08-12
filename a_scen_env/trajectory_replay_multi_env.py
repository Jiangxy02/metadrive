"""
基于不同MetaDrive环境的轨迹重放示例

展示如何将TrajectoryReplayEnv与不同的基础环境结合使用
"""

import sys
sys.path.append('/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive')

from metadrive.envs import MetaDriveEnv, SafeMetaDriveEnv
from trajectory_replay import TrajectoryReplayEnv
from trajectory_loader import load_trajectory


class TrajectoryReplayWithSafeEnv(SafeMetaDriveEnv):
    """
    基于SafeMetaDriveEnv的轨迹重放环境
    结合了安全强化学习环境的事故处理能力和轨迹重放功能
    """
    
    def __init__(self, trajectory_dict, config=None):
        # 设置SafeMetaDriveEnv的默认配置
        safe_config = {
            "accident_prob": 0.3,           # 30%概率出现事故场景
            "crash_vehicle_done": False,    # 碰撞时不立即结束
            "crash_object_done": False,     # 撞物体时不立即结束  
            "cost_to_reward": True,         # 将cost转换为reward信号
            "map": "SSS",                   # 短地图
            "traffic_density": 0.1,         # 适度交通密度
            "use_render": True,
            "manual_control": True,
        }
        
        if config:
            safe_config.update(config)
            
        # 复制trajectory_dict避免修改原数据
        self.trajectory_dict = trajectory_dict.copy()
        
        # 将车辆-1设置为主车，从背景车辆中移除
        self.main_vehicle_trajectory = None
        if -1 in self.trajectory_dict:
            self.main_vehicle_trajectory = self.trajectory_dict.pop(-1)
            print(f"Vehicle -1 will be used as the main car (agent)")
            print(f"Remaining {len(self.trajectory_dict)} vehicles will be background vehicles")
        
        # 初始化SafeMetaDriveEnv
        super().__init__(safe_config)
        
        # 添加轨迹重放相关的属性（从TrajectoryReplayEnv复制）
        self._step_count = 0
        self._simulation_time = 0.0
        self._trajectory_start_time = None
        
        # 背景车管理
        self.ghost_vehicles = {}
        
        # 从TrajectoryReplayEnv导入控制管理器
        from control_mode_manager import ControlModeManager
        self.control_manager = ControlModeManager(
            engine=self.engine,
            main_vehicle_trajectory=self.main_vehicle_trajectory
        )

    def reset(self):
        """重置环境，结合SafeMetaDriveEnv和轨迹重放功能"""
        obs = super().reset()
        
        # 重置轨迹重放相关状态
        self._step_count = 0
        self._simulation_time = 0.0
        self.ghost_vehicles = {}
        
        # 设置控制管理器
        self.control_manager.set_agent(self.agent)
        self.control_manager.reset_modes()
        self.control_manager.bind_hotkeys()
        self.control_manager.initialize_policies()
        
        # 设置主车初始状态
        if self.main_vehicle_trajectory and len(self.main_vehicle_trajectory) > 0:
            initial_state = self.main_vehicle_trajectory[0]
            self.agent.set_position([initial_state["x"], initial_state["y"]])
            self.agent.set_heading_theta(initial_state["heading"])
            import numpy as np
            direction = [np.cos(initial_state["heading"]), np.sin(initial_state["heading"])]
            self.agent.set_velocity(direction, initial_state["speed"])
            print(f"Main car initialized at position: ({initial_state['x']:.1f}, {initial_state['y']:.1f})")
        
        return obs

    def step(self, action):
        """执行一步，结合安全环境和轨迹重放"""
        # 更新仿真时间
        self._simulation_time += 0.1  # 简化的时间步长
        
        # 使用控制管理器获取动作
        self.control_manager.set_step_count(self._step_count)
        action, action_info = self.control_manager.get_action(action)
        
        # 执行SafeMetaDriveEnv的step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 添加cost信息（SafeMetaDriveEnv特有）
        cost, cost_info = self.cost_function(self.agent.id)
        info.update(cost_info)
        
        # 重放背景车（简化版本）
        self._replay_background_vehicles()
        
        self._step_count += 1
        
        # 添加控制模式信息
        control_status = self.control_manager.get_control_status()
        info.update(control_status)
        info["action_source"] = action_info.get("source", "unknown")
        info["safety_cost"] = cost
        
        return obs, reward, terminated or truncated, info

    def _replay_background_vehicles(self):
        """简化的背景车重放（可以根据需要完善）"""
        # 这里可以添加背景车重放逻辑
        # 为了演示，我们跳过复杂的实现
        pass


def create_different_trajectory_environments():
    """创建基于不同基础环境的轨迹重放环境"""
    
    # 加载轨迹数据
    csv_path = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/scenario_vehicles_9_0_1_2_3_4_5_6_7_8_test3_20250304_150943_row001_filtered.csv"
    
    print("正在加载轨迹数据...")
    traj_data = load_trajectory(
        csv_path=csv_path, 
        normalize_position=False,
        max_duration=10,  # 加载10秒数据用于测试
        use_original_position=False,
        translate_to_origin=True,
        target_fps=50.0,
        use_original_timestamps=True
    )
    
    environments = {}
    
    # 1. 基于标准MetaDriveEnv的轨迹重放（原版）
    print("\n创建标准轨迹重放环境...")
    try:
        environments['standard'] = TrajectoryReplayEnv(
            traj_data.copy(),
            config={
                "map": "SSS",
                "use_render": False,  # 避免窗口冲突
                "manual_control": True,
            }
        )
        print("✅ 标准轨迹重放环境创建成功")
    except Exception as e:
        print(f"❌ 标准环境创建失败: {e}")
    
    # 2. 基于SafeMetaDriveEnv的轨迹重放（新版）
    print("\n创建安全轨迹重放环境...")
    try:
        environments['safe'] = TrajectoryReplayWithSafeEnv(
            traj_data.copy(),
            config={
                "map": "SSS",
                "use_render": False,
                "accident_prob": 0.5,  # 50%概率出现事故
            }
        )
        print("✅ 安全轨迹重放环境创建成功")
    except Exception as e:
        print(f"❌ 安全环境创建失败: {e}")
    
    return environments


def test_environments():
    """测试不同的轨迹重放环境"""
    
    environments = create_different_trajectory_environments()
    
    for env_name, env in environments.items():
        print(f"\n=== 测试 {env_name} 环境 ===")
        
        try:
            # 重置环境
            obs = env.reset()
            print(f"环境类型: {type(env).__name__}")
            print(f"主车位置: {env.agent.position}")
            
            # 运行几步测试
            for i in range(3):
                action = [0.0, 0.0]  # 默认动作
                obs, reward, done, info = env.step(action)
                
                # 显示不同环境的特有信息
                if env_name == 'safe' and 'safety_cost' in info:
                    print(f"Step {i}: reward={reward:.3f}, cost={info['safety_cost']:.3f}, control={info.get('Control Mode', 'unknown')}")
                else:
                    print(f"Step {i}: reward={reward:.3f}, control={info.get('Control Mode', 'unknown')}")
                
                if done:
                    print(f"Episode结束于step {i}")
                    break
            
            env.close()
            print(f"✅ {env_name} 环境测试完成")
            
        except Exception as e:
            print(f"❌ {env_name} 环境测试失败: {e}")


def demonstrate_environment_modification():
    """演示如何修改trajectory_replay.py来使用不同环境"""
    
    print("=== 如何修改trajectory_replay.py使用不同环境 ===")
    print("""
1. 方法1: 修改继承的基类
   # 原始代码
   class TrajectoryReplayEnv(MetaDriveEnv):
       
   # 修改为
   class TrajectoryReplayEnv(SafeMetaDriveEnv):  # 或其他环境类

2. 方法2: 创建新的子类（推荐）
   class TrajectoryReplaySafe(TrajectoryReplayEnv, SafeMetaDriveEnv):
       def __init__(self, trajectory_dict, config=None):
           # 添加安全环境特有配置
           super().__init__(trajectory_dict, config)

3. 方法3: 使用组合模式
   class TrajectoryReplayWithCustomEnv:
       def __init__(self, base_env_class, trajectory_dict, config):
           self.base_env = base_env_class(config)
           # 添加轨迹重放功能

4. 方法4: 配置驱动的环境选择
   def create_trajectory_env(env_type='standard', trajectory_dict, config):
       env_classes = {
           'standard': MetaDriveEnv,
           'safe': SafeMetaDriveEnv, 
           'multi_agent': MultiAgentMetaDrive,
       }
       base_class = env_classes[env_type]
       return type('TrajectoryReplayCustom', (TrajectoryReplayEnv, base_class), {})(trajectory_dict, config)
    """)


if __name__ == "__main__":
    print("MetaDrive环境类型演示")
    print("=" * 50)
    
    # 演示环境修改方法
    demonstrate_environment_modification()
    
    print("\n" + "=" * 50)
    
    # 测试不同环境
    test_environments() 