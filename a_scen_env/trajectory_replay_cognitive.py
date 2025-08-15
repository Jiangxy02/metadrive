"""
轨迹重放环境 - 认知模块集成版本
基于trajectory_replay.py，集成认知模块实现人类认知建模
感知误差和执行延迟仅在PPO模式下生效
"""

import pandas as pd
import numpy as np
from metadrive.envs import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.engine.engine_utils import get_global_config
from metadrive.constants import HELP_MESSAGE
import sys
import os

# 添加认知模块路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
cognitive_module_dir = os.path.join(parent_dir, 'cognitive_module')
if cognitive_module_dir not in sys.path:
    sys.path.insert(0, cognitive_module_dir)

# 导入认知模块
from cognitive_wrappers import PerceptionWrapper, CognitiveBiasWrapper, DelayWrapper
from cognitive_perception_module import CognitivePerceptionModule
from cognitive_bias_module import CognitiveBiasModule  # 新增：导入认知偏差模块

# 导入可视化模块
from cognitive_visualization import CognitiveDataRecorder, CognitiveVisualizer

# 导入原有模块
from control_mode_manager import ControlModeManager
from trajectory_loader import TrajectoryLoader, load_trajectory
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation
from observation_recorder import ObservationRecorder


class CognitiveDelayModule:
    """
    认知延迟模块 - 独立实现，不依赖gym环境
    仅在PPO模式下应用执行延迟
    """
    def __init__(self, delay_steps=2, enable_smoothing=True, smoothing_factor=0.3):
        self.delay_steps = delay_steps
        self.enable_smoothing = enable_smoothing
        self.smoothing_factor = smoothing_factor
        
        from collections import deque
        self.buffer = deque(maxlen=delay_steps + 1)
        self.previous_action = np.array([0.0, 0.0])
        
    def process_action(self, action, is_ppo_mode=False):
        """
        处理动作，仅在PPO模式下应用延迟
        
        Args:
            action: 原始动作
            is_ppo_mode: 是否为PPO专家模式
            
        Returns:
            处理后的动作
        """
        if not is_ppo_mode:
            return action  # 非PPO模式直接返回原始动作
            
        # 确保action是numpy数组
        action = np.array(action, dtype=np.float32)
        
        # 记录原始命令供可视化使用
        self._last_commanded_action = action.copy()

        # 动作平滑（如果启用）
        if self.enable_smoothing:
            smoothed_action = (
                self.smoothing_factor * action +
                (1 - self.smoothing_factor) * self.previous_action
            )
            self.previous_action = smoothed_action.copy()
        else:
            smoothed_action = action
        
        # 添加到延迟缓冲区
        self.buffer.append(smoothed_action.copy())
        
        # 获取延迟后的动作
        if len(self.buffer) <= self.delay_steps:
            # 初始几步返回零动作
            delayed_action = np.array([0.0, 0.0])
        else:
            # 返回延迟的动作
            delayed_action = self.buffer[0]
        
        return delayed_action
    
    def reset(self):
        """重置延迟模块"""
        self.buffer.clear()
        self.previous_action = np.array([0.0, 0.0])


class TrajectoryReplayEnvCognitive(MetaDriveEnv):
    """
    轨迹重放环境 - 认知模块集成版本
    
    在原有功能基础上，集成认知建模模块：
    - PerceptionModule: 感知误差和卡尔曼滤波（仅PPO模式）
    - CognitiveBiasModule: TTA认知偏差（所有模式）
    - DelayModule: 执行延迟（仅PPO模式）
    """
    
    def __init__(self, trajectory_dict, config=None):
        # 处理配置参数
        user_config = config.copy() if config else {}
        
        # 认知模块配置
        self.enable_cognitive_modules = user_config.pop("enable_cognitive_modules", False)
        self.cognitive_config = user_config.pop("cognitive_config", {})
        
        # 可视化配置
        self.enable_visualization = user_config.pop("enable_visualization", True)
        
        # 创建基于时间戳的输出目录
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/fig_cog"
        self.visualization_output_dir = f"{base_dir}/cognitive_analysis_{timestamp}"
        
        # 初始化认知模块（在super().__init__之前）
        self.perception_module = None
        self.cognitive_bias_module = None  # 新增：初始化认知偏差模块
        self.delay_module = None
        
        # 初始化可视化模块
        self.data_recorder = None
        self.visualizer = None
        
        if self.enable_cognitive_modules:
            self._initialize_cognitive_modules()
            
        if self.enable_visualization:
            self._initialize_visualization()
            
        # 保存原始的trajectory_dict用于背景车
        self.enable_background_vehicles = user_config.get("enable_background_vehicles", False)
        
        # 复制trajectory_dict避免修改原数据
        original_trajectory_dict = trajectory_dict.copy()
        
        # 将车辆-1设置为主车
        self.main_vehicle_trajectory = None
        if -1 in original_trajectory_dict:
            self.main_vehicle_trajectory = original_trajectory_dict.pop(-1)
            print(f"Vehicle -1 will be used as the main car (agent)")
        else:
            print("Warning: Vehicle -1 not found in trajectory data")
        
        # 根据enable_background_vehicles参数决定是否保留背景车数据
        if self.enable_background_vehicles:
            self.trajectory_dict = original_trajectory_dict
            print(f"Loaded {len(self.trajectory_dict)} background vehicles from CSV")
        else:
            self.trajectory_dict = {}
            print("⚠️  Background vehicles disabled")
            
        # 计算最大步数
        if self.trajectory_dict:
            self.max_step = max(len(traj) for traj in self.trajectory_dict.values())
        elif self.main_vehicle_trajectory:
            self.max_step = len(self.main_vehicle_trajectory)
        else:
            self.max_step = 1000
            
        self._step_count = 0
        self._simulation_time = 0.0
        self._trajectory_start_time = None
        
        # 实时时间跟踪
        import time
        self._real_start_time = None
        self._real_time_module = time
        
        # 设置默认配置
        default_config = {
            "use_render": True,
            "map": "S"*8,
            "manual_control": True,
            "controller": "keyboard",
            "start_seed": 0,
            "map_region_size": 2048,
            "drivable_area_extension": 50,
            "horizon": 10000,
            "traffic_density": 0.0,
            "accident_prob": 0.0,
            "static_traffic_object": False,
            "vehicle_config": {
                "show_navi_mark": True,
                "show_dest_mark": True,
                "show_line_to_dest": True,
                "show_line_to_navi_mark": True,
                "show_navigation_arrow": True,
                "navigation_module": NodeNetworkNavigation,
                "destination": None,
                "spawn_lane_index": None,
            }
        }
        
        # 处理其他用户配置
        user_config.pop("enable_background_vehicles", None)
        
        self.enable_realtime = user_config.pop("enable_realtime", True)
        self.target_fps = user_config.pop("target_fps", 50.0)
        self._last_step_time = None
        
        self.end_on_crash = user_config.pop("end_on_crash", False)
        self.end_on_out_of_road = user_config.pop("end_on_out_of_road", False)
        self.end_on_arrive_dest = user_config.pop("end_on_arrive_dest", False)
        self.end_on_horizon = user_config.pop("end_on_horizon", False)
        
        self.background_vehicle_update_mode = user_config.pop("background_vehicle_update_mode", "position")
        self.disable_ppo_expert = user_config.pop("disable_ppo_expert", False)
        
        # 观测记录器配置
        self.enable_observation_recording = user_config.pop("enable_observation_recording", False)
        self.observation_recorder = None
        if self.enable_observation_recording:
            session_name = user_config.pop("recording_session_name", None)
            output_dir = user_config.pop("recording_output_dir", "observation_logs")
            self.observation_recorder = ObservationRecorder(output_dir=output_dir, session_name=session_name)
            print("✅ 观测记录器已启用")
        
        if user_config:
            default_config.update(user_config)
        
        # 调用父类初始化
        super().__init__(default_config)
        
        # 获取仿真时间步长
        try:
            self.physics_world_step_size = self.engine.physics_world.static_world.getPhysicsWorldStepSize()
            print(f"MetaDrive physics step size: {self.physics_world_step_size:.6f} seconds")
        except AttributeError:
            self.physics_world_step_size = 0.02
            print(f"Using default physics step size: {self.physics_world_step_size:.6f} seconds")
        
        # 初始化控制模式管理器
        self.control_manager = ControlModeManager(
            engine=self.engine,
            main_vehicle_trajectory=self.main_vehicle_trajectory
        )
        
        # 背景车缓存
        self.ghost_vehicles = {}
        
        # 重写max_step
        self.max_step = 10000
        
        print("\n=== 认知模块状态 ===")
        if self.enable_cognitive_modules:
            print("✅ 认知模块已启用")
            if self.perception_module:
                print(f"  - 感知模块: sigma={self.perception_module.sigma}, kalman={self.perception_module.enable_kalman}")
            if self.delay_module:
                print(f"  - 延迟模块: delay_steps={self.delay_module.delay_steps}, smoothing={self.delay_module.enable_smoothing}")
            print("  - 认知模块仅在PPO专家模式下生效")
        else:
            print("❌ 认知模块未启用")
        print("=" * 30 + "\n")
    
    def _initialize_cognitive_modules(self):
        """初始化认知模块"""
        print("正在初始化认知模块...")
        
        # 感知模块配置 - 使用新的雷达噪声注入模块
        perception_config = self.cognitive_config.get('perception', {})
        
        # 转换配置格式以适配新的噪声模块
        noise_config = {
            'sigma0': perception_config.get('sigma', 0.5),  # 基础噪声
            'k': perception_config.get('k', 0.02),          # 距离相关系数
            
            # 漏检配置
            'p_miss0': perception_config.get('p_miss0', 0.01),  # 基础漏检概率
            'far_distance': perception_config.get('far_distance', 50.0),
            
            # 误检配置（保守默认值）
            'p_false': perception_config.get('p_false', 0.0001), # 误检概率（大幅降低）
            'near_min': perception_config.get('near_min', 1.0),  # 误检最近距离
            'near_max': perception_config.get('near_max', 5.0),  # 误检最远距离
            
            # 角度抖动
            'angle_jitter_steps': perception_config.get('angle_jitter_steps', 1),
            
            # 时间相关性配置
            'use_ar1': perception_config.get('use_ar1', True),   # 启用AR(1)
            'rho': perception_config.get('rho', 0.8),            # AR(1)系数
            'use_lowpass': perception_config.get('use_lowpass', False),
            'alpha': perception_config.get('alpha', 0.7),
            
            # KF配置
            'use_kf': perception_config.get('use_kf', True),
            'kf_dt': perception_config.get('kf_dt', 0.1),
            'kf_q': perception_config.get('kf_q', 0.5),
            'kf_sigma_a': perception_config.get('kf_sigma_a', 3.0),
            'kf_q_scale': perception_config.get('kf_q_scale', 100.0),
            'kf_r_floor': perception_config.get('kf_r_floor', 1e-4),
            'kf_init_std_pos': perception_config.get('kf_init_std_pos', 5.0),
            'kf_init_std_vel': perception_config.get('kf_init_std_vel', 10.0),
        }
        
        self.perception_module = CognitivePerceptionModule(noise_config)
        print("  ✅ 感知模块已初始化（雷达噪声注入器）")
        
        # 延迟模块配置
        delay_config = self.cognitive_config.get('delay', {})
        if delay_config.get('enable', False):
            self.delay_module = CognitiveDelayModule(
                delay_steps=delay_config.get('delay_steps', 2),
                enable_smoothing=delay_config.get('enable_smoothing', True),
                smoothing_factor=delay_config.get('smoothing_factor', 0.3)
            )
            print("  ✅ 延迟模块已初始化")
        
        # 认知偏差模块配置（新增）
        cognitive_bias_config = self.cognitive_config.get('cognitive_bias', {})
        if cognitive_bias_config.get('enable', False):
            # 创建认知偏差模块
            bias_config = {
                'inverse_tta_coef': cognitive_bias_config.get('inverse_tta_coef', 1.0),
                'tta_threshold': cognitive_bias_config.get('tta_threshold', 1.0),
                'adaptive_bias': cognitive_bias_config.get('adaptive_bias', True),
                'adaptation_rate': cognitive_bias_config.get('adaptation_rate', 0.01),
                'min_adaptive_factor': cognitive_bias_config.get('min_adaptive_factor', 0.5),
                'max_adaptive_factor': cognitive_bias_config.get('max_adaptive_factor', 2.0),
                'history_length': cognitive_bias_config.get('history_length', 100),
                'verbose': cognitive_bias_config.get('verbose', False),
                
                # 视觉厌恶参数
                'visual_detection_distance': cognitive_bias_config.get('visual_detection_distance', 50.0),
                'visual_detection_angle': cognitive_bias_config.get('visual_detection_angle', 30.0),
                'visual_aversion_strength': cognitive_bias_config.get('visual_aversion_strength', 0.5)
            }
            self.cognitive_bias_module = CognitiveBiasModule(bias_config)
            print("  ✅ 认知偏差模块已初始化（TTA风险厌恶）")

    def _initialize_visualization(self):
        """初始化可视化模块"""
        print("正在初始化可视化模块...")
        
        try:
            self.data_recorder = CognitiveDataRecorder()
            self.visualizer = CognitiveVisualizer(output_dir=self.visualization_output_dir)
            print(f"✅ 可视化模块已初始化，输出目录: {self.visualization_output_dir}")
        except Exception as e:
            print(f"❌ 可视化模块初始化失败: {e}")
            self.enable_visualization = False
    
    def reset(self):
        """重置环境，包括认知模块"""
        # 清理背景车
        self._cleanup_ghost_vehicles()
        
        # 调用父类reset
        reset_result = super().reset()
        
        # 处理reset返回值（可能是(obs, info)元组或单独的obs）
        if isinstance(reset_result, tuple) and len(reset_result) == 2:
            obs, reset_info = reset_result
        else:
            obs = reset_result
            reset_info = {}
        
        # 重置计数器和时间
        self._step_count = 0
        self._simulation_time = 0.0
        self._real_start_time = self._real_time_module.time()
        self._last_step_time = self._real_start_time
        self.ghost_vehicles = {}
        
        # 初始化轨迹起始时间
        if self.main_vehicle_trajectory and "timestamp" in self.main_vehicle_trajectory[0]:
            self._trajectory_start_time = self.main_vehicle_trajectory[0]["timestamp"]
        else:
            self._trajectory_start_time = 0.0
        
        # 设置控制管理器
        self.control_manager.set_agent(self.agent)
        self.control_manager.reset_modes()
        self.control_manager.bind_hotkeys()
        self.control_manager.initialize_policies()
        
        # ===== 新增：设置自定义目标点 =====
        self._set_custom_destination()
        
        # ===== 新增：调试导航信息 =====
        self._debug_navigation_info()
        
        # 设置主车初始状态
        if self.main_vehicle_trajectory and len(self.main_vehicle_trajectory) > 0:
            initial_state = self.main_vehicle_trajectory[0]
            self.agent.set_position([initial_state["x"], initial_state["y"]])
            self.agent.set_heading_theta(initial_state["heading"])
            direction = [np.cos(initial_state["heading"]), np.sin(initial_state["heading"])]
            self.agent.set_velocity(direction, initial_state["speed"])
            print(f"Main car initialized at original position: ({initial_state['x']:.1f}, {initial_state['y']:.1f})")
            print(f"Initial heading: {initial_state['heading']:.2f} rad, speed: {initial_state['speed']:.1f} m/s")

            # ===== 新增：在主车位置设置后修复车道检测问题 =====
            self._fix_lane_detection()
        
        # 重置认知模块
        if self.enable_cognitive_modules:
            # 附加噪声雷达到环境（在环境完全初始化后）
            if self.perception_module:
                success = self.perception_module.attach_to_env(self)
                if success:
                    print("  ✅ 噪声雷达已附加到环境")
                else:
                    print("  ⚠️ 噪声雷达附加失败，将跳过雷达噪声注入")
                self.perception_module.reset()
            
            # 附加和重置认知偏差模块
            if self.cognitive_bias_module:
                success = self.cognitive_bias_module.attach_to_env(self)
                if success:
                    print("  ✅ 认知偏差模块已附加到环境")
                else:
                    print("  ⚠️ 认知偏差模块附加失败，将跳过TTA偏差调整")
                self.cognitive_bias_module.reset()
            
            if self.delay_module:
                self.delay_module.reset()
            print("认知模块已重置")
        
        # 重置可视化数据记录器
        if self.enable_visualization and self.data_recorder:
            self.data_recorder.reset_data()
        
        # 处理观测（可能应用感知误差）
        obs = self._process_observation(obs)
        
        # # 调试：打印观测数据结构并保存到文件
        # with open("/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/metadrive_observation_analysis.txt", "w") as f:
        #     f.write("=== MetaDrive 观测数据结构分析 ===\n")
        #     f.write(f"观测数据类型: {type(obs)}\n")
            
        #     if isinstance(obs, dict):
        #         f.write(f"观测字典键数量: {len(obs.keys())}\n")
        #         f.write(f"观测字典键: {list(obs.keys())}\n\n")
                
        #         for key, value in obs.items():
        #             f.write(f"键 '{key}':\n")
        #             f.write(f"  类型: {type(value)}\n")
        #             f.write(f"  形状: {getattr(value, 'shape', 'N/A')}\n")
                    
        #             if hasattr(value, 'shape') and len(value.shape) == 1:
        #                 if value.shape[0] <= 20:  # 小数组显示全部
        #                     f.write(f"  内容: {value}\n")
        #                 else:  # 大数组显示前20个
        #                     f.write(f"  内容(前20个): {value[:20]}\n")
        #                     f.write(f"  内容(后10个): {value[-10:]}\n")
        #             elif hasattr(value, 'shape') and len(value.shape) == 2:
        #                 f.write(f"  内容(前3行): {value[:3] if value.shape[0] >= 3 else value}\n")
        #             f.write("\n")
                    
        #     elif isinstance(obs, (list, tuple, np.ndarray)):
        #         obs_array = np.array(obs)
        #         f.write(f"观测数组形状: {obs_array.shape}\n")
        #         f.write(f"观测数组数据类型: {obs_array.dtype}\n")
        #         f.write(f"观测数组最小值: {np.min(obs_array)}\n")
        #         f.write(f"观测数组最大值: {np.max(obs_array)}\n")
        #         f.write(f"观测数组均值: {np.mean(obs_array)}\n\n")
                
        #         # 分析数组结构
        #         f.write("=== 观测向量结构推测 ===\n")
        #         if obs_array.shape[0] == 259:
        #             f.write("这是MetaDrive的标准观测向量(259维)，通常包含:\n")
        #             f.write("- 前几维: 主车状态 (位置、速度、朝向等)\n")
        #             f.write("- 中间部分: 激光雷达数据 (通常240维)\n") 
        #             f.write("- 后面部分: 车道线检测、侧面检测器等\n\n")
                    
        #             f.write(f"前20维(主车状态): {obs_array[:20]}\n")
        #             f.write(f"第21-260维(可能是激光雷达): 长度{len(obs_array[20:])}")
        #             f.write(f"激光雷达数据范围: [{np.min(obs_array[20:]):.3f}, {np.max(obs_array[20:]):.3f}]\n")
        #             f.write(f"激光雷达数据均值: {np.mean(obs_array[20:]):.3f}\n")
                
        #         f.write(f"\n完整观测向量:\n{obs_array}\n")
            
        #     f.write("================================\n")
        
        # print("观测数据分析已保存到: /tmp/metadrive_observation_analysis.txt")
        
        
        # 返回与父类相同的格式
        if isinstance(reset_result, tuple):
            return obs, reset_info
        else:
            return obs
    
    def _get_ego_state_for_cognitive_modules(self):
        """
        为认知模块获取主车状态信息
        
        Returns:
            dict: 包含主车状态的字典
        """
        agent = self.agent
        
        # 认知模块需要的关键状态信息
        ego_state = {
            # 用于感知模块的位置信息
            'position_x': agent.position[0],
            'position_y': agent.position[1],
            
            # 用于延迟模块的动力学信息
            'speed': agent.speed,
            'heading': agent.heading_theta,
            'velocity_x': agent.velocity[0],
            'velocity_y': agent.velocity[1],
            
            # 用于安全评估的车道信息
            'on_lane': agent.on_lane,
            'out_of_road': agent.out_of_road,
            'dist_to_left_side': agent.dist_to_left_side,
            'dist_to_right_side': agent.dist_to_right_side
        }
        
        return ego_state
    
    def _process_observation(self, obs):
        """
        处理观测数据（兼容性方法）
        注意：新的状态处理主要在step()方法中的process_vehicle_state()进行
        """
        if not self.enable_cognitive_modules or not self.perception_module:
            return obs
        
        # 调试：打印观测数据类型和结构（仅第一次）
        if not hasattr(self, '_obs_type_printed'):
            print(f"\n[调试] 观测数据类型: {type(obs)}")
            if isinstance(obs, dict):
                print(f"[调试] 观测字典键: {list(obs.keys())[:5]}...")  # 只打印前5个键
            elif isinstance(obs, tuple):
                print(f"[调试] 观测是元组，长度: {len(obs)}")
                if len(obs) == 2:
                    print(f"[调试] 元组第一个元素类型: {type(obs[0])}")
                    print(f"[调试] 元组第二个元素类型: {type(obs[1])}")
                    # 在Gym中，reset()通常返回(observation, info)
                    # 我们只需要observation部分
                    if isinstance(obs[0], dict):
                        print(f"[调试] 实际观测是字典，键: {list(obs[0].keys())[:5]}...")
            elif isinstance(obs, (list, np.ndarray)):
                print(f"[调试] 观测数组形状: {np.array(obs).shape}")
            self._obs_type_printed = True
        
        # 检查是否为PPO专家模式
        is_ppo_mode = (
            self.control_manager.expert_mode and 
            not self.disable_ppo_expert and
            hasattr(self.agent, 'expert_takeover') and 
            self.agent.expert_takeover
        )
        
        # 保留原有的观测处理逻辑（兼容性，但主要处理已在step中完成）
        if is_ppo_mode:
            # 获取主车状态
            ego_state = np.array([
                self.agent.position[0],
                self.agent.position[1]
            ])
            
            # 应用感知处理（这主要是兼容性处理，核心处理在process_vehicle_state中）
            processed_obs = self.perception_module.process_observation(
                obs, ego_state, is_ppo_mode=True
            )
            
            return processed_obs
        
        return obs
    
    def step(self, action):
        """执行一步仿真，集成认知模块处理"""
        # 实时控制
        if self.enable_realtime and self._last_step_time is not None:
            current_time = self._real_time_module.time()
            target_step_duration = 1.0 / self.target_fps
            elapsed_since_last_step = current_time - self._last_step_time
            
            if elapsed_since_last_step < target_step_duration:
                sleep_duration = target_step_duration - elapsed_since_last_step
                self._real_time_module.sleep(sleep_duration)
            
            self._last_step_time = self._real_time_module.time()
        
        # 更新仿真时间
        decision_repeat = self.engine.global_config.get('decision_repeat', 1)
        effective_time_step = self.physics_world_step_size * decision_repeat
        self._simulation_time += effective_time_step
        
        # 检查是否为PPO专家模式
        is_ppo_mode = (
            self.control_manager.expert_mode and 
            not self.disable_ppo_expert and
            hasattr(self.agent, 'expert_takeover') and 
            self.agent.expert_takeover and
            not self.control_manager.use_trajectory_for_main  # 非轨迹重放模式
        )
        
        # 更新控制管理器
        self.control_manager.set_step_count(self._step_count)
        
        # 获取动作
        action, action_info = self.control_manager.get_action(action)
        
        # 应用执行延迟（仅PPO模式）
        if self.enable_cognitive_modules and self.delay_module and is_ppo_mode:
            original_action = action.copy() if isinstance(action, np.ndarray) else list(action)
            action = self.delay_module.process_action(action, is_ppo_mode=True)
            
            if self._step_count % 50 == 0:
                action_diff = np.linalg.norm(np.array(action) - np.array(original_action))
                print(f"[认知延迟] PPO模式 - 应用执行延迟 (delay={self.delay_module.delay_steps} steps, diff={action_diff:.3f})")
        
        # === 旧逻辑：已废弃，噪声现在在雷达传感器层注入 ===
        # 不再在此处修改agent状态，噪声直接在雷达传感器的perceive方法中注入
        # if self.enable_cognitive_modules and self.perception_module and is_ppo_mode:
        #     ego_state = self._get_ego_state_for_cognitive_modules()
        #     self.perception_module.process_vehicle_state(agent=self.agent, ego_state=ego_state, is_ppo_mode=True)
        #     if self._step_count % 50 == 0:
        #         print(f"[认知感知] PPO模式 - 对agent状态应用感知噪声和卡尔曼滤波")
        
        # 执行动作
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 应用认知偏差调整奖励（仅PPO模式）
        if self.enable_cognitive_modules and self.cognitive_bias_module and is_ppo_mode:
            original_reward = reward
            adjusted_reward, bias_info = self.cognitive_bias_module.process_reward(
                original_reward=reward,
                env=self,
                info=info,
                is_ppo_mode=is_ppo_mode
            )
            reward = adjusted_reward
            
            # 将偏差信息添加到info字典
            info['cognitive_bias_info'] = bias_info
            info['original_reward'] = original_reward
            info['bias_applied'] = bias_info['bias_applied']
            
            if self._step_count % 50 == 0 and bias_info['bias_active']:
                print(f"[认知偏差] PPO模式 - 应用TTA风险厌恶 (inverse_tta={bias_info['inverse_tta']:.3f}, bias={bias_info['bias_applied']:.3f})")
        
        # 保留原有的观测处理方法（兼容性）
        obs = self._process_observation(obs)
        
        # 记录认知数据（如果启用）
        if self.enable_visualization and self.data_recorder:
            self.data_recorder.record_step(
                step_count=self._step_count,
                timestamp=self._simulation_time,
                env=self,
                obs=obs,
                action=action,
                action_info=action_info,
                reward=reward,
                info=info,
                cognitive_active=is_ppo_mode and self.enable_cognitive_modules
            )
        
        # 记录观测（如果启用）
        if self.observation_recorder:
            self.observation_recorder.record_step(
                env=self,
                action=action,
                action_info=action_info,
                obs=obs,
                reward=reward,
                info=info,
                step_count=self._step_count
            )
        
        # 重放背景车
        self._replay_all_vehicles_by_time()
        
        # 清理已结束的背景车
        self._cleanup_finished_trajectories()
        
        self._step_count += 1
        
        # 检查结束条件
        crash_flag = info.get("crash", False) or info.get("crash_vehicle", False)
        out_of_road_flag = info.get("out_of_road", False)
        arrive_dest_flag = info.get("arrive_dest", False)
        horizon_reached_flag = (self._step_count >= self.max_step)
        
        should_end = False
        if crash_flag and self.end_on_crash:
            should_end = True
        if out_of_road_flag and self.end_on_out_of_road:
            should_end = True
        if arrive_dest_flag and self.end_on_arrive_dest:
            should_end = True
        if horizon_reached_flag and self.end_on_horizon:
            should_end = True
        
        done = bool(should_end)
        
        # 添加信息
        info["simulation_time"] = self._simulation_time
        info["cognitive_modules_active"] = self.enable_cognitive_modules and is_ppo_mode
        
        # 添加控制模式信息
        control_status = self.control_manager.get_control_status()
        info.update(control_status)
        info["action_source"] = action_info.get("source", "unknown")
        
        return obs, reward, done, info
    
    def render(self, *args, **kwargs):
        """渲染环境，显示认知模块状态"""
        render_text = kwargs.get("text", {})
        
        # 获取控制状态
        control_status = self.control_manager.get_control_status()
        render_text.update(control_status)
        
        # 添加认知模块状态
        if self.enable_cognitive_modules:
            is_ppo_mode = (
                self.control_manager.expert_mode and 
                not self.disable_ppo_expert and
                hasattr(self.agent, 'expert_takeover') and 
                self.agent.expert_takeover
            )
            
            cognitive_status = "Active (PPO)" if is_ppo_mode else "Inactive"
            render_text["Cognitive Modules"] = cognitive_status
            
            if self.perception_module:
                render_text["Perception"] = f"sigma={self.perception_module.sigma:.2f}"
            if self.delay_module:
                render_text["Delay"] = f"{self.delay_module.delay_steps} steps"
        
        # 其他信息
        render_text.update({
            "Step": f"{self._step_count}/{self.max_step}",
            "Simulation Time": f"{self._simulation_time:.1f}s",
            "Main Car Speed": f"{self.agent.speed:.1f} m/s",
            "Background Vehicles": f"{len(self.ghost_vehicles)}",
        })
        
        kwargs["text"] = render_text
        return super().render(*args, **kwargs)
    
    def _cleanup_ghost_vehicles(self):
        """清理背景车"""
        for vid, vehicle in self.ghost_vehicles.items():
            try:
                vehicle.destroy()
            except:
                pass
        self.ghost_vehicles = {}
    
    def _cleanup_finished_trajectories(self):
        """清理已结束轨迹的背景车"""
        if not self.enable_background_vehicles:
            return
        
        vehicles_to_remove = []
        for vid, vehicle in self.ghost_vehicles.items():
            if vid in self.trajectory_dict:
                traj = self.trajectory_dict[vid]
                if self._is_trajectory_finished(traj, self._simulation_time):
                    vehicles_to_remove.append(vid)
        
        for vid in vehicles_to_remove:
            if vid in self.ghost_vehicles:
                vehicle = self.ghost_vehicles[vid]
                try:
                    vehicle.destroy()
                except Exception as e:
                    pass
                del self.ghost_vehicles[vid]
    
    def _is_trajectory_finished(self, trajectory, sim_time):
        """检查轨迹是否已结束"""
        if not trajectory:
            return True
        
        target_time = self._trajectory_start_time + sim_time
        
        if "timestamp" in trajectory[0]:
            last_timestamp = trajectory[-1]["timestamp"]
            return target_time > last_timestamp
        else:
            max_steps = len(trajectory)
            current_step = int(sim_time / self.physics_world_step_size)
            return current_step >= max_steps
    
    def _get_trajectory_state_at_time(self, trajectory, sim_time):
        """根据仿真时间获取轨迹状态"""
        if not trajectory:
            return None
        
        target_time = self._trajectory_start_time + sim_time
        
        if "timestamp" in trajectory[0]:
            # 基于时间戳查找
            timestamps = [point["timestamp"] for point in trajectory]
            
            if target_time <= timestamps[0]:
                return trajectory[0]
            elif target_time >= timestamps[-1]:
                return trajectory[-1]
            
            # 线性插值
            for i in range(len(timestamps) - 1):
                if timestamps[i] <= target_time <= timestamps[i + 1]:
                    t0, t1 = timestamps[i], timestamps[i + 1]
                    p0, p1 = trajectory[i], trajectory[i + 1]
                    
                    alpha = (target_time - t0) / (t1 - t0) if t1 != t0 else 0.0
                    
                    interpolated_state = {
                        "x": p0["x"] + alpha * (p1["x"] - p0["x"]),
                        "y": p0["y"] + alpha * (p1["y"] - p0["y"]),
                        "speed": p0["speed"] + alpha * (p1["speed"] - p0["speed"]),
                        "heading": p0["heading"] + alpha * (p1["heading"] - p0["heading"]),
                        "timestamp": target_time
                    }
                    
                    return interpolated_state
        else:
            # 基于步数索引
            step_index = int(sim_time / self.physics_world_step_size)
            if 0 <= step_index < len(trajectory):
                return trajectory[step_index]
        
        return None
    
    def _replay_all_vehicles_by_time(self):
        """重放背景车轨迹"""
        if not self.enable_background_vehicles:
            return
        
        for vid, traj in self.trajectory_dict.items():
            state = self._get_trajectory_state_at_time(traj, self._simulation_time)
            
            if state is None:
                if vid in self.ghost_vehicles:
                    vehicle = self.ghost_vehicles[vid]
                    try:
                        vehicle.destroy()
                    except:
                        pass
                    del self.ghost_vehicles[vid]
                continue
            
            if vid not in self.ghost_vehicles:
                # 创建新背景车
                vehicle_config = self.engine.global_config["vehicle_config"].copy()
                vehicle_config.update({
                    "show_navi_mark": False,
                    "show_dest_mark": False,
                    "show_line_to_dest": False,
                    "show_line_to_navi_mark": False,
                    "show_navigation_arrow": False,
                    "use_special_color": False,
                })
                
                v = self.engine.spawn_object(
                    DefaultVehicle, 
                    vehicle_config=vehicle_config, 
                    position=[0, 0], 
                    heading=0
                )
                
                self.ghost_vehicles[vid] = v
            else:
                v = self.ghost_vehicles[vid]
            
            # 更新背景车位置
            v.set_position([state["x"], state["y"]])
            v.set_heading_theta(state["heading"])
            
            speed_magnitude = state["speed"]
            if speed_magnitude > 0.01:
                direction = [np.cos(state["heading"]), np.sin(state["heading"])]
                v.set_velocity(direction, speed_magnitude)
            else:
                v.set_velocity([1.0, 0.0], 0.0)
    def close(self):
        """关闭环境"""
        # 生成认知感知模块的cog_influence可视化（在detach_from_env之前）
        if self.enable_cognitive_modules and self.perception_module and hasattr(self.perception_module, 'generate_visualization'):
            try:
                print(f"\n开始生成认知感知影响分析图表...")
                cog_influence_dir = self.perception_module.generate_visualization(env=self)
                print(f"✅ 认知感知影响图表已保存到 {cog_influence_dir}")
            except Exception as e:
                print(f"❌ 生成认知感知影响图表失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 生成认知偏差模块的可视化（新增）
        if self.enable_cognitive_modules and self.cognitive_bias_module and hasattr(self.cognitive_bias_module, 'generate_visualization'):
            try:
                print(f"\n开始生成认知偏差分析图表...")
                bias_dir = self.cognitive_bias_module.generate_visualization(env=self)
                print(f"✅ 认知偏差分析图表已保存到 {bias_dir}")
                
                # 打印统计信息
                stats = self.cognitive_bias_module.get_statistics()
                print(f"\n认知偏差统计信息:")
                print(f"  总步数: {stats['total_steps']}")
                print(f"  激活率: {stats['activation_rate']:.2%}")
                print(f"  平均偏差: {stats['average_bias']:.4f}")
                print(f"  总偏差: {stats['total_bias']:.2f}")
            except Exception as e:
                print(f"❌ 生成认知偏差分析图表失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 分离噪声雷达，恢复原始传感器（在可视化之后）
        if self.enable_cognitive_modules and self.perception_module:
            self.perception_module.detach_from_env()
        
        # 分离认知偏差模块（新增）
        if self.enable_cognitive_modules and self.cognitive_bias_module:
            self.cognitive_bias_module.detach_from_env()
        
        # 生成可视化图表
        if self.enable_visualization and self.data_recorder and self.visualizer:
            try:
                from datetime import datetime
                session_name = f"cognitive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                print(f"\n开始生成认知分析图表...")
                self.visualizer.generate_all_plots(self.data_recorder, session_name)
                print(f"✅ 认知分析图表已保存到 {self.visualization_output_dir}")
            except Exception as e:
                print(f"❌ 生成可视化图表失败: {e}")
        
        if self.observation_recorder:
            self.observation_recorder.finalize_recording()

        self._cleanup_ghost_vehicles()
        super().close()

    def _debug_navigation_info(self):
        """
        调试导航信息，检查导航系统是否正确配置
        """
        if hasattr(self.agent, 'navigation') and self.agent.navigation:
            nav = self.agent.navigation
            print(f"\n=== Navigation Debug Info ===")
            print(f"Navigation module: {type(nav).__name__}")
            print(f"Current lane: {nav.current_lane.index if nav.current_lane else 'None'}")
            print(f"Destination: {getattr(nav, 'final_lane', 'Not set')}")
            print(f"Route length: {len(getattr(nav, 'route', []))}")
            print(f"Route completion: {nav.route_completion:.3f}")
            
            # 检查是否有有效的导航路径
            if hasattr(nav, 'route') and nav.route and len(nav.route) > 1:
                print(f"✅ Navigation route established: {nav.route[:3]}{'...' if len(nav.route) > 3 else ''}")
            else:
                print(f"❌ Warning: No valid navigation route found!")
                print(f"   This may cause PPO expert to remain stationary")
                
                # 🔧 自动修复导航路径
                print(f"🚀 尝试自动修复导航路径...")
                try:
                    from fix_navigation_route import fix_navigation_for_env
                    success = fix_navigation_for_env(self)
                    if success:
                        print(f"✅ 导航路径修复成功!")
                        # 重新检查路径
                        if hasattr(nav, 'route') and nav.route and len(nav.route) > 1:
                            print(f"✅ 修复后路径: {nav.route[:3]}{'...' if len(nav.route) > 3 else ''}")
                    else:
                        print(f"❌ 导航路径修复失败，PPO可能无法正常工作")
                except ImportError:
                    print(f"⚠️ 导航修复模块未找到，请检查 fix_navigation_route.py 文件")
                except Exception as e:
                    print(f"❌ 导航修复过程中出错: {e}")
                
            # 检查目标距离
            if hasattr(nav, 'distance_to_destination'):
                print(f"Distance to destination: {nav.distance_to_destination:.1f}m")
            
            # 显示自定义目标点信息
            if hasattr(self, 'custom_destination'):
                dest = self.custom_destination
                agent_pos = self.agent.position
                distance_to_custom_dest = np.sqrt((agent_pos[0] - dest[0])**2 + (agent_pos[1] - dest[1])**2)
                print(f"Custom destination: ({dest[0]:.1f}, {dest[1]:.1f})")
                print(f"Distance to custom dest: {distance_to_custom_dest:.1f}m")
            
            print(f"=============================\n")
        else:
            print(f"❌ Error: Agent has no navigation module!")

    def _set_custom_destination(self):
        """
        设置自定义目标点：所有车辆轨迹中x坐标的最大值，y坐标设置为合适的车道位置
        """
        # 计算轨迹数据中x坐标的最大值
        max_x = float('-inf')
        target_y = 0.0  # 默认y坐标
        
        # 根据enable_background_vehicles参数决定是否包含背景车
        if self.enable_background_vehicles and self.trajectory_dict:
            for vehicle_id, trajectory in self.trajectory_dict.items():
                for point in trajectory:
                    if point["x"] > max_x:
                        max_x = point["x"]
                        target_y = point["y"]  # 使用达到最大x时的y坐标作为参考
        
        # 如果有主车轨迹，也包含在计算中
        if self.main_vehicle_trajectory:
            for point in self.main_vehicle_trajectory:
                if point["x"] > max_x:
                    max_x = point["x"]
                    target_y = point["y"]
        
        if max_x == float('-inf'):
            print("Warning: Could not calculate destination from trajectory data")
            # 如果没有轨迹数据，设置一个默认的远处目标
            max_x = 500.0  # 默认500米远的目标
            target_y = 0.0
            print(f"Using default destination: ({max_x:.1f}, {target_y:.1f})")
        
        if not self.enable_background_vehicles:
            print("Note: Destination calculated from main vehicle trajectory only (background vehicles disabled)")
            
        target_position = [max_x, target_y]
        print(f"\n=== Custom Destination Setup ===")
        print(f"Calculated destination: ({max_x:.1f}, {target_y:.1f})")
        
        # 尝试找到最接近目标位置的车道
        try:
            # 使用MetaDrive的车道定位功能找到最近的车道
            if hasattr(self.engine, 'current_map') and self.engine.current_map:
                current_map = self.engine.current_map
                
                # 查找最接近目标位置的车道
                closest_lane = None
                min_distance = float('inf')
                
                for road in current_map.road_network.graph.keys():
                    for lane_index in current_map.road_network.graph[road].keys():
                        lane = current_map.road_network.get_lane((road, lane_index))
                        if lane:
                            # 计算车道中心线上最接近目标点的位置
                            lane_length = lane.length
                            # 检查车道末端位置
                            end_position = lane.position(lane_length, 0)
                            distance = np.sqrt((end_position[0] - max_x)**2 + (end_position[1] - target_y)**2)
                            
                            if distance < min_distance:
                                min_distance = distance
                                closest_lane = lane
                                # 更新目标y坐标为车道中心
                                target_y = end_position[1]
                
                if closest_lane:
                    target_position = [max_x, target_y]
                    print(f"Adjusted destination to nearest lane: ({max_x:.1f}, {target_y:.1f})")
                    print(f"Target lane: {closest_lane.index}")
                    
                    # 设置导航目标 - 修复版本
                    if hasattr(self.agent, 'navigation') and self.agent.navigation:
                        # 🔧 修复：使用PG地图的正确导航设置
                        navigation_success = self._fix_pg_map_navigation()
                        
                        if not navigation_success:
                            # 备用方案：尝试原有逻辑
                            try:
                                # 查找包含目标位置的车道索引
                                target_lane_index = closest_lane.index
                                self.agent.navigation.set_route(
                                    self.agent.navigation.current_lane.index,
                                    target_lane_index
                                )
                                print(f"✅ Navigation route set to lane: {target_lane_index}")
                                navigation_success = True
                            except Exception as e:
                                print(f"⚠️  Could not set navigation route: {e}")
                                # 手动设置目标位置
                                if hasattr(self.agent.navigation, 'destination_point'):
                                    self.agent.navigation.destination_point = target_position
                                    print(f"🔧 Fall back to manual destination: {target_position}")
                        
                        # 如果所有方案都失败，调用外部修复模块
                        if not navigation_success:
                            print(f"🚨 所有导航设置都失败，调用外部修复模块...")
                            try:
                                from fix_navigation_route import fix_navigation_for_env
                                success = fix_navigation_for_env(self)
                                if success:
                                    print(f"✅ 外部修复模块成功修复导航!")
                                else:
                                    print(f"❌ 外部修复模块也无法修复导航")
                            except ImportError:
                                print(f"⚠️ 外部修复模块未找到")
                            except Exception as repair_e:
                                print(f"❌ 外部修复模块出错: {repair_e}")
                else:
                    print(f"Warning: Could not find suitable lane for destination")
                    
        except Exception as e:
            print(f"Error in destination setup: {e}")
            
        # 存储目标位置供调试使用
        self.custom_destination = target_position
        print(f"Final destination: ({target_position[0]:.1f}, {target_position[1]:.1f})")
        print(f"================================\n")

    def _fix_pg_map_navigation(self):
        """
        修复PG地图（程序化生成地图）的导航路径
        专门针对 "S"*8 类型的直线地图
        
        Returns:
            bool: 修复是否成功
        """
        print(f"🔧 尝试修复PG地图导航路径...")
        
        try:
            # 获取当前地图和道路网络
            current_map = self.engine.current_map
            road_network = current_map.road_network
            
            # 获取当前车道
            current_lane = self.agent.navigation.current_lane
            if not current_lane:
                print(f"❌ 无法获取当前车道")
                return False
            
            current_lane_index = current_lane.index
            print(f"📍 当前车道: {current_lane_index}")
            
            # 策略1: 查找地图中的最后一个车道段作为目标
            all_road_segments = list(road_network.graph.keys())
            print(f"🗺️ 地图包含 {len(all_road_segments)} 个道路段: {all_road_segments}")
            
            # 对于 "S"*8 类型的地图，查找最远的车道
            target_lane_index = None
            max_distance = 0
            
            for road_start in all_road_segments:
                for road_end in road_network.graph[road_start].keys():
                    for lane_idx, lane in road_network.graph[road_start][road_end].items():
                        if lane:
                            # 计算车道末端位置
                            lane_end_pos = lane.position(lane.length, 0)
                            # 计算距离当前位置的距离
                            current_pos = self.agent.position
                            distance = np.sqrt((lane_end_pos[0] - current_pos[0])**2 + 
                                             (lane_end_pos[1] - current_pos[1])**2)
                            
                            if distance > max_distance:
                                max_distance = distance
                                target_lane_index = (road_start, road_end, lane_idx)
                                print(f"🎯 找到更远的目标车道: {target_lane_index}, 距离: {distance:.1f}m")
            
            if target_lane_index:
                print(f"🎯 设置导航路径:")
                print(f"  起始车道: {current_lane_index}")
                print(f"  目标车道: {target_lane_index}")
                print(f"  目标距离: {max_distance:.1f}m")
                
                # 尝试设置路径
                self.agent.navigation.set_route(current_lane_index, target_lane_index)
                
                # 验证路径是否成功设置
                if hasattr(self.agent.navigation, 'route') and self.agent.navigation.route:
                    print(f"✅ PG地图导航路径设置成功!")
                    print(f"📍 路径: {self.agent.navigation.route[:3]}{'...' if len(self.agent.navigation.route) > 3 else ''}")
                    return True
                else:
                    print(f"❌ 路径设置后验证失败")
                    return False
            else:
                print(f"❌ 未找到合适的目标车道")
                return False
                
        except Exception as e:
            print(f"❌ PG地图导航修复失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _find_next_lane(self, current_lane, road_network):
        """寻找当前车道的下一个车道"""
        
        try:
            current_index = current_lane.index
            
            # 解析当前车道索引
            if len(current_index) >= 3:
                current_start = current_index[0]
                current_end = current_index[1]
                lane_idx = current_index[2]
                
                # 在道路网络中寻找以current_end为起点的车道
                if current_end in road_network.graph:
                    for next_end in road_network.graph[current_end].keys():
                        lanes = road_network.graph[current_end][next_end]
                        
                        # 处理不同的lanes数据结构
                        if hasattr(lanes, 'items'):
                            lane_items = lanes.items()
                        elif isinstance(lanes, (list, tuple)):
                            lane_items = enumerate(lanes)
                        else:
                            continue
                        
                        for next_lane_idx, next_lane in lane_items:
                            if next_lane and next_lane_idx == lane_idx:  # 保持相同的车道编号
                                return next_lane
                                
                        # 如果没有找到相同编号的车道，返回第一个可用车道
                        for next_lane_idx, next_lane in lane_items:
                            if next_lane:
                                return next_lane
            
            return None
            
        except Exception as e:
            print(f"⚠️ 查找下一个车道失败: {e}")
            return None

    def _fix_lane_detection(self):
        """修复车道检测问题 - 根据主车实际位置重新检测正确的当前车道"""
        
        print("🔧 开始修复车道检测...")
        
        try:
            agent = self.agent
            navigation = agent.navigation
            current_map = self.engine.current_map
            road_network = current_map.road_network
            agent_pos = agent.position
            
            print(f"📍 主车实际位置: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f})")
            print(f"❌ 错误检测车道: {navigation.current_lane.index}")
            
            # 找到主车真正所在的车道
            best_lane = None
            min_distance = float('inf')
            
            for road_start in road_network.graph.keys():
                for road_end in road_network.graph[road_start].keys():
                    lanes = road_network.graph[road_start][road_end]
                    
                    # 处理不同的lanes数据结构
                    if hasattr(lanes, 'items'):
                        lane_items = lanes.items()
                    elif isinstance(lanes, (list, tuple)):
                        lane_items = enumerate(lanes)
                    else:
                        continue
                    
                    for lane_idx, lane in lane_items:
                        if lane:
                            try:
                                # 计算主车在此车道上的位置
                                local_coords = lane.local_coordinates(agent_pos)
                                longitudinal = local_coords[0]
                                lateral = local_coords[1]
                                
                                # 检查主车是否在此车道上
                                is_on_lane = (0 <= longitudinal <= lane.length) and (abs(lateral) < 5)
                                
                                if is_on_lane:
                                    # 计算距离车道中心的距离作为优先级
                                    distance = abs(lateral)
                                    if distance < min_distance:
                                        min_distance = distance
                                        best_lane = lane
                                        
                            except Exception as e:
                                continue
            
            if best_lane:
                print(f"✅ 找到正确车道: {best_lane.index}")
                print(f"🎯 车道位置: ({best_lane.position(0, 0)[0]:.1f}, {best_lane.position(0, 0)[1]:.1f}) → ({best_lane.position(best_lane.length, 0)[0]:.1f}, {best_lane.position(best_lane.length, 0)[1]:.1f})")
                
                # 1. 强制更新当前车道
                navigation._current_lane = best_lane
                
                # 2. 更新参考车道 - 这是检查点计算的基础
                navigation.current_ref_lanes = [best_lane]
                print(f"✅ 更新当前参考车道: {best_lane.index}")
                
                # 3. 寻找下一个车道作为next_ref_lanes
                next_lane = self._find_next_lane(best_lane, road_network)
                if next_lane:
                    navigation.next_ref_lanes = [next_lane]
                    print(f"✅ 更新下一个参考车道: {next_lane.index}")
                else:
                    navigation.next_ref_lanes = [best_lane]  # 如果没有下一个车道，使用当前车道
                    print(f"⚠️ 未找到下一个车道，使用当前车道")
                
                # 4. 重置检查点索引
                if hasattr(navigation, '_target_checkpoints_index'):
                    navigation._target_checkpoints_index = [0, 1]
                    print(f"✅ 重置检查点索引: [0, 1]")
                
                # 5. 更新导航状态
                navigation.update_localization(agent)
                
                print(f"✅ 车道检测和导航路径修复成功!")
                return True
            else:
                print(f"❌ 无法找到主车所在的正确车道")
                return False
                
        except Exception as e:
            print(f"❌ 车道检测修复失败: {e}")
            import traceback
            traceback.print_exc()
            return False


# 测试代码
if __name__ == "__main__":
    # 测试CSV路径
    csv_path = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/scenario_vehicles_9_0_1_2_3_4_5_6_7_8_test3_20250304_150943_row001_filtered.csv"
    
    # 加载轨迹数据
    traj_data = load_trajectory(
        csv_path=csv_path,
        normalize_position=False,
        max_duration=100,
        use_original_position=False,
        translate_to_origin=True,
        target_fps=50.0,
        use_original_timestamps=True
    )
    
    print(f"\n加载了 {len(traj_data)} 辆车的轨迹数据")
    print(f"车辆ID: {list(traj_data.keys())}")
    
    # 认知模块配置
    COGNITIVE_CONFIG = {
        'perception': {
            'enable': True,
            'sigma': 0.3,
            'enable_kalman': True,
            'process_noise': 0.1,
            'dt': 0.02
        },
        'cognitive_bias': {
            'enable': True,  # 启用认知偏差模块
            'inverse_tta_coef': 1.5,  # TTA偏差系数
            'tta_threshold': 1.0,  # TTA阈值（降低以便更容易触发视觉厌恶）
            'adaptive_bias': True,  # 启用自适应偏差
            'adaptation_rate': 0.01,
            'min_adaptive_factor': 0.5,
            'max_adaptive_factor': 2.0,
            'verbose': True,  # 开启详细日志查看视觉厌恶效果
            
            # 视觉厌恶参数
            'visual_detection_distance': 50.0,  # 检测距离50米
            'visual_detection_angle': 30.0,     # 车头方向±30度
            'visual_aversion_strength': 0.8     # 视觉厌恶强度
        },
        'delay': {
            'enable': True,
            'delay_steps': 2,
            'enable_smoothing': True,
            'smoothing_factor': 0.3
        }
    }
    
    # 创建认知环境
    env = TrajectoryReplayEnvCognitive(
        traj_data,
        config={
            'use_render': True,
            'manual_control': True,
            'enable_background_vehicles': True,
            'background_vehicle_update_mode': "position",
            'enable_realtime': True,
            'target_fps': 50.0,
            
            # 启用认知模块
            'enable_cognitive_modules': True,
            'cognitive_config': COGNITIVE_CONFIG
        }
    )
    
    obs = env.reset()
    
    print("\n环境已初始化")
    print("主车位置:", env.agent.position)
    print("控制模式: PPO Expert (默认)")
    print("\n快捷键说明:")
    print("  T: 切换 PPO专家/手动控制 模式")
    print("  E: 开关 PPO专家接管")
    print("  R: 开关 轨迹重放模式")
    print("  W/A/S/D: 手动控制车辆")
    print("\n认知模块仅在PPO专家模式下生效!")
    
    # 主循环
    try:
        for i in range(1000):
            env.render()
            
            action = [0.0, 0.0]
            obs, reward, done, info = env.step(action)
            
            if i % 50 == 0:
                mode = info.get('Control Mode', 'unknown')
                cognitive_active = info.get('cognitive_modules_active', False)
                print(f"Step {i}: 控制模式={mode}, 认知模块={'激活' if cognitive_active else '未激活'}, 速度={env.agent.speed:.2f}")
            
            if done:
                print("环境结束")
                break
                
    except KeyboardInterrupt:
        print("\n用户中断，关闭环境...")
    finally:
        env.close() 