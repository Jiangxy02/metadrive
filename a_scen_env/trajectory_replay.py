
import pandas as pd
import numpy as np
from metadrive.envs import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.engine.engine_utils import get_global_config
from metadrive.constants import HELP_MESSAGE

# 导入新的控制模式管理器和轨迹加载器
from control_mode_manager import ControlModeManager
from trajectory_loader import TrajectoryLoader, load_trajectory

# ===== 新增：导入导航模块 =====
from metadrive.component.navigation_module.node_network_navigation import NodeNetworkNavigation

# ===== 新增：导入观测记录器 =====
from observation_recorder import ObservationRecorder


class TrajectoryReplayEnv(MetaDriveEnv):
    """
    华为场景背景车轨迹重放环境
    
    功能概述：
    - 支持三种主车控制模式：
      1) PPO专家接管（默认）
      2) 键盘手动控制（W/A/S/D）
      3) 轨迹重放（使用CSV中车辆-1的状态逐步回放）
    - 支持从CSV加载多车轨迹：车辆-1作为主车，其余作为"背景车"（ghost vehicles），
      背景车仅用于渲染与参考，不影响主车的动力学与奖励。
    - 支持根据配置开关控制"碰撞/出界/到达终点/步数上限"是否结束仿真（在本类层面实现）。
    - 基于仿真时间的轨迹同步：解决CSV频率与MetaDrive更新频率不匹配的问题
    - 支持背景车启用/禁用控制：可设置enable_background_vehicles参数决定是否显示背景车
    
    主要配置参数：
    - enable_background_vehicles (bool): 是否启用背景车，默认True
    - background_vehicle_update_mode (str): 背景车更新模式，"position"或"dynamics"
    - enable_realtime (bool): 是否启用实时模式，默认True
    - target_fps (float): 目标帧率，默认50.0
    - disable_ppo_expert (bool): 是否禁用PPO专家，默认False
    
    使用说明（渲染窗口中热键）：
    - T/t：在 PPO Expert 与 Manual Control 之间切换
    - E/e：开启/关闭 PPO Expert 接管
    - R/r：开启/关闭 轨迹重放模式（车辆-1）
    - M/m：一键强制进入手动模式（关闭专家接管与轨迹重放）
    - W/A/S/D：加速/转向/刹车（在手动模式下生效）
    """
    def __init__(self, trajectory_dict, config=None):
        # 先处理配置参数，获取enable_background_vehicles设置
        user_config = config.copy() if config else {}
        self.enable_background_vehicles = user_config.get("enable_background_vehicles", False)
        
        # 复制trajectory_dict避免修改原数据
        original_trajectory_dict = trajectory_dict.copy()
        
        # 将车辆-1设置为主车，从背景车辆中移除
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
            self.trajectory_dict = {}  # 清空背景车数据
            print("⚠️  Background vehicles disabled - CSV background vehicle data skipped")
            
        # 计算最大步数（如果没有背景车，使用主车轨迹长度）
        if self.trajectory_dict:
            self.max_step = max(len(traj) for traj in self.trajectory_dict.values())
        elif self.main_vehicle_trajectory:
            self.max_step = len(self.main_vehicle_trajectory)
        else:
            self.max_step = 1000  # 默认步数
            
        self._step_count = 0
        
        # ===== 新增：仿真时间跟踪 =====
        self._simulation_time = 0.0  # 仿真开始以来的总时间（秒）
        self._trajectory_start_time = None  # 轨迹数据的起始时间戳
        
        # ===== 新增：实时时间跟踪 =====
        import time
        self._real_start_time = None  # 实际开始时间
        self._real_time_module = time  # 保存time模块引用
        
        # 设置手动驾驶
        default_config = {
            "use_render": True,          # 是否开启渲染（True 表示显示可视化窗口）
            "map": "S"*8,                # 地图结构：连续 8 个直线段（S 表示 Straight），
                                        # 每段约 40-80 米，总长约 320-640 米，
                                        # 适合长距离驾驶测试，目标距离合理
            "manual_control": True,      # 是否开启手动控制（True 则允许键盘/手柄控制）
            "controller": "keyboard",    # 控制方式："keyboard" 表示使用键盘（WASD）驾驶

            "start_seed": 0,             # 随机种子（0 表示固定地图生成结果，方便复现）

            "map_region_size": 2048,     # 地图渲染区域大小（单位：像素，2048 相对 4096 性能更好）
            "drivable_area_extension": 50,  # 可行驶区域的额外扩展（米），防止车辆出界过早结束

            "horizon": 10000,            # 单个 episode 最大步数（适合长时间驾驶场景）
            
            # ===== 禁用自动交通生成 =====
            "traffic_density": 0.0,      # 设置为0.0完全禁用MetaDrive的自动交通生成
                                        # 确保只有CSV中指定的车辆出现在场景中
            
            # ===== 新增：禁用静态障碍物 =====
            "accident_prob": 0.0,        # 设置为0.0禁用随机事故和静态障碍物
            "static_traffic_object": False,  # 禁用静态交通对象
            
            # 车辆显示配置（启用导航系统，为PPO expert提供明确目标）
            "vehicle_config": {
                "show_navi_mark": True,          # 开启导航目标点标记
                "show_dest_mark": True,          # 开启目的地标记
                "show_line_to_dest": True,       # 开启通往目的地的虚线
                "show_line_to_navi_mark": True,  # 开启通往导航标记的虚线
                "show_navigation_arrow": True,   # 开启导航方向箭头
                
                # ===== 新增：确保导航模块正确配置 =====
                "navigation_module": NodeNetworkNavigation,  # 明确指定导航模块
                "destination": None,             # 让系统自动分配目标
                "spawn_lane_index": None,        # 让系统自动选择起始车道
            }
        }
        
        # 自定义"结束条件开关"不属于MetaDrive底层可识别的配置键，
        # 因此在本类中以实例属性保存，并从用户传入的config中pop后再传给父类。
        # user_config已在前面创建，这里移除已处理的参数
        user_config.pop("enable_background_vehicles", None)  # 移除已处理的参数
        
        # ===== 新增：帧率控制 =====
        self.enable_realtime = user_config.pop("enable_realtime", True)  # 是否启用实时模式
        self.target_fps = user_config.pop("target_fps", 50.0)  # 目标帧率
        self._last_step_time = None  # 上一步的时间
        
        self.end_on_crash = user_config.pop("end_on_crash", False)  # 当车辆发生碰撞时是否立即结束
        self.end_on_out_of_road = user_config.pop("end_on_out_of_road", False)  # 当车辆驶出道路时是否立即结束
        self.end_on_arrive_dest = user_config.pop("end_on_arrive_dest", False)  # 当车辆到达目的地时是否立即结束
        self.end_on_horizon = user_config.pop("end_on_horizon", False)  # 当到达最大时间步（horizon）时是否结束

        # ===== 新增：背景车更新机制控制参数 =====
        self.background_vehicle_update_mode = user_config.pop("background_vehicle_update_mode", "position")
        # 可选值：
        # - "position": 使用CSV位置数据直接更新（原kinematic模式）
        # - "dynamics": 使用CSV中的speed_x, speed_y通过动力学模型更新（物理模式）
        
        print(f"Background vehicle update mode: {self.background_vehicle_update_mode}")
        if self.background_vehicle_update_mode not in ["position", "dynamics"]:
            print(f"Warning: Unknown background_vehicle_update_mode '{self.background_vehicle_update_mode}', defaulting to 'position'")
            self.background_vehicle_update_mode = "position"

        # enable_background_vehicles参数已在初始化开始时处理

        # ===== 新增：PPO专家禁用标志（用于训练） =====
        self.disable_ppo_expert = user_config.pop("disable_ppo_expert", False)
        if self.disable_ppo_expert:
            print("PPO expert disabled for training mode")

        # ===== 新增：观测记录器配置 =====
        self.enable_observation_recording = user_config.pop("enable_observation_recording", False)
        self.observation_recorder = None
        if self.enable_observation_recording:
            session_name = user_config.pop("recording_session_name", None)
            output_dir = user_config.pop("recording_output_dir", "observation_logs")
            self.observation_recorder = ObservationRecorder(output_dir=output_dir, session_name=session_name)
            print("✅ 观测记录器已启用")

        if user_config:
            default_config.update(user_config)
        
        super().__init__(default_config)
        
        # 获取仿真时间步长（必须在super().__init__之后）
        try:
            self.physics_world_step_size = self.engine.physics_world.static_world.getPhysicsWorldStepSize()
            print(f"MetaDrive physics step size: {self.physics_world_step_size:.6f} seconds")
        except AttributeError:
            # 如果无法获取，使用默认值
            self.physics_world_step_size = 0.02  # 50Hz
            print(f"Using default physics step size: {self.physics_world_step_size:.6f} seconds")
        
        # 检查轨迹数据的时间步长并设置同步
        self._setup_time_synchronization()
        
        # decision_repeat信息将在reset()中显示，此时engine尚未完全初始化
        
        # 初始化控制模式管理器
        self.control_manager = ControlModeManager(
            engine=self.engine,
            main_vehicle_trajectory=self.main_vehicle_trajectory
        )
        
        # 背景车缓存
        self.ghost_vehicles = {}  # 存储背景车对象
        
        # 重写 max_step 以允许比轨迹更长的驾驶时间
        self.max_step = 10000  # 允许 10000 步，无论轨迹长度如何

    def reset(self):
        """
        重置环境：
        - 重置计步器与背景车缓存；
        - 初始化控制模式管理器；
        - 若存在车辆-1的轨迹，则将主车初始化到其轨迹的第一个状态（位置/朝向/速度）。
        """
        # 清理之前的背景车
        self._cleanup_ghost_vehicles()
        
        obs = super().reset()
        self._step_count = 0
        self._simulation_time = 0.0  # 重置仿真时间
        self._real_start_time = self._real_time_module.time()  # 记录实际开始时间
        self._last_step_time = self._real_start_time  # 重置步进时间
        self.ghost_vehicles = {}  # 存储背景车对象
        
        # 显示decision_repeat信息（engine此时已完全初始化）
        if not hasattr(self, '_decision_repeat_displayed'):
            decision_repeat = self.engine.global_config.get('decision_repeat', 1)
            effective_time_step = self.physics_world_step_size * decision_repeat
            print(f"MetaDrive decision_repeat: {decision_repeat}")
            print(f"Effective time step per env.step(): {effective_time_step:.6f} seconds ({1/effective_time_step:.1f} Hz)")
            if decision_repeat > 1:
                print(f"⚠️  主车每步实际移动时间是背景车的 {decision_repeat} 倍，已自动修正时间同步")
            self._decision_repeat_displayed = True
        
        # 初始化轨迹起始时间
        self._initialize_trajectory_start_time()
        
        # 在engine可用后设置disable_ppo_expert标志
        if hasattr(self, 'disable_ppo_expert') and self.disable_ppo_expert and self.engine:
            self.engine.global_config["disable_ppo_expert"] = True
        
        # 设置主车实例到控制管理器
        self.control_manager.set_agent(self.agent)
        
        # 重置控制模式并绑定热键
        self.control_manager.reset_modes()
        self.control_manager.bind_hotkeys()
        
        # 初始化控制策略
        self.control_manager.initialize_policies()
        
        # ===== 新增：设置自定义目标点 =====
        self._set_custom_destination()
        
        # ===== 新增：调试导航信息 =====
        self._debug_navigation_info()
        
        # 设置主车初始状态为车辆-1的初始状态
        if self.main_vehicle_trajectory and len(self.main_vehicle_trajectory) > 0:
            initial_state = self.main_vehicle_trajectory[0]
            # 使用原始位置，不进行偏移
            self.agent.set_position([initial_state["x"], initial_state["y"]])
            self.agent.set_heading_theta(initial_state["heading"])
            # 设置初始速度（按照朝向分解为方向向量与标量速度）
            direction = [np.cos(initial_state["heading"]), np.sin(initial_state["heading"])]
            self.agent.set_velocity(direction, initial_state["speed"])
            print(f"Main car initialized at original position: ({initial_state['x']:.1f}, {initial_state['y']:.1f})")
            print(f"Initial heading: {initial_state['heading']:.2f} rad, speed: {initial_state['speed']:.1f} m/s")
            
            # ===== 新增：在主车位置设置后修复车道检测问题 =====
            self._fix_lane_detection()
        
        return obs

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

    def _check_and_fix_checkpoint_issue(self):
        """
        检查和修复引导点问题
        
        问题：如果检查点在主车后方，会导致route_completion为负数，PPO认为需要倒退
        """
        print(f"🔍 检查引导点问题...")
        
        try:
            agent = self.agent
            navigation = agent.navigation
            
            # 检查是否有引导点在后方
            has_backward_checkpoint = False
            
            try:
                checkpoint1, checkpoint2 = navigation.get_checkpoints()
                agent_pos = agent.position[:2]
                
                for i, checkpoint in enumerate([checkpoint1, checkpoint2]):
                    ckpt_pos = checkpoint[:2]
                    
                    # 计算方向向量（从主车到检查点）
                    direction_vec = np.array(ckpt_pos) - np.array(agent_pos)
                    
                    # 计算主车朝向
                    heading = agent.heading_theta
                    heading_vec = np.array([np.cos(heading), np.sin(heading)])
                    
                    # 检查点是否在前方（点积 > 0表示前方）
                    dot_product = np.dot(direction_vec, heading_vec)
                    is_forward = dot_product > 0
                    
                    distance = np.sqrt(direction_vec[0]**2 + direction_vec[1]**2)
                    direction_str = "前方" if is_forward else "后方"
                    
                    print(f"  检查点{i+1}: ({ckpt_pos[0]:.1f}, {ckpt_pos[1]:.1f}), " +
                          f"距离={distance:.1f}m, 方向={direction_str}")
                    
                    if not is_forward:
                        has_backward_checkpoint = True
                        print(f"  ❌ 检查点{i+1}在主车后方!")
                        
            except Exception as e:
                print(f"⚠️ 无法获取检查点信息: {e}")
            
            # 检查路径完成度
            route_completion = getattr(navigation, 'route_completion', 0)
            travelled_length = getattr(navigation, 'travelled_length', 0)
            
            print(f"  路径完成度: {route_completion:.3f}")
            print(f"  已行驶距离: {travelled_length:.2f}m")
            
            # 如果发现问题，进行修复
            if has_backward_checkpoint or route_completion < 0 or travelled_length < 0:
                print(f"🔧 发现引导点问题，开始自动修复...")
                
                # 修复方法1: 重置已行驶距离
                if hasattr(navigation, 'travelled_length'):
                    old_travelled = navigation.travelled_length
                    navigation.travelled_length = 0.0
                    print(f"  重置已行驶距离: {old_travelled:.2f} → 0.0")
                
                # 修复方法2: 重置参考车道位置
                if hasattr(navigation, '_last_long_in_ref_lane') and hasattr(navigation, 'current_ref_lanes'):
                    if navigation.current_ref_lanes:
                        ref_lane = navigation.current_ref_lanes[0]
                        current_long, _ = ref_lane.local_coordinates(agent.position)
                        navigation._last_long_in_ref_lane = current_long
                        print(f"  重置参考车道位置: {navigation._last_long_in_ref_lane:.2f}")
                
                # 修复方法3: 如果仍有问题，强制重新设置导航
                new_completion = getattr(navigation, 'route_completion', -1)
                if new_completion < 0:
                    print(f"  路径完成度仍为负数，尝试重新设置导航...")
                    try:
                        success = self._fix_pg_map_navigation()
                        if success:
                            print(f"  ✅ 导航重新设置成功")
                        else:
                            print(f"  ⚠️ 导航重设失败，使用基础修复")
                            # 强制设置为小的正数
                            if hasattr(navigation, 'total_length') and navigation.total_length > 0:
                                navigation.travelled_length = 0.01 * navigation.total_length
                                print(f"  强制设置路径完成度为 0.01")
                    except Exception as e:
                        print(f"  ❌ 导航重设过程出错: {e}")
                
                # 验证修复效果
                final_completion = getattr(navigation, 'route_completion', -1)
                print(f"  修复后路径完成度: {final_completion:.3f}")
                
                if final_completion >= 0:
                    print(f"✅ 引导点问题修复成功!")
                else:
                    print(f"❌ 引导点问题仍然存在")
            else:
                print(f"✅ 没有检测到引导点问题")
                
        except Exception as e:
            print(f"❌ 引导点检查过程出错: {e}")

    def _debug_ppo_action_info(self, action, action_info):
        """
        调试PPO专家动作信息
        """
        print(f"\n=== PPO Action Debug (Step {self._step_count}) ===")
        print(f"Action source: {action_info.get('source', 'unknown')}")
        print(f"Action success: {action_info.get('success', 'unknown')}")
        print(f"Action values: [{action[0]:.3f}, {action[1]:.3f}] (steering, throttle)")
        
        # 检查主车状态
        print(f"Agent position: ({self.agent.position[0]:.1f}, {self.agent.position[1]:.1f})")
        print(f"Agent speed: {self.agent.speed:.2f} m/s")
        print(f"Agent heading: {self.agent.heading_theta:.3f} rad")
        
        # 检查导航状态
        if hasattr(self.agent, 'navigation') and self.agent.navigation:
            nav = self.agent.navigation
            print(f"Route completion: {nav.route_completion:.3f}")
            if hasattr(nav, 'distance_to_destination'):
                print(f"Distance to dest: {nav.distance_to_destination:.1f}m")
        
        # 显示自定义目标点信息
        if hasattr(self, 'custom_destination'):
            dest = self.custom_destination
            agent_pos = self.agent.position
            distance_to_custom_dest = np.sqrt((agent_pos[0] - dest[0])**2 + (agent_pos[1] - dest[1])**2)
            print(f"Custom destination: ({dest[0]:.1f}, {dest[1]:.1f})")
            print(f"Distance to custom dest: {distance_to_custom_dest:.1f}m")
        
        # 检查专家takeover状态
        if hasattr(self.agent, 'expert_takeover'):
            print(f"Expert takeover: {self.agent.expert_takeover}")
        
        # 输出错误信息（如果有）
        if 'error' in action_info:
            print(f"❌ Error: {action_info['error']}")
        
        print(f"=======================================\n")

    def step(self, action):
        """
        单步仿真：使用控制模式管理器确定动作并推进环境。
        同时在本函数中重放所有"背景车"的位置/朝向/速度（不会影响主车物理）。
        最后根据本类实例属性（end_on_*）决定是否结束仿真。
        """
        # ===== 实时控制：确保仿真以目标帧率运行 =====
        if self.enable_realtime and self._last_step_time is not None:
            current_time = self._real_time_module.time()
            target_step_duration = 1.0 / self.target_fps
            elapsed_since_last_step = current_time - self._last_step_time
            
            if elapsed_since_last_step < target_step_duration:
                sleep_duration = target_step_duration - elapsed_since_last_step
                self._real_time_module.sleep(sleep_duration)
            
            self._last_step_time = self._real_time_module.time()
        
        # 更新仿真时间（考虑decision_repeat）
        decision_repeat = self.engine.global_config.get('decision_repeat', 1)
        effective_time_step = self.physics_world_step_size * decision_repeat
        self._simulation_time += effective_time_step
        
        # 检查是否为轨迹重放模式，如果是，先同步主车状态
        if self.control_manager.use_trajectory_for_main and self.main_vehicle_trajectory:
            main_state = self._get_trajectory_state_at_time(self.main_vehicle_trajectory, self._simulation_time)
            if main_state:
                self.agent.set_position([main_state["x"], main_state["y"]])
                self.agent.set_heading_theta(main_state["heading"])
                
                import numpy as np
                direction = [np.cos(main_state["heading"]), np.sin(main_state["heading"])]
                self.agent.set_velocity(direction, main_state["speed"])
                
                print(f"Step {self._step_count}: Main car trajectory replay - Position: ({main_state['x']:.1f}, {main_state['y']:.1f}), Speed: {main_state['speed']:.1f} m/s")
        
        # 更新控制管理器的步骤计数
        self.control_manager.set_step_count(self._step_count)
        
        # 使用控制管理器获取动作
        action, action_info = self.control_manager.get_action(action)
        
        # ===== 新增：调试PPO专家动作 =====
        if self._step_count % 20 == 0:  # 每20步输出一次调试信息
            self._debug_ppo_action_info(action, action_info)
        
        # 推进主车（在轨迹重放模式下，这主要是为了保持环境状态一致性）
        obs, reward, terminated, truncated, info = super().step(action)
        
        # ===== 新增：记录观测状态 =====
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
        
        # 重放背景车（基于仿真时间而非步数）
        self._replay_all_vehicles_by_time()
        
        self._step_count += 1
        
        # 每10步输出速度对比信息，检查同步效果
        if self._step_count % 10 == 0:
            self._print_speed_comparison()
        
        # 根据实例属性决定是否结束（不依赖父类config中的未知键）
        crash_flag = info.get("crash", False) or info.get("crash_vehicle", False) or info.get("crash_object", False) or info.get("crash_building", False)
        out_of_road_flag = info.get("out_of_road", False)
        arrive_dest_flag = info.get("arrive_dest", False)
        horizon_reached_flag = (self._step_count >= self.max_step)
        
        should_end = False  # 记录当前是否需要结束回合，初始为 False
        if crash_flag and getattr(self, "end_on_crash", True):  
            should_end = True  # 如果发生碰撞且允许"碰撞即结束"，则标记结束
        if out_of_road_flag and getattr(self, "end_on_out_of_road", True):  
            should_end = True  # 如果驶出道路且允许"出界即结束"，则标记结束
        if arrive_dest_flag and getattr(self, "end_on_arrive_dest", True):  
            should_end = True  # 如果到达终点且允许"到达即结束"，则标记结束
        if horizon_reached_flag and getattr(self, "end_on_horizon", True):  
            should_end = True  # 如果到达最大时间步且允许"超时即结束"，则标记结束

        # done由should_end决定，忽略父类的terminated/truncated时也能继续运行
        done = bool(should_end)

        # 附加诊断信息
        info["termination_overridden"] = (terminated or truncated) and (not should_end)
        info["crash_flag"] = crash_flag
        info["out_of_road_flag"] = out_of_road_flag
        info["arrive_dest_flag"] = arrive_dest_flag
        info["horizon_reached_flag"] = horizon_reached_flag
        info["simulation_time"] = self._simulation_time  # 添加仿真时间信息
        
        # 添加控制模式信息
        control_status = self.control_manager.get_control_status()
        info.update(control_status)
        info["action_source"] = action_info.get("source", "unknown")
        
        return obs, reward, done, info

    def _cleanup_ghost_vehicles(self):
        """
        清理所有背景车对象
        """
        for vid, vehicle in self.ghost_vehicles.items():
            try:
                vehicle.destroy()
            except:
                pass  # 忽略销毁失败的情况
        self.ghost_vehicles = {}

    def _initialize_trajectory_start_time(self):
        """
        初始化轨迹数据的起始时间戳，用于时间对齐
        """
        if not self.main_vehicle_trajectory:
            self._trajectory_start_time = 0.0
            return
            
        # 使用主车轨迹的第一个时间戳作为起始时间
        if "timestamp" in self.main_vehicle_trajectory[0]:
            self._trajectory_start_time = self.main_vehicle_trajectory[0]["timestamp"]
        else:
            self._trajectory_start_time = 0.0
            
        print(f"Trajectory start time: {self._trajectory_start_time:.3f} seconds")

    def _get_trajectory_state_at_time(self, trajectory, sim_time):
        """
        根据仿真时间获取轨迹中对应的状态
        
        Args:
            trajectory: 车辆轨迹数据列表
            sim_time: 当前仿真时间（从reset开始的秒数）
            
        Returns:
            Dict: 对应时间的车辆状态，如果时间超出范围则返回None
        """
        if not trajectory:
            return None
            
        # 计算目标时间戳
        target_time = self._trajectory_start_time + sim_time
        
        # ===== 新增：基于CSV原始时间戳的精确匹配 =====
        if "original_timestamp" in trajectory[0] and trajectory[0]["original_timestamp"] != 0:
            return self._find_closest_original_timestamp(trajectory, target_time)
        # 如果轨迹中有timestamp字段，使用时间插值
        elif "timestamp" in trajectory[0]:
            return self._interpolate_trajectory_by_time(trajectory, target_time)
        else:
            # 如果没有timestamp，使用步数索引（兜底方案）
            step_index = int(sim_time / self.physics_world_step_size)
            if 0 <= step_index < len(trajectory):
                return trajectory[step_index]
            else:
                return None

    def _find_closest_original_timestamp(self, trajectory, target_time):
        """
        基于CSV原始时间戳查找最接近的轨迹点，确保精确时间匹配
        
        Args:
            trajectory: 轨迹数据列表，包含original_timestamp字段
            target_time: 目标时间戳
            
        Returns:
            Dict: 最接近的轨迹状态
        """
        if not trajectory:
            return None
            
        # 提取所有原始时间戳
        original_timestamps = [point["original_timestamp"] for point in trajectory]
        
        # 如果目标时间在轨迹范围之外
        if target_time < original_timestamps[0]:
            return trajectory[0]
        elif target_time > original_timestamps[-1]:
            return trajectory[-1]
        
        # 查找最接近的时间戳
        closest_idx = min(range(len(original_timestamps)), 
                         key=lambda i: abs(original_timestamps[i] - target_time))
        
        closest_point = trajectory[closest_idx].copy()
        
        # 计算实际时间匹配误差
        time_error = abs(original_timestamps[closest_idx] - target_time)
        closest_point["current_time_error"] = time_error
        
        return closest_point

    def _interpolate_trajectory_by_time(self, trajectory, target_time):
        """
        基于时间戳进行轨迹插值
        
        Args:
            trajectory: 包含timestamp字段的轨迹数据
            target_time: 目标时间戳
            
        Returns:
            Dict: 插值后的状态数据
        """
        # 查找目标时间周围的两个数据点
        timestamps = [point["timestamp"] for point in trajectory]
        
        # 如果目标时间在轨迹范围之外
        if target_time <= timestamps[0]:
            return trajectory[0]
        elif target_time >= timestamps[-1]:
            return trajectory[-1]
        
        # 找到目标时间的位置
        for i in range(len(timestamps) - 1):
            if timestamps[i] <= target_time <= timestamps[i + 1]:
                # 线性插值
                t0, t1 = timestamps[i], timestamps[i + 1]
                p0, p1 = trajectory[i], trajectory[i + 1]
                
                # 插值权重
                alpha = (target_time - t0) / (t1 - t0) if t1 != t0 else 0.0
                
                # 插值计算
                interpolated_state = {
                    "x": p0["x"] + alpha * (p1["x"] - p0["x"]),
                    "y": p0["y"] + alpha * (p1["y"] - p0["y"]),
                    "speed": p0["speed"] + alpha * (p1["speed"] - p0["speed"]),
                    "heading": p0["heading"] + alpha * (p1["heading"] - p0["heading"]),
                    "timestamp": target_time
                }
                
                # 如果有速度分量，也进行插值
                if "speed_x" in p0 and "speed_x" in p1:
                    interpolated_state["speed_x"] = p0["speed_x"] + alpha * (p1["speed_x"] - p0["speed_x"])
                    interpolated_state["speed_y"] = p0["speed_y"] + alpha * (p1["speed_y"] - p0["speed_y"])
                
                return interpolated_state
        
        # 如果没有找到合适的区间，返回最接近的点
        closest_idx = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - target_time))
        return trajectory[closest_idx]

    def _replay_all_vehicles_by_time(self):
        """
        基于仿真时间重放所有背景车辆：
        - 支持两种更新模式：
          1) position模式：按仿真时间获取每辆车当前状态，创建/更新 `DefaultVehicle` 实例，
             直接设置位置（kinematic模式）
          2) dynamics模式：使用CSV中的speed_x, speed_y通过物理引擎更新车辆，
             更真实地模拟车辆运动
        - 当轨迹结束时，当前实现选择移除车辆。
        - 根据enable_background_vehicles参数决定是否启用背景车
        """
        # 如果禁用背景车，直接返回
        if not self.enable_background_vehicles:
            return
            
        for vid, traj in self.trajectory_dict.items():  # 遍历每辆车的轨迹数据
            # 根据仿真时间获取当前状态
            state = self._get_trajectory_state_at_time(traj, self._simulation_time)
            
            if state is None:  # 如果轨迹已结束或无效
                if vid in self.ghost_vehicles:
                    # 移除车辆
                    vehicle = self.ghost_vehicles[vid]
                    vehicle.destroy()
                    del self.ghost_vehicles[vid]
                continue

            if vid not in self.ghost_vehicles:  # 如果该车还没被创建
                # 获取默认车辆配置
                vehicle_config = self.engine.global_config["vehicle_config"].copy()
                # 确保背景车不与主车冲突，且不显示任何导航标记
                vehicle_config.update({
                    "show_navi_mark": False,          # 不显示导航标记
                    "show_dest_mark": False,          # 不显示目的地标记
                    "show_line_to_dest": False,       # 不显示目的地路线
                    "show_line_to_navi_mark": False,  # 不显示导航路线
                    "show_navigation_arrow": False,   # 不显示导航箭头
                    "use_special_color": False,       # 不使用特殊颜色（避免与主车混淆）
                })
                
                # 根据更新模式配置车辆物理属性
                if self.background_vehicle_update_mode == "position":
                    # position模式：使用kinematic模式，减少物理影响
                    vehicle_config.update({
                        "mass": 1,                        # 极小质量（降低物理影响）
                        "no_wheel_friction": True,        # 禁用车轮摩擦
                    })
                elif self.background_vehicle_update_mode == "dynamics":
                    # dynamics模式：使用正常物理参数，让车辆参与物理模拟
                    vehicle_config.update({
                        "mass": 1100,                     # 正常质量
                        "no_wheel_friction": False,       # 启用车轮摩擦
                    })
                
                # 初始化创建：在位置(0,0)创建一个背景车辆对象
                v = self.engine.spawn_object(DefaultVehicle, vehicle_config=vehicle_config, position=[0, 0], heading=0)

                # 根据更新模式配置物理属性
                if self.background_vehicle_update_mode == "position":
                    # position模式：配置为kinematic模式
                    if hasattr(v, '_body') and hasattr(v._body, 'disable'):
                        try:
                            v._body.disable()  # 禁用物理体，使其不受物理引擎影响
                        except:
                            pass

                    # 尝试设置为Kinematic模式（不产生物理碰撞，但位置可更新）
                    if hasattr(v, '_body') and hasattr(v._body, 'setKinematic'):
                        try:
                            v._body.setKinematic(True)
                        except:
                            pass
                elif self.background_vehicle_update_mode == "dynamics":
                    # dynamics模式：保持正常物理模式（非kinematic）
                    if hasattr(v, '_body') and hasattr(v._body, 'setKinematic'):
                        try:
                            v._body.setKinematic(False)  # 确保不是kinematic模式
                        except:
                            pass

                self.ghost_vehicles[vid] = v  # 保存该车辆实例
            else:
                v = self.ghost_vehicles[vid]  # 已存在则直接取出

            # 根据更新模式选择不同的车辆更新方式
            if self.background_vehicle_update_mode == "position":
                self._update_vehicle_by_position(v, state)
            elif self.background_vehicle_update_mode == "dynamics":
                self._update_vehicle_by_dynamics(v, state, vid)

    def _update_vehicle_by_position(self, vehicle, state):
        """
        位置模式：直接更新车辆位置，使用预计算的稳定heading
        """
        # 车辆定位更新：更新车辆位置和朝向
        vehicle.set_position([state["x"], state["y"]])
        
        # 使用轨迹数据中预计算的稳定heading
        heading = state.get("heading", 0.0)
        vehicle.set_heading_theta(heading)
        
        # 设置速度大小（用于显示和诊断），方向与heading一致
        speed_magnitude = state["speed"]
        if speed_magnitude > 0.01:
            direction = [np.cos(heading), np.sin(heading)]
            vehicle.set_velocity(direction, speed_magnitude)
        else:
            # 静止时保持朝向但速度为0
            direction = [np.cos(heading), np.sin(heading)]
            vehicle.set_velocity(direction, 0.0)

    def _update_vehicle_by_dynamics(self, vehicle, state, vehicle_id):
        """
        动力学模式：使用CSV中的speed_x, speed_y通过物理引擎更新车辆
        使用轨迹数据中预计算的稳定heading，避免基于瞬时速度的跳动
        """
        # 获取CSV中的速度分量
        speed_x = state.get("speed_x", 0.0)
        speed_y = state.get("speed_y", 0.0)
        
        # 使用轨迹数据中预计算的稳定heading
        heading = state.get("heading", 0.0)
        
        # 设置车辆朝向（使用稳定的预计算heading）
        vehicle.set_heading_theta(heading)
        
        # 设置速度向量（使用原始的speed_x, speed_y）
        speed_magnitude = np.sqrt(speed_x**2 + speed_y**2)
        if speed_magnitude > 0.01:
            # 归一化方向向量
            direction = [speed_x / speed_magnitude, speed_y / speed_magnitude]
            vehicle.set_velocity(direction, speed_magnitude)
        else:
            # 速度为零时停止车辆
            vehicle.set_velocity([1.0, 0.0], 0.0)
        
        # 可选：微调位置以确保与CSV数据同步
        # 这有助于防止由于物理模拟误差导致的位置偏移
        current_pos = vehicle.position
        target_pos = [state["x"], state["y"]]
        pos_error = np.sqrt((current_pos[0] - target_pos[0])**2 + (current_pos[1] - target_pos[1])**2)
        
        # 如果位置偏差过大，进行位置校正
        if pos_error > 2.0:  # 偏差超过2米时进行校正
            vehicle.set_position(target_pos)
            print(f"Background vehicle {vehicle_id} position corrected: error={pos_error:.2f}m")


    def _print_speed_comparison(self):
        """
        打印主车和背景车的速度对比信息
        """
        print(f"\n=== Speed Comparison (Step {self._step_count}, Sim Time: {self._simulation_time:.3f}s) ===")
        
        # 主车速度信息
        main_actual_speed = self.agent.speed
        main_expected_speed = "N/A"
        if self.main_vehicle_trajectory:
            main_state = self._get_trajectory_state_at_time(self.main_vehicle_trajectory, self._simulation_time)
            if main_state:
                main_expected_speed = f"{main_state['speed']:.1f}"
            
        print(f"Main Car: Actual={main_actual_speed:.1f} m/s, Expected={main_expected_speed} m/s")
        
        # 背景车速度信息
        if self.enable_background_vehicles and self.ghost_vehicles:
            print(f"Background Vehicles:")
            for vid, traj in self.trajectory_dict.items():
                if vid in self.ghost_vehicles:
                    bg_vehicle = self.ghost_vehicles[vid]
                    actual_speed = bg_vehicle.speed if hasattr(bg_vehicle, 'speed') else 0.0
                    
                    # 根据当前仿真时间获取期望状态
                    expected_state = self._get_trajectory_state_at_time(traj, self._simulation_time)
                    if expected_state:
                        expected_speed = expected_state["speed"]
                        position = bg_vehicle.position if hasattr(bg_vehicle, 'position') else [0, 0]
                        print(f"  Vehicle {vid}: Actual={actual_speed:.1f} m/s, Expected={expected_speed:.1f} m/s, Pos=({position[0]:.1f}, {position[1]:.1f})")
                    else:
                        print(f"  Vehicle {vid}: Trajectory ended")
        elif not self.enable_background_vehicles:
            print("Background Vehicles: Disabled")
        
        print("=" * 50)


    def _setup_time_synchronization(self):
        """
        设置时间同步，分析轨迹数据的时间特性
        """
        if not self.main_vehicle_trajectory:
            print("Warning: No main vehicle trajectory for time synchronization")
            return
            
        # 检查轨迹数据是否包含时间戳信息
        if len(self.main_vehicle_trajectory) > 1:
            first_point = self.main_vehicle_trajectory[0]
            second_point = self.main_vehicle_trajectory[1]
            
            if "timestamp" in first_point and "timestamp" in second_point:
                csv_dt = second_point["timestamp"] - first_point["timestamp"]
                print(f"\nTime Synchronization Analysis:")
                print(f"  CSV interpolated step size: {csv_dt:.6f} seconds ({1/csv_dt:.1f} Hz)")
                print(f"  MetaDrive physics step: {self.physics_world_step_size:.6f} seconds ({1/self.physics_world_step_size:.1f} Hz)")
                
                # 计算时间步长比率
                step_ratio = csv_dt / self.physics_world_step_size
                print(f"  Time step ratio (CSV/Physics): {step_ratio:.2f}")
                
                if abs(step_ratio - 1.0) > 0.1:  # 如果差异超过10%
                    print(f"  ⚠️  Warning: Significant time step mismatch!")
                    print(f"     This may cause speed inconsistencies.")
                    print(f"     Consider adjusting CSV interpolation or physics step size.")
                else:
                    print(f"  ✅ Time steps are well synchronized")
                    
                self.csv_dt = csv_dt
            else:
                print("Warning: Trajectory data missing timestamp information")
                self.csv_dt = 0.05  # 默认50ms
        else:
            print("Warning: Insufficient trajectory data for time analysis")
            self.csv_dt = 0.05


    def render(self, *args, **kwargs):
        """
        渲染环境并在HUD上显示当前控制模式、步骤计数、主车状态与背景车数量等信息。
        """
        render_text = kwargs.get("text", {})
        
        # 从控制管理器获取状态信息
        control_status = self.control_manager.get_control_status()
        
        render_text.update(control_status)
        # 计算实际经过的时间
        real_elapsed_time = self._real_time_module.time() - self._real_start_time if self._real_start_time else 0.0
        time_ratio = self._simulation_time / real_elapsed_time if real_elapsed_time > 0 else 0.0
        
        # 计算到自定义目标点的距离
        distance_to_dest = "N/A"
        if hasattr(self, 'custom_destination'):
            dest = self.custom_destination
            agent_pos = self.agent.position
            distance_to_dest = f"{np.sqrt((agent_pos[0] - dest[0])**2 + (agent_pos[1] - dest[1])**2):.1f}m"
        
        render_text.update({
            "Step": f"{self._step_count}/{self.max_step}",
            "Simulation Time": f"{self._simulation_time:.1f}s",
            "Real Time": f"{real_elapsed_time:.1f}s",
            "Time Ratio": f"{time_ratio:.1f}x",
            "Physics Step": f"{self.physics_world_step_size:.3f}s",
            "Realtime Mode": "ON" if self.enable_realtime else "OFF",
            "Target FPS": f"{self.target_fps:.0f}",
            "Main Car Position": f"({self.agent.position[0]:.1f}, {self.agent.position[1]:.1f})",
            "Main Car Speed": f"{self.agent.speed:.1f} m/s",
            "Distance to Destination": distance_to_dest,
            "Background Vehicles": f"{len(self.ghost_vehicles)}" + ("" if self.enable_background_vehicles else " (Disabled)"),
        })
        kwargs["text"] = render_text
        return super().render(*args, **kwargs)

    def close(self):
        """
        关闭环境并清理资源
        """
        # ===== 新增：结束观测记录 =====
        if self.observation_recorder:
            self.observation_recorder.finalize_recording()
            
        self._cleanup_ghost_vehicles()
        super().close()


if __name__ == "__main__":
    csv_path = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/scenario_vehicles_9_0_1_2_3_4_5_6_7_8_test3_20250304_150943_row001_filtered.csv"
    
    # 使用新的轨迹加载器加载数据
    # 🎯 新增：支持两种时间戳模式
    # - use_original_timestamps=False: 重采样到固定频率 (target_fps)
    # - use_original_timestamps=True: 使用CSV原始时间戳，确保精确匹配
    traj_data = load_trajectory(
        csv_path=csv_path, 
        normalize_position=False,  # 不归一化
        max_duration=100,  # 只加载前100秒
        use_original_position=False,  # 不使用原始位置
        translate_to_origin=True,  # 平移到道路起点
        target_fps=50.0,  # 目标频率（仅在use_original_timestamps=False时使用）
        use_original_timestamps=True  # 🔥 启用原始时间戳精确匹配！
    )
    
    print(f"\nLoaded {len(traj_data)} vehicles from CSV")
    print(f"Vehicle IDs: {list(traj_data.keys())}")
    
    # Create environment, enable manual control and PPO expert
    # 演示背景车控制选项：
    # - enable_background_vehicles=True: 显示所有CSV中的背景车（默认）
    # - enable_background_vehicles=False: 只显示主车，跳过背景车加载
    
    # 示例1：启用背景车（默认行为）
    env = TrajectoryReplayEnv(
        traj_data, 
        config=dict(
            use_render=True, 
            manual_control=True,
            background_vehicle_update_mode="position",  # 可选: "position" 或 "dynamics"
            enable_background_vehicles=False,  # 是否启用背景车（默认True）
            enable_realtime=True,  # 启用实时模式，使仿真以真实时间速度运行
            target_fps=50.0,       # 目标帧率，匹配物理步长 (50Hz = 0.02s per step)
        )
    )
    
    # 示例2：禁用背景车（纯净单车环境）
    # env = TrajectoryReplayEnv(
    #     traj_data, 
    #     config=dict(
    #         use_render=True, 
    #         manual_control=True,
    #         enable_background_vehicles=False,  # 🔥 禁用背景车，只有主车
    #         enable_realtime=True,
    #         target_fps=50.0,
    #     )
    # )
    
    obs = env.reset()
    
    print("\nEnvironment initialized, main car position:", env.agent.position)
    print("Main car heading:", env.agent.heading_theta)
    print("Control mode: PPO Expert (default)")
    print("\nShortcut instructions:")
    print("  T: Toggle PPO Expert/Manual Control mode")
    print("  E: Toggle PPO Expert mode switch")
    print("  R: Toggle trajectory replay mode (Vehicle -1)")
    print("  W/A/S/D: Manual control vehicle")
    
    # Main loop
    try:
        for i in range(1000):
            # Render environment
            env.render()
            
            # Process action based on control mode
            action = [0.0, 0.0]  # 默认动作，由控制管理器内部处理
            
            obs, reward, done, info = env.step(action)
            
            if i % 10 == 0:
                print(f"Step {i}: Control mode={info['Control Mode']}, Position={env.agent.position}, Speed={env.agent.speed:.2f}")
            
            if done:
                print("Environment finished")
                break
                
    except KeyboardInterrupt:
        print("\nUser interrupted, closing environment...")
    finally:
        env.close()
