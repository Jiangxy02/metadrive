
import pandas as pd
import numpy as np
from metadrive.envs import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.engine.engine_utils import get_global_config
from metadrive.constants import HELP_MESSAGE

# 导入新的控制模式管理器和轨迹加载器
from control_mode_manager import ControlModeManager
from trajectory_loader import TrajectoryLoader, load_trajectory


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
    
    使用说明（渲染窗口中热键）：
    - T/t：在 PPO Expert 与 Manual Control 之间切换
    - E/e：开启/关闭 PPO Expert 接管
    - R/r：开启/关闭 轨迹重放模式（车辆-1）
    - M/m：一键强制进入手动模式（关闭专家接管与轨迹重放）
    - W/A/S/D：加速/转向/刹车（在手动模式下生效）
    """
    def __init__(self, trajectory_dict, config=None):
        # 复制trajectory_dict避免修改原数据
        self.trajectory_dict = trajectory_dict.copy()
        
        # 将车辆-1设置为主车，从背景车辆中移除
        self.main_vehicle_trajectory = None
        if -1 in self.trajectory_dict:
            self.main_vehicle_trajectory = self.trajectory_dict.pop(-1)
            print(f"Vehicle -1 will be used as the main car (agent)")
            print(f"Remaining {len(self.trajectory_dict)} vehicles will be background vehicles")
        else:
            print("Warning: Vehicle -1 not found in trajectory data")
            
        self.max_step = max(len(traj) for traj in trajectory_dict.values())
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
            "map": "S"*20,               # 地图结构：连续 20 个直线段（S 表示 Straight），
                                        # 每段约 40-80 米，总长约 800-1600 米，
                                        # 适合需要 100 秒左右行驶的长直道路场景
            "manual_control": True,      # 是否开启手动控制（True 则允许键盘/手柄控制）
            "controller": "keyboard",    # 控制方式："keyboard" 表示使用键盘（WASD）驾驶

            "start_seed": 0,             # 随机种子（0 表示固定地图生成结果，方便复现）

            "map_region_size": 2048,     # 地图渲染区域大小（单位：像素，2048 相对 4096 性能更好）
            "drivable_area_extension": 50,  # 可行驶区域的额外扩展（米），防止车辆出界过早结束

            "horizon": 10000,            # 单个 episode 最大步数（适合长时间驾驶场景）
            
            # ===== 禁用自动交通生成 =====
            "traffic_density": 0.0,      # 设置为0.0完全禁用MetaDrive的自动交通生成
                                        # 确保只有CSV中指定的车辆出现在场景中
            
            # 车辆显示配置（全部关闭导航标记，使视野更干净）
            "vehicle_config": {
                "show_navi_mark": False,         # 关闭导航目标点标记
                "show_dest_mark": False,         # 关闭目的地标记
                "show_line_to_dest": False,      # 关闭通往目的地的虚线
                "show_line_to_navi_mark": False, # 关闭通往导航标记的虚线
                "show_navigation_arrow": False,  # 关闭导航方向箭头
            }
        }
        
        # 自定义"结束条件开关"不属于MetaDrive底层可识别的配置键，
        # 因此在本类中以实例属性保存，并从用户传入的config中pop后再传给父类。
        user_config = config.copy() if config else {}  # 复制一份用户传入的配置，避免直接修改原字典
        
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
        
        # 设置主车实例到控制管理器
        self.control_manager.set_agent(self.agent)
        
        # 重置控制模式并绑定热键
        self.control_manager.reset_modes()
        self.control_manager.bind_hotkeys()
        
        # 初始化控制策略
        self.control_manager.initialize_policies()
        
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
        
        return obs

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
        # exit()        
        # 推进主车（在轨迹重放模式下，这主要是为了保持环境状态一致性）
        obs, reward, terminated, truncated, info = super().step(action)
        
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
        """
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
        if self.ghost_vehicles:
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
            "Background Vehicles": len(self.ghost_vehicles),
        })
        kwargs["text"] = render_text
        return super().render(*args, **kwargs)


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
    # 演示新的背景车更新机制：
    # - "position": 使用CSV位置直接更新（原kinematic模式）
    # - "dynamics": 使用CSV速度通过物理引擎更新（物理模式）
    env = TrajectoryReplayEnv(
        traj_data, 
        config=dict(
            use_render=True, 
            manual_control=True,
            background_vehicle_update_mode="position",  # 可选: "position" 或 "dynamics"
            enable_realtime=True,  # 启用实时模式，使仿真以真实时间速度运行
            target_fps=50.0,       # 目标帧率，匹配物理步长 (50Hz = 0.02s per step)
        )
    )
    
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
