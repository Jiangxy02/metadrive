
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
        self.end_on_crash = user_config.pop("end_on_crash", False)  # 当车辆发生碰撞时是否立即结束
        self.end_on_out_of_road = user_config.pop("end_on_out_of_road", False)  # 当车辆驶出道路时是否立即结束
        self.end_on_arrive_dest = user_config.pop("end_on_arrive_dest", False)  # 当车辆到达目的地时是否立即结束
        self.end_on_horizon = user_config.pop("end_on_horizon", False)  # 当到达最大时间步（horizon）时是否结束

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
        self.ghost_vehicles = {}  # 存储背景车对象
        
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
        # 检查是否为轨迹重放模式，如果是，先同步主车状态
        if (self.control_manager.use_trajectory_for_main and 
            self.main_vehicle_trajectory and 
            self._step_count < len(self.main_vehicle_trajectory)):
            
            # 在轨迹重放模式下，主车应该与背景车同步更新
            main_state = self.main_vehicle_trajectory[self._step_count]
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
        
        # 推进主车（在轨迹重放模式下，这主要是为了保持环境状态一致性）
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 重放背景车（与主车同步）
        self._replay_all_vehicles()
        
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
            should_end = True  # 如果发生碰撞且允许“碰撞即结束”，则标记结束
        if out_of_road_flag and getattr(self, "end_on_out_of_road", True):  
            should_end = True  # 如果驶出道路且允许“出界即结束”，则标记结束
        if arrive_dest_flag and getattr(self, "end_on_arrive_dest", True):  
            should_end = True  # 如果到达终点且允许“到达即结束”，则标记结束
        if horizon_reached_flag and getattr(self, "end_on_horizon", True):  
            should_end = True  # 如果到达最大时间步且允许“超时即结束”，则标记结束

        # done由should_end决定，忽略父类的terminated/truncated时也能继续运行
        done = bool(should_end)

        # 附加诊断信息
        info["termination_overridden"] = (terminated or truncated) and (not should_end)
        info["crash_flag"] = crash_flag
        info["out_of_road_flag"] = out_of_road_flag
        info["arrive_dest_flag"] = arrive_dest_flag
        info["horizon_reached_flag"] = horizon_reached_flag
        
        # 添加控制模式信息
        control_status = self.control_manager.get_control_status()
        info.update(control_status)
        info["action_source"] = action_info.get("source", "unknown")
        
        return obs, reward, done, info

    def _replay_all_vehicles(self):
        """
        重放所有背景车辆：
        - 按时间步取每辆车当前状态，创建/更新 `DefaultVehicle` 实例；
        - 尽可能降低其物理影响（质量、摩擦、设置为Kinematic等），仅用于可视化与相对关系展示；
        - 当轨迹结束时，当前实现选择"保持最后位置不移除"（可按需改为销毁）。
        """
        for vid, traj in self.trajectory_dict.items():  # 遍历每辆车的轨迹数据
            if self._step_count >= len(traj):  # 如果当前步数已经超过该车轨迹长度
                # 如果轨迹结束，保持车辆在最后位置或移除它
                if vid in self.ghost_vehicles:
                    # 选项1：保持车辆在最后位置（静态）
                    # 车辆将保持在最后已知位置
                    # continue
                    # 选项2：移除车辆（取消注释下面代码以启用）
                    vehicle = self.ghost_vehicles[vid]
                    vehicle.destroy()
                    del self.ghost_vehicles[vid]
                    continue  # 车辆已销毁，跳过后续处理
                else:
                    continue  # 轨迹结束且车辆不存在，跳过该车

            state = traj[self._step_count]  # 取当前时间步的车辆状态
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
                    # 减少碰撞影响
                    "mass": 1,                        # 极小质量（降低物理影响）
                    "no_wheel_friction": True,        # 禁用车轮摩擦
                })
                # 初始化创建：在位置(0,0)创建一个背景车辆对象
                v = self.engine.spawn_object(DefaultVehicle, vehicle_config=vehicle_config, position=[0, 0], heading=0)

                # 配置：尝试禁用物理体（让车辆不参与物理碰撞）
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

                self.ghost_vehicles[vid] = v  # 保存该车辆实例
            else:
                v = self.ghost_vehicles[vid]  # 已存在则直接取出

            # 车辆定位更新：只更新车辆位置，朝向保持x正方向
            v.set_position([state["x"], state["y"]])
            # 朝向始终保持x正方向（0度），避免旋转导致的视觉问题
            v.set_heading_theta(0.0)
            # 设置速度大小（用于显示和诊断），但方向始终向前
            direction = [1.0, 0.0]  # x正方向
            v.set_velocity(direction, state["speed"])  # 设置速度大小，方向固定


    def _print_speed_comparison(self):
        """
        打印主车和背景车的速度对比信息
        """
        print(f"\n=== Speed Comparison (Step {self._step_count}) ===")
        
        # 主车速度信息
        main_actual_speed = self.agent.speed
        main_expected_speed = "N/A"
        if self.main_vehicle_trajectory and self._step_count < len(self.main_vehicle_trajectory):
            main_expected_speed = self.main_vehicle_trajectory[self._step_count]["speed"]
            
        print(f"Main Car: Actual={main_actual_speed:.1f} m/s, Expected={main_expected_speed} m/s")
        
        # 背景车速度信息
        if self.ghost_vehicles:
            print(f"Background Vehicles:")
            for vid, traj in self.trajectory_dict.items():
                if vid in self.ghost_vehicles and self._step_count < len(traj):
                    bg_vehicle = self.ghost_vehicles[vid]
                    actual_speed = bg_vehicle.speed if hasattr(bg_vehicle, 'speed') else 0.0
                    expected_speed = traj[self._step_count]["speed"]
                    position = bg_vehicle.position if hasattr(bg_vehicle, 'position') else [0, 0]
                    print(f"  Vehicle {vid}: Actual={actual_speed:.1f} m/s, Expected={expected_speed:.1f} m/s, Pos=({position[0]:.1f}, {position[1]:.1f})")
        
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
        render_text.update({
            "Step": f"{self._step_count}/{self.max_step}",
            "Main Car Position": f"({self.agent.position[0]:.1f}, {self.agent.position[1]:.1f})",
            "Main Car Speed": f"{self.agent.speed:.1f} m/s",
            "Background Vehicles": len(self.ghost_vehicles),
        })
        kwargs["text"] = render_text
        return super().render(*args, **kwargs)


if __name__ == "__main__":
    csv_path = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    
    # 使用新的轨迹加载器加载数据
    traj_data = load_trajectory(
        csv_path=csv_path, 
        normalize_position=False,  # 不归一化
        max_duration=100,  # 只加载前100秒
        use_original_position=False,  # 不使用原始位置
        translate_to_origin=True,  # 平移到道路起点
        target_fps=50.0  # 使用50Hz匹配MetaDrive物理更新频率
    )
    
    print(f"\nLoaded {len(traj_data)} vehicles from CSV")
    print(f"Vehicle IDs: {list(traj_data.keys())}")
    
    # Create environment, enable manual control and PPO expert
    env = TrajectoryReplayEnv(
        traj_data, 
        config=dict(
            use_render=True, 
            manual_control=True,
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
