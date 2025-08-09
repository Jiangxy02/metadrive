"""
控制模式管理器模块 (Control Mode Manager)

功能：
- 封装控制模式切换逻辑（PPO专家、手动控制、轨迹重放）
- 管理按键绑定与热键响应
- 提供统一的控制策略初始化接口
- 支持控制模式状态查询与显示

使用方式：
1. 创建 ControlModeManager 实例
2. 在环境 reset 时调用 initialize_policies()
3. 在环境 step 时调用 get_action()
4. 在渲染时调用 get_control_status() 获取显示信息
"""

from metadrive.policy.manual_control_policy import ManualControlPolicy
from metadrive.examples import expert


class ControlModeManager:
    """
    控制模式管理器
    
    管理三种控制模式：
    1. PPO 专家接管模式
    2. 键盘手动控制模式  
    3. 轨迹重放模式
    """
    
    def __init__(self, engine, main_vehicle_trajectory=None):
        """
        初始化控制模式管理器
        
        Args:
            engine: MetaDrive 引擎实例
            main_vehicle_trajectory: 主车轨迹数据（用于轨迹重放模式）
        """
        self.engine = engine
        self.main_vehicle_trajectory = main_vehicle_trajectory
        
        # 控制模式状态
        self.expert_mode = True  # 是否使用 PPO 专家
        self.use_trajectory_for_main = False  # 是否使用轨迹重放
        self.control_mode_changed = False  # 模式是否已切换
        
        # 控制策略
        self.manual_policy = None  # 手动控制策略
        self.agent = None  # 主车实例
        
        # 步骤计数器（用于轨迹重放）
        self.step_count = 0
        
    def set_agent(self, agent):
        """设置主车实例"""
        self.agent = agent
        
    def set_step_count(self, step_count):
        """设置当前步骤计数"""
        self.step_count = step_count
        
    def bind_hotkeys(self):
        """
        绑定控制模式切换热键
        仅在渲染模式下生效
        """
        if not self.engine or not self.engine.global_config.get("use_render", False):
            return
            
        # 小写/大写兼容的按键绑定
        hotkey_mappings = [
            ("t", "T", self.toggle_control_mode),
            ("e", "E", self.toggle_expert_mode), 
            ("r", "R", self.toggle_trajectory_mode),
            ("m", "M", self.force_manual_mode)
        ]
        
        for lower_key, upper_key, callback in hotkey_mappings:
            self.engine.accept(lower_key, callback)
            self.engine.accept(upper_key, callback)
            
        # 显示热键说明
        print("控制模式切换热键:")
        print("  T: 切换 PPO专家/手动控制 模式")
        print("  E: 开关 PPO专家接管")
        print("  R: 开关 轨迹重放模式 (车辆-1)")
        print("  M: 强制进入手动模式 (关闭专家接管与轨迹重放)")
        print("  键盘控制: W(前进), S(后退), A(左转), D(右转)")
        
    def initialize_policies(self):
        """
        初始化控制策略
        需要在 agent 创建后调用
        """
        self._ensure_manual_policy()
        
        # 设置默认专家接管状态
        if hasattr(self.agent, 'expert_takeover'):
            self.agent.expert_takeover = True
            
    def _ensure_manual_policy(self):
        """确保手动控制策略已初始化"""
        if self.manual_policy is None and self.agent is not None:
            seed = 0
            if self.engine and hasattr(self.engine, 'global_config'):
                seed = self.engine.global_config.get("seed", 0)
            
            self.manual_policy = ManualControlPolicy(
                obj=self.agent,
                seed=seed,
                enable_expert=True
            )
            
    def toggle_control_mode(self):
        """
        切换控制模式：PPO专家 <-> 手动控制
        切到手动时会自动禁用轨迹模式并确保手动策略可用
        """
        self.expert_mode = not self.expert_mode
        mode_name = "PPO Expert" if self.expert_mode else "Manual Control"
        print(f"控制模式切换为: {mode_name}")
        self.control_mode_changed = True
        
        # 若切到手动，确保启用手动控制、禁用轨迹模式
        if not self.expert_mode:
            # 强制开启手动控制
            if self.engine and hasattr(self.engine, 'global_config'):
                self.engine.global_config["manual_control"] = True
            
            # 禁用轨迹模式，避免覆盖键盘动作
            if self.use_trajectory_for_main:
                self.use_trajectory_for_main = False
                print("轨迹重放模式已禁用，以允许手动控制")
                
            # 确保手动策略可用
            self._ensure_manual_policy()
            
    def toggle_expert_mode(self):
        """开关 PPO 专家接管"""
        if hasattr(self.agent, 'expert_takeover'):
            self.agent.expert_takeover = not self.agent.expert_takeover
            status = "开启" if self.agent.expert_takeover else "关闭"
            print(f"PPO 专家接管 {status}")
        else:
            print("PPO 专家模式不可用")
            
    def toggle_trajectory_mode(self):
        """开关轨迹重放模式"""
        if self.main_vehicle_trajectory is not None:
            self.use_trajectory_for_main = not self.use_trajectory_for_main
            status = "开启" if self.use_trajectory_for_main else "关闭"
            print(f"主车轨迹重放模式 (车辆-1) {status}")
        else:
            print("无车辆-1轨迹数据")
            
    def force_manual_mode(self):
        """一键强制进入手动模式"""
        self.expert_mode = False
        
        if hasattr(self.agent, 'expert_takeover'):
            self.agent.expert_takeover = False
            
        if self.engine and hasattr(self.engine, 'global_config'):
            self.engine.global_config["manual_control"] = True
        
        if self.use_trajectory_for_main:
            self.use_trajectory_for_main = False
            
        self._ensure_manual_policy()
        print("强制进入手动控制模式 (专家接管已关闭，轨迹重放已关闭)")
        
    def get_action(self, default_action):
        """
        根据当前控制模式获取动作
        
        Args:
            default_action: 默认动作（当其他模式不可用时使用）
            
        Returns:
            tuple: (action, action_info)
            action: 计算得到的动作
            action_info: 动作来源信息字典
        """
        action_info = {"source": "default", "success": True}
        
        # 优先级1: 轨迹重放模式
        if self.use_trajectory_for_main and self.main_vehicle_trajectory:
            if self.step_count < len(self.main_vehicle_trajectory):
                # 轨迹重放模式：状态由主环境step函数统一设置，这里只返回标识
                action_info = {"source": "trajectory", "success": True}
                return [0.0, 0.0], action_info  # 空动作，状态由环境直接设置
                
        # 优先级2: PPO 专家模式
        elif self.expert_mode and hasattr(self.agent, 'expert_takeover') and self.agent.expert_takeover:
            try:
                action = expert(self.agent)
                action_info = {"source": "expert", "success": True}
                return action, action_info
            except (ValueError, AssertionError) as e:
                print(f"PPO 专家控制失败，切换到手动控制: {e}")
                self.expert_mode = False
                action_info = {"source": "expert", "success": False, "error": str(e)}
                return [0.0, 0.0], action_info
                
        # 优先级3: 手动控制模式
        elif not self.expert_mode and self.engine.global_config["manual_control"] and self.manual_policy:
            action = self.manual_policy.act(agent_id=self.agent.id)
            action_info = {"source": "manual", "success": True}
            return action, action_info
            
        # 优先级4: 使用传入的默认动作
        else:
            action_info = {"source": "default", "success": True}
            return default_action, action_info
            
    def get_control_status(self):
        """
        获取当前控制状态信息（用于渲染显示）
        
        Returns:
            dict: 控制状态信息
        """
        # 确定当前控制模式
        if self.use_trajectory_for_main:
            control_mode = "trajectory replay(vehicle-1)"
        elif self.expert_mode:
            control_mode = "PPO expert"
        else:
            control_mode = "manual control"
            
        return {
            "Control Mode": control_mode,
            "PPO Expert Status": "开启" if getattr(self.agent, 'expert_takeover', False) else "关闭",
            "Trajectory Mode": "开启" if self.use_trajectory_for_main else "关闭",
        }
        
    def reset_modes(self):
        """重置控制模式到默认状态"""
        self.expert_mode = True
        self.control_mode_changed = False
        self.use_trajectory_for_main = False
        self.step_count = 0 