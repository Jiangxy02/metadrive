"""
认知模块可视化分析模块
生成感知偏差、执行延迟、综合表现的可视化图表
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class CognitiveDataRecorder:
    """认知数据记录器"""
    
    def __init__(self):
        self.reset_data()
    
    def reset_data(self):
        """重置所有数据"""
        # 时间序列
        self.timestamps = []
        self.step_counts = []
        
        # 感知相关数据
        self.true_positions_x = []
        self.true_positions_y = []
        self.noisy_positions_x = []
        self.noisy_positions_y = []
        self.filtered_positions_x = []
        self.filtered_positions_y = []
        self.observation_errors = []
        self.effective_sigmas = []
        self.ego_target_distances = []
        
        # 执行延迟相关数据
        self.commanded_steers = []
        self.executed_steers = []
        self.commanded_throttles = []
        self.executed_throttles = []
        self.action_differences = []
        self.vehicle_speeds = []
        
        # 综合表现数据
        self.rewards = []
        self.crash_flags = []
        self.out_of_road_flags = []
        self.arrive_dest_flags = []
        self.distances_to_destination = []
        
        # 认知模块状态
        self.cognitive_active_flags = []
        self.ppo_mode_flags = []
        
        print("认知数据记录器已重置")
    
    def record_step(self, step_count, timestamp, env, obs, action, action_info, 
                   reward, info, cognitive_active=False):
        """记录一步的数据"""
        # 基础信息
        self.step_counts.append(step_count)
        self.timestamps.append(timestamp)
        self.cognitive_active_flags.append(cognitive_active)
        
        # 检查是否为PPO模式
        is_ppo_mode = (
            hasattr(env, 'control_manager') and
            env.control_manager.expert_mode and 
            not getattr(env, 'disable_ppo_expert', False) and
            hasattr(env.agent, 'expert_takeover') and 
            env.agent.expert_takeover and
            not env.control_manager.use_trajectory_for_main
        )
        self.ppo_mode_flags.append(is_ppo_mode)
        
        # 主车真实位置
        true_pos = env.agent.position
        self.true_positions_x.append(true_pos[0])
        self.true_positions_y.append(true_pos[1])
        
        # 感知数据（仅在认知模块激活时记录详细数据）
        if cognitive_active and hasattr(env, 'perception_module') and env.perception_module:
            # 这里需要从认知模块获取处理过程中的数据
            self.noisy_positions_x.append(getattr(env.perception_module, '_last_noisy_x', true_pos[0]))
            self.noisy_positions_y.append(getattr(env.perception_module, '_last_noisy_y', true_pos[1]))
            self.filtered_positions_x.append(getattr(env.perception_module, '_last_filtered_x', true_pos[0]))
            self.filtered_positions_y.append(getattr(env.perception_module, '_last_filtered_y', true_pos[1]))
            
            # 观测误差
            noisy_x = getattr(env.perception_module, '_last_noisy_x', true_pos[0])
            noisy_y = getattr(env.perception_module, '_last_noisy_y', true_pos[1])
            error = np.sqrt((noisy_x - true_pos[0])**2 + (noisy_y - true_pos[1])**2)
            self.observation_errors.append(error)
            
            # 有效噪声强度
            self.effective_sigmas.append(getattr(env.perception_module, '_last_effective_sigma', 0.0))
            
            # 主车-目标车距离
            if hasattr(env, 'custom_destination'):
                dest = env.custom_destination
                distance = np.sqrt((true_pos[0] - dest[0])**2 + (true_pos[1] - dest[1])**2)
                self.ego_target_distances.append(distance)
            else:
                self.ego_target_distances.append(0.0)
        else:
            # 非认知模式或模块未激活时，记录真实值
            self.noisy_positions_x.append(true_pos[0])
            self.noisy_positions_y.append(true_pos[1])
            self.filtered_positions_x.append(true_pos[0])
            self.filtered_positions_y.append(true_pos[1])
            self.observation_errors.append(0.0)
            self.effective_sigmas.append(0.0)
            self.ego_target_distances.append(0.0)
        
        # 执行延迟数据
        if cognitive_active and hasattr(env, 'delay_module') and env.delay_module:
            # 从延迟模块获取命令vs执行的动作
            commanded_action = getattr(env.delay_module, '_last_commanded_action', action)
            executed_action = action
            
            self.commanded_steers.append(commanded_action[0] if len(commanded_action) > 0 else 0.0)
            self.commanded_throttles.append(commanded_action[1] if len(commanded_action) > 1 else 0.0)
            self.executed_steers.append(executed_action[0] if len(executed_action) > 0 else 0.0)
            self.executed_throttles.append(executed_action[1] if len(executed_action) > 1 else 0.0)
            
            # 动作差异
            if len(commanded_action) >= 2 and len(executed_action) >= 2:
                diff = np.sqrt((commanded_action[0] - executed_action[0])**2 + 
                              (commanded_action[1] - executed_action[1])**2)
                self.action_differences.append(diff)
            else:
                self.action_differences.append(0.0)
        else:
            # 非认知模式时，命令=执行
            steer = action[0] if len(action) > 0 else 0.0
            throttle = action[1] if len(action) > 1 else 0.0
            
            self.commanded_steers.append(steer)
            self.commanded_throttles.append(throttle)
            self.executed_steers.append(steer)
            self.executed_throttles.append(throttle)
            self.action_differences.append(0.0)
        
        # 车辆速度
        self.vehicle_speeds.append(env.agent.speed)
        
        # 综合表现数据
        self.rewards.append(reward)
        self.crash_flags.append(info.get("crash", False) or info.get("crash_vehicle", False))
        self.out_of_road_flags.append(info.get("out_of_road", False))
        self.arrive_dest_flags.append(info.get("arrive_dest", False))
        
        # 到目标距离
        if hasattr(env.agent, 'navigation') and hasattr(env.agent.navigation, 'distance_to_destination'):
            self.distances_to_destination.append(env.agent.navigation.distance_to_destination)
        elif hasattr(env, 'custom_destination'):
            dest = env.custom_destination
            distance = np.sqrt((true_pos[0] - dest[0])**2 + (true_pos[1] - dest[1])**2)
            self.distances_to_destination.append(distance)
        else:
            self.distances_to_destination.append(0.0)


class CognitiveVisualizer:
    """认知模块可视化器"""
    
    def __init__(self, output_dir="fig_cog"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        print(f"认知可视化器初始化，输出目录: {output_dir}")
    
    def generate_all_plots(self, recorder: CognitiveDataRecorder, session_name="cognitive_analysis"):
        """生成所有可视化图表"""
        if len(recorder.timestamps) == 0:
            print("⚠️ 没有数据可用于生成图表")
            return
        
        print(f"开始生成认知分析图表，共 {len(recorder.timestamps)} 个数据点...")
        
        # 1. 感知偏差相关图表
        self._plot_position_curves(recorder, session_name)
        self._plot_observation_error(recorder, session_name)
        self._plot_effective_sigma(recorder, session_name)
        
        # 2. 执行延迟相关图表
        self._plot_action_comparison(recorder, session_name)
        self._plot_action_difference(recorder, session_name)
        self._plot_speed_response(recorder, session_name)
        
        # 3. 综合表现图表
        self._plot_reward_curve(recorder, session_name)
        self._plot_distance_to_destination(recorder, session_name)
        
        print(f"✅ 所有图表已生成完成，保存在 {self.output_dir} 目录")
    
    def _plot_position_curves(self, recorder, session_name):
        """绘制位置观测曲线：真值 vs 加噪后 vs 卡尔曼滤波后"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        times = np.array(recorder.step_counts)
        
        # X坐标曲线
        ax1.plot(times, recorder.true_positions_x, 'b-', label='True', linewidth=2)
        ax1.plot(times, recorder.noisy_positions_x, 'r--', label='Noisy', alpha=0.7)
        ax1.plot(times, recorder.filtered_positions_x, 'g-', label='Kalman Filtered', linewidth=1.5)
        
        # 标记认知模块激活区间
        cognitive_active = np.array(recorder.cognitive_active_flags)
        if np.any(cognitive_active):
            active_regions = self._get_active_regions(cognitive_active)
            for start, end in active_regions:
                ax1.axvspan(times[start], times[end], alpha=0.2, color='yellow', 
                           label='Cognitive module active' if start == active_regions[0][0] else "")
        
        ax1.set_xlabel('Simulation step')
        ax1.set_ylabel('X coordinate (m)')
        ax1.set_title(f'X coordinate position observation curve - {session_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Y坐标曲线
        ax2.plot(times, recorder.true_positions_y, 'b-', label=' True', linewidth=2)
        ax2.plot(times, recorder.noisy_positions_y, 'r--', label='Noisy', alpha=0.7)
        ax2.plot(times, recorder.filtered_positions_y, 'g-', label='Kalman Filtered', linewidth=1.5)
        
        # 标记认知模块激活区间
        if np.any(cognitive_active):
            for start, end in active_regions:
                ax2.axvspan(times[start], times[end], alpha=0.2, color='yellow', 
                           label='Cognitive module active' if start == active_regions[0][0] else "")
        
        ax2.set_xlabel('Simulation step')
        ax2.set_ylabel('Y coordinate (m)')
        ax2.set_title(f'Y coordinate position observation curve - {session_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'position_curves_{session_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 位置观测曲线已生成")
    
    def _plot_observation_error(self, recorder, session_name):
        """绘制观测误差随时间变化"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        times = np.array(recorder.step_counts)
        errors = np.array(recorder.observation_errors)
        
        ax.plot(times, errors, 'r-', linewidth=1.5, label='Observation error |Noisy observation - True|')
        
        # 计算并显示统计信息
        if len(errors) > 0:
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            ax.axhline(mean_error, color='orange', linestyle='--', alpha=0.7, label=f'Average error: {mean_error:.3f}m')
            ax.text(0.02, 0.95, f'Maximum error: {max_error:.3f}m', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 标记认知模块激活区间
        cognitive_active = np.array(recorder.cognitive_active_flags)
        if np.any(cognitive_active):
            active_regions = self._get_active_regions(cognitive_active)
            for start, end in active_regions:
                ax.axvspan(times[start], times[end], alpha=0.2, color='yellow', 
                          label='Cognitive module active' if start == active_regions[0][0] else "")
        
        ax.set_xlabel('Simulation step')
        ax.set_ylabel('Observation error (m)')
        ax.set_title(f'Observation error over time - {session_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'observation_error_{session_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 观测误差曲线已生成")
    
    def _plot_effective_sigma(self, recorder, session_name):
        """绘制有效噪声强度与距离的对照曲线"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        times = np.array(recorder.step_counts)
        sigmas = np.array(recorder.effective_sigmas)
        distances = np.array(recorder.ego_target_distances)
        
        # 有效噪声强度曲线
        ax1.plot(times, sigmas, 'purple', linewidth=1.5, label='Effective noise strength σ_eff(t)')
        ax1.set_xlabel('Simulation step')
        ax1.set_ylabel('σ_eff')
        ax1.set_title(f'Effective noise strength over time - {session_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 主车-目标距离曲线
        ax2.plot(times, distances, 'brown', linewidth=1.5, label='Ego-target distance d(t)')
        ax2.set_xlabel('Simulation step')
        ax2.set_ylabel('Distance (m)')
        ax2.set_title(f'Ego-target distance over time - {session_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 标记认知模块激活区间
        cognitive_active = np.array(recorder.cognitive_active_flags)
        if np.any(cognitive_active):
            active_regions = self._get_active_regions(cognitive_active)
            for start, end in active_regions:
                ax1.axvspan(times[start], times[end], alpha=0.2, color='yellow', 
                           label='Cognitive module active' if start == active_regions[0][0] else "")
                ax2.axvspan(times[start], times[end], alpha=0.2, color='yellow', 
                           label='Cognitive module active' if start == active_regions[0][0] else "")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'effective_sigma_{session_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 有效噪声强度曲线已生成")
    
    def _plot_action_comparison(self, recorder, session_name):
        """绘制动作前后对比：命令 vs 执行"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        times = np.array(recorder.step_counts)
        
        # 转向对比
        ax1.plot(times, recorder.commanded_steers, 'b-', label='Steering command', linewidth=1.5)
        ax1.plot(times, recorder.executed_steers, 'r--', label='Actual execution', linewidth=1.5)
        ax1.set_xlabel('Simulation step')
        ax1.set_ylabel('Steering value')
        ax1.set_title(f'Steering command vs actual execution - {session_name}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 油门对比
        ax2.plot(times, recorder.commanded_throttles, 'b-', label='Throttle command', linewidth=1.5)
        ax2.plot(times, recorder.executed_throttles, 'r--', label='Actual execution', linewidth=1.5)
        ax2.set_xlabel('Simulation step')
        ax2.set_ylabel('Throttle value')
        ax2.set_title(f'Throttle command vs actual execution - {session_name}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 标记认知模块激活区间
        cognitive_active = np.array(recorder.cognitive_active_flags)
        if np.any(cognitive_active):
            active_regions = self._get_active_regions(cognitive_active)
            for start, end in active_regions:
                ax1.axvspan(times[start], times[end], alpha=0.2, color='yellow', 
                           label='Cognitive module active' if start == active_regions[0][0] else "")
                ax2.axvspan(times[start], times[end], alpha=0.2, color='yellow', 
                           label='Cognitive module active' if start == active_regions[0][0] else "")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'action_comparison_{session_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 动作对比曲线已生成")
    
    def _plot_action_difference(self, recorder, session_name):
        """绘制动作差异曲线"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        times = np.array(recorder.step_counts)
        diffs = np.array(recorder.action_differences)
        
        ax.plot(times, diffs, 'red', linewidth=1.5, label='Action difference |Actual execution - Command|')
        
        # 计算并显示统计信息
        if len(diffs) > 0:
            mean_diff = np.mean(diffs)
            max_diff = np.max(diffs)
            ax.axhline(mean_diff, color='orange', linestyle='--', alpha=0.7, label=f'Average difference: {mean_diff:.3f}')
            ax.text(0.02, 0.95, f'Maximum difference: {max_diff:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # 标记认知模块激活区间
        cognitive_active = np.array(recorder.cognitive_active_flags)
        if np.any(cognitive_active):
            active_regions = self._get_active_regions(cognitive_active)
            for start, end in active_regions:
                ax.axvspan(times[start], times[end], alpha=0.2, color='yellow', 
                          label='Cognitive module active' if start == active_regions[0][0] else "")
        
        ax.set_xlabel('Simulation step')
        ax.set_ylabel('Action difference')
        ax.set_title(f'Action difference over time - {session_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'action_difference_{session_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 动作差异曲线已生成")
    
    def _plot_speed_response(self, recorder, session_name):
        """绘制延迟对车辆响应的影响：速度变化"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        times = np.array(recorder.step_counts)
        speeds = np.array(recorder.vehicle_speeds)
        
        ax.plot(times, speeds, 'green', linewidth=1.5, label='Vehicle speed')
        
        # 标记认知模块激活区间
        cognitive_active = np.array(recorder.cognitive_active_flags)
        if np.any(cognitive_active):
            active_regions = self._get_active_regions(cognitive_active)
            for start, end in active_regions:
                ax.axvspan(times[start], times[end], alpha=0.2, color='yellow', 
                          label='Cognitive module active' if start == active_regions[0][0] else "")
        
        # 添加速度统计信息
        if len(speeds) > 0:
            mean_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            ax.axhline(mean_speed, color='orange', linestyle='--', alpha=0.7, label=f'Average speed: {mean_speed:.1f} m/s')
            ax.text(0.02, 0.95, f'Maximum speed: {max_speed:.1f} m/s', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax.set_xlabel('Simulation step')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title(f'Vehicle speed over time - {session_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'speed_response_{session_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 速度响应曲线已生成")
    
    def _plot_reward_curve(self, recorder, session_name):
        """绘制奖励曲线并标注事件"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        times = np.array(recorder.step_counts)
        rewards = np.array(recorder.rewards)
        
        ax.plot(times, rewards, 'blue', linewidth=1.5, label='奖励值')
        
        # 标注事件
        crash_flags = np.array(recorder.crash_flags)
        out_of_road_flags = np.array(recorder.out_of_road_flags)
        arrive_dest_flags = np.array(recorder.arrive_dest_flags)
        
        # 标记碰撞事件
        crash_indices = np.where(crash_flags)[0]
        if len(crash_indices) > 0:
            ax.scatter(times[crash_indices], rewards[crash_indices], 
                      color='red', s=100, marker='x', label='Crash', zorder=5)
        
        # 标记出界事件
        out_indices = np.where(out_of_road_flags)[0]
        if len(out_indices) > 0:
            ax.scatter(times[out_indices], rewards[out_indices], 
                      color='orange', s=100, marker='s', label='Out of Road', zorder=5)
        
        # 标记到达目标事件
        arrive_indices = np.where(arrive_dest_flags)[0]
        if len(arrive_indices) > 0:
            ax.scatter(times[arrive_indices], rewards[arrive_indices], 
                      color='green', s=100, marker='o', label='Arrive Destination', zorder=5)
        
        # 标记认知模块激活区间
        cognitive_active = np.array(recorder.cognitive_active_flags)
        if np.any(cognitive_active):
            active_regions = self._get_active_regions(cognitive_active)
            for start, end in active_regions:
                ax.axvspan(times[start], times[end], alpha=0.2, color='yellow', 
                          label='Cognitive module active' if start == active_regions[0][0] else "")
        
        # 计算累积奖励
        if len(rewards) > 0:
            cumulative_reward = np.sum(rewards)
            ax.text(0.02, 0.95, f'Cumulative reward: {cumulative_reward:.2f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax.set_xlabel('Simulation step')
        ax.set_ylabel('Reward value')
        ax.set_title(f'Reward curve and event annotation - {session_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'reward_curve_{session_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 奖励曲线已生成")
    
    def _plot_distance_to_destination(self, recorder, session_name):
        """绘制到目标距离曲线"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        times = np.array(recorder.step_counts)
        distances = np.array(recorder.distances_to_destination)
        
        ax.plot(times, distances, 'teal', linewidth=1.5, label='Distance to destination')
        
        # 标记认知模块激活区间
        cognitive_active = np.array(recorder.cognitive_active_flags)
        if np.any(cognitive_active):
            active_regions = self._get_active_regions(cognitive_active)
            for start, end in active_regions:
                ax.axvspan(times[start], times[end], alpha=0.2, color='yellow', 
                            label='Cognitive module active' if start == active_regions[0][0] else "")
        
        # 添加距离统计信息
        if len(distances) > 0 and np.max(distances) > 0:
            start_distance = distances[0] if len(distances) > 0 else 0
            end_distance = distances[-1] if len(distances) > 0 else 0
            min_distance = np.min(distances[distances > 0]) if np.any(distances > 0) else 0
            
            ax.text(0.02, 0.95, f'Start distance: {start_distance:.1f}m', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            ax.text(0.02, 0.85, f'End distance: {end_distance:.1f}m', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            if min_distance > 0:
                ax.text(0.02, 0.75, f'Minimum distance: {min_distance:.1f}m', transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
        
        ax.set_xlabel('Simulation step')
        ax.set_ylabel('Distance (m)')
        ax.set_title(f'Distance to destination over time - {session_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'distance_to_destination_{session_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 到目标距离曲线已生成")
    
    def _get_active_regions(self, active_flags):
        """获取激活区间"""
        regions = []
        start = None
        
        for i, active in enumerate(active_flags):
            if active and start is None:
                start = i
            elif not active and start is not None:
                regions.append((start, i-1))
                start = None
        
        # 处理最后一个区间
        if start is not None:
            regions.append((start, len(active_flags)-1))
        
        return regions 