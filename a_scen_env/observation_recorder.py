#!/usr/bin/env python3
"""
观测状态记录器 - 用于记录和分析主车的观测数据
功能：
1. 记录每一步的完整观测状态
2. 记录车辆动态信息（位置、速度、方向等）
3. 记录导航信息（路径完成度、目标距离等）
4. 记录PPO专家动作信息
5. 输出多种格式（CSV、JSON、分析报告）
"""

import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import os

class ObservationRecorder:
    """
    观测状态记录器
    
    功能概述：
    - 记录每一步的主车观测状态
    - 记录车辆动态信息和导航信息
    - 记录PPO专家的动作和决策信息
    - 支持多种输出格式和分析功能
    """
    
    def __init__(self, output_dir="observation_logs", session_name=None):
        """
        初始化记录器
        
        Args:
            output_dir (str): 输出目录
            session_name (str): 会话名称，如果为None则自动生成时间戳
        """
        self.output_dir = output_dir
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_name = session_name
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据存储
        self.step_data = []  # 存储每一步的数据
        self.summary_stats = {}  # 统计信息
        
        # 输出文件路径
        self.csv_path = os.path.join(output_dir, f"{session_name}_observations.csv")
        self.json_path = os.path.join(output_dir, f"{session_name}_observations.json")
        self.analysis_path = os.path.join(output_dir, f"{session_name}_analysis.txt")
        
        print(f"📊 观测记录器初始化完成")
        print(f"   会话名称: {session_name}")
        print(f"   输出目录: {output_dir}")
        print(f"   CSV文件: {self.csv_path}")
        print(f"   JSON文件: {self.json_path}")
        print(f"   分析报告: {self.analysis_path}")
    
    def record_step(self, env, action, action_info, obs, reward, info, step_count):
        """
        记录单步观测数据
        
        Args:
            env: 环境实例
            action: 执行的动作
            action_info: 动作信息
            obs: 观测状态
            reward: 奖励值
            info: 环境信息
            step_count: 步数
        """
        try:
            # 基础步骤信息
            step_record = {
                'step': step_count,
                'timestamp': datetime.now().isoformat(),
                'simulation_time': getattr(env, '_simulation_time', 0.0),
            }
            
            # 车辆状态信息
            agent = env.agent
            step_record.update({
                # 位置和运动
                'pos_x': float(agent.position[0]),
                'pos_y': float(agent.position[1]),
                'speed': float(agent.speed),
                'heading': float(agent.heading_theta),
                'velocity_x': float(agent.velocity[0]) if hasattr(agent, 'velocity') else 0.0,
                'velocity_y': float(agent.velocity[1]) if hasattr(agent, 'velocity') else 0.0,
                
                # 车道和道路信息
                'on_lane': getattr(agent, 'on_lane', None),
                'out_of_road': getattr(agent, 'out_of_road', None),
                'dist_to_left_side': getattr(agent, 'dist_to_left_side', None),
                'dist_to_right_side': getattr(agent, 'dist_to_right_side', None),
                
                # 碰撞状态
                'crash_vehicle': getattr(agent, 'crash_vehicle', None),
                'crash_object': getattr(agent, 'crash_object', None),
                'crash_sidewalk': getattr(agent, 'crash_sidewalk', None),
            })
            
            # 导航信息
            if hasattr(agent, 'navigation') and agent.navigation:
                nav = agent.navigation
                step_record.update({
                    'nav_route_completion': getattr(nav, 'route_completion', 0.0),
                    'nav_distance_to_dest': getattr(nav, 'distance_to_destination', None),
                    'nav_current_lane': str(nav.current_lane.index) if nav.current_lane else None,
                    'nav_route_length': len(getattr(nav, 'route', [])),
                    'nav_checkpoints_count': len(getattr(nav, 'checkpoints', [])),
                })
                
                # 当前车道位置信息
                if nav.current_lane:
                    try:
                        long_pos, lat_pos = nav.current_lane.local_coordinates(agent.position)
                        step_record.update({
                            'lane_longitudinal_pos': float(long_pos),
                            'lane_lateral_pos': float(lat_pos),
                            'lane_length': float(nav.current_lane.length),
                        })
                    except:
                        step_record.update({
                            'lane_longitudinal_pos': None,
                            'lane_lateral_pos': None,
                            'lane_length': None,
                        })
            else:
                step_record.update({
                    'nav_route_completion': None,
                    'nav_distance_to_dest': None,
                    'nav_current_lane': None,
                    'nav_route_length': 0,
                    'nav_checkpoints_count': 0,
                    'lane_longitudinal_pos': None,
                    'lane_lateral_pos': None,
                    'lane_length': None,
                })
            
            # 自定义目标点信息
            if hasattr(env, 'custom_destination'):
                dest = env.custom_destination
                distance_to_custom = np.sqrt((agent.position[0] - dest[0])**2 + (agent.position[1] - dest[1])**2)
                step_record.update({
                    'custom_dest_x': float(dest[0]),
                    'custom_dest_y': float(dest[1]),
                    'distance_to_custom_dest': float(distance_to_custom),
                })
            else:
                step_record.update({
                    'custom_dest_x': None,
                    'custom_dest_y': None,
                    'distance_to_custom_dest': None,
                })
            
            # 动作信息
            step_record.update({
                'action_steering': float(action[0]) if len(action) > 0 else 0.0,
                'action_throttle': float(action[1]) if len(action) > 1 else 0.0,
                'action_source': action_info.get('source', 'unknown'),
                'action_success': action_info.get('success', None),
            })
            
            # 环境反馈信息
            step_record.update({
                'reward': float(reward),
                'control_mode': info.get('Control Mode', 'unknown'),
                'expert_takeover': getattr(agent, 'expert_takeover', None),
            })
            
            # 观测状态信息
            if obs is not None:
                obs_array = np.array(obs)
                step_record.update({
                    'obs_shape': list(obs_array.shape),
                    'obs_mean': float(np.mean(obs_array)),
                    'obs_std': float(np.std(obs_array)),
                    'obs_min': float(np.min(obs_array)),
                    'obs_max': float(np.max(obs_array)),
                })
                
                # 记录观测向量的前几个关键值（通常是速度、转向等）
                if len(obs_array) >= 10:
                    step_record.update({
                        'obs_0_speed_related': float(obs_array[0]),
                        'obs_1_steering_related': float(obs_array[1]),
                        'obs_2': float(obs_array[2]),
                        'obs_3': float(obs_array[3]),
                        'obs_4': float(obs_array[4]),
                    })
                
                # 保存完整观测向量（可选，数据量大）
                if len(obs_array) <= 100:  # 只有在观测向量不太大时才保存
                    step_record['obs_full'] = obs_array.tolist()
            else:
                step_record.update({
                    'obs_shape': None,
                    'obs_mean': None,
                    'obs_std': None,
                    'obs_min': None,
                    'obs_max': None,
                })
            
            # 添加到记录列表
            self.step_data.append(step_record)
            
            # 每100步保存一次（防止数据丢失）
            if len(self.step_data) % 100 == 0:
                self._save_data()
                
        except Exception as e:
            print(f"⚠️  记录步骤 {step_count} 时出错: {e}")
    
    def _save_data(self):
        """保存数据到文件"""
        if not self.step_data:
            return
            
        try:
            # 保存CSV格式
            df = pd.DataFrame(self.step_data)
            df.to_csv(self.csv_path, index=False)
            
            # 保存JSON格式
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.step_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"⚠️  保存数据时出错: {e}")
    
    def finalize_recording(self):
        """结束记录，生成最终分析报告"""
        if not self.step_data:
            print("⚠️  没有记录任何数据")
            return
            
        print(f"📊 正在生成分析报告...")
        
        # 保存最终数据
        self._save_data()
        
        # 生成分析报告
        self._generate_analysis_report()
        
        print(f"✅ 观测记录完成！")
        print(f"   总步数: {len(self.step_data)}")
        print(f"   CSV文件: {self.csv_path}")
        print(f"   JSON文件: {self.json_path}")
        print(f"   分析报告: {self.analysis_path}")
    
    def _generate_analysis_report(self):
        """生成分析报告"""
        try:
            df = pd.DataFrame(self.step_data)
            
            with open(self.analysis_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("MetaDrive 主车观测状态分析报告\n")
                f.write("="*80 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"会话名称: {self.session_name}\n")
                f.write(f"总步数: {len(self.step_data)}\n\n")
                
                # 基础统计信息
                f.write("📊 基础统计信息\n")
                f.write("-" * 40 + "\n")
                f.write(f"仿真总时长: {df['simulation_time'].max():.2f} 秒\n")
                f.write(f"平均速度: {df['speed'].mean():.2f} m/s\n")
                f.write(f"最大速度: {df['speed'].max():.2f} m/s\n")
                f.write(f"最小速度: {df['speed'].min():.2f} m/s\n")
                f.write(f"速度标准差: {df['speed'].std():.2f} m/s\n\n")
                
                # 位置信息
                f.write("📍 位置轨迹信息\n")
                f.write("-" * 40 + "\n")
                f.write(f"起始位置: ({df['pos_x'].iloc[0]:.1f}, {df['pos_y'].iloc[0]:.1f})\n")
                f.write(f"结束位置: ({df['pos_x'].iloc[-1]:.1f}, {df['pos_y'].iloc[-1]:.1f})\n")
                f.write(f"X轴移动距离: {df['pos_x'].iloc[-1] - df['pos_x'].iloc[0]:.1f} m\n")
                f.write(f"Y轴移动距离: {df['pos_y'].iloc[-1] - df['pos_y'].iloc[0]:.1f} m\n")
                f.write(f"总位移: {np.sqrt((df['pos_x'].iloc[-1] - df['pos_x'].iloc[0])**2 + (df['pos_y'].iloc[-1] - df['pos_y'].iloc[0])**2):.1f} m\n\n")
                
                # 停车分析
                f.write("🚗 停车行为分析\n")
                f.write("-" * 40 + "\n")
                low_speed_steps = df[df['speed'] < 0.5]
                stopped_steps = df[df['speed'] < 0.1]
                f.write(f"低速步数 (<0.5 m/s): {len(low_speed_steps)} ({len(low_speed_steps)/len(df)*100:.1f}%)\n")
                f.write(f"停车步数 (<0.1 m/s): {len(stopped_steps)} ({len(stopped_steps)/len(df)*100:.1f}%)\n")
                
                if len(stopped_steps) > 0:
                    f.write(f"首次停车位置: ({stopped_steps['pos_x'].iloc[0]:.1f}, {stopped_steps['pos_y'].iloc[0]:.1f})\n")
                    f.write(f"最后停车位置: ({stopped_steps['pos_x'].iloc[-1]:.1f}, {stopped_steps['pos_y'].iloc[-1]:.1f})\n")
                
                # 导航分析
                f.write("\n🧭 导航状态分析\n")
                f.write("-" * 40 + "\n")
                nav_completion = df['nav_route_completion'].dropna()
                if len(nav_completion) > 0:
                    f.write(f"路径完成度范围: {nav_completion.min():.3f} - {nav_completion.max():.3f}\n")
                    f.write(f"路径完成度变化: {nav_completion.max() - nav_completion.min():.3f}\n")
                    
                    # 检查路径完成度是否卡住
                    completion_stuck = nav_completion.std() < 0.001
                    f.write(f"路径完成度是否卡住: {'是' if completion_stuck else '否'}\n")
                else:
                    f.write("无导航数据\n")
                
                # 动作分析
                f.write("\n🎮 动作分析\n")
                f.write("-" * 40 + "\n")
                f.write(f"转向动作范围: {df['action_steering'].min():.3f} - {df['action_steering'].max():.3f}\n")
                f.write(f"油门动作范围: {df['action_throttle'].min():.3f} - {df['action_throttle'].max():.3f}\n")
                f.write(f"平均转向: {df['action_steering'].mean():.3f}\n")
                f.write(f"平均油门: {df['action_throttle'].mean():.3f}\n")
                
                # 负油门分析（刹车行为）
                negative_throttle = df[df['action_throttle'] < 0]
                f.write(f"负油门步数 (刹车): {len(negative_throttle)} ({len(negative_throttle)/len(df)*100:.1f}%)\n")
                if len(negative_throttle) > 0:
                    f.write(f"平均刹车强度: {negative_throttle['action_throttle'].mean():.3f}\n")
                
                # 动作源分析
                action_sources = df['action_source'].value_counts()
                f.write(f"\n动作来源统计:\n")
                for source, count in action_sources.items():
                    f.write(f"  {source}: {count} ({count/len(df)*100:.1f}%)\n")
                
                # 奖励分析
                f.write("\n💰 奖励分析\n")
                f.write("-" * 40 + "\n")
                f.write(f"总奖励: {df['reward'].sum():.2f}\n")
                f.write(f"平均奖励: {df['reward'].mean():.4f}\n")
                f.write(f"奖励标准差: {df['reward'].std():.4f}\n")
                f.write(f"最高奖励: {df['reward'].max():.4f}\n")
                f.write(f"最低奖励: {df['reward'].min():.4f}\n")
                
                # 问题检测
                f.write("\n⚠️  问题检测\n")
                f.write("-" * 40 + "\n")
                
                issues = []
                
                # 检测停车问题
                if len(stopped_steps) > len(df) * 0.3:
                    issues.append("车辆停车时间过长 (>30%)")
                
                # 检测导航问题
                if len(nav_completion) > 0 and nav_completion.std() < 0.001:
                    issues.append("导航路径完成度卡住不动")
                
                # 检测负油门问题
                if len(negative_throttle) > len(df) * 0.5:
                    issues.append("过多刹车行为 (>50%)")
                
                # 检测位移问题
                total_displacement = np.sqrt((df['pos_x'].iloc[-1] - df['pos_x'].iloc[0])**2 + 
                                           (df['pos_y'].iloc[-1] - df['pos_y'].iloc[0])**2)
                if total_displacement < 50:  # 总位移小于50米
                    issues.append("总位移过小，可能存在前进问题")
                
                if issues:
                    for i, issue in enumerate(issues, 1):
                        f.write(f"{i}. {issue}\n")
                else:
                    f.write("未检测到明显问题\n")
                
                # 关键时刻分析
                f.write("\n📋 关键时刻分析\n")
                f.write("-" * 40 + "\n")
                
                # 速度下降时刻
                speed_drops = []
                for i in range(1, len(df)):
                    speed_change = df['speed'].iloc[i] - df['speed'].iloc[i-1]
                    if speed_change < -5.0:  # 速度下降超过5 m/s
                        speed_drops.append((i, speed_change, df['pos_x'].iloc[i], df['pos_y'].iloc[i]))
                
                if speed_drops:
                    f.write(f"发现 {len(speed_drops)} 次显著减速事件:\n")
                    for step, change, x, y in speed_drops[:5]:  # 只显示前5个
                        f.write(f"  步骤 {step}: 速度下降 {abs(change):.1f} m/s，位置 ({x:.1f}, {y:.1f})\n")
                else:
                    f.write("未发现显著减速事件\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("分析报告结束\n")
                f.write("="*80 + "\n")
                
        except Exception as e:
            print(f"⚠️  生成分析报告时出错: {e}")
    
    def get_current_stats(self):
        """获取当前统计信息"""
        if not self.step_data:
            return {}
            
        df = pd.DataFrame(self.step_data)
        return {
            'total_steps': len(self.step_data),
            'current_position': (df['pos_x'].iloc[-1], df['pos_y'].iloc[-1]),
            'current_speed': df['speed'].iloc[-1],
            'average_speed': df['speed'].mean(),
            'stopped_percentage': len(df[df['speed'] < 0.1]) / len(df) * 100,
        } 