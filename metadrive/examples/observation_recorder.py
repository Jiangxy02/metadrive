#!/usr/bin/env python3
"""
观测状态记录器 - 用于记录和分析主车的观测数据
功能：
1. 记录每一步的完整观测状态
2. 记录车辆动态信息（位置、速度、方向等）
3. 记录导航信息（路径完成度、目标距离等）
4. 记录PPO专家动作信息
5. 输出多种格式（CSV、JSON、分析报告）

修改版：增强了对MetaDrive官方环境的兼容性，更好的错误处理
"""
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import os


class ObservationRecorder:
    """
    观测状态记录器类
    
    用于记录车辆仿真过程中的所有关键信息，包括：
    - 车辆状态（位置、速度、朝向等）
    - 导航信息（路径进度、目标距离等）
    - 动作信息（转向、油门等）
    - 环境反馈（奖励、碰撞等）
    - 观测向量统计
    """
    
    def __init__(self, output_dir="observation_logs", session_name=None):
        """
        初始化观测记录器
        
        Args:
            output_dir: 输出目录
            session_name: 会话名称，用于文件命名
        """
        self.output_dir = output_dir
        self.session_name = session_name or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 文件路径
        self.csv_path = os.path.join(self.output_dir, f"{self.session_name}_observations.csv")
        self.json_path = os.path.join(self.output_dir, f"{self.session_name}_observations.json")
        self.report_path = os.path.join(self.output_dir, f"{self.session_name}_analysis.txt")
        
        # 数据存储
        self.data = []
        
        print(f"📊 观测记录器初始化完成")
        print(f"   会话名称: {self.session_name}")
        print(f"   输出目录: {self.output_dir}")
        print(f"   CSV文件: {self.csv_path}")
        print(f"   JSON文件: {self.json_path}")
        print(f"   分析报告: {self.report_path}")
    
    def record_step(self, env, action, action_info, obs, reward, info, step_count):
        """
        记录单步观测数据
        
        Args:
            env: 环境实例
            action: 动作
            action_info: 动作信息
            obs: 观测
            reward: 奖励
            info: 环境信息
            step_count: 步数
        """
        try:
            # 基本步骤信息
            step_record = {
                'step': step_count,
                'timestamp': datetime.now().isoformat(),
                'simulation_time': getattr(env, '_simulation_time', 0.0),
            }
            
            # 安全获取智能体
            agent = None
            if hasattr(env, 'agent') and env.agent is not None:
                agent = env.agent
            elif hasattr(env, 'current_track_agent') and env.current_track_agent is not None:
                agent = env.current_track_agent
            elif hasattr(env, 'env') and hasattr(env.env, 'current_track_agent'):
                agent = env.env.current_track_agent
            
            if agent is None:
                print(f"⚠️  无法获取智能体信息，跳过步骤 {step_count}")
                return
                
            # 车辆状态信息
            step_record.update({
                # 位置和运动
                'pos_x': float(agent.position[0]) if hasattr(agent, 'position') and agent.position is not None else 0.0,
                'pos_y': float(agent.position[1]) if hasattr(agent, 'position') and agent.position is not None and len(agent.position) > 1 else 0.0,
                'speed': float(agent.speed) if hasattr(agent, 'speed') else 0.0,
                'heading': float(agent.heading_theta) if hasattr(agent, 'heading_theta') else 0.0,
                'velocity_x': float(agent.velocity[0]) if hasattr(agent, 'velocity') and agent.velocity is not None and len(agent.velocity) > 0 else 0.0,
                'velocity_y': float(agent.velocity[1]) if hasattr(agent, 'velocity') and agent.velocity is not None and len(agent.velocity) > 1 else 0.0,
                
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
                    'nav_current_lane': str(nav.current_lane.index) if nav.current_lane and hasattr(nav.current_lane, 'index') else None,
                    'nav_route_length': len(getattr(nav, 'route', [])),
                    'nav_checkpoints_count': len(getattr(nav, 'checkpoints', [])),
                })
                
                # 当前车道位置信息
                if nav.current_lane and hasattr(agent, 'position') and agent.position is not None:
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
            
            # 自定义目标点信息 (安全处理)
            if hasattr(env, 'custom_destination') and env.custom_destination is not None:
                try:
                    dest = env.custom_destination
                    if hasattr(agent, 'position') and agent.position is not None and len(dest) >= 2:
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
                except Exception as e:
                    step_record.update({
                        'custom_dest_x': None,
                        'custom_dest_y': None,
                        'distance_to_custom_dest': None,
                    })
            else:
                step_record.update({
                    'custom_dest_x': None,
                    'custom_dest_y': None,
                    'distance_to_custom_dest': None,
                })
            
            # 动作信息 (安全处理)
            try:
                if action is not None:
                    step_record.update({
                        'action_steering': float(action[0]) if len(action) > 0 else 0.0,
                        'action_throttle': float(action[1]) if len(action) > 1 else 0.0,
                    })
                else:
                    step_record.update({
                        'action_steering': 0.0,
                        'action_throttle': 0.0,
                    })
                    
                step_record.update({
                    'action_source': action_info.get('source', 'unknown') if action_info else 'unknown',
                    'action_success': action_info.get('success', None) if action_info else None,
                })
            except Exception as e:
                step_record.update({
                    'action_steering': 0.0,
                    'action_throttle': 0.0,
                    'action_source': 'error',
                    'action_success': False,
                })
            
            # 环境反馈信息
            step_record.update({
                'reward': float(reward) if reward is not None else 0.0,
                'control_mode': getattr(env, 'control_mode', 'unknown'),
                'expert_takeover': getattr(agent, 'expert_takeover', None) if agent else None,
            })
            
            # 观测向量统计 (安全处理)
            if obs is not None:
                try:
                    if isinstance(obs, dict):
                        # 处理字典形式的观测（如RGB相机）
                        obs_array = obs.get('vector', obs.get('lidar', None))
                        if obs_array is not None:
                            obs_flat = np.array(obs_array).flatten()
                        else:
                            # 尝试获取第一个numpy数组
                            for key, value in obs.items():
                                if isinstance(value, np.ndarray):
                                    obs_flat = value.flatten()
                                    break
                            else:
                                obs_flat = np.array([])
                    else:
                        # 处理数组形式的观测
                        obs_flat = np.array(obs).flatten()
                    
                    if len(obs_flat) > 0:
                        step_record.update({
                            'obs_shape': str(list(obs_flat.shape)),
                            'obs_mean': float(np.mean(obs_flat)),
                            'obs_std': float(np.std(obs_flat)),
                            'obs_min': float(np.min(obs_flat)),
                            'obs_max': float(np.max(obs_flat)),
                        })
                        
                        # 添加前几个观测值
                        for i in range(min(5, len(obs_flat))):
                            step_record[f'obs_{i}'] = float(obs_flat[i])
                    else:
                        step_record.update({
                            'obs_shape': '[0]',
                            'obs_mean': 0.0,
                            'obs_std': 0.0,
                            'obs_min': 0.0,
                            'obs_max': 0.0,
                        })
                except Exception as e:
                    step_record.update({
                        'obs_shape': 'error',
                        'obs_mean': 0.0,
                        'obs_std': 0.0,
                        'obs_min': 0.0,
                        'obs_max': 0.0,
                    })
            else:
                step_record.update({
                    'obs_shape': 'none',
                    'obs_mean': 0.0,
                    'obs_std': 0.0,
                    'obs_min': 0.0,
                    'obs_max': 0.0,
                })
            
            # 添加到数据列表
            self.data.append(step_record)
            
        except Exception as e:
            print(f"⚠️  记录步骤 {step_count} 时出错: {e}")
            # 记录基本信息，避免完全丢失
            basic_record = {
                'step': step_count,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            self.data.append(basic_record)
    
    def _save_data(self):
        """保存数据到文件"""
        if not self.data:
            print("⚠️  没有记录任何数据")
            return
        
        try:
            # 保存CSV
            df = pd.DataFrame(self.data)
            df.to_csv(self.csv_path, index=False)
            print(f"✅ CSV数据已保存: {self.csv_path}")
            
            # 保存JSON
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False, default=str)
            print(f"✅ JSON数据已保存: {self.json_path}")
            
        except Exception as e:
            print(f"❌ 保存数据时出错: {e}")
    
    def finalize_recording(self):
        """结束记录并生成分析报告"""
        self._save_data()
        self._generate_analysis_report()
    
    def _generate_analysis_report(self):
        """生成分析报告"""
        if not self.data:
            return
            
        try:
            with open(self.report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("观测数据分析报告\n")
                f.write("=" * 80 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"会话名称: {self.session_name}\n")
                f.write(f"记录步数: {len(self.data)}\n\n")
                
                # 数据统计
                df = pd.DataFrame(self.data)
                
                if 'speed' in df.columns:
                    f.write("🚗 车辆运动统计\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  平均速度: {df['speed'].mean():.2f} m/s\n")
                    f.write(f"  最大速度: {df['speed'].max():.2f} m/s\n")
                    f.write(f"  最小速度: {df['speed'].min():.2f} m/s\n")
                    
                    # 停车分析
                    stopped_steps = (df['speed'] < 0.1).sum()
                    f.write(f"  停车时间: {stopped_steps}/{len(df)} 步 ({stopped_steps/len(df)*100:.1f}%)\n\n")
                
                if 'action_source' in df.columns:
                    f.write("🎮 控制模式统计\n")
                    f.write("-" * 40 + "\n")
                    action_counts = df['action_source'].value_counts()
                    for source, count in action_counts.items():
                        f.write(f"  {source}: {count} 步 ({count/len(df)*100:.1f}%)\n")
                    f.write("\n")
                
                if 'reward' in df.columns:
                    f.write("🏆 奖励统计\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"  总奖励: {df['reward'].sum():.2f}\n")
                    f.write(f"  平均奖励: {df['reward'].mean():.3f}\n")
                    f.write(f"  奖励范围: [{df['reward'].min():.3f}, {df['reward'].max():.3f}]\n\n")
                
                # 问题诊断
                f.write("❗ 问题诊断\n")
                f.write("-" * 40 + "\n")
                issues = []
                
                if 'speed' in df.columns and (df['speed'] < 0.1).sum() / len(df) > 0.5:
                    issues.append("车辆长时间停车")
                
                if 'action_throttle' in df.columns and (df['action_throttle'] < 0).sum() / len(df) > 0.3:
                    issues.append("频繁刹车行为")
                
                if 'reward' in df.columns and df['reward'].mean() < 0:
                    issues.append("平均奖励为负")
                
                if issues:
                    for issue in issues:
                        f.write(f"  ⚠️  {issue}\n")
                else:
                    f.write("  ✅ 未检测到明显问题\n")
            
            print(f"✅ 分析报告已生成: {self.report_path}")
            
        except Exception as e:
            print(f"❌ 生成分析报告时出错: {e}")
    
    def get_current_stats(self):
        """获取当前统计信息"""
        if not self.data:
            return {"total_steps": 0}
        
        df = pd.DataFrame(self.data)
        stats = {
            "total_steps": len(self.data),
        }
        
        if 'speed' in df.columns:
            stats.update({
                "current_speed": df['speed'].iloc[-1] if len(df) > 0 else 0.0,
                "avg_speed": df['speed'].mean(),
                "stopped_percentage": (df['speed'] < 0.1).sum() / len(df) * 100,
            })
        
        if 'pos_x' in df.columns and 'pos_y' in df.columns:
            stats.update({
                "current_position": (df['pos_x'].iloc[-1], df['pos_y'].iloc[-1]) if len(df) > 0 else (0.0, 0.0),
            })
        
        return stats 