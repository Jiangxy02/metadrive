#!/usr/bin/env python
"""
官方MetaDrive单智能体环境示例 + 观测状态记录功能

基于原始的drive_in_single_agent_env.py，集成了观测记录器功能，
用于记录和分析车辆在标准MetaDrive环境中的行为表现。

功能特性：
1. 保持原有的键盘控制和自动驾驶功能
2. 集成观测状态记录器，记录每一步的详细数据
3. 支持lidar和rgb_camera两种观测模式
4. 自动生成分析报告和可视化图表

使用方法：
python drive_in_single_agent_env_with_recorder.py --observation lidar
python drive_in_single_agent_env_with_recorder.py --observation rgb_camera
"""
import argparse
import logging
import random
import sys
import os

import cv2
import numpy as np

from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.constants import HELP_MESSAGE

# 导入观测记录器
from observation_recorder import ObservationRecorder


class MetaDriveWithRecorder:
    """
    带观测记录功能的MetaDrive环境包装器
    """
    def __init__(self, config, enable_recording=True, session_name=None, max_steps=1000):
        self.env = MetaDriveEnv(config)
        self.enable_recording = enable_recording
        self.max_steps = max_steps
        self.step_count = 0
        
        # 初始化观测记录器
        if self.enable_recording:
            output_dir = "metadrive_official_logs"
            if session_name is None:
                session_name = "official_metadrive_analysis"
            self.recorder = ObservationRecorder(output_dir=output_dir, session_name=session_name)
            print(f"✅ 观测记录器已启用，输出目录：{output_dir}")
        else:
            self.recorder = None
    
    def reset(self, seed=None):
        """重置环境"""
        self.step_count = 0
        return self.env.reset(seed=seed)
    
    def step(self, action):
        """执行一步并记录观测数据"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 记录观测数据
        if self.recorder and self.step_count < self.max_steps:
            # 构造action_info（官方环境没有明确的控制模式管理器）
            action_info = {
                "source": "expert" if self.env.current_track_agent.expert_takeover else "manual",
                "success": True
            }
            
            # 使用适配器模式记录数据
            self.recorder.record_step(
                env=self._create_env_adapter(),
                action=action,
                action_info=action_info,
                obs=obs,
                reward=reward,
                info=info,
                step_count=self.step_count
            )
        
        self.step_count += 1
        return obs, reward, terminated, truncated, info
    
    def _create_env_adapter(self):
        """
        创建环境适配器，使官方MetaDriveEnv兼容我们的记录器接口
        """
        class EnvAdapter:
            def __init__(self, metadrive_env):
                self.env = metadrive_env
                self.agent = metadrive_env.current_track_agent
                self.custom_destination = None  # 官方环境没有自定义目标点
            
            @property
            def enable_background_vehicles(self):
                return True  # 官方环境默认有背景车辆
        
        return EnvAdapter(self.env)
    
    def render(self, **kwargs):
        """渲染环境"""
        return self.env.render(**kwargs)
    
    def close(self):
        """关闭环境并保存记录"""
        if self.recorder:
            print(f"\n📊 正在保存观测记录... (共{self.step_count}步)")
            self.recorder.finalize_recording()
            print("✅ 观测记录已保存并生成分析报告")
        
        self.env.close()
    
    @property
    def current_track_agent(self):
        """访问当前主车"""
        return self.env.current_track_agent
    
    @property
    def agent(self):
        """访问当前主车（别名）"""
        return self.env.current_track_agent


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="MetaDrive官方示例 + 观测记录功能")
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
    parser.add_argument("--enable_recording", action="store_true", default=True, help="启用观测记录功能")
    parser.add_argument("--session_name", type=str, default="official_metadrive_analysis", help="记录会话名称")
    parser.add_argument("--max_steps", type=int, default=1000, help="最大记录步数")
    args = parser.parse_args()
    
    # 环境配置
    config = dict(
        # controller="steering_wheel",
        use_render=True,
        manual_control=True,
        traffic_density=0.1,
        num_scenarios=10000,
        random_agent_model=False,
        random_lane_width=True,
        random_lane_num=True,
        on_continuous_line_done=False,
        out_of_route_done=True,
        vehicle_config=dict(show_lidar=True, show_navi_mark=True, show_line_to_navi_mark=True),
        # debug=True,
        # debug_static_world=True,
        map=4,  # seven block
        start_seed=10,
    )
    
    # RGB相机观测配置
    if args.observation == "rgb_camera":
        config.update(
            dict(
                image_observation=True,
                sensors=dict(rgb_camera=(RGBCamera, 400, 300)),
                interface_panel=["rgb_camera", "dashboard"]
            )
        )
    
    # 创建带记录功能的环境
    env = MetaDriveWithRecorder(
        config=config,
        enable_recording=args.enable_recording,
        session_name=args.session_name,
        max_steps=args.max_steps
    )
    
    print(f"\n🚗 MetaDrive官方示例 + 观测记录功能")
    print(f"观测模式：{args.observation}")
    print(f"观测记录：{'启用' if args.enable_recording else '禁用'}")
    print(f"最大记录步数：{args.max_steps}")
    print(f"会话名称：{args.session_name}")
    
    try:
        # 重置环境
        o, _ = env.reset(seed=21)
        print(HELP_MESSAGE)
        
        # 启用专家模式
        env.current_track_agent.expert_takeover = True
        
        # 观测信息
        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("观测是字典格式，包含numpy数组：", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            print("观测是numpy数组，形状：", o.shape)
        
        # 主循环
        for i in range(1, args.max_steps + 1):
            # 执行步骤
            o, r, tm, tc, info = env.step([0, 0])
            
            # 渲染
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_agent.expert_takeover else "off",
                    "Current Observation": args.observation,
                    "Keyboard Control": "W,A,S,D",
                    "Recording": "Enabled" if args.enable_recording else "Disabled",
                    "Step": f"{i}/{args.max_steps}",
                    "Position": f"({env.current_track_agent.position[0]:.1f}, {env.current_track_agent.position[1]:.1f})",
                    "Speed": f"{env.current_track_agent.speed:.1f} m/s",
                }
            )
            
            # 打印导航信息（每10步一次）
            if i % 10 == 0:
                print(f"Step {i}: Navigation: {info.get('navigation_command', 'N/A')}, "
                      f"Position: ({env.current_track_agent.position[0]:.1f}, {env.current_track_agent.position[1]:.1f}), "
                      f"Speed: {env.current_track_agent.speed:.2f} m/s")
            
            # RGB相机显示
            if args.observation == "rgb_camera":
                cv2.imshow('RGB Image in Observation', o["image"][..., -1])
                cv2.waitKey(1)
            
            # 检查是否到达目标或超过最大步数
            if (tm or tc) and info.get("arrive_dest", False):
                print(f"🎯 到达目标！重置环境...")
                env.reset(env.env.current_seed + 1)
                env.current_track_agent.expert_takeover = True
            elif i >= args.max_steps:
                print(f"📈 达到最大记录步数 ({args.max_steps})，停止记录")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️  用户中断，正在保存数据...")
    except Exception as e:
        print(f"\n❌ 运行错误：{e}")
    finally:
        env.close()
        print("\n🏁 程序结束")


if __name__ == "__main__":
    main() 