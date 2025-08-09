"""
轨迹数据加载和处理模块 (Trajectory Data Loader and Processor)

功能：
- CSV 轨迹数据的读取和解析
- 时间范围过滤（如前100秒数据）
- 坐标变换（平移、归一化等）
- 轨迹数据格式化和验证
- 车辆初始位置分析和可视化

支持的变换模式：
1. translate_to_origin: 平移变换，将车辆-1置于道路起点
2. use_original_position: 保持原始坐标不变
3. normalize_position: 归一化变换（历史功能）

使用方式：
```python
from trajectory_loader import TrajectoryLoader

loader = TrajectoryLoader()
traj_data = loader.load_trajectory(
    csv_path="path/to/file.csv",
    max_duration=100,
    translate_to_origin=True
)
```
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple


class TrajectoryLoader:
    """
    轨迹数据加载器
    
    负责从CSV文件加载车辆轨迹数据，并提供多种坐标变换选项
    """
    
    def __init__(self, verbose: bool = True):
        """
        初始化轨迹加载器
        
        Args:
            verbose: 是否输出详细的处理信息
        """
        self.verbose = verbose
        
    def load_trajectory(self, 
                       csv_path: str, 
                       normalize_position: bool = False, 
                       max_duration: Optional[float] = 100, 
                       use_original_position: bool = False, 
                       translate_to_origin: bool = True,
                       target_fps: float = 20.0) -> Dict[int, List[Dict]]:
        """
        加载CSV轨迹数据，并按需进行时间裁剪与位置平移
        
        Args:
            csv_path: CSV文件路径
            normalize_position: 是否归一化位置（默认False，通常不使用本分支）
            max_duration: 仅加载最前面的若干秒数据（默认100秒）
            use_original_position: 是否保留原始坐标（不平移/缩放）
            translate_to_origin: 是否进行平移，使车辆-1位于可驾驶道路的合理起点
            target_fps: 目标帧率，用于时间插值同步（默认20Hz，即0.05秒步长）
        
        Returns:
            Dict[int, List[Dict]]: 车辆轨迹字典
            格式: {vehicle_id: [{"x": float, "y": float, "speed": float, "heading": float}, ...]}
            
        处理策略：
        - 若 translate_to_origin=True：
          以车辆-1的初始 (x,y) 为参考，整体平移到目标起点（x=200,y=7.0），
          保留各车辆的相对位置与速度（速度不缩放不变更）。
        - 若 use_original_position=True：直接使用原始坐标，不做平移（可能导致出界）。
        - 若 normalize_position=True：使用参考点做平移（历史功能，默认不使用）。
        """
        # 第一步：读取和排序CSV数据
        df = self._load_csv_data(csv_path)
        
        # 第二步：时间过滤
        if max_duration is not None:
            df = self._filter_by_duration(df, max_duration)
            
        # 第三步：位置变换
        if translate_to_origin:
            df = self._apply_translation_transform(df)
        elif use_original_position:
            self._display_original_positions(df)
        elif normalize_position:
            df = self._apply_normalization_transform(df)
            
        # 第四步：转换为轨迹字典格式
        trajectory_dict = self._convert_to_trajectory_dict(df)
        
        # 第五步：输出统计信息
        self._print_trajectory_summary(trajectory_dict)
        
        return trajectory_dict
        
    def _load_csv_data(self, csv_path: str) -> pd.DataFrame:
        """读取和排序CSV数据"""
        if self.verbose:
            print(f"Loading trajectory data from: {csv_path}")
            
        df = pd.read_csv(csv_path)
        df = df.sort_values(["vehicle_id", "timestamp"])
        
        if self.verbose:
            print(f"Loaded {len(df)} data points for {df['vehicle_id'].nunique()} vehicles")
            
        return df
        
    def _filter_by_duration(self, df: pd.DataFrame, max_duration: float) -> pd.DataFrame:
        """按时间长度过滤数据"""
        min_timestamp = df['timestamp'].min()
        max_timestamp = min_timestamp + max_duration
        filtered_df = df[df['timestamp'] <= max_timestamp]
        
        if self.verbose:
            print(f"Filtering data to first {max_duration} seconds")
            print(f"  Timestamp range: [{min_timestamp:.1f}, {filtered_df['timestamp'].max():.1f}]")
            print(f"  Total frames: {len(filtered_df)}")
            
        return filtered_df
        
    def _apply_translation_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用平移变换，将车辆-1置于道路起点"""
        # 获取平移参考点和偏移量
        translate_x, translate_y, ref_info = self._calculate_translation_offset(df)
        
        if self.verbose:
            print(f"\nUsing vehicle -1 initial position as reference: {ref_info}")
            print(f"Translation offset: ({translate_x:.1f}, {translate_y:.1f})")
            
        # 显示原始位置范围
        self._display_position_range(df, "Original position range")
        
        # 应用平移变换
        df_translated = df.copy()
        df_translated['position_x'] = df['position_x'] + translate_x
        df_translated['position_y'] = df['position_y'] + translate_y
        # 速度不需要改变，因为速度是相对的
        
        # 显示变换后位置范围
        self._display_position_range(df_translated, "Translated position range (vehicle -1 at x=200)")
        
        # 显示各车辆初始位置和相对关系
        self._display_vehicle_positions(df_translated)
        
        return df_translated
        
    def _calculate_translation_offset(self, df: pd.DataFrame) -> Tuple[float, float, str]:
        """计算平移偏移量"""
        # 找到车辆-1的初始位置作为参考点
        vehicle_minus1 = df[df['vehicle_id'] == -1]
        if not vehicle_minus1.empty:
            # 使用车辆-1的初始位置
            ref_x = vehicle_minus1.iloc[0]['position_x']
            ref_y = vehicle_minus1.iloc[0]['position_y']
            ref_info = f"({ref_x:.1f}, {ref_y:.1f})"
            
            # 计算平移量，使车辆-1从x=200开始（给后面的车辆留空间），y在道路中心线
            translate_x = 200.0 - ref_x
            translate_y = 7.0 - ref_y  # 道路中心线在y=7.0（根据MetaDrive默认设置）
        else:
            # 如果没有车辆-1，使用最小值作为参考
            min_x = df['position_x'].min()
            min_y = df['position_y'].min()
            ref_info = f"minimum position ({min_x:.1f}, {min_y:.1f})"
            translate_x = 200.0 - min_x
            translate_y = 7.0 - min_y
            
        return translate_x, translate_y, ref_info
        
    def _display_original_positions(self, df: pd.DataFrame):
        """显示原始位置信息（不变换模式）"""
        if not self.verbose:
            return
            
        print(f"\nUsing original positions without transformation")
        self._display_position_range(df, "Position range")
        
        # 显示每个车辆的初始位置
        initial_positions = df.groupby('vehicle_id').first()[['position_x', 'position_y']]
        print(f"\nInitial vehicle positions:")
        for vid in initial_positions.index:
            x, y = initial_positions.loc[vid]
            print(f"  Vehicle {vid}: ({x:.1f}, {y:.1f})")
            
    def _apply_normalization_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用归一化变换（历史功能）"""
        # 找到车辆-1的初始位置作为参考点
        vehicle_minus1 = df[df['vehicle_id'] == -1]
        if not vehicle_minus1.empty:
            # 使用车辆-1的初始位置作为场景起点
            ref_x = vehicle_minus1.iloc[0]['position_x']
            ref_y = vehicle_minus1.iloc[0]['position_y']
            if self.verbose:
                print(f"Using vehicle -1 initial position as reference: ({ref_x:.1f}, {ref_y:.1f})")
        else:
            # 如果没有车辆-1，使用最小值
            ref_x = df['position_x'].min()
            ref_y = df['position_y'].min()
            if self.verbose:
                print(f"Vehicle -1 not found, using minimum position as reference")
        
        # 将所有位置相对于参考点平移
        df_normalized = df.copy()
        df_normalized['position_x'] = df['position_x'] - ref_x + 5.0
        df_normalized['position_y'] = df['position_y'] - ref_y + 10.0
        
        if self.verbose:
            self._display_position_range(df_normalized, "Normalized position range")
            
        return df_normalized
        
    def _display_position_range(self, df: pd.DataFrame, title: str):
        """显示位置范围信息"""
        if not self.verbose:
            return
            
        print(f"\n{title}:")
        print(f"  X: [{df['position_x'].min():.1f}, {df['position_x'].max():.1f}]")
        print(f"  Y: [{df['position_y'].min():.1f}, {df['position_y'].max():.1f}]")
        
    def _display_vehicle_positions(self, df: pd.DataFrame):
        """显示各车辆初始位置和相对关系"""
        if not self.verbose:
            return
            
        initial_positions = df.groupby('vehicle_id').first()[['position_x', 'position_y']]
        print(f"\nInitial vehicle positions after translation:")
        
        # 获取车辆-1的新位置
        if -1 in initial_positions.index:
            v1_x, v1_y = initial_positions.loc[-1]
            print(f"  Vehicle -1 (main): ({v1_x:.1f}, {v1_y:.1f}) [Reference]")
            
            # 显示其他车辆相对于车辆-1的位置
            for vid in sorted(initial_positions.index):
                if vid != -1:
                    x, y = initial_positions.loc[vid]
                    rel_x = x - v1_x
                    rel_y = y - v1_y
                    distance = np.sqrt(rel_x**2 + rel_y**2)
                    print(f"  Vehicle {vid}: ({x:.1f}, {y:.1f}) [Relative: x={rel_x:+.1f}, y={rel_y:+.1f}, dist={distance:.1f}m]")
        else:
            for vid in initial_positions.index:
                x, y = initial_positions.loc[vid]
                print(f"  Vehicle {vid}: ({x:.1f}, {y:.1f})")
                
    def _convert_to_trajectory_dict(self, df: pd.DataFrame) -> Dict[int, List[Dict]]:
        """将DataFrame转换为轨迹字典格式，并进行时间插值同步"""
        grouped = df.groupby("vehicle_id")
        trajectory_dict = {}

        # 分析时间间隔
        timestamps = df['timestamp'].unique()
        timestamps = np.sort(timestamps)
        
        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            avg_interval = np.mean(intervals)
            min_interval = np.min(intervals)
            max_interval = np.max(intervals)
            
            if self.verbose:
                print(f"\nCSV时间戳分析:")
                print(f"  平均间隔: {avg_interval:.6f} 秒")
                print(f"  最小间隔: {min_interval:.6f} 秒") 
                print(f"  最大间隔: {max_interval:.6f} 秒")
                print(f"  总时长: {timestamps[-1] - timestamps[0]:.3f} 秒")
        
        # 使用固定时间步长进行插值（假设MetaDrive默认为20Hz，即0.05秒）
        target_dt = 1.0 / target_fps  # 使用传入的target_fps
        start_time = timestamps[0]
        end_time = timestamps[-1]
        
        # 生成统一的时间网格
        uniform_timestamps = np.arange(start_time, end_time + target_dt, target_dt)
        
        if self.verbose:
            print(f"\n时间插值设置:")
            print(f"  目标步长: {target_dt:.6f} 秒 ({target_fps:.1f} Hz)")
            print(f"  插值后帧数: {len(uniform_timestamps)}")

        for vid, group in grouped:
            group = group.reset_index(drop=True)
            
            # 对每个车辆进行时间插值
            interpolated_traj = self._interpolate_vehicle_trajectory(group, uniform_timestamps)
            trajectory_dict[int(vid)] = interpolated_traj
            
        return trajectory_dict
    
    def _interpolate_vehicle_trajectory(self, vehicle_df: pd.DataFrame, target_timestamps: np.ndarray) -> List[Dict]:
        """
        对单个车辆的轨迹进行时间插值
        
        Args:
            vehicle_df: 单个车辆的数据
            target_timestamps: 目标时间戳数组
            
        Returns:
            List[Dict]: 插值后的轨迹点列表
        """
        original_timestamps = vehicle_df['timestamp'].values
        
        # 使用线性插值
        interp_x = np.interp(target_timestamps, original_timestamps, vehicle_df['position_x'].values)
        interp_y = np.interp(target_timestamps, original_timestamps, vehicle_df['position_y'].values)
        interp_speed_x = np.interp(target_timestamps, original_timestamps, vehicle_df['speed_x'].values)
        interp_speed_y = np.interp(target_timestamps, original_timestamps, vehicle_df['speed_y'].values)
        
        interpolated_traj = []
        for i in range(len(target_timestamps)):
            speed = np.sqrt(interp_speed_x[i]**2 + interp_speed_y[i]**2)
            # 简化朝向处理：始终朝向x正方向（0度）
            heading = 0.0
            
            interpolated_traj.append({
                "x": interp_x[i],
                "y": interp_y[i],
                "speed": speed,
                "heading": heading,
                "timestamp": target_timestamps[i]  # 添加时间戳信息
            })
            
        return interpolated_traj
        
    def _print_trajectory_summary(self, trajectory_dict: Dict[int, List[Dict]]):
        """输出轨迹数据统计摘要"""
        if not self.verbose:
            return
            
        print(f"\nLoaded trajectories for {len(trajectory_dict)} vehicles")
        print(f"Trajectory lengths: {[len(traj) for traj in trajectory_dict.values()]}")
        
    def get_vehicle_info(self, trajectory_dict: Dict[int, List[Dict]]) -> Dict:
        """
        获取轨迹数据的统计信息
        
        Args:
            trajectory_dict: 轨迹数据字典
            
        Returns:
            Dict: 包含车辆数量、轨迹长度等统计信息
        """
        if not trajectory_dict:
            return {"vehicle_count": 0, "trajectory_lengths": []}
            
        return {
            "vehicle_count": len(trajectory_dict),
            "vehicle_ids": list(trajectory_dict.keys()),
            "trajectory_lengths": [len(traj) for traj in trajectory_dict.values()],
            "max_trajectory_length": max(len(traj) for traj in trajectory_dict.values()),
            "min_trajectory_length": min(len(traj) for traj in trajectory_dict.values()),
            "avg_trajectory_length": np.mean([len(traj) for traj in trajectory_dict.values()]),
            "has_main_vehicle": -1 in trajectory_dict.keys()
        }
        
    def validate_trajectory_data(self, trajectory_dict: Dict[int, List[Dict]]) -> Tuple[bool, List[str]]:
        """
        验证轨迹数据的完整性和有效性
        
        Args:
            trajectory_dict: 轨迹数据字典
            
        Returns:
            Tuple[bool, List[str]]: (是否有效, 错误消息列表)
        """
        errors = []
        
        # 检查是否为空
        if not trajectory_dict:
            errors.append("轨迹数据为空")
            return False, errors
            
        # 检查是否有主车（车辆-1）
        if -1 not in trajectory_dict:
            errors.append("缺少主车数据（车辆-1）")
            
        # 检查每个轨迹的完整性
        for vid, traj in trajectory_dict.items():
            if not traj:
                errors.append(f"车辆{vid}的轨迹为空")
                continue
                
            # 检查轨迹点的必要字段
            required_fields = ["x", "y", "speed", "heading"]
            for i, point in enumerate(traj):
                for field in required_fields:
                    if field not in point:
                        errors.append(f"车辆{vid}轨迹点{i}缺少字段'{field}'")
                    elif not isinstance(point[field], (int, float)):
                        errors.append(f"车辆{vid}轨迹点{i}字段'{field}'类型错误")
                        
        return len(errors) == 0, errors


def load_trajectory(csv_path: str, 
                   normalize_position: bool = False, 
                   max_duration: Optional[float] = 100, 
                   use_original_position: bool = False, 
                   translate_to_origin: bool = True,
                   target_fps: float = 20.0) -> Dict[int, List[Dict]]:
    """
    便捷的轨迹加载函数（保持向后兼容性）
    
    这是TrajectoryLoader.load_trajectory的简化接口
    """
    loader = TrajectoryLoader(verbose=True)
    return loader.load_trajectory(
        csv_path=csv_path,
        normalize_position=normalize_position,
        max_duration=max_duration,
        use_original_position=use_original_position,
        translate_to_origin=translate_to_origin,
        target_fps=target_fps
    )


if __name__ == "__main__":
    # 测试轨迹加载器
    print("=" * 60)
    print("轨迹数据加载器测试")
    print("=" * 60)
    
    csv_path = "/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/scenario_vehicles_8_0_1_2_3_4_5_6_7_test3_20250304_162012_row070_filtered.csv"
    
    # 创建加载器实例
    loader = TrajectoryLoader(verbose=True)
    
    # 加载轨迹数据
    traj_data = loader.load_trajectory(
        csv_path=csv_path,
        max_duration=100,
        translate_to_origin=True
    )
    
    # 获取统计信息
    info = loader.get_vehicle_info(traj_data)
    print(f"\n=== 轨迹数据统计 ===")
    print(f"车辆数量: {info['vehicle_count']}")
    print(f"车辆ID: {info['vehicle_ids']}")
    print(f"最大轨迹长度: {info['max_trajectory_length']}")
    print(f"最小轨迹长度: {info['min_trajectory_length']}")
    print(f"平均轨迹长度: {info['avg_trajectory_length']:.1f}")
    print(f"包含主车: {'是' if info['has_main_vehicle'] else '否'}")
    
    # 验证数据
    is_valid, errors = loader.validate_trajectory_data(traj_data)
    print(f"\n=== 数据验证结果 ===")
    print(f"数据有效性: {'✅ 有效' if is_valid else '❌ 无效'}")
    if errors:
        print("错误信息:")
        for error in errors:
            print(f"  - {error}")
    
    print(f"\n✅ 轨迹数据加载器测试完成") 