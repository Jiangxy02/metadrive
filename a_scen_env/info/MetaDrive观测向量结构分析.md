# MetaDrive 观测向量结构分析

## 📋 概述

基于实际运行结果和源码分析，MetaDrive环境的观测向量是一个259维的numpy数组，包含了主车状态、传感器数据和导航信息等多种类型的信息。

## 🔍 观测向量结构详解

### 观测数据基本信息
- **数据类型**: `numpy.ndarray`
- **形状**: `(259,)`
- **数据范围**: 主要在[0, 1]区间，经过归一化处理
- **编码格式**: `float32`

### 具体维度分解

#### 1. 主车状态信息 (前19维)

根据`StateObservation.vehicle_state()`方法的实现：

```python
# 维度 0-1: 侧面检测器数据 (如果启用) 或 左右边界距离
obs[0:2] = [lateral_to_left, lateral_to_right]  # 归一化的左右道路边界距离

# 维度 2: 朝向差异
obs[2] = heading_diff  # 主车朝向与当前车道朝向的差异 (弧度, 归一化)

# 维度 3: 当前速度  
obs[3] = current_speed  # 主车当前速度 (m/s, 归一化)

# 维度 4: 当前转向
obs[4] = current_steering  # 当前转向角 (归一化)

# 维度 5: 上一帧油门/刹车
obs[5] = last_throttle_brake  # 上一帧的油门/刹车值

# 维度 6: 上一帧转向
obs[6] = last_steering  # 上一帧的转向值

# 维度 7: 偏航率
obs[7] = yaw_rate  # 车辆偏航角速度 (归一化)

# 维度 8: 车道线检测器数据 (如果启用) 或 车道横向位置
obs[8] = lateral_position  # 在当前车道上的横向位置偏移

# 维度 9-18: 导航信息 (如果启用导航模块)
# 每个检查点包含5个值，通常有2个检查点
obs[9:14] = [
    distance_to_checkpoint_forward,   # 到检查点的前向距离投影
    distance_to_checkpoint_lateral,   # 到检查点的横向距离投影  
    lane_radius,                      # 车道曲率半径 (0表示直线)
    clockwise_flag,                   # 顺时针(1)或逆时针(0)标志
    lane_angle                        # 车道角度
]
obs[14:19] = [...]  # 下一个检查点的相同信息
```

#### 2. 激光雷达数据 (第20-259维)

```python
# 维度 19-258: 240个激光雷达射线的距离测量值
# 激光雷达配置: num_lasers=240, distance=50m
obs[19:259] = lidar_cloud_points  # 每个值表示该方向上最近障碍物的归一化距离

# 激光雷达扫描模式:
# - 360度环形扫描
# - 240个等间隔的射线 (每1.5度一个射线)
# - 最大检测距离: 50米
# - 返回值: 归一化距离 [0,1], 1表示50米内无障碍物
```

## 📊 传感器配置详情

### 默认传感器配置
```python
# metadrive/envs/base_env.py 中的默认配置
vehicle_config = {
    "lidar": {
        "num_lasers": 240,        # 激光雷达射线数量
        "distance": 50,           # 最大检测距离(米)
        "num_others": 0,          # 其他车辆信息数量
        "gaussian_noise": 0.0,    # 高斯噪声
        "dropout_prob": 0.0       # 丢失概率
    },
    "side_detector": {
        "num_lasers": 0,          # 侧面检测器射线数(0表示禁用)
        "distance": 50,           # 检测距离
        "gaussian_noise": 0.0,    # 噪声
        "dropout_prob": 0.0       # 丢失概率
    },
    "lane_line_detector": {
        "num_lasers": 0,          # 车道线检测器射线数(0表示禁用)
        "distance": 20,           # 检测距离
        "gaussian_noise": 0.0,    # 噪声
        "dropout_prob": 0.0       # 丢失概率
    }
}

# 传感器注册
sensors = {
    "lidar": (Lidar,),
    "side_detector": (SideDetector,),
    "lane_line_detector": (LaneLineDetector,)
}
```

## 🔧 观测处理流程

### 1. 原始数据采集
```
物理引擎 → 车辆状态获取 → 传感器数据采集 → 导航信息计算
```

### 2. 数据归一化处理
```python
# 距离归一化: distance / max_distance
# 角度归一化: angle / (2 * π) 
# 速度归一化: speed / max_speed
# 位置归一化: position / map_region_size
```

### 3. 观测向量组装
```python
def observe(self, vehicle):
    ego_state = self.vehicle_state(vehicle)           # 主车状态 (19维)
    if self.navi_dim > 0:
        navi_info = vehicle.navigation.get_navi_info() # 导航信息 (10维)
        obs = np.concatenate([ego_state, navi_info])   # 拼接状态
    lidar_data = self.lidar_observe(vehicle)          # 激光雷达 (240维)
    full_obs = np.concatenate([obs, lidar_data])      # 完整观测向量
    return full_obs.astype(np.float32)
```

## 🎯 认知模块中的观测处理

### 感知模块处理
```python
class CognitivePerceptionModule:
    def process_observation(self, obs):
        # 仅处理前2维位置信息 (x, y坐标)
        if len(obs) >= 2:
            true_x, true_y = obs[0], obs[1]
            # 添加感知噪声
            noisy_x = true_x + np.random.normal(0, self.sigma_x)
            noisy_y = true_y + np.random.normal(0, self.sigma_y)
            # 卡尔曼滤波
            filtered_x, filtered_y = self._apply_kalman_filter(noisy_x, noisy_y)
            # 更新观测向量
            obs[0] = filtered_x
            obs[1] = filtered_y
        return obs
```

## 📈 实际运行示例

基于运行输出的观测向量分析：
```python
obs = [0.09722222, 0.4861111, 0.5, 0.01234568, 0.5, 0.5, 
       0.5, 0.0, 0.5, 0.55, 0.465, 0.0, 0.5, 0.5, 0.95, 
       0.46500003, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, ...]

# 解读:
obs[0] = 0.097  # 左侧道路边界距离 (归一化)
obs[1] = 0.486  # 右侧道路边界距离 (归一化) 
obs[2] = 0.5    # 朝向差异 (中性值)
obs[3] = 0.012  # 当前速度 (很低的速度)
obs[4] = 0.5    # 当前转向 (直行)
obs[5] = 0.5    # 上一帧油门/刹车 (中性)
obs[6] = 0.5    # 上一帧转向 (直行)
obs[7] = 0.0    # 偏航率 (无旋转)
obs[8] = 0.5    # 车道横向位置 (车道中心)
obs[9:19] = ... # 导航信息
obs[19:259] = [1.0, 1.0, ...] # 激光雷达数据 (大部分为1.0表示无障碍物)
```

## ⚠️ 注意事项

### 1. 坐标系统
- MetaDrive使用右手坐标系
- X轴: 前进方向  
- Y轴: 左侧方向
- Z轴: 向上方向

### 2. 归一化范围
- 大部分观测值在[0, 1]范围内
- 角度值可能在[-1, 1]范围内
- 某些特殊值可能超出[0, 1]范围

### 3. 传感器精度
- 激光雷达: 1.5度角分辨率
- 检测距离: 50米
- 更新频率: 与物理仿真步长一致(20ms)

### 4. 认知模块影响
- 感知模块仅影响位置相关的观测维度
- 不影响激光雷达等其他传感器数据
- 在PPO模式下才会应用感知偏差

## 📝 总结

MetaDrive的259维观测向量提供了丰富的环境感知信息，包括：
- **车辆状态**: 速度、朝向、转向等动力学信息
- **空间感知**: 道路边界、车道位置、导航信息  
- **环境感知**: 360度激光雷达扫描的障碍物分布
- **历史信息**: 上一帧的控制输入

这种设计使得智能体能够获得全面的环境感知能力，支持复杂的驾驶决策任务。 