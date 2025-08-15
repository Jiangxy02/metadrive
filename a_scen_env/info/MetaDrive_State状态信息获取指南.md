# MetaDrive State 状态信息获取指南

## 📋 概述

MetaDrive环境中，车辆的真实物理状态（State）与经过处理的观测向量（Observation）是两个不同的概念。本文档详细说明如何获取车辆的原始状态信息，以及这些状态信息的含义和用途。

## 🔍 State vs Observation 区别

### State（状态）
- **定义**：车辆在物理世界中的真实状态
- **特点**：未经归一化，保持物理单位和真实数值
- **用途**：用于物理仿真、碰撞检测、精确控制等

### Observation（观测）
- **定义**：经过处理、归一化的感知信息
- **特点**：归一化到[0,1]区间，适合机器学习模型
- **用途**：作为智能体的输入，用于决策和学习

## 🚗 车辆状态信息详解

### 1. 基础运动状态

#### 位置信息
```python
# 获取方式
position = env.agent.position  # 返回 [x, y] 坐标
x = env.agent.position[0]      # X坐标 (米)
y = env.agent.position[1]      # Y坐标 (米)

# 数据类型
# position: numpy.ndarray, shape=(2,)
# 坐标系: MetaDrive世界坐标系，单位为米
# 示例: [200.5, 7.2] 表示车辆在(200.5m, 7.2m)位置
```

#### 速度信息
```python
# 速度大小
speed = env.agent.speed        # 标量速度 (m/s)

# 速度矢量
velocity = env.agent.velocity  # 返回 [vx, vy] 速度矢量
vx = env.agent.velocity[0]     # X方向速度分量 (m/s)
vy = env.agent.velocity[1]     # Y方向速度分量 (m/s)

# 数据类型
# speed: float, 非负值
# velocity: numpy.ndarray, shape=(2,)
# 示例: speed=15.3 表示车速15.3米/秒
#       velocity=[12.2, 9.1] 表示速度矢量
```

#### 朝向信息
```python
# 车辆朝向角度
heading = env.agent.heading_theta  # 朝向角 (弧度)

# 数据类型
# heading_theta: float
# 范围: [-π, π]
# 说明: 0表示正东方向，π/2表示正北方向
# 示例: heading=1.57 表示朝向北方(π/2弧度)
```

### 2. 车道和道路信息

#### 车道状态
```python
# 是否在车道上
on_lane = env.agent.on_lane           # bool

# 是否出界
out_of_road = env.agent.out_of_road   # bool

# 到车道边界的距离
dist_to_left = env.agent.dist_to_left_side    # 到左侧边界距离 (米)
dist_to_right = env.agent.dist_to_right_side  # 到右侧边界距离 (米)

# 数据类型和含义
# on_lane: True表示在有效车道上，False表示不在车道上
# out_of_road: True表示车辆已出界
# dist_to_left/right: float, 正值表示距离，单位为米
```

#### 车道参考信息
```python
# 当前车道
current_lane = env.agent.lane         # 当前所在车道对象

# 车道中心线位置
if current_lane:
    # 获取车道的局部坐标
    longitudinal, lateral = current_lane.local_coordinates(env.agent.position)
    # longitudinal: 沿车道方向的距离
    # lateral: 相对车道中心线的横向偏移
```

### 3. 碰撞状态信息

```python
# 各种碰撞状态
crash_vehicle = env.agent.crash_vehicle     # 是否撞车
crash_object = env.agent.crash_object       # 是否撞物体
crash_sidewalk = env.agent.crash_sidewalk   # 是否撞人行道

# 数据类型
# 所有碰撞状态: bool
# True表示发生对应类型的碰撞，False表示无碰撞
```

### 4. 导航信息

```python
# 导航模块
navigation = env.agent.navigation

# 导航信息
if navigation:
    # 获取导航路径信息
    navi_info = navigation.get_navi_info()  # 10维导航向量
    
    # 当前路径
    current_lanes = navigation.current_ref_lanes
    
    # 目标点信息
    destination = navigation.final_lane
    
    # 路径完成度
    route_completion = navigation.route_completion
```

## 🔧 状态信息获取方法

### 方法1：直接访问属性

```python
def get_vehicle_state(env):
    """获取车辆完整状态信息"""
    agent = env.agent
    
    state_info = {
        # 基础运动信息
        'position': {
            'x': float(agent.position[0]),
            'y': float(agent.position[1])
        },
        'velocity': {
            'speed': float(agent.speed),
            'vx': float(agent.velocity[0]),
            'vy': float(agent.velocity[1])
        },
        'orientation': {
            'heading': float(agent.heading_theta)
        },
        
        # 车道信息
        'lane_info': {
            'on_lane': bool(agent.on_lane),
            'out_of_road': bool(agent.out_of_road),
            'dist_to_left': float(agent.dist_to_left_side),
            'dist_to_right': float(agent.dist_to_right_side)
        },
        
        # 碰撞状态
        'collision': {
            'crash_vehicle': bool(agent.crash_vehicle),
            'crash_object': bool(agent.crash_object), 
            'crash_sidewalk': bool(agent.crash_sidewalk)
        }
    }
    
    return state_info

# 使用示例
env = TrajectoryReplayEnvCognitive(...)
obs = env.reset()
state = get_vehicle_state(env)
print(f"车辆位置: ({state['position']['x']:.2f}, {state['position']['y']:.2f})")
print(f"车辆速度: {state['velocity']['speed']:.2f} m/s")
```

### 方法2：使用getattr安全访问

```python
def safe_get_vehicle_state(env):
    """安全获取车辆状态（处理可能不存在的属性）"""
    agent = env.agent
    
    state_info = {
        # 位置和运动
        'pos_x': float(getattr(agent, 'position', [0, 0])[0]),
        'pos_y': float(getattr(agent, 'position', [0, 0])[1]),
        'speed': float(getattr(agent, 'speed', 0.0)),
        'heading': float(getattr(agent, 'heading_theta', 0.0)),
        'velocity_x': float(getattr(agent, 'velocity', [0, 0])[0]),
        'velocity_y': float(getattr(agent, 'velocity', [0, 0])[1]),
        
        # 车道和道路信息
        'on_lane': getattr(agent, 'on_lane', None),
        'out_of_road': getattr(agent, 'out_of_road', None),
        'dist_to_left_side': getattr(agent, 'dist_to_left_side', None),
        'dist_to_right_side': getattr(agent, 'dist_to_right_side', None),
        
        # 碰撞状态
        'crash_vehicle': getattr(agent, 'crash_vehicle', None),
        'crash_object': getattr(agent, 'crash_object', None),
        'crash_sidewalk': getattr(agent, 'crash_sidewalk', None),
    }
    
    return state_info
```

### 方法3：实时状态监控

```python
class VehicleStateMonitor:
    """车辆状态实时监控器"""
    
    def __init__(self, env):
        self.env = env
        self.state_history = []
    
    def record_current_state(self, step_count=None):
        """记录当前状态"""
        agent = self.env.agent
        
        current_state = {
            'step': step_count,
            'timestamp': time.time(),
            'simulation_time': getattr(self.env, '_simulation_time', 0.0),
            
            # 核心状态
            'position': list(agent.position),
            'speed': float(agent.speed),
            'heading': float(agent.heading_theta),
            'velocity': list(agent.velocity),
            
            # 车道状态
            'on_lane': bool(agent.on_lane),
            'out_of_road': bool(agent.out_of_road),
            'dist_to_left': float(agent.dist_to_left_side),
            'dist_to_right': float(agent.dist_to_right_side),
            
            # 碰撞检测
            'crash_vehicle': bool(agent.crash_vehicle),
            'crash_object': bool(agent.crash_object),
            'crash_sidewalk': bool(agent.crash_sidewalk)
        }
        
        self.state_history.append(current_state)
        return current_state
    
    def get_state_changes(self):
        """获取状态变化信息"""
        if len(self.state_history) < 2:
            return None
            
        prev_state = self.state_history[-2]
        curr_state = self.state_history[-1]
        
        changes = {
            'position_change': np.linalg.norm(
                np.array(curr_state['position']) - np.array(prev_state['position'])
            ),
            'speed_change': curr_state['speed'] - prev_state['speed'],
            'heading_change': curr_state['heading'] - prev_state['heading'],
            'time_delta': curr_state['simulation_time'] - prev_state['simulation_time']
        }
        
        return changes

# 使用示例
env = TrajectoryReplayEnvCognitive(...)
monitor = VehicleStateMonitor(env)

obs = env.reset()
for step in range(100):
    action = [0.0, 0.5]  # 示例动作
    obs, reward, done, info = env.step(action)
    
    # 记录当前状态
    state = monitor.record_current_state(step)
    print(f"Step {step}: Position=({state['position'][0]:.2f}, {state['position'][1]:.2f}), "
          f"Speed={state['speed']:.2f}")
    
    # 分析状态变化
    if step > 0:
        changes = monitor.get_state_changes()
        print(f"  位置变化: {changes['position_change']:.3f}m, "
              f"速度变化: {changes['speed_change']:.3f}m/s")
    
    if done:
        break
```

## 📊 State与Observation的对应关系

```python
def compare_state_and_observation(env):
    """对比真实状态和观测向量"""
    # 获取原始状态
    agent = env.agent
    true_state = {
        'position_x': agent.position[0],
        'position_y': agent.position[1], 
        'speed': agent.speed,
        'heading': agent.heading_theta,
        'dist_to_left': agent.dist_to_left_side,
        'dist_to_right': agent.dist_to_right_side
    }
    
    # 获取观测向量
    obs = env._get_observation()
    
    # 对比分析
    print("=== State vs Observation 对比 ===")
    print(f"真实位置: ({true_state['position_x']:.3f}, {true_state['position_y']:.3f})")
    print(f"观测位置: obs[0]={obs[0]:.3f}, obs[1]={obs[1]:.3f}")
    print(f"  说明: 观测值是归一化的道路边界距离，不是绝对位置")
    
    print(f"\n真实速度: {true_state['speed']:.3f} m/s")
    print(f"观测速度: obs[3]={obs[3]:.3f}")
    print(f"  说明: 观测值是归一化的速度")
    
    print(f"\n真实朝向: {true_state['heading']:.3f} rad")
    print(f"观测朝向差: obs[2]={obs[2]:.3f}")
    print(f"  说明: 观测值是与车道朝向的差异，不是绝对朝向")
    
    print(f"\n真实边界距离: 左={true_state['dist_to_left']:.3f}m, 右={true_state['dist_to_right']:.3f}m")
    print(f"观测边界距离: obs[0]={obs[0]:.3f}, obs[1]={obs[1]:.3f}")
    print(f"  说明: 观测值是归一化后的边界距离")
    
    return true_state, obs
```

## 🎯 认知模块中的应用

```python
def get_ego_state_for_cognitive_modules(env):
    """为认知模块获取主车状态信息"""
    agent = env.agent
    
    # 认知模块需要的关键状态信息
    ego_state = {
        # 用于感知模块的位置信息
        'true_position': np.array([agent.position[0], agent.position[1]]),
        
        # 用于计算主车-目标车距离
        'position_x': agent.position[0],
        'position_y': agent.position[1],
        
        # 用于延迟模块的动力学信息
        'speed': agent.speed,
        'heading': agent.heading_theta,
        'velocity': np.array([agent.velocity[0], agent.velocity[1]]),
        
        # 用于安全评估的车道信息
        'on_lane': agent.on_lane,
        'out_of_road': agent.out_of_road,
        'dist_to_left_side': agent.dist_to_left_side,
        'dist_to_right_side': agent.dist_to_right_side
    }
    
    return ego_state

# 在认知模块中的使用示例
class CognitivePerceptionModule:
    def process_observation(self, obs, env):
        # 获取真实的主车状态
        ego_state = get_ego_state_for_cognitive_modules(env)
        true_x, true_y = ego_state['true_position']
        
        # 基于真实位置计算感知噪声
        sigma_x = self.calculate_dynamic_sigma(ego_state)
        
        # 处理观测...
        return processed_obs
```

## ⚠️ 重要注意事项

### 1. 坐标系统
- **MetaDrive坐标系**: 右手坐标系，X轴向前，Y轴向左，Z轴向上
- **原点位置**: 通常以地图中心或起始点为原点
- **单位**: 所有距离单位为米，角度单位为弧度

### 2. 数据类型
- **位置和速度**: numpy.ndarray 或 float
- **布尔状态**: bool 类型
- **角度**: float，范围 [-π, π]

### 3. 访问安全性
- 使用 `getattr()` 可以安全访问可能不存在的属性
- 某些属性在特定条件下可能为 `None`
- 建议在访问前先检查属性是否存在

### 4. 性能考虑
- 状态信息获取是轻量级操作
- 可以在每个仿真步骤中安全调用
- 避免频繁的深拷贝操作

## 📝 总结

MetaDrive的状态信息提供了车辆的真实物理状态，这些信息对于：
- **精确控制**: 需要物理单位的真实数值
- **碰撞检测**: 需要准确的位置和速度信息  
- **性能分析**: 需要未经处理的原始数据
- **认知建模**: 需要真实状态用于计算感知偏差

与归一化的观测向量相比，状态信息保持了原始的物理意义，是实现高精度仿真和分析的重要基础。 