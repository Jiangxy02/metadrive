# MetaDrive PPO训练系统文件说明

## 📖 概述

本目录包含完整的MetaDrive强化学习PPO训练系统，支持随机场景生成、可视化训练监控、模型测试等功能。

---

## 🗂️ 核心文件结构

### 1. **random_scenario_generator.py** 🎲
**功能**: 随机场景生成器  
**核心类**: `RandomScenarioGenerator`, `MetaDriveRandomEnv`

#### 主要功能:
- **多维度随机化**:
  - 🗺️ 地图结构: 长度、复杂度、车道数、弯道密度
  - 🚗 交通环境: 车流密度、事故概率、交通模式
  - 🌤️ 环境条件: 天气、光照、地形
  - 🎯 任务难度: 起始位置、目标距离、时间限制

- **三种训练模式**:
  - `"random"`: 完全随机场景，适合基础训练
  - `"curriculum"`: 课程学习，难度递增
  - `"safe"`: 安全驾驶场景，强调安全性

#### 核心方法:
```python
# 生成基础随机场景
generate_basic_scenarios(num_scenarios=100)

# 生成课程学习场景
generate_curriculum_scenarios(max_difficulty=5)

# 生成安全驾驶场景
generate_safe_scenarios(accident_prob_range=(0.1, 0.3))
```

#### 使用示例:
```python
from random_scenario_generator import RandomScenarioGenerator, MetaDriveRandomEnv

# 创建生成器
generator = RandomScenarioGenerator(seed=42)

# 创建训练环境
env = MetaDriveRandomEnv(
    generator=generator,
    scenario_type="random",
    base_config={"use_render": True}
)
```

---

### 2. **sb3_ppo_integration.py** 🔗
**功能**: Stable Baselines3集成适配器  
**核心类**: `SB3TrajectoryReplayWrapper`

#### 主要功能:
- **SB3兼容性**: 确保与Stable Baselines3的gym.Env接口完全兼容
- **环境包装**: 处理observation_space和action_space
- **奖励设计**: 提供可配置的奖励函数
- **格式统一**: 统一step返回格式（5元组）

#### 奖励机制:
```python
reward_config = {
    "success_reward": 10.0,        # 成功完成奖励
    "crash_penalty": -5.0,         # 碰撞惩罚
    "out_of_road_penalty": -3.0,   # 出路惩罚
    "speed_reward_factor": 0.1,    # 速度奖励系数
    "steering_penalty_factor": 0.02 # 转向惩罚系数
}
```

#### 核心方法:
```python
# 训练PPO模型
train_sb3_ppo(
    csv_path="trajectory.csv",
    total_timesteps=50000,
    learning_rate=3e-4
)

# 加载和测试模型
load_and_test_sb3_model(
    model_path="trained_model.zip",
    csv_path="test_trajectory.csv"
)
```

---

### 3. **visual_training_monitor.py** 📊
**功能**: 可视化训练监控系统  
**核心类**: `VisualTrainingEnv`, `VisualTrainingCallback`

#### 主要功能:
- **实时渲染**: MetaDrive窗口显示训练场景
- **训练统计**: 收集episode奖励、长度、动作、速度历史
- **实时图表**: Matplotlib显示训练曲线
- **HUD显示**: 在MetaDrive窗口显示训练信息

#### 可视化内容:
1. **Episode Rewards**: 奖励值变化趋势
2. **Episode Lengths**: 每个episode持续步数
3. **Action Distribution**: 转向vs油门的动作分布热图
4. **Speed Distribution**: 车辆速度分布直方图

#### 训练环境特性:
```python
class VisualTrainingEnv(MetaDriveRandomEnv):
    def render_training_info(self):
        """在MetaDrive HUD显示训练信息"""
        training_text = {
            "Episode": f"{self.episode_count}",
            "Steps": f"{self.step_count}",
            "Reward": f"{self.current_episode_reward:.2f}",
            "Speed": f"{self.agent.speed:.1f} m/s"
        }
```

#### 关键配置:
```python
visual_config = {
    "render_mode": "human",
    "window_size": (1200, 800),
    "show_sensors": True,
    "show_trajectory": True,
    "record_video": False
}
```

---

### 4. **train_ppo_with_random_scenarios.py** 🎯
**功能**: 主PPO训练脚本  
**核心类**: `MetaDriveRandomWrapper`

#### 主要功能:
- **环境包装**: 将MetaDrive环境包装为SB3兼容格式
- **并行训练**: 支持多进程训练加速
- **超参数管理**: 完整的训练参数配置
- **日志记录**: TensorBoard和CSV日志

#### 训练流程:
1. **环境创建**: 创建随机场景环境
2. **模型初始化**: 配置PPO模型参数
3. **训练执行**: 开始强化学习训练
4. **模型保存**: 保存训练好的模型

#### 核心配置:
```python
training_config = {
    "total_timesteps": 100000,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95
}
```

#### 命令行使用:
```bash
python train_ppo_with_random_scenarios.py \
    --scenario-type random \
    --total-timesteps 50000 \
    --learning-rate 3e-4 \
    --log-dir ./logs/ppo_training
```

---

### 5. **start_visual_training.py** 🎬
**功能**: 可视化训练启动器  
**类型**: 命令行启动脚本

#### 主要功能:
- **一键启动**: 简化的可视化训练启动
- **参数配置**: 完整的命令行参数支持
- **用户友好**: 清晰的使用说明和提示

#### 支持参数:
```bash
# 基础配置
--scenario-type {random,curriculum,safe}
--total-timesteps 50000
--seed 42

# 训练参数
--learning-rate 3e-4
--n-steps 1024
--batch-size 64

# 可视化配置
--window-size 1200 800
--log-freq 500
--plot-freq 2000

# 输出配置
--log-dir ./logs/ppo_visual
--device {auto,cpu,cuda}
```

#### 使用示例:
```bash
# 基础训练
python start_visual_training.py

# 自定义配置
python start_visual_training.py \
    --scenario-type curriculum \
    --total-timesteps 100000 \
    --learning-rate 1e-3 \
    --window-size 1600 900
```

---

### 6. **trained_model_test.py** 🧪
**功能**: 训练模型测试工具  
**类型**: 模型评估脚本

#### 主要功能:
- **模型加载**: 加载训练好的PPO模型
- **性能评估**: 在新场景中测试模型性能
- **统计分析**: 详细的测试结果统计
- **渲染选择**: 支持有渲染/无渲染测试

#### 测试环境配置:
```python
test_config = {
    "use_render": False,  # 可选择开启渲染
    "horizon": 1000,      # 更长的测试时间
    "traffic_density": 0.2,
    "num_scenarios": 50,
    "start_seed": 123     # 不同种子测试泛化能力
}
```

#### 评估指标:
- **平均奖励**: 多个episode的平均累积奖励
- **平均长度**: episode平均持续步数
- **最佳性能**: 最高奖励和最长episode
- **成功率**: 完成任务的episode比例

#### 使用示例:
```bash
# 无渲染测试
python trained_model_test.py \
    --model-path ./logs/ppo_training/final_model.zip \
    --episodes 10

# 有渲染测试
python trained_model_test.py \
    --model-path ./logs/ppo_training/final_model.zip \
    --episodes 5 \
    --render
```

---

## 🔧 系统依赖

### Python包依赖:
```bash
# 核心依赖
pip install stable-baselines3[extra]
pip install gymnasium
pip install numpy
pip install matplotlib

# MetaDrive相关
pip install metadrive-simulator

# 可选依赖
pip install tensorboard  # 训练监控
pip install tqdm rich    # 进度条显示
```

### 系统依赖:
```bash
# Ubuntu/Debian
sudo apt-get install -y libxcb-xinerama0 libxcb-xfixes0 \
    libxcb-randr0 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-render-util0 \
    libxcb-shape0 qtbase5-dev qt5-qmake
```

---

## 🚀 快速开始

### 1. 基础训练 (无头模式)
```bash
# 进入项目目录
cd /path/to/metadrive/a_scen_env

# 激活conda环境
conda activate metadrive

# 开始训练
python train_ppo_with_random_scenarios.py \
    --scenario-type random \
    --total-timesteps 10000
```

### 2. 可视化训练
```bash
# 启动可视化训练
python start_visual_training.py \
    --total-timesteps 50000 \
    --device cpu
```

### 3. 模型测试
```bash
# 测试训练好的模型
python trained_model_test.py \
    --model-path ./logs/ppo_training/final_model.zip \
    --episodes 5 \
    --render
```

---

## 📈 训练监控

### TensorBoard可视化:
```bash
# 启动TensorBoard
tensorboard --logdir ./logs

# 在浏览器访问
http://localhost:6006
```

### 日志文件:
- `progress.csv`: 训练进度数据
- `events.out.tfevents.*`: TensorBoard事件文件
- `final_model.zip`: 最终训练模型

---

## 🎛️ 高级配置

### 1. 自定义奖励函数
```python
custom_reward_config = {
    "success_reward": 20.0,
    "crash_penalty": -10.0,
    "speed_reward_factor": 0.2,
    "efficiency_bonus": 5.0
}
```

### 2. 课程学习配置
```python
curriculum_config = {
    "difficulty_progression": "linear",
    "max_difficulty": 5,
    "episodes_per_level": 100
}
```

### 3. 并行训练配置
```python
parallel_config = {
    "n_envs": 4,           # 并行环境数量
    "vec_env_cls": SubprocVecEnv,  # 多进程环境
    "env_kwargs": {}       # 环境参数
}
```

---

## 🐛 故障排除

### 常见问题及解决方案:

1. **Qt平台插件错误**:
   ```bash
   export QT_QPA_PLATFORM=offscreen
   ```

2. **CUDA内存不足**:
   ```bash
   python script.py --device cpu
   ```

3. **依赖包缺失**:
   ```bash
   pip install stable-baselines3[extra]
   pip install tqdm rich
   ```

4. **MetaDrive渲染问题**:
   ```python
   # 使用无头模式
   config = {"use_render": False}
   ```

---

## 📝 开发说明

### 扩展指南:
1. **添加新场景类型**: 在`RandomScenarioGenerator`中添加新的生成方法
2. **自定义奖励函数**: 修改`SB3TrajectoryReplayWrapper`中的奖励计算
3. **新增可视化内容**: 在`VisualTrainingCallback`中添加新的图表
4. **优化训练参数**: 调整PPO超参数以获得更好性能

### 代码结构:
```
a_scen_env/
├── random_scenario_generator.py    # 场景生成核心
├── sb3_ppo_integration.py         # SB3集成适配
├── visual_training_monitor.py     # 可视化监控
├── train_ppo_with_random_scenarios.py  # 主训练脚本
├── start_visual_training.py       # 启动器
├── trained_model_test.py          # 模型测试
└── logs/                          # 训练日志目录
```

---

## 🎉 总结

这套PPO训练系统提供了完整的强化学习训练流程，从随机场景生成到模型训练、可视化监控，再到模型测试评估。系统具有良好的模块化设计，易于扩展和定制，适合研究和实际应用。

**主要特色**:
- 🎲 丰富的随机场景生成
- 📊 实时可视化训练监控  
- 🔧 完善的SB3集成
- 🧪 全面的模型测试评估
- 🚀 简单易用的启动方式 