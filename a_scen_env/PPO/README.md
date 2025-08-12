# PPO训练系统使用说明

## 📋 系统概述

本系统提供了一个完整的PPO（Proximal Policy Optimization）训练框架，专门用于训练`TrajectoryReplayEnv`环境中的自动驾驶智能体。训练好的模型可以直接在`trajectory_replay.py`中作为PPO专家策略使用。

## 🚀 快速开始

### 1. 安装依赖

```bash
# 确保已安装stable-baselines3
pip install stable-baselines3[extra]
pip install gymnasium
```

### 2. 训练模型

最简单的方式是使用启动脚本：

```bash
cd /home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/PPO
python start_training.py
```

或者直接运行训练脚本：

```bash
python train_ppo.py
```

### 3. 评估模型

训练完成后，评估模型性能：

```bash
# 评估最新的模型
python evaluate_ppo.py

# 评估指定模型
python evaluate_ppo.py --model path/to/model.zip

# 评估时不渲染（加快速度）
python evaluate_ppo.py --no-render --episodes 10
```

### 4. 在trajectory_replay.py中使用

训练好的模型会自动被`trajectory_replay.py`检测并使用。只需正常运行trajectory_replay.py，按`T`键即可切换到PPO专家模式。

```bash
cd ..
python trajectory_replay.py
```

## 📁 文件结构

```
PPO/
├── README.md              # 本文档
├── train_ppo.py          # PPO训练主脚本
├── evaluate_ppo.py       # 模型评估脚本
├── ppo_expert.py         # PPO专家策略接口
├── start_training.py     # 交互式训练启动脚本
└── models/               # 训练好的模型存储目录
    └── ppo_trajectory_YYYYMMDD_HHMMSS/
        ├── final_model.zip           # 最终模型
        ├── best_model/               # 最佳模型
        │   └── best_model.zip
        ├── logs/                     # 训练日志
        └── ppo_checkpoint_*.zip      # 检查点
```

## ⚙️ 训练配置

### 环境参数

- `horizon`: 每个episode的最大步数（默认：2000）
- `end_on_crash`: 碰撞时是否结束（默认：True）
- `end_on_out_of_road`: 出界时是否结束（默认：True）
- `map`: 地图配置，"S"*30表示30段直道
- `background_vehicle_update_mode`: 背景车更新模式（"position"或"dynamics"）

### PPO超参数

在`train_ppo.py`中可以调整：

- `learning_rate`: 学习率（默认：3e-4）
- `n_steps`: 每次更新的步数（默认：2048）
- `batch_size`: 批大小（默认：64）
- `n_epochs`: 每次更新的epoch数（默认：10）
- `gamma`: 折扣因子（默认：0.99）
- `clip_range`: PPO裁剪范围（默认：0.2）

### 训练参数

- `total_timesteps`: 总训练步数（默认：500000）
- `n_envs`: 并行环境数量（默认：4）
- `save_freq`: 检查点保存频率（默认：50000步）
- `eval_freq`: 评估频率（默认：10000步）

## 🎮 使用训练好的模型

### 在trajectory_replay.py中使用

系统会自动加载最新的训练模型。控制模式切换热键：

- **T**: 切换PPO专家/手动控制模式
- **E**: 开关PPO专家接管
- **R**: 开关轨迹重放模式
- **M**: 强制进入手动模式
- **W/A/S/D**: 手动控制（前进/左转/后退/右转）

### 程序化使用

```python
from PPO.ppo_expert import expert, set_expert_model

# 使用最新模型
action = expert(agent)

# 使用指定模型
set_expert_model("path/to/model.zip")
action = expert(agent)
```

## 📊 查看训练日志

使用TensorBoard查看训练进度：

```bash
tensorboard --logdir models/ppo_trajectory_*/logs
```

然后在浏览器中打开 http://localhost:6006

## 🔧 高级用法

### 自定义训练配置

```python
from train_ppo import train_ppo

custom_config = {
    "horizon": 3000,
    "map": "S" * 50,  # 更长的地图
    "background_vehicle_update_mode": "dynamics",  # 使用动力学模式
}

model, model_dir = train_ppo(
    total_timesteps=1000000,
    n_envs=8,
    config=custom_config
)
```

### 继续训练已有模型

```python
from stable_baselines3 import PPO
from train_ppo import TrajectoryReplayWrapper

# 加载已有模型
model = PPO.load("models/ppo_trajectory_*/final_model.zip")

# 创建环境
env = TrajectoryReplayWrapper(config)

# 继续训练
model.set_env(env)
model.learn(total_timesteps=500000)
```

## ❓ 常见问题

### Q: 训练过程中内存占用过高？
A: 减少并行环境数量（n_envs）或减少n_steps参数。

### Q: 模型收敛速度慢？
A: 可以尝试：
- 增加训练步数（total_timesteps）
- 调整学习率（learning_rate）
- 修改奖励函数（需要修改环境代码）

### Q: 如何使用多个CSV文件训练？
A: 训练脚本会自动检测并使用`a_scen_env`目录下所有的`scenario_vehicles_*.csv`文件。

### Q: 训练好的模型在trajectory_replay.py中表现不好？
A: 可能需要：
- 增加训练步数
- 确保训练和测试使用相同的环境配置
- 检查观察空间和动作空间是否一致

## 📝 注意事项

1. **GPU加速**: 如果有GPU，stable-baselines3会自动使用GPU加速训练。
2. **训练时间**: 50万步的训练在4个并行环境下大约需要1-2小时（取决于硬件）。
3. **模型选择**: 系统会自动保存最佳模型（best_model）和最终模型（final_model），通常best_model性能更好。
4. **内存管理**: 并行环境会占用较多内存，如果内存不足，减少n_envs参数。

## 🤝 技术支持

如有问题或需要帮助，请查看相关文档或联系开发团队。

---

*最后更新：2025-01-03* 