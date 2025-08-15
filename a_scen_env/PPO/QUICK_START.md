# 🚀 PPO训练系统快速开始指南

## 系统状态

✅ **系统已成功集成！** 所有组件都已就绪。

## 快速开始（3步）

### 步骤1：训练PPO模型

```bash
cd /home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/PPO

# 方式A：交互式训练（推荐）
python start_training.py

# 方式B：直接训练（使用默认参数）
python train_ppo.py
```

训练将自动：
- 使用目录中的所有CSV轨迹文件
- 创建4个并行环境加速训练
- 每10000步评估模型性能
- 保存最佳模型和检查点

### 步骤2：评估训练效果

```bash
# 评估最新模型
python evaluate_ppo.py

# 评估并渲染（观看模型驾驶）
python evaluate_ppo.py --episodes 3
```

### 步骤3：在trajectory_replay.py中使用

训练好的模型会**自动**被trajectory_replay.py检测并加载！

```bash
cd ..
python trajectory_replay.py
```

使用热键控制：
- **T**: 切换PPO专家/手动控制
- **E**: 开关PPO专家接管
- **R**: 开关轨迹重放
- **M**: 强制手动模式

## 文件结构

```
PPO/
├── train_ppo.py          # 核心训练脚本
├── evaluate_ppo.py       # 模型评估工具
├── ppo_expert.py         # 专家策略接口
├── start_training.py     # 交互式启动器
├── test_integration.py   # 系统测试工具
├── README.md            # 详细文档
├── QUICK_START.md       # 本文件
└── models/              # 训练的模型存储在这里
```

## 监控训练进度

使用TensorBoard查看实时训练曲线：

```bash
tensorboard --logdir models/
```

然后打开浏览器访问 http://localhost:6006

## 常见命令

```bash
# 测试系统集成
python test_integration.py

# 快速训练测试（1分钟）
python train_ppo.py  # 修改代码中的total_timesteps=10000

# 长时间训练（获得更好效果）
python train_ppo.py  # 修改代码中的total_timesteps=1000000
```

## 故障排除

### 问题：没有找到PPO模型
**解决**：先运行训练脚本生成模型

### 问题：内存不足
**解决**：减少并行环境数（在train_ppo.py中修改n_envs=2）

### 问题：训练速度慢
**解决**：
1. 确保没有开启渲染（use_render=False）
2. 减少horizon参数
3. 使用GPU（如果可用）

## 系统特点

✨ **自动集成**：训练的模型自动被trajectory_replay.py使用
🚀 **并行训练**：多环境并行加速训练
📊 **实时监控**：TensorBoard可视化训练进度
🎮 **热键切换**：保留所有原有控制模式
🔄 **多场景训练**：自动使用所有CSV文件训练

---

**准备就绪！** 现在可以开始训练您的PPO驾驶智能体了！ 🎉 