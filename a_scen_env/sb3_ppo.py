import os
import gym
import metadrive
from metadrive.envs import SafeMetaDriveEnv  # 或 MetaDriveEnv / GeneralizationRacing
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# -------- 环境工厂（并行创建 N 个子环境） --------
NUM_ENVS = 8  # 你的机器有 24 线程，可以设 12~24，注意显存/CPU占用
def make_env(rank):
    def _init():
        env = SafeMetaDriveEnv({
            "use_render": False,
            "num_scenarios": 1000,
            "start_seed": 0 + rank*10000,     # 不同子环境不同seed
            "horizon": 1000,
            # 奖励权重（可按你之前的配置改）
            "speed_reward_weight": 0.1,
            "progress_reward_weight": 1.0,
            "crash_penalty": -10.0,
            "out_road_penalty": -5.0,
            "completion_bonus": 20.0,
            "step_penalty": -0.01,
            # 训练模式：random/curriculum/safe（不同环境类支持项不同）
            # "environment_num": 100, ...
        })
        return Monitor(env)
    return _init

env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])
env = VecMonitor(env)

# -------- PPO 模型 --------
policy_kwargs = dict(net_arch=[256, 256])  # MLP 两层，每层256
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048 // NUM_ENVS,  # 每个子环境的步数，保证总采样约 2048
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    device="cuda",       # 有GPU就用"cuda"
    verbose=1,
    tensorboard_log="./tb_logs"
)

# -------- 评估回调（边训边测）--------
eval_env = SafeMetaDriveEnv({"use_render": False, "num_scenarios": 100, "horizon": 1000})
eval_env = Monitor(eval_env)
eval_callback = EvalCallback(
    eval_env, best_model_save_path="./ckpts",
    log_path="./eval_logs", eval_freq=10000, n_eval_episodes=10, deterministic=True
)

# -------- 开始训练 --------
total_steps = 1_000_000
model.learn(total_timesteps=total_steps, callback=eval_callback)

# -------- 保存 --------
os.makedirs("ckpts", exist_ok=True)
model.save("ckpts/ppo_metadrive_sb3")

# -------- 简单评估与可视化（可选）--------
# from stable_baselines3.common.evaluation import evaluate_policy
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
# print(mean_reward, std_reward)
