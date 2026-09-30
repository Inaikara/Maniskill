#!/usr/bin/env bash
set -euo pipefail
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill

# 实验参数（按名称排序）
baseline=DP
batch_size=1024        # 每次更新的训练窗口数
control_mode=pd_ee_delta_pos
demo_path=/home/xuwj/dev/maniskill/demos/StackCube-v1/motionplanning/trajectory.filtered.state.pd_ee_delta_pos.physx_cpu.h5
demo_type=motionplanning
demos=980              # 使用全部过滤后的示范
env_id=StackCube-v1
eval_freq=10000         # 评测间隔（iterations）
log_freq=1000          # 日志间隔（iterations）
max_episode_steps=200  # 每个评测 episode 的最大步数
model_id=DP
num_eval_envs=10       # 评测并行环境数
output_dir=/home/xuwj/dev/maniskill/outputs # 权重、日志和评测产物的输出根目录。官方入口使用相对路径 runs/<exp_name>，通过工作目录配置输出位置。
seed=1
sim_backend=physx_cpu  # 评测物理后端
total_iters=30000      # 参数更新次数，不是 epoch 数

exp_name="${env_id}_${baseline}_${seed}"

mkdir -p "${output_dir}"
cd "${output_dir}"
python /home/xuwj/dev/maniskill/examples/baselines/diffusion_policy/train.py \
  --batch-size "${batch_size}" \
  --control-mode "${control_mode}" \
  --demo-path "${demo_path}" \
  --demo-type "${demo_type}" \
  --env-id "${env_id}" \
  --eval-freq "${eval_freq}" \
  --exp-name "${exp_name}" \
  --log-freq "${log_freq}" \
  --max-episode-steps "${max_episode_steps}" \
  --capture-video \
  --num-demos "${demos}" \
  --num-eval-envs "${num_eval_envs}" \
  --seed "${seed}" \
  --sim-backend "${sim_backend}" \
  --total-iters "${total_iters}"
