#!/usr/bin/env bash
set -euo pipefail
source /home/xuwj/.miniconda3/etc/profile.d/conda.sh
conda activate maniskill

# 实验参数（按名称排序）
max_cylinder_attempts=40
max_episodes=100000
max_templates=20000
output_dir=/home/xuwj/dev/maniskill/outputs/stackcube_obstacle
reuse_count=2
seed=0
target_successes=100

mkdir -p "$(dirname "${output_dir}")"
cd /home/xuwj/dev/maniskill
python -m examples.baselines.constrain.collect_constructive \
  --max-cylinder-attempts "${max_cylinder_attempts}" \
  --max-episodes "${max_episodes}" \
  --max-templates "${max_templates}" \
  --output-dir "${output_dir}" \
  --reuse-count "${reuse_count}" \
  --seed "${seed}" \
  --target-successes "${target_successes}"

# 采集成功后，将全部轨迹按编号拼接成同目录下的 dataset.mp4。
python -m examples.baselines.constrain.render_dataset_video \
  --dataset-dir "${output_dir}"
