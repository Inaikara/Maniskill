# Agent 必须遵守

- 开发、安装依赖、训练、评测和 replay 前，必须激活 `maniskill` conda 环境：`conda activate maniskill`。
- 实验代码放在 `examples/baselines/constrain`；修改 ManiSkill 源码须先获得用户明确许可。

## 产物目录

- 训练数据及其采集、验证元信息保存到 `demos/<数据集名>/`；正式障碍演示数据集为 `demos/StackCube-obstacle/`。
- `outputs/` 按用途分为 `ckpt/`（模型权重）和 `exp/`（训练日志、评估场景、结果、视频及验证记录）。
- 实验目录使用时间戳前缀：`outputs/exp/YYYYMMDD_HHMMSS_<实验名>/`；只有日期可追溯的历史实验使用 `YYYYMMDD_<实验名>/`。
- 运行采集、训练或评估时显式指定符合上述分类的输出路径；训练日志与权重分别保存到 `exp/` 和 `ckpt/`。当前产物位置见 `examples/baselines/constrain/OUTPUTS.md`。
- 移动已完成产物时保留其内容和身份哈希；原始 manifest/checkpoint 内的路径作为历史来源保留，通过迁移映射定位现有文件。

## Troubleshooting

- 官方 CPU Diffusion Policy 评测报 `KeyError: 'final_info'`：Gymnasium 1.3 的自动 reset 接口与 baseline 不兼容。在 `maniskill` 环境使用 `pip install gymnasium==0.29.1`；已验证训练与并行评测通过。
- [Issue #1491](https://github.com/mani-skill/ManiSkill/issues/1491) / [修复 PR #1490](https://github.com/mani-skill/ManiSkill/pull/1490)：GPU replay 传入 tensor 种子，触发 `_set_main_rng` 的 numpy 类型断言；失败 replay 未清空缓冲区，会污染下一条成功轨迹。使用前检查上游修复状态。
- 本机 CPU replay 已发现并过滤 10 条污染轨迹。训练使用 `demos/StackCube-v1/motionplanning/trajectory.filtered.state.pd_ee_delta_pos.physx_cpu.h5`（980 条），配套 JSON 同名；原始 replay 文件保留用于追溯。
