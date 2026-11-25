# 三机车库穿缝 · 论文同期基线

本目录从作者于 2024 年 9 月 10 日公开的 `examples/crossing/rl.py` 启动 PPO。正式训练有 24 个独立车库，每个车库 3 架无人机，种子 42。脚本增加检查点、观测与评估记录，保留原版网络、奖励、碰撞和终止行为。[作者训练入口](https://github.com/SJTU-ViSYS-team/VisFly/blob/7b0a4534b57051fd9b00ab4e561496f9abaacab1/examples/crossing/rl.py) · [论文 IV-B 与图 7](https://arxiv.org/html/2407.14783v4#S4.SS2)

本机归档位于 [paper-crossing-20261123](/home/ruan/research/VisFly-local-setup-20261113/paper-crossing-20261123/REPORT.md)。[环境和资产验证](/home/ruan/research/VisFly-local-setup-20261113/paper-crossing-20261123/setup/SOURCES.md)记录作者版本、数据哈希、渲染器编译和实际导入路径。[复现问题](/home/ruan/research/VisFly-local-setup-20261113/paper-crossing-20261123/ISSUES.md)记录原生碰撞、复位、退出以及论文曲线估读的限制。

## 复用本机环境

以下路径在本次工作站上可直接使用。新一次训练需指定空输出目录。若在其他机器运行，将历史 VisFly 放在 `$run/source`，`source/datasets/spy_datasets` 指向 9 月资产，`setup/renderer-lib` 指向历史绑定。本目录的 `protocol.json` 需复制至 `$run/setup/protocol.json`。

```bash
scripts=/home/ruan/research/VisFly/experiments/crossing/2026-11-23
run=/home/ruan/research/VisFly-local-setup-20261113/paper-crossing-20261123
python_bin=/home/ruan/miniconda3/envs/visfly/bin/python
export PYTHONPATH="$run/setup/renderer-lib"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 一场三机完整 PPO 更新，768 个交互
"$python_bin" "$scripts/train_paper.py" "$run" "$run/smoke/new-one-scene" --scenes 1 --budget 768

# 按论文同期参数开新训练
"$python_bin" "$scripts/train_paper.py" "$run" "$run/train-new"

# 用已存百万交互权重观察一个新的开发外调试种子
"$python_bin" "$scripts/evaluate_paper.py" "$run" "$run/train/PPO_1013760.zip" smoke \
    --seeds 3000 3000 --output "$run/smoke/seed-3000"
```

`train/training.csv` 的单位是智能体交互。一次同步控制推进 72 条交互，每次 PPO 采样共 18,432 条。检查点在梯度更新后保存，`PPO_0.zip` 在第一次优化前保存。固定点和滚动点都保存参数、Adam 状态和 SHA-256。滚动点不包含每个仿真实例及全部随机数状态。服务重启后从滚动权重重新采样时，应单独标记续训条件。

## 评估与核查

`evaluate_checkpoints.sh` 等待 `$run/train` 中的四个正式检查点，依次运行开发种子 1000–1099。最后用固定的千万交互权重运行留出种子 20000–20099。已完成的文件按哈希核对后可复用。另跑新实验时，应复制冻结协议并用独立运行根目录保存自己的训练、评估和日志。

```bash
cp "$scripts/evaluate_checkpoints.sh" "$run/setup/evaluate_checkpoints.sh"
VISFLY_SCRIPTS="$scripts" VISFLY_PYTHON="$python_bin" bash "$run/setup/evaluate_checkpoints.sh" "$run"
cohort="$run/evaluation/development/PPO_1013760"
/usr/bin/python3 "$scripts/summarize_evaluation.py" "$run" "$cohort"
/usr/bin/python3 "$scripts/replay_cases.py" "$run" "$cohort" --prefix dev-1m
"$python_bin" "$scripts/official_replay.py" "$run" "$cohort" --prefix official-dev-1m
"$python_bin" "$scripts/compare_curves.py" "$run"
(cd "$scripts" && /usr/bin/python3 -m unittest -v test_geometry_audit)
```

评估采用 CPU 确定性推理及原版视觉仿真。每个种子新建场景，在载入冻结权重后重新设置种子。记录的 `control_xyz` 有初始位置和各控制帧，`substep_xyz` 保存每 2.5 ms 的实际积分位置，`depth` 为策略收到的原始米制深度。几何审计复用 11 月 18 日的场景解析，在系统 Python 中运行 VTK。成功要求三架半径 0.10 m 的球体全程避开墙及机库、机间中心距至少 0.20 m、从真实开口穿过、最终 x 均超过 10 m。每种结局和首碰均可在相应 `episodes.csv` 中逐回合查看。

## 本次实测

| 评估集 | 累计交互 | 原生通过／100 | 严格通过／100 |
|---|---:|---:|---:|
| 开发 | 0 | 0 | 0 |
| 开发 | 1,013,760 | 92 | 75 |
| 开发 | 5,013,504 | 95 | 75 |
| 开发 | 10,008,576 | 99 | 91 |
| 留出 | 10,008,576 | 99 | 87 |

[检查点、耗时与两窗口比较](results/summary.json) · [500 个团队回合](results/episodes.csv) · [逐轮原生训练曲线](results/training.csv)

[完整实验报告](REPORT.md) · [复现问题记录](ISSUES.md) · [独立核验结果](results/verification.json)
