# 安全起点与机间碰撞终止

这组实验检验在三机车库穿缝任务中，将飞行中的机间接触加入训练终止条件，能否提高全队安全通过率。训练直接调用论文同期的 PPO 示例。场景、深度观测、邻机观测、动作、网络、奖励函数及训练超参数都沿用已验收基线。[原版训练入口](https://github.com/SJTU-ViSYS-team/VisFly/blob/7b0a4534b57051fd9b00ab4e561496f9abaacab1/examples/crossing/rl.py) · [基线报告](../2026-11-23/REPORT.md) · [本轮冻结协议](protocol.json)

| 组别 | 起飞约束 | 训练终止 |
|---|---|---|
| 安全起点对照组 `control` | 三对中心距均至少 20 cm | 原版 |
| 机间终止组 `peer` | 与对照组相同 | 新增任意一对中心距低于 20 cm 时全队结束 |

首次起飞和自动复位均调用原版生成器。三对距离中有一对不合格就重新抽取这支队伍。每次控制持续 20 ms，在 8 个实际动力学子步内分别检查两机相对运动线段的最小长度。恰好相切可继续飞行。接触与成功或超时同时出现时，该队记为碰撞，沿用原奖励公式中失败回合的处理。

## 训练与评估

本机旧训练归档是 `/home/ruan/research/VisFly-local-setup-20261113/paper-crossing-20261123`，本轮是同级目录 `peer-clearance-20261125`。以下命令用于在**新的空目录**复现实验。历史源码、资产及渲染器安装步骤见[基线说明](../2026-11-23/README.md)。

```bash
scripts=/home/ruan/research/VisFly/experiments/crossing/2026-11-25
reference=/home/ruan/research/VisFly-local-setup-20261113/paper-crossing-20261123
run=/path/to/new-experiment
visfly_python=/home/ruan/miniconda3/envs/visfly/bin/python
mkdir -p "$run/setup" "$run/train"
ln -s "$reference/source" "$run/source"
ln -s "$reference/setup/renderer-lib" "$run/setup/renderer-lib"
cp "$scripts/protocol.json" "$run/setup/protocol.json"
export PYTHONPATH="$run/setup/renderer-lib" VISFLY_PAPER_RUN="$run"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$visfly_python" "$scripts/test_peer_env.py" -v
"$visfly_python" "$scripts/run_training.py" "$run" control
"$visfly_python" "$scripts/run_training.py" "$run" peer
/usr/bin/python3 "$scripts/run_evaluation.py" "$run" "$reference" --visfly-python "$visfly_python"
```

两组均用种子 42 从零训练。每轮采样 24 个场景 × 3 架飞机 × 256 个控制步，每组完成 543 轮。原版冻结权重、对照权重、机间终止权重均在相同的合法起点和机间终止环境中评估。每种子计一个团队回合，采用确定性动作；逐子步独立核对机间距离、墙体、机库和真实开口。只有三机全部安全穿过开口并越过终点才算成功。

开发集为 1000–1099。先用开发集严格通过数挑选两组各自表现最好的非零检查点，同分时取首碰为机间接触较少者，再同分取较晚者。把入选权重的 SHA-256 写入 `selection.json` 后，才评估 30000–30099 的新留出集。每个训练末权重均接受留出评估。每项测量的 `episodes.csv` 可按种子对应到首碰子步、轨迹、动作、三路真实深度和模型路径。

## 核验与回看

```bash
"$visfly_python" "$scripts/verify_training.py" "$run"
/usr/bin/python3 "$scripts/summarize_pairs.py" "$run"
/usr/bin/python3 "$scripts/present_results.py" "$run"
/usr/bin/python3 "$scripts/archive_results.py" "$run" "$run/results"
```

`training-verification.json` 逐轮核对原始团队事件、TensorBoard、合法复位、优化器更新及终末权重。`paired/paired-metrics.csv` 列出同种子的成功与机间碰撞变化。`figures/paired-case.png` 对齐同一起点下的两组轨迹，并画出实际机间中心距。工作簿导出脚本用带 openpyxl 和 lxml 的 Python 运行，参数为运行目录、原工作簿和新副本路径。
