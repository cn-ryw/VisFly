# cluttered_flight PPO 实验记录

2026 年 11 月 18 日完成了 32 步流程验证、100 万步 PPO 基线，以及从基线恢复到 **2,506,752 步**的续训。最终模型在训练车库到达 **12/15**，在障碍更密的留出车库到达 **8/15**。下一轮可从这个检查点开始，比较课程训练和同等步数的原车库续训。

## 训练和评估

- 32 步快速版用 1 架无人机，`n_steps=32`、`batch_size=32`、`n_epochs=1`。它验证了训练、模型加载、TensorBoard 和仿真回放流程。固定起点评估中 30 次均未到达，25 次在高处接触上方障碍。
- 正式训练用 RTX 4060 Laptop GPU，随机种子 `42`，48 架并行无人机，每架每轮采样 256 步，`batch_size=512`，每轮采样后做 10 轮 PPO 更新。输入是 64 × 64 深度图、飞行状态和目标位置。基线完成 **1,007,616 步**，续训恢复了策略和优化器，再采集 **1,499,136 步**。
- 每个检查点用确定性策略在两个车库各飞 15 次。两个布局沿用同一组坐标，`x` 固定为 1.0 米，`y` 取 −1.8、−0.9、0、0.9、1.8 米，`z` 取 0.65、1.50、2.35 米。评估报告到达、碰撞、超时及其他终止，并记录飞行用时和每回合最近障碍距离。统计来自一个训练种子和两个固定布局。

| 累计训练步数 | 训练车库（44 个障碍） | 留出车库（84 个障碍） |
| ---: | ---: | ---: |
| 32 | 0 / 13 / 0 / 2 | 0 / 15 / 0 / 0 |
| 258,048 | 0 / 15 / 0 / 0 | 0 / 15 / 0 / 0 |
| 503,808 | 1 / 14 / 0 / 0 | 0 / 15 / 0 / 0 |
| 761,856 | 5 / 10 / 0 / 0 | 1 / 14 / 0 / 0 |
| 1,007,616 | 3 / 12 / 0 / 0 | 1 / 14 / 0 / 0 |
| 1,511,424 | 10 / 5 / 0 / 0 | 4 / 11 / 0 / 0 |
| 2,002,944 | 12 / 3 / 0 / 0 | 3 / 12 / 0 / 0 |
| **2,506,752** | **12 / 3 / 0 / 0** | **8 / 7 / 0 / 0** |

表格每格依次为到达、碰撞、超时、其他终止次数。最终模型到达的回合中，训练车库和留出车库的用时中位数分别为 **1.605 秒**和 **1.770 秒**，最近障碍距离中位数分别为 **0.315 米**和 **0.517 米**。

| 实验 | 配置与结果 | 脚本快照 |
| --- | --- | --- |
| 100 万步基线及快速版复评 | [训练配置](baseline_1m_seed42_20261118_203054_vHNu2a/training.json) · [汇总](baseline_1m_seed42_20261118_203054_vHNu2a/summary.json) · [150 条回合记录](baseline_1m_seed42_20261118_203054_vHNu2a/episodes.csv) | [训练](baseline_1m_seed42_20261118_203054_vHNu2a/train.py) · [复评](baseline_1m_seed42_20261118_203054_vHNu2a/evaluate.py) · [回放](baseline_1m_seed42_20261118_203054_vHNu2a/view.py) |
| 250 万步续训 | [续训配置](continued_2p5m_seed42_20261118_211011_tKUqfO/continuation.json) · [汇总](continued_2p5m_seed42_20261118_211011_tKUqfO/summary.json) · [90 条回合记录](continued_2p5m_seed42_20261118_211011_tKUqfO/episodes.csv) | [续训](continued_2p5m_seed42_20261118_211011_tKUqfO/continue_ppo.py) · [复评](continued_2p5m_seed42_20261118_211011_tKUqfO/assess.py) · [回放](continued_2p5m_seed42_20261118_211011_tKUqfO/show.py) · [数值修正](continued_2p5m_seed42_20261118_211011_tKUqfO/stable_ppo.py) |

## 数值恢复

- 在 **2,015,232 步**的更新中策略产生非有限值。诊断重跑在 **2,113,536 步**记录到对数概率比最大值约 **468.9**，指数溢出导致反向传播产生非有限梯度。随后在 **2,211,840 步**遇到梯度范数溢出。
- 最后从 **2,002,944 步**的有效检查点恢复，在对数域计算 PPO 裁剪目标，并等价地缩放损失和裁剪梯度。恢复期间每 8 轮采样保存一次恢复点。最终检查点通过 ZIP 校验，策略和优化器张量均为有限值。
- 稳定化脚本依赖当时的 `PPO.train` 源码结构。归档保留脚本快照供复查。两个配置中的 `repo_revision` 是写入配置时的 Git HEAD，当时的工作树已含本地改动，所以该字段不能单独确定完整代码状态。

## 本机检查点与回放

最终检查点相对仓库根目录的路径如下。

```text
exps/examples/saved/cluttered_flight/continued_2p5m_seed42_20261118_211011_tKUqfO/PPO_2506752.zip
```

SHA-256 为 `072d3a015d6101de8610afcbcb008919eecf3eeb74c1666765b4574046a55d22`。两个运行目录中的正式检查点、有效 TensorBoard 日志和原始训练日志，以及快速版的 `PPO_quickstart_1.zip` 和对应 TensorBoard 日志，均保存在本机。

在有场景资产和 `visfly` 环境的副本中，将本归档下两个同名目录的脚本放到 `exps/examples/saved/cluttered_flight/` 中对应的运行目录。回放需在对应运行目录放好模型 ZIP；续训复评还会读取基线目录的 `summary.json`。运行脚本会向所在运行目录写结果，复跑时使用独立的实验副本。已有本机运行目录可从 `VisFly/` 的父目录直接执行以下命令。

```bash
conda activate visfly
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
run="VisFly/exps/examples/saved/cluttered_flight/continued_2p5m_seed42_20261118_211011_tKUqfO"
python "$run/show.py" --steps 2506752 --case train-02
```

`train-02` 在第 52 步到达，用时 1.56 秒，终点距目标 0.345 米。桌面窗口按 **R** 重播，按 **Esc** 关闭。复评可在独立副本中运行 `python "$run/assess.py"`，它会读取三个续训检查点。重新跑整轮基线时，可在独立副本中用基线目录的 `train.py --batch-size 512`；续训脚本 `continue_ppo.py train` 会根据运行目录的 `continuation.json` 恢复进度，归档版 JSON 记录的是已经完成的这次运行。

## 下一轮对照

从 **2,506,752 步**的同一模型和优化器出发，给原 44 障碍车库续训和从稀疏到密集的课程训练相同的新增步数。两条路线继续用这 30 个固定起点比较到达和碰撞。最终模型在留出车库的 7 次碰撞中，有 4 次在离地不足 0.10 米时结束，其余 3 次为侧向碰撞。复评时可单独看低高度起点的结果。

## 废弃产物清理记录

以下路径相对 `exps/examples/saved/cluttered_flight/`。将这些文件移入系统回收站后，仍可用保留的模型和日志复查训练及评估。

| 路径 | 清理原因 |
| --- | --- |
| `test/PPO_quickstart_1/0.png`、`1.png`、`flight-replay.mp4` | 快速版的飞行状态图、最近障碍距离图和旧视频均可由保留的模型重新生成。 |
| `baseline_1m_seed42_20261118_203054_vHNu2a/eval_preflight.log` | 预检仅跑了两个场景各两个起点，四次均碰撞。正式复评已覆盖这些起点。 |
| `continued_2p5m_seed42_20261118_211011_tKUqfO/tensorboard/PPO_continued_0/events.out.tfevents.1795008772.AI.223166.2` | 首次失败，记录到 2,015,232 步。 |
| `continued_2p5m_seed42_20261118_211011_tKUqfO/tensorboard/PPO_continued_0/events.out.tfevents.1795008885.AI.251863.0` | 诊断重跑，记录到 2,113,536 步。 |
| `continued_2p5m_seed42_20261118_211011_tKUqfO/tensorboard/PPO_continued_0/events.out.tfevents.1795009141.AI.256833.0` | 对数域裁剪重跑，记录到 2,211,840 步。 |
