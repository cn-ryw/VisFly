# 复跑三机穿缝的配对评估

`run_pairs.py` 沿用实验时的 VisFly 官方 `Test`、确定性动作、种子重设和逐步记录。它先用种子 42 对照原始轨迹，再比较两个检查点在种子 1000–1099 的表现。`audit_pairs.py` 读取保存的轨迹，用 VTK 从场景模型重建墙体、测量机体扫过的空间，并复核每帧到障碍物或邻机的最近距离。

## 准备本地归档

推理使用已配置的 VisFly 环境、CUDA 及 `datasets/visfly-beta` 场景。几何核查可使用另一个装有 NumPy、VTK 和支持 `scipy.stats.binomtest` 的 SciPy 的 Python。本次使用系统 `/usr/bin/python3` 完成几何核查。仓库目录名沿用官方包名 `VisFly`。从仓库根目录执行，将路径换为本机实际位置。

```bash
export VISFLY_REPO_ROOT="$PWD"
export VISFLY_EXPERIMENT_ARCHIVE="/path/to/experiment-archive"
replay="$VISFLY_REPO_ROOT/experiments/crossing/2026-11-18/replay"
out="$VISFLY_EXPERIMENT_ARCHIVE/paired-replay"
```

| 归档根目录下的相对位置 | 文件和用途 |
| --- | --- |
| `crossing-corrected-20261118/train/` | `ppo_500736.zip`、`ppo_1000704.zip` 供推理。对应的两个同名 `.json` 供几何核查验证权重指纹。`training-config.json` 记录原代码和配置的哈希。 |
| `crossing-corrected-20261118/evaluation/` | `comparison.json` 记录权重哈希。两个 `steps_<交互数>/seed_42/` 和 `steps_1000704/seed_43/` 目录各含 `episode.json`、`trajectory.npz`，用于校准和已知碰撞回归。 |
| `crossing-paired-1000-1099/evaluation/` | 若直接审计原结果，此处需有两个 `steps_<交互数>/seed_<1000 至 1099>/` 目录树及每回合的 `episode.json`、`trajectory.npz`。仅重新推理时不需要这批原始轨迹。 |

## 运行

先复跑校准，期望 50 万档在第 51 步安全通过，百万档蓝机在第 41 步撞墙。程序同时逐项核对四类初始输入和 16 个逐回合数组。

```bash
conda activate visfly
python "$replay/run_pairs.py" --output-run "$out" --calibration-only
```

用同一个输出目录运行正式评估。完成的回合按权重和轨迹哈希复用。200 个回合跑完后执行几何审计。

```bash
python "$replay/run_pairs.py" --output-run "$out"
/usr/bin/python3 "$replay/audit_pairs.py" --input-run "$out" --output-dir "$out/audit"
```

已有完整原始轨迹时，以下命令直接离线审计历史归档，并将新结果单独保存。

```bash
/usr/bin/python3 "$replay/audit_pairs.py" --output-dir "$out/historical-audit"
```

## 校验记录

移植后实际重放了两个校准回合，校准记录与原件逐字节一致。另对历史 200 回合做了完整的独立几何审计，`pairs.json` 与 `summary.json` 也都与原件逐字节一致。四种配对结局按双方成功、仅 50 万档成功、仅百万档成功和双方失败依次为 0、2、4、94，双侧精确检验 p = 0.6875。

| 原始产物 | SHA-256 |
| --- | --- |
| `evaluation/calibration/verification.json` | `d384faf4ee688f27ad3d597126fd368c4a642f5eb577a7b55a33f32db405aec2` |
| `audit/pairs.json` | `23ea669a48943dd6714d11680e2087b20b0439e1be46c263f282ea6424b723d3` |
| `audit/summary.json` | `6931965f589465672a4c3f7ec36567f1eb83a3ef1f2069bca766b0b60a9e737d` |
