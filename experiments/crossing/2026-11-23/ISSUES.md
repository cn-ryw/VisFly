# 三机穿缝复现问题记录

本机原始归档目录为 `/home/ruan/research/VisFly-local-setup-20261113/paper-crossing-20261123`。

本次正式训练从 2026 年 11 月 23 日启动。以下保留实际触发条件、决定和证据。训练任务、奖励、动作与终止实现沿用历史提交。

| 问题 | 实测现象与处理 | 证据 |
|---|---|---|
| 历史版本和资产布局 | 历史脚本读 `datasets/spy_datasets/configs/garage_crossing`。复用已下载的 2024 年 9 月数据和已编译的同期 Habitat-Sim，作者训练机器的精确依赖组合未公开。本机版本和校验清单单独归档。 | 生效设置（本机 `train/effective.json`）、[历史入口](https://github.com/SJTU-ViSYS-team/VisFly/blob/7b0a4534b57051fd9b00ab4e561496f9abaacab1/examples/crossing/rl.py) |
| 场景注册器打印错误 | 官方全局目录登记的 `garage.glb` 和其他任务若干 navmesh 路径缺失。所选场景加载的是 `garage_v1.glb`。实际三机获得 64×64 深度观测、推进 8 个动力学子步、完成 PPO 更新、保存及重新加载。 | 初始测通日志（本机 `logs/initial-smoke.log`）、单场景完整更新（本机 `smoke/one-scene/complete.json`）、几何标定（本机 `smoke/geometry/calibration.json`） |
| 部分场景复位后的首个观测 | 两个并行场景强制其中一队到时间上限，另一队状态、深度、邻机观测保持正确。已复位队的三个返回状态仍是上一回合末帧。复现训练保留作者行为。定点评估每个种子显式 reset 且在终止时先保留轨迹。对学习效果的影响尚未单独测量。 | 双场景实测（本机 `smoke/runtime-check.json`）、[作者封装](https://github.com/SJTU-ViSYS-team/VisFly/blob/7b0a4534b57051fd9b00ab4e561496f9abaacab1/envs/multiDroneGymEnv.py) |
| 多场景退出时 OpenGL 中止 | 双场景校验写完结果后，原 `close()` 先关闭拥有共享上下文的第一个模拟器，进程退出码 134，报 `GL::Context::current(): no current context`。先按倒序关闭其他场景，再调用原 close。双场景复测和随后独立的双场景 PPO 更新（1,536 交互、完整权重）均退出码 0。已启动的正式训练进程沿用初次启动版本。其最终检查点及完整训练记录在清理之前落盘。 | 第一次退出（本机 `logs/check-runtime.log`）、修复后复测（本机 `logs/check-runtime-reverse.log`）、双场景 PPO 退出验证（本机 `logs/two-scene-close.log`） |
| 原生通过和几何安全可能不同 | 官方以三架机当前 x 均大于 10 米判断成功，原生距离查询还包含虚拟边界。原生机间查询中第一架到其余两架的距离使用一次矩阵范数。严验逐子步测量每一对球形机体。对原生结果与严验不一致的每个回合保留初始状态、原生碰撞与越界标志、首碰对象和时刻。 | [作者距离查询](https://github.com/SJTU-ViSYS-team/VisFly/blob/7b0a4534b57051fd9b00ab4e561496f9abaacab1/utils/sceneDatasets.py)、新旧判据对照（本机 `smoke/geometry/measured_smoke_a.json`） |
| 论文图只公布图像 | 从 arXiv v4 原始 PNG 提取图 7 三机黄色线。遮挡的最末 4.53 万步用最近可见值延展。图中未报告曲线平滑和奖励归一化的明细，本机原始回报单独作图。 | 取点方法和窗口值（本机 `reference/figure7-reference.json`）、原始曲线图（本机 `reference/train_curve.png`） |
| 历史样本仅存控制帧 | 用旧口径重放 200 个历史回合，两个检查点的结果与首碰逐项一致。新实验每 2.5 毫秒记录真实积分位置，颜色和录像用于定位，最终验收使用数值轨迹。 | 历史逐回合对照（本机 `smoke/geometry/historical_control_parity.json`）、回归测试（本机 `smoke/geometry/unittest.log`） |
| 连续复用评估场景导致首帧深度漂移 | 两档策略的 100 个同种子初始状态均相同，97 对首帧深度不相同。原版 reset 设置相机姿态，视觉刚体到 step 才复位。改为每个种子使用新建官方场景，重新运行 0 和约 101 万步开发集；100/100 对的状态、深度、目标、邻机观测哈希均匹配。留出集在修复时尚未运行。 | 首次采集证据（本机 `smoke/paired-reset-v1/README.md`）、修正后配对核查（本机 `evaluation/development/PPO_1013760/summary.json`） |
| 原生漏报机间碰撞 | 101 万检查点的 17 个原生成功、严验失败回合均为机间接触。15 个回合在控制帧的最近中心距仍不少于 10 cm；另外两个甚至低于 10 cm。种子 1064 的第 48 步，1、3 号机中心距 4.35 cm，原生距离分别为 1.021 和 0.995 m。静态查询确认两堵墙均在场，机体过墙与机间间距可单独区分。 | 实测探针（本机 `smoke/native-geometry/README.md`）、[几何复核](results/episodes.csv) |
| 未训练回合的虚拟边界 | 0 步策略 100 队全被原生判碰撞。其中 98 队的终止距离吻合 ENU x = −1 m 的虚拟后边界，余 2 队由邻机触发。95 队全程未碰真实几何且未完成穿缝，记为严验未完成。 | 逐点和逐队测量（本机 `smoke/native-geometry/summary.json`） |

正式训练实际完成 543 轮、10,008,576 条交互与 5,430 个 PPO 梯度训练轮次。末尾原版在打印两次权重保存成功、写完训练参数后，因释放多个图形上下文触发 SIGABRT。两份压缩包中的策略张量和 Adam 状态逐字节相同，最终权重可在独立进程中重新加载。修正只调整多场景资源的关闭顺序，已用双场景完整 PPO 更新验证。进程退出记录（本机 `logs/train-process-exit.txt`） · 最终两个权重对照（本机 `train/final-policy-parity.json`） · 完整训练记录（本机 `train/complete.json`）

开发集四个检查点已逐档完成。排队脚本在等待期间被修改，旧 Bash 进程在切换留出集前读到不完整引号并以 2 退出。核查当时磁盘尚无留出集输出后，直接从同一冻结权重恢复原定的 20000–20099 一轮留出评估。以后把要长时间运行的 shell 脚本复制为单次运行快照，再交给服务执行。排队日志（本机 `logs/evaluation-v2.log`） · 留出评估日志（本机 `logs/held-out-evaluation.log`）

相同的等待脚本改写也使最终报告队列在读取后续命令时退出。将已核对语法的汇总脚本复制为固定快照后，成功复用既有 500 条轨迹，完成曲线、录像、权重与逐文件核验、工作簿写入。训练和留出试飞均无需重跑。汇总快照（本机 `setup/as-launched/finalize-retry.sh`） · 恢复日志（本机 `logs/finalize-retry.log`） · 工作簿校验（本机 `workbook/installation.json`）
