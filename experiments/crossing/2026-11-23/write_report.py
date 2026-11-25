"""Publish measured metrics and their traceable evidence once the full run is verified."""
import argparse
import json
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def main():
    args=argparse.ArgumentParser(description=__doc__)
    args.add_argument("run",type=Path)
    run=args.parse_args().run.resolve()
    protocol=read(run/"setup/protocol.json")
    verification=read(run/"verification.json")
    paths=[run/f"evaluation/development/PPO_{n}" for n in protocol["post_update_checkpoints"]]
    paths.append(run/"evaluation/held-out/PPO_10008576")
    results=[read(p/"summary.json") for p in paths]
    final,hold=results[-2:]
    ck=verification["archived_weights"]
    times={w["agent_steps"]:w["elapsed_seconds"] for w in ck}
    complete=read(run/"train/complete.json")
    windows=read(run/"figures/curve-windows.json")["windows"]
    diag=read(run/"figures/paired-diagnostics.json")
    held_wins=hold["counts"]["strict"].get("success",0)
    chart_pass=all(w["meets_threshold"] for w in windows)
    hold_pass=held_wins>=protocol["held_out"]["required_strict_successes_of_100"]
    finalwins=final["counts"]["strict"].get("success",0)
    lines=["# VisFly 三机车库穿缝复现报告","","2026 年 11 月 23 日启动，11 月 24 日完成。",
           "",f"原版训练曲线两个窗口{'均已达标' if chart_pass else '未全部达标'}。最终冻结权重在 100 个独立留出起点，作者环境通过 **{hold['counts']['native'].get('success',0)} 队**，三机全程安全穿缝 **{held_wins} 队**。本次约定的独立验收线为 80/100，实测{'达到' if hold_pass else '低于'}该线。[验收约定](setup/protocol.json) · [留出明细](evaluation/held-out/PPO_10008576/episodes.csv)",
           "","## 实测结果","",
           "| 评估集 | 累计智能体交互 | 原生通过 | 旧验收通过 | 严验通过 | 严验碰撞 | 超时 | 绕行 | 未完成 | 平均控制步 | 训练耗时 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    metrics=[]
    for s,p in zip(results,paths):
        count=s["counts"]; strict=count["strict"]; st=s["agent_steps"]
        split="留出" if s["split"]=="held-out" else "开发"
        label=f"[{split}]({p.relative_to(run)}/episodes.csv)"
        line=[label,f"{st:,}",count["native"].get("success",0),count["legacy"].get("success",0),
              *[strict.get(k,0) for k in ("success","collision","timeout","bypass","incomplete")],
              f"{s['mean_control_steps']:.2f}",f"{times[st]/60:.2f} 分"]
        lines.append("| "+" | ".join(map(str,line))+" |")
        metrics.append(dict(split=s["split"],steps=st,counts=count,episodes=100,mean_controls=s["mean_control_steps"],
                            training_seconds=times[st],sha256=s["checkpoint_sha256"],strict_wilson95=s["strict_success_wilson95"]))
    lo,hi=hold["strict_success_wilson95"]
    lines.extend(["","每行分母均为 100 个团队。原生按作者定义的三机越线结局计数。旧验收对照此前控制帧口径。严验使用每个 20 ms 控制周期内的 8 个实测积分子步，沿相邻子步扫掠半径 10 cm 的机体，要求不碰实物、机间中心距至少 20 cm、全队从真实开口穿过且最终 x 均大于 10 m。未完成表示原生先终止，未测出实物或机体碰撞且三机尚未完成穿缝。",
                  "",f"四个开发检查点共用种子 1000–1099，100 组首帧状态、深度、目标、邻机观测逐字节相同。留出使用种子 20000–20099，100 组起点均与开发起点不同。按团队回合计算，留出严格成功率的 Wilson 95% 区间为 {lo:.1%}–{hi:.1%}。范围限于同一车库和一次训练种子。[数据完整性](verification.json)",
                  "","## 与论文图 7 的对照","","| 训练窗口（交互） | 论文读图均值 | 本机原生训练均值 | 差值 | 约定下限 |","|---|---:|---:|---:|---:|"])
    for v in windows:
        lines.append(f"| {v['low']:,}–{v['high']:,} | {v['paper']:.2%} | {v['actual']:.2%} | {v['difference_points']:+.2f} 个百分点 | {v['required']:.2%} |")
    lines.extend(["",f"本机从零连续完成 {complete['rollouts']} 轮 PPO 采样，共 {complete['agent_steps']:,} 条智能体交互，相当于 {complete['agent_steps']//3:,} 个团队控制步。24 个场景、每场 3 机，种子 42。耗时 {complete['elapsed_s']/3600:.2f} 小时。论文 IV-B 报告约 1.1 小时。作者原生日志按最近 100 个智能体结局计算，一支队伍贡献三个相同的成功标志。独立试飞以一支队伍计一次。[实测曲线](figures/training-vs-figure7.png) · [窗口统计](figures/curve-windows.json) · [论文 IV-B 与图 7](https://arxiv.org/html/2407.14783v4#S4.SS2)",
                  "","论文原图的最后约 4.53 万步被边框遮挡，采用末个可见值延展。作者未公开生成图 7 的逐回合日志、准确绘图平滑和奖励缩放、训练权重及完整提交组合。本机结果可与公开曲线比较。所用 2024 年 9 月官方代码、644 个数据文件、渲染器来源与实际导入二进制均已核查。[取点细节](reference/figure7-reference.json) · [来源和依赖](setup/SOURCES.md)",
                  "","## 失败原因与训练量","",
                  f"同一版本和训练种子下，开发集严验随四个检查点依次为 {'、'.join(str(r['counts']['strict'].get('success',0))+'/100' for r in results[:4])}。从零步到约 101 万步已经学会穿越车库；最终的原生与安全差距可用冻结轨迹直接核对。"])
    middle=next(x for x in diag["pairs"] if x["earlier"]=="PPO_1013760" and x["later"]=="PPO_5013504")
    finalpair=next(x for x in diag["pairs"] if x["earlier"]=="PPO_5013504" and x["later"]=="PPO_10008576")
    lines.extend(["",f"相同开发起点从约 101 万到 501 万交互，{len(middle['changed_to_success'])} 队由败转胜，{len(middle['changed_to_failure'])} 队由胜转败。再从 501 万到 1,001 万交互，{len(finalpair['changed_to_success'])} 队由败转胜，{len(finalpair['changed_to_failure'])} 队由胜转败，安全通过净增 {len(finalpair['changed_to_success'])-len(finalpair['changed_to_failure'])} 队。这一次连续训练中，延长预算改善了后段的安全通过数。[逐种子配对](figures/paired-diagnostics.json)"])
    for s,desc in [(final,"最终开发"),(hold,"独立留出")]:
        misses=s["native_success_strict_failure_first_group"]
        phases=s["first_contact_group"]
        labels={"drone":"机间","wall":"穿缝墙","stage":"机库或地面","drone+wall":"机间和墙体"}
        fmt=lambda m:"、".join(f"{labels.get(k,k)} {n} 队" for k,n in m.items()) or "0 队"
        lines.append(f"\n{desc}的全部首次接触为{fmt(phases)}。其中原生通过但严验失败的首次接触为{fmt(misses)}。初始帧已经发生接触的回合有 {len(s['initial_contact_seeds'])} 队。")
    lines.extend(["","101 万步的 25 个严验碰撞均先碰到邻机，整批 100 条轨迹对两侧墙的最小机体净空为 15.62 cm。17 个原生成功、严验失败中，15 队在全部控制帧的最小中心距仍不少于 10 cm，低于两机半径之和的安全距离。种子 1064 第 48 步，1、3 号机中心仅距 4.35 cm，原生却分别报告约 1.02 和 1.00 m。历史代码把第一架机到其余两架的位置差一起求矩阵范数，因此可能漏掉最近的邻机。[实测距离与轨迹](smoke/native-geometry/README.md)",
                  "","未训练时的失败机制不同。0 步的 98 队在后方虚拟边界附近触发原生碰撞，余 2 队由邻机触发。严验找到 5 队发生机间接触，95 队未完成。历史修正配置的 1,000,704 交互旧验收为 4/100。它与本次配置同时存在多个差异，即使种子编号相同，也只有 34/100 个初始位置逐项一致，不能据此估计某个单项改动的因果作用。[历史条件核查](reference/historical-starts-comparison.json) · [复现问题记录](ISSUES.md)"])
    lines.extend(["","| 约百万交互时的条件 | 11 月 18 日修正配置 | 本次论文同期公开示例 |",
                  "|---|---|---|","| 并行车库及每场机数 | 1 × 3 | 24 × 3 |",
                  "| 控制周期／动力学步长 | 30 ms／30 ms | 20 ms／2.5 ms |",
                  "| 显式动作排队 | 2 个控制步（60 ms） | 同一控制步执行新指令 |",
                  "| PPO 每轮交互／小批大小／训练轮数 | 768／256／5 | 18,432／768／10 |",
                  "| 开发集旧口径安全通过 | 4/100 | 90/100 |",
                  "","相近交互预算下已有明显差异，旧版结果的原因不能单独归为训练量。比较同一训练运行的四个权重，才固定了上述配置。"])
    failures=hold["counts"]["strict"].get("collision",0)
    initial_count=len(hold["initial_contact_seeds"])
    peer_ranges=diag["groups"]["held-out"]["peer_range_among_these"]
    missed=sum(hold["native_success_strict_failure_first_group"].values())
    lines.extend(["",f"最终留出集首次接触的 {failures} 队中，{initial_count} 队在初始采样时两机已相距不足 20 cm，另 {failures-initial_count} 队在飞行中接触。{missed} 个原生成功、严验失败的回合中，{peer_ranges.get('control_center_between_10_and_20cm',0)} 队控制帧最近中心距在 10–20 cm 之间，{peer_ranges.get('control_center_below_10cm',0)} 队曾低于 10 cm。留出成绩按预先冻结的入组规则保留全部 100 个种子。[留出逐子步审计](evaluation/held-out/PPO_10008576/geometry.json)"])
    case=diag["matched_case"]
    if case:
        lines.extend(["",f"同种子 {case['seed']} 在 {case['earlier']['checkpoint']} 严验失败，在 {case['later']['checkpoint']} 安全通过。对齐的俯视轨迹、机间距离、滚转动作、机载深度来自同一初始观测。两条轨迹首次出现至少 10 cm 横向位置差的控制步为 {case['first_y_difference_at_least_10cm_step']}。[配对对照图](figures/matched-trajectory-depth.png) · [四档逐种子对照](figures/paired-diagnostics.json)"])
    lines.extend(["","## 可复用的基线","",
                  "训练时保持作者的策略、奖励、终止条件和多场景复位行为。确定性评估使用 CPU 策略推理，每个种子使用新场景。首次复用同一场景时，两个检查点有 97/100 个首帧深度哈希不一致；新建场景后所有开发检查点均通过核对。[问题复盘](ISSUES.md) · [启动配置](train/effective.json)",
                  "","| 交互数 | 冻结权重及 SHA-256 |","|---:|---|"])
    for v in ck:
        st=v["agent_steps"]; lines.append(f"| {st:,} | [PPO_{st}.zip](train/PPO_{st}.zip)  `{v['sha256']}` |")
    for r in read(run/"figures/final-replays.json"):
        name="真正安全穿缝" if r["geometry_outcome"]=="success" else "严格验收失败"
        lines.append(f"\n最终策略{ name }的开发种子为 {r['seed']}。[俯视、深度和净空录像](figures/{Path(r['video']).name})")
    official=run/"figures/official-final-replays.json"
    if official.is_file():
        for v in read(official):
            lines.append(f"\n种子 {v['seed']} 的原始策略位置和姿态也可在作者的 Habitat 固定俯视相机中回放。[官方场景俯视录像](figures/{Path(v['video']).name})")
    lines.extend(["","下一项实验优先检验**训练中的机间终止条件**。以本次原版训练为对照，增加一条团队终止规则。任意一对机体的中心距在子步内低于 20 cm，立即判团队碰撞。按同一训练种子、场景、PPO 配置、奖励函数和预算从零训练。比较相同开发起点的机间首碰数和严格安全通过数，并为新方法另留未用过的最终评估种子。是否改善学习效果需由这次对照给出。"])
    lines.extend(["","## 归档和退出核验","",
                  "正式训练在写完最终权重后，释放旧版 Habitat 多场景图形上下文时触发 SIGABRT。作者入口和记录回调保存的最终 ZIP 逐字节相同，四份检查点均已重新加载/核验，500 个团队回合已完成。倒序释放其他场景后，双场景完整 PPO 验证退出码为 0。留出评估和最终资料汇总曾因排队脚本在运行时被改写而停在阶段交界，已从既有数据恢复并核验。[退出记录](logs/train-process-exit.txt) · [最终权重字节核对](train/final-policy-parity.json) · [完整问题记录](ISSUES.md)",
                  "","工作区《VisFly三机穿越基线实验表.xlsx》新增“论文同期三机基线”页，含 500 条逐回合实测和原始文件位置。原有 5 页数据与 295 个公式已核验。[表格写入记录](workbook/installation.json)"])
    (run/"REPORT.md").write_text("\n".join(lines)+"\n")
    (run/"baseline-results.json").write_text(json.dumps(dict(training=complete,windows=windows,cohorts=metrics,
               paper_curve_threshold_met=chart_pass,independent_strict_threshold_met=hold_pass,
               archive=str(run)),ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
