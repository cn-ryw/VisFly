"""Compare logged original PPO metrics with predeclared Figure 7 pixel windows."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from figure7_reference import window_mean


def run_comparison(run):
    train = run/"train"
    with (train/"training.csv").open() as f:
        rows = list(csv.DictReader(f))
    progress = json.loads((train/"progress.json").read_text())["agent_steps"]
    rows = [r for r in rows if int(r["agent_steps"]) <= progress]
    x = np.array([int(row["agent_steps"]) for row in rows])
    success = np.array([float(r["native_success_rate"]) for r in rows])
    reward = np.array([float(r["native_reward_mean"]) for r in rows])
    tb = EventAccumulator(str(train/"PPO_paper_crossing_1"),size_guidance={"scalars":0}).Reload()
    for tag, recorded in [("rollout/success_rate", success), ("rollout/ep_rew_mean", reward)]:
        scalars = tb.Scalars(tag)
        # PPO writes TensorBoard just after the callback flushes each CSV row.
        common = min(len(scalars),len(x))
        assert common >= len(x)-1
        assert np.array_equal(x[:common], [s.step for s in scalars[:common]])
        np.testing.assert_allclose(recorded[:common], [s.value for s in scalars[:common]], rtol=1e-6, atol=1e-7)
    if (train/"complete.json").exists():
        assert common == len(x) and int(x[-1]) == json.loads((train/"complete.json").read_text())["agent_steps"]
    protocol = json.loads((run/"setup/protocol.json").read_text())
    ref = json.loads((run/"reference/figure7-reference.json").read_text())
    comp = []
    for low, high in protocol["figure_comparison"]["windows_agent_steps"]:
        paper = ref["windows"]["success_rate"]["means"][f"{low}-{high}"]
        achieved = window_mean(x, success, low, high) if x[0] <= low and x[-1] >= high else None
        cutoff = paper-protocol["figure_comparison"]["permitted_shortfall_absolute"]
        comp.append(dict(low=low,high=high,paper=paper,required=cutoff,actual=achieved,
                         difference_points=(achieved-paper)*100 if achieved is not None else None,
                         meets_threshold=achieved>=cutoff if achieved is not None else None))
    results = dict(agent_steps=int(x[-1]),rows=len(x),tensorboard_rows_verified=common,
                   final_sample_rate=float(success[-1]),windows=comp)
    out = run/"figures"
    out.mkdir(exist_ok=True)
    (out/"curve-windows.json").write_text(json.dumps(results,indent=2)+"\n")
    paper_rows = list(csv.DictReader((run/"reference/figure7-digitized.csv").open()))
    paper_rows = [r for r in paper_rows if r["series"]=="success_rate"]
    paper_x = np.array([float(r["step"]) for r in paper_rows])/1e6
    paper_y = np.array([float(r["plotted_value"]) for r in paper_rows])*100
    plt.rcParams.update({"font.family":"Noto Sans CJK JP","axes.unicode_minus":False,"font.size":11})
    fig, axes = plt.subplots(2,1,figsize=(10.6,7.8),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    ax, reward_ax = axes
    ax.plot(x/1e6,100*success,color="#287a91",lw=1.1,label="本机原生 PPO · 每轮日志")
    ax.plot(paper_x,paper_y,color="#ad6b07",lw=2,label="论文图 7 · Swarm Crossing 数字化")
    for item in comp:
        for panel in axes:
            panel.axvspan(item["low"]/1e6,item["high"]/1e6,color="#d0b986",alpha=.18,zorder=0)
        ax.hlines(100*item["required"],item["low"]/1e6,item["high"]/1e6,colors="#8b5848",linestyles="dashed",lw=1.5)
    ax.set(ylim=(-2,103),ylabel="原生训练成功率（%）")
    ax.legend(loc="lower right",frameon=False)
    reward_ax.plot(x/1e6,reward,color="#373e50",lw=1.25)
    reward_ax.set(xlabel="累计智能体交互（百万条）",ylabel="本机原始回合回报",xlim=(0,10.08))
    for panel in axes:
        panel.grid(axis="y",color="#d5dce0",lw=.7)
        panel.spines[["right","top"]].set_visible(False)
    fig.suptitle("三机车库穿缝  |  同期官方 PPO 与论文训练曲线",fontsize=17,x=.10,ha="left")
    fig.text(.10,.025,"阴影为约定的比较窗口，虚线为验收下限。论文奖励纵轴的换算未公开。",fontsize=10,color="#49535b")
    fig.tight_layout(rect=(.015,.055,.995,.95))
    fig.savefig(out/"training-vs-figure7.png",dpi=165)
    plt.close(fig)
    print(json.dumps(results))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    run_comparison(parser.parse_args().run.resolve())
