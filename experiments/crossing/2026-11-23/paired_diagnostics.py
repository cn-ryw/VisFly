"""Identify when safe crossings change on matched starts and localize remaining contacts."""
import argparse
from collections import Counter
import csv
from itertools import combinations
import json
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry_audit import build_geometry
from replay_cases import COLORS


def read(p):
    return json.loads(p.read_text())


def as_rows(p):
    with (p/"episodes.csv").open() as f:
        return {int(r["seed"]):r for r in csv.DictReader(f)}


def focus_contact(row,geo):
    if not geo["first_contact"]:
        return None
    if row["first_contact_step"]=="0":
        return "initial"
    points=np.asarray(geo["first_contact_xyz"])[np.asarray(geo["first_contact"]["drones"])-1]
    x=float(np.mean(points[:,0]))
    if x < 7.8:
        return "approach"
    return "wall_slab" if x <= 8.2 else "exit"


def group_diagnostic(directory):
    rows=as_rows(directory)
    geo={d["seed"]:d for d in read(directory/"geometry.json")}
    outcome=Counter()
    overlap=Counter()
    phase=Counter()
    gap_min=None
    for seed,row in rows.items():
        dist=geo[seed]["min_wall_clearance_m"]
        gap_min=dist if gap_min is None else min(gap_min,dist)
        if geo[seed]["first_contact"]:
            phase[focus_contact(row,geo[seed])]+=1
        if row["native_outcome"]!="success" or row["strict_outcome"]=="success":
            continue
        outcome[row["first_contact_group"] or row["strict_outcome"]]+=1
        with np.load(directory/f"seed_{seed}.npz") as d:
            xyz=d["control_xyz"]
        at_frame=min(float(np.min(np.linalg.norm(xyz[:,a]-xyz[:,b],axis=1))) for a,b in combinations(range(3),2))
        m=geo[seed]["min_peer_m"]
        if at_frame < .10-1e-12:
            overlap["control_center_below_10cm"]+=1
        elif at_frame < .20-1e-12:
            overlap["control_center_between_10_and_20cm"]+=1
        elif m < .20-1e-12:
            overlap["only_between_controls_center_below_20cm"]+=1
        else:
            overlap["no_peer_contact"]+=1
    return {"native_success_strict_failure_first_contact":dict(outcome),"peer_range_among_these":dict(overlap),
            "all_first_contact_phases":dict(phase),"minimum_gap_wall_surface_clearance_m":gap_min}


def paired(a,b):
    ra,rb=map(as_rows,(a,b))
    labels=Counter()
    seeds={}
    for seed,row in ra.items():
        key=("S" if row["strict_outcome"]=="success" else "F")+("S" if rb[seed]["strict_outcome"]=="success" else "F")
        labels[key]+=1
        seeds.setdefault(key,[]).append(seed)
    n=labels["FS"]+labels["SF"]
    k=min(labels["FS"],labels["SF"])
    p=min(1.,2*sum(comb(n,i) for i in range(k+1))/2**n) if n else 1.
    return dict(earlier=a.name,later=b.name,strict_result_counts=dict(labels),
                changed_to_success=seeds.get("FS",[]),changed_to_failure=seeds.get("SF",[]),exact_paired_p=p)


def nearest(xyz):
    return np.min([np.linalg.norm(xyz[:,a]-xyz[:,b],axis=1) for a,b in combinations(range(3),2)],axis=0)


def draw_case(run, earlier, later, seed, geometry):
    aa,bb=[read(p/f"seed_{seed}.json") for p in (earlier,later)]
    assert aa["initial_observations_sha256"]==bb["initial_observations_sha256"]
    groups=[[v for v in read(p/"geometry.json") if v["seed"]==seed][0] for p in (earlier,later)]
    blobs=[np.load(p/f"seed_{seed}.npz") for p in (earlier,later)]
    try:
        n=min(d["control_xyz"].shape[0] for d in blobs)
        ydiff=np.abs(blobs[0]["control_xyz"][:n,:,1]-blobs[1]["control_xyz"][:n,:,1])
        index=np.argwhere(ydiff>=.10)
        if len(index):
            step,drone=map(int,index[0])
        else:
            step,drone=n-1,int(ydiff[-1].argmax())
        details=dict(seed=seed,first_y_difference_at_least_10cm_step=int(index[0,0]) if len(index) else None,
                     frame_shown=step,drone_number=drone+1,paired_initial_obs=aa["initial_observations_sha256"],
                     earlier=dict(checkpoint=earlier.name,native=aa["native_outcome"],strict=groups[0]["strict_outcome"]),
                     later=dict(checkpoint=later.name,native=bb["native_outcome"],strict=groups[1]["strict_outcome"]),
                     delta_y_m=float(ydiff[step,drone]),
                     roll_action_at_control=[float(d["action"][max(0,step-1),drone,1]) for d in blobs],
                     z_m=[float(d["control_xyz"][step,drone,2]) for d in blobs],
                     y_m=[float(d["control_xyz"][step,drone,1]) for d in blobs])
        plt.rcParams.update({"font.family":"Noto Sans CJK JP","axes.unicode_minus":False,"font.size":10})
        fig=plt.figure(figsize=(12,8.3))
        grid=fig.add_gridspec(3,4,height_ratios=[1.2,1,1],width_ratios=[1,1,1,1])
        map_axes=[fig.add_subplot(grid[0,:2]),fig.add_subplot(grid[0,2:])]
        peer_ax=fig.add_subplot(grid[1,:2]); roll_ax=fig.add_subplot(grid[1,2:])
        depths=[fig.add_subplot(grid[2,0:2]),fig.add_subplot(grid[2,2:])]
        xmin=min(-1.5,min(float(d["control_xyz"][...,0].min()) for d in blobs)-.5)
        xmax=max(14.,max(float(d["control_xyz"][...,0].max()) for d in blobs)+.5)
        ybound=max(3.,max(float(np.max(abs(d["control_xyz"][...,1]))) for d in blobs)+.5)
        for j,(p,d,check,ax,depthax) in enumerate(zip((earlier,later),blobs,groups,map_axes,depths)):
            c=d["control_xyz"]
            label=f'{int(p.name.split("_")[1]):,} 条交互'
            result={"success":"安全通过","collision":"碰撞","incomplete":"未完成","timeout":"超时","bypass":"绕行"}[check["strict_outcome"]]
            for a in range(3):
                ax.plot(c[:,a,0],c[:,a,1],c=COLORS[a],lw=1.6,label=f"{a+1} 号机")
                ax.plot(*c[step,a,:2],"o",color=COLORS[a],ms=3)
            ax.axvline(10,c="#947739",ls="--",lw=1)
            for y0,y1 in ((-5,geometry.dimensions[1]),(geometry.dimensions[2],5)):
                ax.fill_between([7.8,8.2],y0,y1,color="#596871")
            ax.set(xlim=(xmin,xmax),ylim=(-ybound,ybound),xlabel="x（m）",ylabel="y（m）",title=f"{label} · {result}")
            ax.legend(loc="lower left",ncol=3,frameon=False,fontsize=8)
            series="#687682" if j==0 else "#097d83"
            ts=np.arange(len(c))
            peer_ax.plot(ts,nearest(c),color=series,label=label)
            roll_ax.plot(ts[1:],d["action"][:,drone,1],color=series,label=label)
            depthax.imshow(d["depth"][step,drone,0],cmap="gray_r",vmin=0,vmax=10,interpolation="nearest")
            depthax.set_title(f"{label}  ·  {drone+1} 号机实际深度  |  第 {step} 步")
            depthax.axis("off")
        peer_ax.axhline(.20,color="#b44438",lw=1,ls="--")
        peer_ax.set(xlabel="控制步",ylabel="最小机间中心距（m）")
        roll_ax.set(xlabel="控制步（动作施加后）",ylabel=f"{drone+1} 号机归一化滚转角速度指令")
        for ax in (peer_ax,roll_ax):
            ax.axvline(step,c="#8a8276",ls=":",lw=1)
            ax.legend(frameon=False,fontsize=8)
        fig.suptitle(f"同一种子 {seed} 的两次闭环飞行",fontsize=18)
        fig.text(.5,.015,"轨迹圆点与深度图取同一控制时刻。虚线 0.20 m 表示两架半径 0.10 m 飞机的接触界限。",ha="center",fontsize=10)
        fig.tight_layout(rect=(0,.035,1,.96))
        fig.savefig(run/"figures/matched-trajectory-depth.png",dpi=150)
        plt.close(fig)
        return details
    finally:
        for d in blobs:
            d.close()


def main():
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("run",type=Path)
    run=arg.parse_args().run.resolve()
    protocol=read(run/"setup/protocol.json")
    dev=run/"evaluation/development"
    cohorts=[dev/f"PPO_{s}" for s in protocol["post_update_checkpoints"] if (dev/f"PPO_{s}/summary.json").exists()]
    summaries={p.name:group_diagnostic(p) for p in cohorts}
    hold=run/"evaluation/held-out/PPO_10008576"
    if (hold/"summary.json").is_file():
        summaries["held-out"]=group_diagnostic(hold)
    pairs=[paired(a,b) for a,b in combinations(cohorts,2)]
    case=None
    # Use the nearest earlier development checkpoint with a failure on a later successful seed.
    if len(cohorts)>1:
        choice=next((p for p in reversed(pairs) if p["later"]==cohorts[-1].name and p["changed_to_success"]),None)
        if choice:
            geom=build_geometry(run/"source/datasets/spy_datasets")
            case=draw_case(run,dev/choice["earlier"],dev/choice["later"],choice["changed_to_success"][0],geom)
    (run/"figures/paired-diagnostics.json").write_text(json.dumps(dict(groups=summaries,pairs=pairs,matched_case=case),
                                                                 ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
