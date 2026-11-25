"""Present completed, paired evaluations of the 20 cm peer-termination experiment."""
import argparse
import csv
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


ARMS = ("control", "peer")
ARM = {"baseline": "历史末权重", "control": "对照组", "peer": "机间终止组"}
SPLIT = {"development": "开发集", "holdout": "留出集"}
OUTCOME = {"success": "安全穿缝", "collision": "碰撞", "timeout": "超时",
           "bypass": "绕行", "incomplete": "未完成"}
COLOR = {"baseline": "#59646c", "control": "#186296", "peer": "#ab4923"}
DRONE_COLORS = ("#b53b3b", "#19845e", "#3068a3")
CONTACT_M = 0.20
SUBSTEP_S = 0.0025


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@dataclass
class Cohort:
    folder: Path
    summary: dict
    records: dict


def read_cohort(folder):
    summary = read_json(folder / "summary.json")
    records = {}
    for row in csv_rows(folder / "episodes.csv"):
        for name in ("seed", "control_steps", "peer_first", "common_peer_stop", "first_control_step", "first_substep"):
            row[name] = int(row[name]) if row[name] else None
        for name in ("initial_min_peer_m", "min_peer_center_m", "min_wall_surface_clearance_m"):
            row[name] = float(row[name])
        if row["seed"] in records or row["strict_outcome"] not in OUTCOME:
            raise ValueError(f"Invalid evaluated episode in {folder}")
        records[row["seed"]] = row
    checks = (len(records) == summary["episodes"],
              set(records) == set(range(summary["first"], summary["last"] + 1)),
              sum(r["strict_outcome"] == "success" for r in records.values()) == summary["strict_successes"],
              sum(r["peer_first"] for r in records.values()) == summary["peer_first"],
              min(r["initial_min_peer_m"] for r in records.values()) >= CONTACT_M)
    if not all(checks):
        raise ValueError(f"Episode records do not match the audit in {folder}")
    return Cohort(folder, summary, records)


def load_experiment(run):
    protocol_file = run / "setup/protocol.json"
    spec = read_json(protocol_file)
    frozen = read_json(run / "selection.json")
    parity = read_json(run / "setup/initialization-parity.json")
    if not (parity["policy_tensors_identical"] and parity["initial_observations_identical"]):
        raise ValueError("Shared zero-step results require identical initial policies and observations")
    protocol_sha = hashlib.sha256(protocol_file.read_bytes()).hexdigest()
    if frozen["protocol_sha256"] != protocol_sha:
        raise ValueError("Selection belongs to a different protocol")
    steps = spec["checkpoints"]
    final = steps[-1]
    best = frozen["best_by_arm"]
    # Both initial policies have identical parameters. The runner evaluates control/0 once.
    development = [("baseline", final), ("control", 0)]
    development.extend((arm, step) for step in steps if step for arm in ARMS)
    held = [("baseline", final)]
    for arm in ARMS:
        held.extend((arm, step) for step in sorted({final, best[arm]}))
    identities = {(r["arm"], r["step"]): r["sha256"] for r in frozen["holdout_weights"]}
    if set(identities) != set(held) or len(identities) != len(frozen["holdout_weights"]):
        raise ValueError("Holdout roster does not match the fixed final and selected weights")
    cohorts = {}
    for split, roster in (("development", development), ("holdout", held)):
        ref = None
        for arm, step in roster:
            cohort = read_cohort(run / "evaluation" / split / arm / f"steps_{step}")
            s = cohort.summary
            if (s["arm"], s["agent_steps"], s["split"], s["protocol_sha256"]) != (arm, step, split, protocol_sha):
                raise ValueError(f"Different experiment in {cohort.folder}")
            if [s["first"], s["last"]] != spec["evaluation"][split] or not s["common_peer_termination"]:
                raise ValueError(f"Different seeds or termination in {cohort.folder}")
            if split == "holdout":
                dev_digest = cohorts["development", arm, step].summary["checkpoint_sha256"]
                if s["checkpoint_sha256"] != identities[arm, step] or s["checkpoint_sha256"] != dev_digest:
                    raise ValueError(f"Holdout checkpoint differs from frozen development weight in {cohort.folder}")
            states = {seed: r["first_observation_sha256"] for seed, r in cohort.records.items()}
            if ref is not None and states != ref:
                raise ValueError(f"Unpaired starting states in {cohort.folder}")
            ref = states
            cohorts[split, arm, step] = cohort
    return spec, best, cohorts


def role(arm, step, final, best):
    if arm == "baseline":
        return "历史末权重复测"
    if step == 0:
        return "两组共同初始策略"
    if step == final:
        return "末权重兼开发最优" if step == best[arm] else "末权重"
    return "开发最优" if step == best[arm] else "中间权重"


def paired_counts(control, peer):
    counts = [0, 0, 0, 0]  # both pass, only control, only peer, neither
    if control.records.keys() != peer.records.keys():
        raise ValueError("Cohorts use different seeds")
    for seed, row in control.records.items():
        a = row["strict_outcome"] == "success"
        b = peer.records[seed]["strict_outcome"] == "success"
        counts[(0 if a else 2) + (0 if b else 1)] += 1
    return counts


def choose_case(control, peer):
    def priority(seed):
        a, b = control.records[seed], peer.records[seed]
        passed_a, passed_b = a["strict_outcome"] == "success", b["strict_outcome"] == "success"
        if a["peer_first"] and passed_b:
            return 0
        if passed_a and b["peer_first"]:
            return 1
        if passed_a != passed_b:
            return 2
        if a["peer_first"] or b["peer_first"]:
            return 3
        return 4
    return min(control.records, key=lambda seed: (priority(seed), seed))


def configure_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    names = {font.name for font in font_manager.fontManager.ttflist}
    family = next((name for name in ("Noto Sans CJK SC", "Noto Sans CJK JP", "AR PL UMing CN") if name in names), None)
    if family is None:
        raise RuntimeError("Install a CJK font such as fonts-noto-cjk before rendering the Chinese figures")
    plt.rcParams.update({"font.family": family, "font.size": 10, "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})
    return plt


def million(step):
    return f"{step / 1e6:.2f}"


def comparison_figure(run, spec, best, cohorts, output):
    plt = configure_plot()
    steps = spec["checkpoints"]
    final = steps[-1]
    fig, axes = plt.subplots(2, 2, figsize=(14.7, 9.0))
    fig.subplots_adjust(left=.075, right=.975, bottom=.155, top=.87, wspace=.22, hspace=.56)
    fig.suptitle("安全起点下，增加机间终止对三机穿缝的影响", x=.075, y=.975, ha="left", fontsize=21)
    dev = spec["evaluation"]["development"]
    held = spec["evaluation"]["holdout"]
    fig.text(.075, .916, f"训练种子 {spec['train_seed']}   |   开发种子 {dev[0]}–{dev[1]}   |   独立留出种子 {held[0]}–{held[1]}   |   每个种子一个三机团队回合", fontsize=11)
    native, devpass, devpeer, heldax = axes.flat
    for ax, title in zip(axes.flat, ("训练时最近 100 个团队回合的通过率", "开发集 · 严格安全穿缝",
                                    "开发集 · 首次接触涉及另一架无人机", "独立留出集 · 冻结权重")):
        ax.set_title(title, loc="left", fontsize=14, pad=8)
        ax.grid(axis="y", color="#dde2e5", linewidth=.65)
        ax.set_axisbelow(True)
        ax.set_ylim(-4, 109)
        ax.set_yticks(range(0, 101, 25))
        ax.set_ylabel("团队占比（%）")
    for arm in ARMS:
        training = csv_rows(run / "train" / arm / "training.csv")
        if not training or int(training[-1]["agent_steps"]) != final:
            raise ValueError(f"Final training curve missing for {arm}")
        native.plot([int(x["agent_steps"])/1e6 for x in training], [float(x["teams_last100"])*100 for x in training],
                    color=COLOR[arm], label=ARM[arm], lw=1.45, alpha=.87)
    native.legend(frameon=False, fontsize=9, loc="lower right")
    for ax, key in ((devpass, "strict_successes"), (devpeer, "peer_first")):
        for arm in ARMS:
            series = [cohorts["development", arm if step else "control", step].summary for step in steps]
            ax.plot(np.array(steps)/1e6, [r[key]*100/r["episodes"] for r in series], color=COLOR[arm],
                    label=ARM[arm], marker=("o" if arm == "control" else "s"), lw=2, ms=5)
        s = cohorts["development", "baseline", final].summary
        ax.scatter([final/1e6], [s[key]*100/s["episodes"]], s=82, marker="x", linewidths=2,
                   color=COLOR["baseline"], label="历史末权重复测", zorder=5)
    peer_values = [100*c.summary["peer_first"]/c.summary["episodes"] for (split, _, _), c in cohorts.items()
                   if split == "development"]
    top = min(100, max(10, 10*int(np.ceil(max(peer_values)/10))))
    devpeer.set_ylim(-top*.035, top*1.08)
    devpeer.set_yticks(np.linspace(0, top, 6))
    devpass.legend(loc="lower right", frameon=False, fontsize=9)
    for ax in (native, devpass, devpeer):
        ax.set_xlabel("累计智能体交互（百万条）")
        ax.set_xlim(-.2, final/1e6+.5)
    hold = [(arm, step, c.summary) for (split, arm, step), c in cohorts.items() if split == "holdout"]
    def label(arm, step):
        if arm == "baseline":
            return "历史末权重"
        if best[arm] == final or step == final:
            return ARM[arm] + "·末"
        return ARM[arm] + f"·择优\n{million(step)} 百万"
    xs = np.arange(len(hold))
    for i, (arm, step, s) in enumerate(hold):
        pct = 100*s["strict_successes"]/s["episodes"]
        lo, hi = np.array(s["strict_success_wilson95"])*100
        heldax.bar(i-.16, pct, width=.31, color=COLOR[arm], alpha=.84)
        heldax.errorbar(i-.16, pct, yerr=[[max(0., pct-lo)], [max(0., hi-pct)]], color="#27343c", capsize=3, lw=1)
        peer_pct = 100*s["peer_first"]/s["episodes"]
        heldax.bar(i+.16, peer_pct, width=.31, color="white", edgecolor=COLOR[arm], hatch="///")
        heldax.text(i+.16, peer_pct+1.5, str(s["peer_first"]), ha="center", fontsize=9)
        heldax.text(i-.16, max(2, pct-8), str(s["strict_successes"]), color="white" if pct >= 12 else COLOR[arm],
                    ha="center", fontsize=9, weight="bold")
    heldax.set_xticks(xs, [label(arm, step) for arm, step, _ in hold], fontsize=8.5)
    minimum = {split: 100*min(c.summary["minimum_start_m"] for (name, _, _), c in cohorts.items() if name == split)
               for split in SPLIT}
    fig.text(.075, .077, f"统一评估实测最小起点间距为开发集 {minimum['development']:.2f} cm、留出集 {minimum['holdout']:.2f} cm。", fontsize=9.5)
    fig.text(.075, .049, "训练曲线按各组训练环境的终止规则计数。开发集与留出集使用统一终止及子步几何验收。0 步策略只评估一次。", fontsize=9.5)
    fig.text(.075, .022, "留出图实心柱为严格安全穿缝（误差线为 Wilson 95% 区间），斜纹柱为首次机间接触。末权重同时入选时只展示一次。", fontsize=9.5)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def swept_distances(xyz):
    """Minimum of three simultaneous 3-D separations over each measured substep chord."""
    curves = []
    for a, b in combinations(range(3), 2):
        rel = xyz[:, a] - xyz[:, b]
        delta = np.diff(rel, axis=0)
        length2 = np.einsum("ij,ij->i", delta, delta)
        fraction = np.divide(-np.einsum("ij,ij->i", rel[:-1], delta), length2,
                             out=np.zeros_like(length2), where=length2 > 0).clip(0, 1)
        curves.append(np.r_[np.linalg.norm(rel[0]), np.linalg.norm(rel[:-1] + fraction[:, None]*delta, axis=1)])
    return np.min(curves, axis=0)


def case_data(cohort, seed):
    trajectory = cohort.folder / f"seed_{seed}.npz"
    with np.load(trajectory, allow_pickle=False) as archive:
        raw = archive["substep_xyz"]
        if hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest() != cohort.records[seed]["trajectory_sha256"]:
            raise ValueError(f"Trajectory no longer matches audited observations: {trajectory}")
        xyz = raw.astype(np.float64)
    geom = next(r for r in read_json(cohort.folder / "geometry.json") if r["seed"] == seed)
    curve = swept_distances(xyz)
    row = cohort.records[seed]
    if (len(xyz) != 8*row["control_steps"]+1 or geom["strict_outcome"] != row["strict_outcome"]
            or abs(curve.min()-row["min_peer_center_m"]) > 1e-8 or curve[0] < CONTACT_M):
        raise ValueError(f"Trace does not match audited episode {trajectory}")
    return xyz, curve, geom


def contact_description(geometry):
    first = geometry["first_contact"]
    if not first:
        return "全程无几何接触"
    names = {"wall_negative_y": "负 y 侧墙", "wall_positive_y": "正 y 侧墙"}
    labels = []
    for detail in first["details"]:
        obj = detail["object"]
        ids = "、".join(str(drone) for drone in detail["drones"])
        if obj.startswith("drone_"):
            labels.append(f"{ids} 号机相碰")
        else:
            surface = names.get(obj, "机库结构" if obj.startswith("stage_") else obj)
            labels.append(f"{ids} 号机接触{surface}")
    return f"第 {first['step']} 步第 {first['substep']} 子步，" + " / ".join(labels)


def case_figure(run, control, peer, output, metadata):
    plt = configure_plot()
    from matplotlib.lines import Line2D
    seed = choose_case(control, peer)
    data = {a: case_data(c, seed) for a, c in zip(ARMS, (control, peer))}
    if not np.array_equal(data["control"][0][0], data["peer"][0][0]):
        raise ValueError(f"Seed {seed} starts from different positions")
    all_xy = np.concatenate([v[0][:, :, :2].reshape(-1, 2) for v in data.values()])
    x_min = min(-.25, all_xy[:, 0].min()-.3)
    x_max = max(10.25, all_xy[:, 0].max()+.3)
    y_min = min(-1.25, all_xy[:, 1].min()-.28)
    y_max = max(1.25, all_xy[:, 1].max()+.28)
    fig = plt.figure(figsize=(14, 9.0))
    fig.suptitle(f"同一起点的两次推理  ·  开发种子 {seed}", x=.075, y=.98, ha="left", fontsize=21)
    fig.text(.075, .922, f"两组均使用 {control.summary['agent_steps']:,} 条交互的末权重。初始最近中心距 {data['control'][1][0]*100:.1f} cm。", fontsize=11)
    grids = fig.add_gridspec(2, 2, height_ratios=[1.03, 1.42], left=.075, right=.974, bottom=.15, top=.85, wspace=.16, hspace=.45)
    distance = fig.add_subplot(grids[1, :])
    info = {"split": control.summary["split"], "seed": seed, "agent_steps": control.summary["agent_steps"],
            "selection_rule": "从开发集依次查找下列情形，取首个非空类别中的最小种子。",
            "selection_priority": ["对照首次机间接触、机间终止组通过", "对照通过、机间终止组首次机间接触",
                                   "其他仅一组通过", "至少一组首次机间接触", "其余回合"],
            "paired_strict_success": dict(zip(("both", "control_only", "peer_only", "neither"), paired_counts(control, peer))),
            "arms": {}}
    contacts = False
    for j, (arm, cohort) in enumerate(zip(ARMS, (control, peer))):
        ax = fig.add_subplot(grids[0, j])
        xyz, curve, geom = data[arm]
        row = cohort.records[seed]
        wall_x, gap_lo, gap_hi, _, _ = geom["aperture_x_ylo_yhi_zlo_zhi_m"]
        ax.vlines([wall_x, wall_x], [y_min, gap_hi], [gap_lo, y_max], color="#677177", linewidth=3)
        ax.axvline(10, color="#757f84", ls=":", lw=1.2)
        for i, color in enumerate(DRONE_COLORS):
            xy = xyz[:, i, :2]
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.7)
            ax.scatter(*xy[0], s=30, marker="s", edgecolor=color, facecolor="white", zorder=4)
            ax.scatter(*xy[-1], s=29, marker="o", color=color, zorder=4)
        time = np.arange(len(curve))*SUBSTEP_S
        distance.plot(time, curve*100, color=COLOR[arm], ls="-" if arm == "control" else "--",
                      lw=2, label=f"{ARM[arm]} · 最近 {curve.min()*100:.2f} cm")
        first = geom["first_contact"]
        if first:
            contacts = True
            index = first["sample"]
            ids = np.array(first["drones"])-1
            ax.scatter(xyz[index, ids, 0], xyz[index, ids, 1], marker="x", s=85, color="#292c31", lw=1.6, zorder=5)
            distance.scatter(time[index], curve[index]*100, marker="x", s=65, color=COLOR[arm], zorder=5)
        ax.set(title=f"{ARM[arm]} · {OUTCOME[row['strict_outcome']]}\n{contact_description(geom)}",
               xlim=(x_min, x_max), ylim=(y_min, y_max), xlabel="前进方向 x（m）")
        if j == 0:
            ax.set_ylabel("横向 y（m）")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#e6e9ea", lw=.5)
        info["arms"][arm] = {"checkpoint_sha256": cohort.summary["checkpoint_sha256"], "row": row,
                             "first_contact": first, "source": str(cohort.folder.relative_to(run) / f"seed_{seed}.npz")}
    distance.axhline(CONTACT_M*100, color="#6a6260", ls=":", label="机间接触阈值 20 cm")
    distance.set(xlabel="仿真时间（s）", ylabel="三对中最小中心距（cm）", title="三维机间距离随时间的变化")
    distance.grid(axis="y", color="#dde2e5")
    distance.legend(loc="best", frameon=False, fontsize=10)
    legend = [Line2D([], [], color=c, lw=2, label=f"{i} 号机") for i, c in enumerate(DRONE_COLORS, 1)]
    fig.legend(handles=legend, loc="center right", bbox_to_anchor=(.975, .93), ncol=3, frameon=False)
    note = "叉号标在首次接触子步的终点。" if contacts else ""
    fig.text(.075, .075, f"俯视图中空方块为起点，实心圆为末点，灰实线为墙体在缝口平面的截线，竖虚线为 x = 10 m。{note}", fontsize=9.5)
    fig.text(.075, .044, "距离曲线逐个 2.5 ms 物理子步计算两机同时运动的最近距离，画在子步终点。各条曲线到本组回合结束为止。", fontsize=9.5)
    fig.savefig(output, dpi=170)
    plt.close(fig)
    metadata.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path, help="默认写入运行目录的 figures 子目录")
    args = parser.parse_args()
    run = args.run.resolve()
    spec, best, cohorts = load_experiment(run)
    output = args.output_dir or run / "figures"
    output.mkdir(exist_ok=True, parents=True)
    final = spec["checkpoints"][-1]
    control, peer = [cohorts["development", arm, final] for arm in ARMS]
    with TemporaryDirectory(dir=output) as pending:
        staging = Path(pending)
        comparison_figure(run, spec, best, cohorts, staging / "comparison.png")
        case_figure(run, control, peer, staging / "paired-case.png", staging / "paired-case.json")
        for name in ("comparison.png", "paired-case.png", "paired-case.json"):
            (staging / name).replace(output / name)
    print(f"{output} · {len(cohorts)} 组评估，配对示例种子 {choose_case(control, peer)}")


if __name__ == "__main__":
    main()
