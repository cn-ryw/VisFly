"""Pair complete, audited crossing cohorts by seed. Run with the experiment directory."""
import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
from math import comb, isclose, isfinite
from pathlib import Path
from statistics import mean, median
from tempfile import NamedTemporaryFile


ARMS = ("baseline", "control", "peer")
SPLITS = ("development", "holdout")
EQUALITY_M = 1e-12


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def keyed(rows):
    by_seed = {int(row["seed"]): row for row in rows}
    if len(by_seed) != len(rows):
        raise ValueError("Duplicate seed in a cohort")
    return by_seed


def load_cohort(run, split, arm, step, limits):
    path = run / "evaluation" / split / arm / f"steps_{step}"
    summary = read_json(path / "summary.json")
    with (path / "episodes.csv").open(encoding="utf-8", newline="") as file:
        rows = keyed(list(csv.DictReader(file)))
    geometry = keyed(read_json(path / "geometry.json"))
    first, last = limits
    expected = set(range(first, last + 1))
    if (summary["split"], summary["arm"], summary["agent_steps"]) != (split, arm, step):
        raise ValueError(f"Wrong cohort identity in {path}")
    if rows.keys() != expected or geometry.keys() != expected or summary["episodes"] != len(expected):
        raise ValueError(f"Missing or extra paired seeds in {path}")
    for seed, row in rows.items():
        geo = geometry[seed]
        first_contact = geo["first_contact"] or {}
        objects = first_contact.get("objects", [])
        groups = sorted({name.split("_")[0] for name in objects})
        peer_first = "drone" in groups
        gap = float(row["min_peer_center_m"])
        if (row["strict_outcome"] != geo["strict_outcome"] or
                (int(row["peer_first"]) != int(peer_first)) or
                row["first_objects"] != "+".join(objects) or row["first_groups"] != "+".join(groups) or
                not isfinite(gap) or not isclose(gap, geo["min_peer_m"], rel_tol=0, abs_tol=EQUALITY_M)):
            raise ValueError(f"Geometry and episode CSV differ in {path}, seed {seed}")
        row["success"] = geo["strict_success"]
        row["peer_first"] = peer_first
        row["gap_m"] = gap
        row["outcome_with_contact"] = (f"collision/{row['first_groups']}" if objects else row["strict_outcome"])
    if (sum(r["success"] for r in rows.values()) != summary["strict_successes"] or
            sum(r["peer_first"] for r in rows.values()) != summary["peer_first"]):
        raise ValueError(f"Episode counts disagree with {path}/summary.json")
    return dict(path=str(path.relative_to(run)), summary=summary, rows=rows, geometry=geometry)


def metric(changes, favorable_event):
    """Changes holds counts for (left event, right event); 'event' is untransformed."""
    both_true = changes[True, True]
    both_false = changes[False, False]
    new = changes[False, True]
    lost = changes[True, False]
    n = both_true + both_false + new + lost
    favorable, unfavorable = (new, lost) if favorable_event else (lost, new)
    discordant = new + lost
    low = min(new, lost)
    p = min(1., 2 * sum(comb(discordant, i) for i in range(low + 1)) / 2 ** discordant)
    return dict(paired_teams=n, left_events=both_true+lost, right_events=both_true+new,
                right_minus_left_events=new-lost, right_minus_left_pp=100*(new-lost)/n,
                favorable=favorable, unfavorable=unfavorable,
                both_favorable=both_true if favorable_event else both_false,
                both_unfavorable=both_false if favorable_event else both_true,
                favorable_net_pp=100*(favorable-unfavorable)/n, mcnemar_exact_two_sided_p=p)


def compare(left, right):
    rows = []
    success_cells, peer_cells, transitions = Counter(), Counter(), Counter()
    for seed in sorted(left["rows"]):
        a, b = left["rows"][seed], right["rows"][seed]
        if (a["first_observation_sha256"] != b["first_observation_sha256"] or
                float(a["initial_min_peer_m"]) != float(b["initial_min_peer_m"])):
            raise ValueError(f"Different starts for {left['path']} and {right['path']}, seed {seed}")
        success_cells[a["success"], b["success"]] += 1
        peer_cells[a["peer_first"], b["peer_first"]] += 1
        transitions[a["outcome_with_contact"], b["outcome_with_contact"]] += 1
        record = dict(seed=seed, initial_min_peer_m=float(a["initial_min_peer_m"]),
                      initial_state_sha256=a["first_observation_sha256"],
                      left_trace=f"{left['path']}/seed_{seed}.npz",
                      right_trace=f"{right['path']}/seed_{seed}.npz")
        for label, row in (("left", a), ("right", b)):
            for key in ("outcome_with_contact", "peer_first", "common_peer_stop", "first_objects",
                        "first_control_step", "first_substep", "control_steps", "gap_m", "trajectory_sha256"):
                record[f"{label}_{key}"] = row[key]
        record["right_minus_left_gap_m"] = b["gap_m"] - a["gap_m"]
        rows.append(record)

    common = [r for r in rows if r["left_outcome_with_contact"] == r["right_outcome_with_contact"] == "success"]
    delta = [r["right_minus_left_gap_m"] for r in common]
    larger = sum(d > EQUALITY_M for d in delta)
    smaller = sum(d < -EQUALITY_M for d in delta)
    clearance = dict(both_succeeded=len(delta), right_larger=larger, right_smaller=smaller,
                     equal_within_1e_12_m=len(delta)-larger-smaller,
                     left_median_minimum_m=median(r["left_gap_m"] for r in common) if common else None,
                     right_median_minimum_m=median(r["right_gap_m"] for r in common) if common else None,
                     paired_mean_delta_m=mean(delta) if delta else None,
                     paired_median_delta_m=median(delta) if delta else None,
                     paired_min_delta_m=min(delta, default=None), paired_max_delta_m=max(delta, default=None))
    summary = dict(strict_success=metric(success_cells, True), first_peer_contact=metric(peer_cells, False),
                   clearance_among_common_successes=clearance,
                   outcome_transitions=[dict(left=a, right=b, teams=n) for (a,b),n in sorted(transitions.items())])
    return summary, rows


def example_cases(left, right, rows):
    """Snapshot one seed per observed outcome transition, or clearance direction for shared successes."""
    candidates = defaultdict(list)
    for row in rows:
        a, b = row["left_outcome_with_contact"], row["right_outcome_with_contact"]
        group = f"{a} -> {b}"
        if a == b == "success":
            delta = row["right_minus_left_gap_m"]
            direction = "larger" if delta > EQUALITY_M else "smaller" if delta < -EQUALITY_M else "equal"
            group += f" / right clearance {direction}"
        candidates[group].append(row)
    result = []
    for group, items in sorted(candidates.items()):
        # The closest observed clearance change to the stratum median, then lowest seed on a tie.
        middle = median(r["right_minus_left_gap_m"] for r in items)
        chosen = min(items, key=lambda r: (abs(r["right_minus_left_gap_m"] - middle), r["seed"]))
        seed = chosen["seed"]
        case = dict(transition=group, teams_in_stratum=len(items), seed=seed,
                    paired_clearance_delta_m=chosen["right_minus_left_gap_m"])
        for side, cohort in (("left", left), ("right", right)):
            geo = cohort["geometry"][seed]
            case[side] = dict(trace=f"{cohort['path']}/seed_{seed}.npz", steps=geo["control_steps"],
                              strict_outcome=geo["strict_outcome"], first_contact=geo["first_contact"],
                              min_peer_center_m=geo["min_peer_m"], min_wall_surface_m=geo["min_wall_clearance_m"])
        result.append(case)
    return result


def comparisons(split, steps, available, selected, final):
    result = defaultdict(list)
    for step in sorted(steps):
        for a, b in combinations(ARMS, 2):
            if (a, step) in available and (b, step) in available:
                result[(a, step), (b, step)].append("same_training_steps")
    for a, b in combinations(ARMS, 2):
        result[(a, final), (b, final)].append("final_weights")
        result[(a, selected.get(a, final)), (b, selected.get(b, final))].append("development_selected_weights")
    for arm, best in selected.items():
        if best != final:
            result[(arm, final), (arm, best)].append("own_final_to_selected")
    for left, right in result:
        if left not in available or right not in available:
            raise ValueError(f"Missing audited {split} cohort for {left} versus {right}")
    return result


def write_result(path, value, *, csv_output=False):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8", newline="") as stream:
        temp = Path(stream.name)
        try:
            if csv_output:
                writer = csv.DictWriter(stream, fieldnames=list(value[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(value)
            else:
                json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)


def summarize(run):
    spec = read_json(run / "setup/protocol.json")
    fixed = read_json(run / "selection.json")
    final = spec["actual_agent_interactions_per_arm"]
    steps = set(spec["checkpoints"])
    selected = {}
    for arm in ("control", "peer"):
        choice = fixed["best_by_arm"][arm]
        selected[arm] = int(choice.get("step", choice.get("agent_steps")) if isinstance(choice, dict) else choice)
        if selected[arm] not in steps or selected[arm] == 0:
            raise ValueError(f"Selected checkpoint for {arm} is outside the protocol")
    identities = {(w["arm"], w["step"]):w["sha256"] for w in fixed["holdout_weights"]}
    result, metrics, paired_rows, cases = {}, [], [], {}
    for split in SPLITS:
        ready = {(arm, step) for arm in ARMS for step in steps
                 if (run / "evaluation" / split / arm / f"steps_{step}" / "summary.json").is_file()}
        targets = comparisons(split, steps, ready, selected, final)
        cohorts = {}
        for key in sorted({k for pair in targets for k in pair}):
            arm, step = key
            cohort = load_cohort(run, split, arm, step, spec["evaluation"][split])
            meta = cohort["summary"]
            if meta["protocol_sha256"] != fixed["protocol_sha256"]:
                raise ValueError(f"Different protocol for {cohort['path']}")
            if split == "holdout" and identities.get(key) != meta["checkpoint_sha256"]:
                raise ValueError(f"Holdout weight differs from the frozen selection in {cohort['path']}")
            cohorts[key] = cohort
        for (a, b), purposes in targets.items():
            left, right = cohorts[a], cohorts[b]
            key = f"{split}/{a[0]}_{a[1]}__{b[0]}_{b[1]}"
            info, rows = compare(left, right)
            info.update(split=split, purposes=purposes, left_cohort=left["path"], right_cohort=right["path"],
                        left_checkpoint_sha256=left["summary"]["checkpoint_sha256"],
                        right_checkpoint_sha256=right["summary"]["checkpoint_sha256"])
            result[key] = info
            for name in ("strict_success", "first_peer_contact"):
                metrics.append(dict(comparison=key, metric=name, **info[name]))
            paired_rows.extend(dict(comparison=key, **r) for r in rows)
            cases[key] = example_cases(left, right, rows)
    notes = dict(
        unit="按种子配对的一次完整三机回合。先计算全队列统计，再从观察到的转移中选取病例。",
        effects="原始事件差为右侧减左侧。有利指严格通过增加，或首次接触对象包含邻机的回合减少。同一子步并列首次接触保留全部对象。",
        tests="各项 p 值为对不一致配对的双侧精确 McNemar 检验，无多重比较校正。开发集用于选模，留出集检验冻结权重。",
        clearance="间距为三对飞机全程动力学子步扫掠最小中心距，单位为米。共同成功的配对单列，与首碰事件分别统计。",
        cases="按两侧实际结果及首次碰撞对象类别分层。双方均成功时再按间距变化方向分层。各层取最接近本层间距变化中位数的种子，并列取最小种子。")
    dest = run / "paired"
    dest.mkdir(exist_ok=True)
    write_result(dest / "paired-metrics.csv", metrics, csv_output=True)
    write_result(dest / "seed-pairs.csv", paired_rows, csv_output=True)
    write_result(dest / "representative-cases.json", cases)
    write_result(dest / "comparisons.json", dict(notes=notes, comparisons=result))
    print(f"{len(result)} paired comparisons, {len(paired_rows)} seed pairs, {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    summarize(parser.parse_args().run.resolve())
