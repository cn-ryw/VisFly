"""Recompute the published crossing results using only Python's standard library."""
import csv
import json
from collections import Counter, defaultdict
from fractions import Fraction
from math import comb, isclose
from pathlib import Path
from statistics import mean, median


DATA = Path(__file__).resolve().parent / "results"
BUDGETS = ("500736", "1000704")


def read_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def read_csv(name):
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def check(condition, description):
    if not condition:
        raise ValueError(description)


def verify_pairs():
    summary = read_json("paired-summary.json")
    rows = read_csv("paired-episodes.csv")
    expected_seeds = list(range(summary["seeds"][0], summary["seeds"][1] + 1))
    check([int(row["seed"]) for row in rows] == expected_seeds, "Development seeds")
    check(len(rows) == summary["n_pairs"] == 100, "Number of pairs")
    check(sum(int(row["initial_match"]) for row in rows) == summary["matched_initial_pairs"] == len(rows),
          "Matched initial observations")
    pair_counts = Counter()
    outcomes = defaultdict(Counter)
    objects = defaultdict(Counter)
    categories = defaultdict(Counter)
    steps = defaultdict(list)
    for row in rows:
        success = []
        for budget in BUDGETS:
            prefix = budget + "_"
            end = row[prefix + "outcome"]
            length = int(row[prefix + "steps"])
            step = row[prefix + "first_contact_step"]
            drones = row[prefix + "first_contact_drones"]
            contacts = set(filter(None, row[prefix + "first_contact_objects"].split("|")))
            if end == "collision":
                check(step != "" and 0 <= int(step) <= length and drones and contacts,
                      f"Invalid contact at seed {row['seed']}, budget {budget}")
            else:
                check(not (step or drones or contacts), "Unexpected contact in non-collision episode")
            success.append(end == "success")
            outcomes[budget][end] += 1
            steps[budget].append(length)
            objects[budget].update(contacts)
            categories[budget].update({c.split("_")[0] for c in contacts})
        key = {(False, False): "both_fail", (True, False): "only_500736",
               (False, True): "only_1000704", (True, True): "both_success"}[tuple(success)]
        pair_counts[key] += 1
    for key, value in summary["pair_counts"].items():
        check(pair_counts[key] == value, "Paired outcome " + key)
    for budget in BUDGETS:
        report = summary["budgets"][budget]
        for end in ("success", "collision", "timeout", "bypass"):
            check(outcomes[budget][end] == report[end], budget + " " + end)
        check(isclose(mean(steps[budget]), report["mean_length"]), "Mean episode length")
        check(objects[budget] == Counter(report["first_contact_objects"]), "First contact objects")
        check(categories[budget] == Counter(report["first_contact_categories"]), "First contact categories")
    a, b = (pair_counts["only_" + budget] for budget in BUDGETS)
    discordant = a + b
    tail = sum(comb(discordant, k) for k in range(min(a, b) + 1))
    p_value = min(Fraction(1), Fraction(2 * tail, 2 ** discordant))
    test = summary["exact_mcnemar"]
    check(isclose(float(p_value), test["p_value"]) and discordant == test["discordant"], "Exact McNemar p-value")
    return summary, float(p_value)


def verify_approach():
    summary = read_json("approach-summary.json")
    remaining = summary["nominal_wall_front_x_m"] - summary["section_x_m"]
    band = summary["center_band_m"]
    groups = defaultdict(list)
    for row in read_csv("approach-x6.csv"):
        y = float(row["lateral_m"])
        estimate = y + remaining * float(row["lateral_velocity_mps"]) / float(row["forward_speed_mps"])
        groups[int(row["budget"]), row["group"]].append((abs(y), abs(estimate)))
    check(set(groups) == {(r["budget"], r["group"]) for r in summary["stats"]}, "Approach groups")
    for report in summary["stats"]:
        samples = groups[report["budget"], report["group"]]
        check(len(samples) == report["usable_x6"] == report["drone_trajectories"], "Drone count at x=6")
        check(sum(y > band for y, _ in samples) == report["outside_at_section"], "Alignment at x=6")
        check(sum(y > band for _, y in samples) == report["would_be_outside_at_wall"], "Projected alignment")
        check(isclose(median(y for y, _ in samples), report["median_abs_lateral_m"]), "Median offset")


def verify_checkpoints():
    training = read_json("training.json")
    for record in training["checkpoints"]:
        rows = record["development_episodes"]
        check([row["seed"] for row in rows] == training["development_seeds"], "Original five development seeds")
        counts = Counter(row["end"] for row in rows)
        check(all(counts[key] == n for key, n in record["development_outcomes"].items()), "Checkpoint outcomes")
        check(sum(row["geometric_team_success"] for row in rows) == record["development_verified_successes"],
              "Geometric checkpoint successes")
        check(isclose(mean(row["length"] for row in rows), record["development_mean_length"]), "Checkpoint length")


if __name__ == "__main__":
    summary, p = verify_pairs()
    verify_approach()
    verify_checkpoints()
    successes = [summary["budgets"][b]["success"] for b in BUDGETS]
    print(f"{summary['n_pairs']} paired starts verified. Safe passes: {successes}. Exact McNemar p = {p}.")
    print("Approach projection and three historical checkpoint summaries match.")
