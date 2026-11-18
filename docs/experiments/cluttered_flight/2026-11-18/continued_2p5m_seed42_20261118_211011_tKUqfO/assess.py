import argparse
import csv
import json
import time

from continue_ppo import BASELINE, CHECKPOINTS, MILESTONES, RUN, SEED, write_json

import evaluate as baseline_eval
from stable_baselines3.common.utils import set_random_seed
from VisFly.utils.algorithms.PPO import PPO


def save_results(rows, summaries):
    temporary = RUN / "episodes.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RUN / "episodes.csv")
    best = max(
        (int(step) for step in summaries if int(step) != 32),
        key=lambda step: (
            summaries[str(step)]["train"]["outcomes"].get("success", 0),
            -summaries[str(step)]["train"]["outcomes"].get("collision", 0),
            step,
        ),
    )
    if best in MILESTONES:
        candidates = [row for row in rows if row["checkpoint_steps"] == best]
        selected = next((row for row in candidates if row["outcome"] == "success"), None)
        if selected is None:
            selected = min(candidates, key=lambda row: row["final_goal_distance_m"])
    else:
        selected = json.loads((BASELINE / "summary.json").read_text())["showcase"]
        assert selected["checkpoint_steps"] == best
    write_json(RUN / "summary.json", {
        "models": summaries,
        "showcase": selected,
        "new_episodes": len(rows),
    })


def evaluate(wait):
    rows = []
    summaries = json.loads((BASELINE / "summary.json").read_text())["models"]
    for steps, checkpoint in CHECKPOINTS.items():
        if wait and not checkpoint.is_file():
            print(f"WAITING_FOR_CHECKPOINT {steps}", flush=True)
            deadline = time.monotonic() + 3600
            while not checkpoint.is_file() and time.monotonic() < deadline:
                time.sleep(5)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

        summaries[str(steps)] = {}
        for layout, scene in baseline_eval.LAYOUTS.items():
            set_random_seed(SEED)
            env = baseline_eval.make_eval_env(layout)
            try:
                model = PPO.load(checkpoint, env=env, device="cpu")
                assert model.num_timesteps == steps
                env.reset(is_test=True)
                episode_rows = []
                for index, start in enumerate(baseline_eval.STARTS, 1):
                    episode_rows.append({
                        "checkpoint_steps": steps,
                        "layout": layout,
                        "scene": scene.name,
                        "case_id": f"{layout}-{index:02d}",
                        "start_x_m": start[0],
                        "start_y_m": start[1],
                        "start_z_m": start[2],
                        **baseline_eval.episode(env, model, start),
                    })
                summaries[str(steps)][layout] = baseline_eval.summarise(episode_rows)
                print(f"EVALUATED {steps} {layout} {summaries[str(steps)][layout]}", flush=True)
                rows.extend(episode_rows)
            finally:
                env.close()
        save_results(rows, summaries)
    assert len(rows) == len({(row["checkpoint_steps"], row["case_id"]) for row in rows}) == 90
    print("EVALUATION_COMPLETE 90", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    evaluate(parser.parse_args().wait)
