import argparse
import json
import os
from pathlib import Path


def desktop_session():
    if os.environ.get("DISPLAY"):
        return
    for process in Path("/proc").glob("[0-9]*/comm"):
        try:
            if process.stat().st_uid != os.getuid() or process.read_text().strip() != "Xwayland":
                continue
            settings = dict(
                entry.split(b"=", 1)
                for entry in (process.parent / "environ").read_bytes().split(b"\0")
                if b"=" in entry
            )
            for key in ("DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR"):
                if key.encode() in settings:
                    os.environ[key] = settings[key.encode()].decode()
            if os.environ.get("DISPLAY"):
                return
        except (FileNotFoundError, PermissionError):
            continue
    raise RuntimeError("No local X11/Xwayland display is available")


def view():
    desktop_session()
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

    import cv2
    import numpy as np
    from stable_baselines3.common.utils import set_random_seed

    from evaluate import CHECKPOINTS, LAYOUTS, RUN, SEED, STARTS, classify, make_eval_env, reset_case
    from VisFly.utils.algorithms.PPO import PPO

    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, choices=CHECKPOINTS)
    parser.add_argument("--case", choices=[f"{layout}-{n:02d}" for layout in LAYOUTS for n in range(1, 16)])
    args = parser.parse_args()
    summary = json.loads((RUN / "summary.json").read_text())
    case = args.case or summary["showcase"]["case_id"]
    checkpoint_steps = args.steps or summary["showcase"]["checkpoint_steps"]
    layout, index = case.rsplit("-", 1)
    start = STARTS[int(index) - 1]

    set_random_seed(SEED)
    env = make_eval_env(layout, display=True)
    title = "VisFly - cluttered_flight PPO baseline"
    frame = np.full((638, 1376, 3), (36, 29, 24), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def show(observation, step, status):
        scene = cv2.cvtColor(env.render(points=env.target)[0], cv2.COLOR_RGBA2BGR)
        depth = (np.clip(np.asarray(observation["depth"])[0, 0], 0, 10) * 25.5).astype(np.uint8)
        depth = cv2.resize(depth, (384, 384), interpolation=cv2.INTER_NEAREST)
        frame[48:588, :960] = scene
        frame[48:432, 976:1360] = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
        frame[:48] = (36, 29, 24)
        frame[588:] = (36, 29, 24)
        cv2.putText(frame, f"Follow camera | {case} | PPO {checkpoint_steps:,} steps", (18, 31), font, 0.75, (247, 247, 247), 2, cv2.LINE_AA)
        cv2.putText(frame, "Onboard depth (64 x 64)", (976, 31), font, 0.72, (247, 247, 247), 2, cv2.LINE_AA)
        cv2.putText(frame, "Dark: near  |  Light: far", (976, 463), font, 0.6, (201, 216, 231), 1, cv2.LINE_AA)
        goal_distance = (env.target[0] - env.position[0]).norm().item()
        clearance = env.collision_dis[0].item()
        status_line = f"{status} | Step {step} | Goal {goal_distance:.2f}m | Clearance {clearance:.2f}m"
        cv2.putText(frame, status_line, (18, 620), font, 0.62, (234, 235, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, "Space: pause/resume   R: replay   Esc: close", (830, 620), font, 0.59, (201, 216, 231), 1, cv2.LINE_AA)
        cv2.imshow(title, frame)

    try:
        model = PPO.load(CHECKPOINTS[checkpoint_steps], env=env, device="cpu")
        env.reset(is_test=True)
        observation = reset_case(env, start)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(title, 1296, 600)
        steps, finished, paused = 0, False, False
        show(observation, steps, "Ready")
        initial_key = cv2.waitKey(700) & 0xff
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1 or initial_key in (27, ord("q")):
            raise RuntimeError("Simulator window closed before the replay")
        print(f"WINDOW_VISIBLE: {title} | {case}", flush=True)

        while True:
            key = cv2.waitKey(60 if not finished and not paused else 100) & 0xff
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("r"), ord("R")):
                observation = reset_case(env, start)
                steps, finished, paused = 0, False, False
                show(observation, steps, "Running")
            elif key == ord(" ") and not finished:
                paused = not paused
                show(observation, steps, "Paused" if paused else "Running")
            if paused or finished:
                continue

            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, info = env.step(action, is_test=True)
            steps += 1
            finished = bool(np.asarray(done)[0])
            status = classify(env, info[0]) if finished else "Running"
            show(observation, steps, status)
            if finished:
                print(f"REPLAY_COMPLETE: {case} {status} {steps} steps", flush=True)
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    view()
