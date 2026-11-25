"""Use the pinned paper checkout and renderer without modifying the installed VisFly."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

REVISION = "7b0a4534b57051fd9b00ab4e561496f9abaacab1"
ASSET_REVISION = "134192bd343b618764e4497abb00e2dfd23a0b81"
RENDERER_REVISION = "3f89600c942a4dee1866a589acd6e59741952a9f"


def write_json(path, data):
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def load_example(run):
    source = run / "source"
    head = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    if head != REVISION:
        raise ValueError(f"Expected historical VisFly {REVISION}, found {head}")
    os.chdir(source)
    sys.path.insert(0, str(source))
    sys.argv = [str(source / "examples/crossing/rl.py"), "-t", "1", "-c", "paper_crossing"]
    import habitat_sim._ext.habitat_sim_bindings as bindings
    if (run / "setup/renderer-lib").resolve() not in Path(bindings.__file__).resolve().parents:
        raise ValueError(f"Run using the historical renderer, loaded {bindings.__file__}")
    torch.set_num_threads(1)
    from examples.crossing import rl
    return rl


def make_env(rl, scenes=1):
    import copy
    return rl.MultiNavigationEnv(num_agent_per_scene=3, num_scene=scenes,
                                 random_kwargs=copy.deepcopy(rl.random_kwargs), visual=True,
                                 max_episode_steps=256, scene_kwargs={"path": rl.scene_path})


def reset_seed(env, seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    env.envs.set_seed(seed)
    torch.manual_seed(seed)
    return env.reset()


def close_env(env):
    # The September scene manager closes the first simulator only. Its renderer owns
    # the shared GL context. Release the other simulators' resources before it.
    for scene in reversed(env.envs.sceneManager.scenes[1:]):
        scene.close()
    env.close()
