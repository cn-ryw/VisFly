#!/usr/bin/env bash
set -eu
run=$(realpath "$1")
script_dir=${VISFLY_SCRIPTS:-$(cd -- "$(dirname -- "$0")" && pwd)}
python_bin=${VISFLY_PYTHON:-python}
export PYTHONPATH="$run/setup/renderer-lib"
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet PYTHONDONTWRITEBYTECODE=1
for checkpoint in 0 1013760 5013504 10008576; do
    marker="$run/train/PPO_$checkpoint.json"
    until test -e "$marker"; do
        if test -e "$run/train/failure.json" && ! test -e "$run/train/complete.json"; then
            exit 1
        fi
        sleep 30
    done
    "$python_bin" -u "$script_dir/evaluate_paper.py" "$run" "$run/train/PPO_$checkpoint.zip" development
done
"$python_bin" -u "$script_dir/evaluate_paper.py" "$run" "$run/train/PPO_10008576.zip" held-out
