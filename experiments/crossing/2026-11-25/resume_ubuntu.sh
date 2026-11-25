#!/usr/bin/env bash
# Validate the shutdown archive by default. Use --start after returning to Ubuntu.
set -euo pipefail
root=${1:?Pass the peer-clearance experiment directory}
mode=${2:---check}
root=$(realpath "$root")
here=$(cd "$(dirname "$0")" && pwd)
python_bin=${VISFLY_PYTHON:-"$HOME/miniconda3/envs/visfly/bin/python"}
reference=${VISFLY_REFERENCE:-"$(dirname "$root")/paper-crossing-20261123"}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-"/run/user/$(id -u)"}
export DBUS_SESSION_BUS_ADDRESS=${DBUS_SESSION_BUS_ADDRESS:-"unix:path=$XDG_RUNTIME_DIR/bus"}
for arm in control peer; do
    /usr/bin/python3 -B "$here/cold_resume_training.py" "$root" "$arm" --check-only
done
case "$mode" in
    --check) exit 0 ;;
    --start) ;;
    *) echo "Use --check or --start" >&2; exit 2 ;;
esac
for original in visfly-crossing-safe-control-20261125 visfly-crossing-safe-peer-v2-20261125; do
    if systemctl --user is-active --quiet "$original.service"; then
        echo "Original training process still exists. Leave it frozen until the intended OS switch." >&2
        exit 1
    fi
done
mkdir -p "$root/logs"
for arm in control peer; do
    unit="visfly-crossing-cold-$arm-20261125"
    if [[ -f "$root/train/$arm/complete.json" ]] || systemctl --user is-active --quiet "$unit"; then
        continue
    fi
    systemd-run --user --unit="$unit" --setenv="PYTHONPATH=$root/setup/renderer-lib" \
        --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=PYTHONUNBUFFERED=1 \
        --setenv=MAGNUM_LOG=quiet --setenv=HABITAT_SIM_LOG=quiet \
        --setenv=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        --property="StandardOutput=append:$root/logs/cold-$arm.log" \
        --property="StandardError=append:$root/logs/cold-$arm.log" \
        "$python_bin" -Bu "$here/cold_resume_training.py" "$root" "$arm"
done
unit=visfly-crossing-cold-evaluation-20261125
if [[ ! -f "$root/evaluation/complete.json" ]] && ! systemctl --user is-active --quiet "$unit"; then
    systemd-run --user --unit="$unit" --property="StandardOutput=append:$root/logs/evaluation.log" \
        --property="StandardError=append:$root/logs/evaluation.log" \
        /usr/bin/python3 -Bu "$here/run_evaluation.py" "$root" "$reference" --visfly-python "$python_bin"
fi
