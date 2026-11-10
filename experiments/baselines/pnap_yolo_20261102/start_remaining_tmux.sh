#!/usr/bin/env bash
set -Eeuo pipefail
cd /datacc05/huotianlin/code-hu/PAC-yolo
session=pnap_20261102
run=experiments/baselines/pnap_yolo_20261102

if tmux has-session -t "$session" 2>/dev/null && (( $# == 0 )); then
    printf 'Session %s already exists; attach with: tmux attach -t %s\n' "$session" "$session" >&2
    exit 1
fi

# Keep the session alive even if a short validation fails before other windows start.
if ! tmux has-session -t "$session" 2>/dev/null; then
    tmux new-session -d -s "$session" -n monitor -c "$PWD"
fi

sources=(yolov2 yolov3 yolov5m fasterrcnn yolov4)
if (( $# > 0 )); then sources=("$@"); fi
for source_model in "${sources[@]}"; do
    case "$source_model" in
        yolov2) gpu=3 ;;
        yolov3) gpu=4 ;;
        yolov5m) gpu=5 ;;
        fasterrcnn) gpu=6 ;;
        yolov4) gpu=7 ;;
        *) printf 'Unknown source: %s\n' "$source_model" >&2; exit 1 ;;
    esac
    run_id=${source_model}_gpu${gpu}
    # Expandable segments reduce allocator fragmentation without changing precision/batch size.
    offload="--offload-detector"
    if [[ "$source_model" == fasterrcnn ]]; then offload=""; fi
    command="set -o pipefail; env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $run/.venv/bin/python -B $run/controller.py --sources $source_model --gpus $gpu --run-id $run_id --smoke-steps 2 --record-launch $offload 2>&1 | tee -a $run/logs/controller_$run_id.log"
    if tmux list-windows -t "$session" -F '#{window_name}' | rg -qx "$source_model"; then
        # Without -k, tmux refuses to respawn a window whose process is still alive.
        tmux respawn-window -t "$session:$source_model" -c "$PWD" "bash -c '$command'"
    else
        tmux new-window -d -t "$session" -n "$source_model" -c "$PWD" "bash -c '$command'"
    fi
    tmux set-option -w -t "$session:$source_model" remain-on-exit on
done

printf 'Started %s; attach with: tmux attach -t %s\n' "$session" "$session"
tmux list-windows -t "$session"
