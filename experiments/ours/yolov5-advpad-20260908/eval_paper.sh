#!/usr/bin/env bash
# Evaluates every patch produced by train_paper.sh, one group per GPU, plus the clean and
# gray-patch baselines. Framing (letterbox / squash) and EOT match how each patch was trained.
#
#   ./eval_paper.sh          # GPUs 0-6
#   ./eval_paper.sh 3 4      # a subset

cd "$(dirname "$0")" || exit 1
if [ $# -gt 0 ]; then GPUS=("$@"); else GPUS=(0 1 2 3 4 5 6); fi

COMMON=(--data data/inria.yaml --weights yolov5s.pt --imgsz 640 --batch-size 32
        --workers 4 --patch-frac 0.25 --project runs/val --exist-ok)

mkdir -p logs/val

ev() {  # ev <tag> <extra val.py args...>
    tag=$1
    shift
    if python val.py "${COMMON[@]}" --name "$tag" "$@" >"logs/val/$tag.log" 2>&1; then
        echo "  ok   $tag"
    else
        echo "  FAIL $tag -- see logs/val/$tag.log"
    fi
}

# Baselines: no patch at all, and the gray Table 1 control, in both framings.
for mode in lb sq; do
    base=(--device "${GPUS[0]}")
    [ "$mode" = sq ] && base+=(--squash)
    ev "${mode}_clean" "${base[@]}"
    ev "${mode}_gray" "${base[@]}" --patch gray
done

group() {
    gpu=$1
    echo "[gpu $gpu] start"
    for mode in lb sq; do
        args=(--device "$gpu")
        [ "$mode" = sq ] && args+=(--squash)
        for s in advpatch p1 p2 r1 r2 t1 t2; do
            ev "${mode}_g${gpu}_${s}" "${args[@]}" --patch "runs/patch/${mode}_g${gpu}_${s}/best.pt"
        done
    done
    echo "[gpu $gpu] done"
}

for g in "${GPUS[@]}"; do group "$g" & done
wait
echo "all evaluations finished"
