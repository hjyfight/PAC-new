#!/usr/bin/env bash
# Runs N independent seed-repeats of the full Table 1 sweep and evaluates each patch right
# after its group trains, so partial results stay usable if the sweep is stopped early.
# One group = {AdvPatch, CAPGen-P/R/T x Bc1/Bc2} x {letterbox, squash} = 14 patches.
# GPU g takes groups g, g+NGPU, g+2*NGPU, ... so every card stays busy to the end.
#
#   ./run_sweep.sh            # 100 groups on GPUs 0-6
#   ./run_sweep.sh 20         # 20 groups
#   ./run_sweep.sh 100 2 3    # 100 groups, only on GPUs 2 and 3

set -Eeuo pipefail

cd "$(dirname "$0")" || exit 1
N=100
[ $# -gt 0 ] && { N=$1; shift; }
if [ $# -gt 0 ]; then GPUS=("$@"); else GPUS=(0 1 2 3 4 5 6); fi

BASE=${SEED_BASE:-$(date +%s)}  # paired experiments may reuse an earlier seed sequence
PROJ=${PATCH_PROJECT:-runs/patch100}
VPROJ=${VAL_PROJECT:-runs/val100}
LOGS=${PATCH_LOGS:-logs100}
MODES=${PATCH_MODES:-"lb sq"}
OLD_PROJ=${COMPARE_PROJECT:-}
mkdir -p "$LOGS/train" "$LOGS/val" "$PROJ" "$VPROJ"

TRAIN=(--data data/inria.yaml --weights yolov5s.pt --patch-size 300 --imgsz 640
       --epochs 200 --batch-size 8 --lr 0.03 --workers 4 --exist-ok --project "$PROJ")
VAL=(--data data/inria.yaml --weights yolov5s.pt --imgsz 640 --batch-size 32
     --workers 4 --patch-frac 0.25 --no-eot --seed 0 --exist-ok --project "$VPROJ")

train_one() {
    n=$1
    shift
    echo "[TRAIN] $n $(date -Is)"
    python train_patch.py "${TRAIN[@]}" --name "$n" "$@" >"$LOGS/train/$n.log" 2>&1 || {
        echo "FAIL train $n -- see $LOGS/train/$n.log" >&2
        return 1
    }
    echo "[DONE] train $n $(date -Is)"
}

eval_one() {
    n=$1
    shift
    echo "[EVAL] $n $(date -Is)"
    python val.py "${VAL[@]}" --name "$n" "$@" >"$LOGS/val/$n.log" 2>&1 || {
        echo "FAIL val $n -- see $LOGS/val/$n.log" >&2
        return 1
    }
    echo "[DONE] eval $n $(date -Is)"
}

group() {
    gid=$1
    gpu=$2
    seed=$((BASE + gid))
    tag=$(printf "s%03d" "$gid")
    echo "$tag  seed $seed  gpu $gpu  $(date -Is)" >>"$PROJ/seeds.txt"

    for mode in $MODES; do
        a=(--device "$gpu" --seed "$seed")
        v=(--device "$gpu")
        [ "$mode" = sq ] && { a+=(--squash); v+=(--squash); }
        pre="${mode}_${tag}"

        # Re-evaluate old checkpoints with the same digital evaluation settings.
        if [ -n "$OLD_PROJ" ]; then
            for s in advpatch p1 p2 r1 r2 t1 t2; do
                eval_one "old_${pre}_${s}" "${v[@]}" --patch "$OLD_PROJ/${pre}_${s}/best.pt"
            done
        fi

        train_one "${pre}_advpatch" "${a[@]}" --method advpatch --nps-weight 0.01
        eval_one "${pre}_advpatch" "${v[@]}" --patch "$PROJ/${pre}_advpatch/best.pt"
        src="$PROJ/${pre}_advpatch/best.pt"
        for bc in 1 2; do
            train_one "${pre}_p${bc}" "${a[@]}" --method capgen-p --base-colors "bc$bc" --source "$src"
            eval_one "${pre}_p${bc}" "${v[@]}" --patch "$PROJ/${pre}_p${bc}/best.pt"
            train_one "${pre}_r${bc}" "${a[@]}" --method capgen-r --base-colors "bc$bc"
            eval_one "${pre}_r${bc}" "${v[@]}" --patch "$PROJ/${pre}_r${bc}/best.pt"
        done
        for bc in 1 2; do
            train_one "${pre}_t${bc}" "${a[@]}" --method capgen-t --base-colors "bc$bc" --tv-weight 0 --nps-weight 0
            eval_one "${pre}_t${bc}" "${v[@]}" --patch "$PROJ/${pre}_t${bc}/best.pt"
        done
    done
    echo "[$tag] done on gpu $gpu  $(date -Is)"
}

worker() {  # takes groups slot, slot+NGPU, slot+2*NGPU, ...
    slot=$1
    ngpu=${#GPUS[@]}
    gpu=${GPUS[$slot]}
    for ((gid = slot; gid < N; gid += ngpu)); do group "$gid" "$gpu"; done
}

# Clean and gray-patch baselines are seed-independent, so run them once up front.
for mode in $MODES; do
    b=(--device "${GPUS[0]}")
    [ "$mode" = sq ] && b+=(--squash)
    eval_one "${mode}_clean" "${b[@]}"
    eval_one "${mode}_gray" "${b[@]}" --patch gray
done
echo "baselines done, starting $N groups on GPUs ${GPUS[*]}"

pids=()
for ((i = 0; i < ${#GPUS[@]}; i++)); do
    worker "$i" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then
    echo "sweep failed: at least one worker failed; inspect the logs" >&2
    exit 1
fi
echo "sweep finished: $N groups"
