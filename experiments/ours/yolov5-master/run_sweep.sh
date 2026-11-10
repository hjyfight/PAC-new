#!/usr/bin/env bash
# Runs N independent seed-repeats of the full Table 1 sweep and evaluates each patch right
# after its group trains, so partial results stay usable if the sweep is stopped early.
# One group = {AdvPatch, CAPGen-P/R/T x Bc1/Bc2} x {letterbox, squash} = 14 patches.
# GPU g takes groups g, g+NGPU, g+2*NGPU, ... so every card stays busy to the end.
#
#   ./run_sweep.sh            # 100 groups on GPUs 0-6
#   ./run_sweep.sh 20         # 20 groups
#   ./run_sweep.sh 100 2 3    # 100 groups, only on GPUs 2 and 3

cd "$(dirname "$0")" || exit 1
N=100
[ $# -gt 0 ] && { N=$1; shift; }
if [ $# -gt 0 ]; then GPUS=("$@"); else GPUS=(0 1 2 3 4 5 6); fi

BASE=$(date +%s)  # seed of group i is BASE+i: all distinct, and traceable to the launch time
PROJ=runs/patch100
VPROJ=runs/val100
LOGS=logs100
mkdir -p "$LOGS/train" "$LOGS/val" "$PROJ" "$VPROJ"

TRAIN=(--data data/inria.yaml --weights yolov5s.pt --patch-size 300 --imgsz 640
       --epochs 200 --batch-size 8 --lr 0.03 --workers 4 --exist-ok --project "$PROJ")
VAL=(--data data/inria.yaml --weights yolov5s.pt --imgsz 640 --batch-size 32
     --workers 4 --patch-frac 0.25 --exist-ok --project "$VPROJ")

train_one() {
    n=$1
    shift
    python train_patch.py "${TRAIN[@]}" --name "$n" "$@" >"$LOGS/train/$n.log" 2>&1 ||
        echo "  FAIL train $n -- see $LOGS/train/$n.log"
}

eval_one() {
    n=$1
    shift
    python val.py "${VAL[@]}" --name "$n" "$@" >"$LOGS/val/$n.log" 2>&1 ||
        echo "  FAIL val $n -- see $LOGS/val/$n.log"
}

group() {
    gid=$1
    gpu=$2
    seed=$((BASE + gid))
    tag=$(printf "s%03d" "$gid")
    echo "$tag  seed $seed  gpu $gpu  $(date -Is)" >>"$PROJ/seeds.txt"

    for mode in lb sq; do
        a=(--device "$gpu" --seed "$seed")
        v=(--device "$gpu")
        [ "$mode" = sq ] && { a+=(--squash); v+=(--squash); }
        pre="${mode}_${tag}"

        train_one "${pre}_advpatch" "${a[@]}" --method advpatch --nps-weight 0.01
        for bc in 1 2; do
            src="$PROJ/${pre}_advpatch/best.pt"
            [ -f "$src" ] &&
                train_one "${pre}_p${bc}" "${a[@]}" --method capgen-p --base-colors "bc$bc" --source "$src"
            train_one "${pre}_r${bc}" "${a[@]}" --method capgen-r --base-colors "bc$bc"
            train_one "${pre}_t${bc}" "${a[@]}" --method capgen-t --base-colors "bc$bc" --nps-weight 0
        done

        for s in advpatch p1 p2 r1 r2 t1 t2; do
            [ -f "$PROJ/${pre}_${s}/best.pt" ] &&
                eval_one "${pre}_${s}" "${v[@]}" --patch "$PROJ/${pre}_${s}/best.pt"
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
for mode in lb sq; do
    b=(--device "${GPUS[0]}")
    [ "$mode" = sq ] && b+=(--squash)
    eval_one "${mode}_clean" "${b[@]}"
    eval_one "${mode}_gray" "${b[@]}" --patch gray
done
echo "baselines done, starting $N groups on GPUs ${GPUS[*]}"

for ((i = 0; i < ${#GPUS[@]}; i++)); do worker "$i" & done
wait
echo "sweep finished: $N groups"
