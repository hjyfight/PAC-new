#!/usr/bin/env bash
# Trains and evaluates the CAPGen-T series with no explicit regularizer
# (--tv-weight 0 --nps-weight 0), one group per GPU.
#
# CAPGen states that robustness and stealth come from the EOT framework and the color
# constraint, NOT from explicit loss terms -- the objective is the detection loss alone.
# The first sweep carried AdvPatch's TV weight (2.5) into CAPGen-T, which actively
# penalizes the hard color boundaries that are CAPGen's defining structure (its TV ran
# an order of magnitude above AdvPatch's and left the 0.1 floor in 19.7% of epochs).
#
# One group = {Bc1, Bc2} x {letterbox, squash} = 4 trainings + 4 evaluations.
# Groups reuse their original seed from seeds.txt, so CAPGen-R remains exactly
# CAPGen-T's initialization and the existing AdvPatch/P/R runs stay comparable.
#
#   ./retrain_t.sh            # 6 groups, one per GPU on 0-5
#   ./retrain_t.sh 10 0 1 2   # 10 groups spread over GPUs 0,1,2

cd "$(dirname "$0")" || exit 1
N=6
[ $# -gt 0 ] && { N=$1; shift; }
if [ $# -gt 0 ]; then GPUS=("$@"); else GPUS=(0 1 2 3 4 5); fi

PROJ=runs/patch100
VPROJ=runs/val100
LOGS=logs100
mkdir -p "$LOGS/train" "$LOGS/val"

TRAIN=(--data data/inria.yaml --weights yolov5s.pt --patch-size 300 --imgsz 640
       --epochs 200 --batch-size 8 --lr 0.03 --workers 4 --exist-ok --project "$PROJ"
       --method capgen-t --tv-weight 0 --nps-weight 0)
VAL=(--data data/inria.yaml --weights yolov5s.pt --imgsz 640 --batch-size 32
     --workers 4 --patch-frac 0.25 --exist-ok --project "$VPROJ")

group() {
    tag=$1
    seed=$2
    gpu=$3
    echo "[$tag] start on gpu $gpu (seed $seed)  $(date -Is)"
    for mode in lb sq; do
        a=(--device "$gpu" --seed "$seed")
        v=(--device "$gpu")
        [ "$mode" = sq ] && { a+=(--squash); v+=(--squash); }
        for bc in 1 2; do
            n="${mode}_${tag}_t${bc}"
            python train_patch.py "${TRAIN[@]}" "${a[@]}" --base-colors "bc$bc" --name "$n" \
                >"$LOGS/train/$n.log" 2>&1 || { echo "  FAIL train $n"; continue; }
            python val.py "${VAL[@]}" "${v[@]}" --patch "$PROJ/$n/best.pt" --name "$n" \
                >"$LOGS/val/$n.log" 2>&1 || echo "  FAIL val $n"
        done
    done
    echo "[$tag] done on gpu $gpu  $(date -Is)"
}

# seeds.txt lines: "sNNN  seed <n>  gpu <g>  <iso8601>"; take the first N groups by tag.
mapfile -t ROWS < <(awk '{print $1, $3}' "$PROJ/seeds.txt" | sort -u | head -n "$N")
echo "CAPGen-T (tv=0, nps=0): ${#ROWS[@]} groups on GPUs ${GPUS[*]}"

worker() {
    slot=$1
    ngpu=${#GPUS[@]}
    for ((i = slot; i < ${#ROWS[@]}; i += ngpu)); do
        set -- ${ROWS[$i]}
        group "$1" "$2" "${GPUS[$slot]}"
    done
}

for ((i = 0; i < ${#GPUS[@]}; i++)); do worker "$i" & done
wait
echo "CAPGen-T retrain finished: ${#ROWS[@]} groups"
