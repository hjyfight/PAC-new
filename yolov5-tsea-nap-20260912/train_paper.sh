#!/usr/bin/env bash
# Trains one full method group per GPU: AdvPatch + CAPGen-P/R/T over base colors Bc1 and Bc2,
# in both letterbox (lb_) and squash (sq_) framing. Groups start 20 s apart and take their seed
# from the launch timestamp, so each GPU is an independent repeat of the whole Table 1 sweep.
# CAPGen-R shares its group's seed with CAPGen-T, so both start from the same color matrix.
#
#   ./train_paper.sh          # GPUs 0-6
#   ./train_paper.sh 3 4      # a subset

cd "$(dirname "$0")" || exit 1
if [ $# -gt 0 ]; then GPUS=("$@"); else GPUS=(0 1 2 3 4 5 6); fi

COMMON=(--data data/inria.yaml --weights yolov5s.pt --patch-size 300 --imgsz 640
        --epochs 200 --batch-size 8 --lr 0.03 --workers 4 --exist-ok)

mkdir -p logs runs/patch

run() {  # run <name> <extra train_patch.py args...>
    name=$1
    shift
    if python train_patch.py "${COMMON[@]}" --name "$name" "$@" >"logs/$name.log" 2>&1; then
        echo "  ok   $name"
    else
        echo "  FAIL $name -- see logs/$name.log"
    fi
}

group() {
    gpu=$1
    sleep $((20 * gpu))  # stagger the groups; the launch time is what makes each seed differ
    seed=$(date +%s)
    echo "gpu $gpu  seed $seed  $(date -Is)" >>runs/patch/seeds.txt
    echo "[gpu $gpu] start, seed $seed"

    for mode in lb sq; do
        pre="${mode}_g${gpu}"
        args=(--device "$gpu" --seed "$seed")
        [ "$mode" = sq ] && args+=(--squash)

        # AdvPatch: color-unrestricted, official NPS weight.
        run "${pre}_advpatch" "${args[@]}" --method advpatch --nps-weight 0.01

        for bc in 1 2; do
            # CAPGen-P: recolor this group's AdvPatch, no optimization.
            src="runs/patch/${pre}_advpatch/best.pt"
            if [ -f "$src" ]; then
                run "${pre}_p${bc}" "${args[@]}" --method capgen-p --base-colors "bc$bc" --source "$src"
            else
                echo "  SKIP ${pre}_p${bc} -- ${pre}_advpatch produced no source patch"
            fi
            # CAPGen-R: untrained matrix. CAPGen-T: trained, NPS off since the palette is fixed.
            run "${pre}_r${bc}" "${args[@]}" --method capgen-r --base-colors "bc$bc"
            run "${pre}_t${bc}" "${args[@]}" --method capgen-t --base-colors "bc$bc" --tv-weight 0 --nps-weight 0
        done
    done
    echo "[gpu $gpu] done"
}

for g in "${GPUS[@]}"; do group "$g" & done
wait
echo "all groups finished"
