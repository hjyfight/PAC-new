"""Recompute the requested ablations from existing raw metrics, read-only."""
from statistics import mean, stdev
from common import MODELS, ROOT, RUN, read, sha, write

NAMES = ("ncq_random3_20261025", "ncq_random3_sum255_20261025", "ncq_random3_sum255_extra5_20261026")


def main():
    evidence, metrics, runs = {}, {}, []
    for name, expected, per_model in zip(NAMES, (132, 66, 96), (22, 11, 16)):
        directory = ROOT / "experiments" / "NCQ" / name
        paths = sorted(directory.glob("sources/*/eval/*/*/metrics.json"))
        assert len(paths) == expected
        for path in paths:
            d = read(path)
            assert d["images"] == 288 and d["targets"] == 589
            assert d["patch_area_fraction"] == 0.25 and d["preprocess"] == "advpatch"
            assert d["imgsz"] == 640 and d["eot"] is False
            assert d["conf_thres"] == 0.001 and d["nms_iou"] == 0.6
            assert 0 <= d["map50"] <= 1
            metrics[name, d["detector"], d["patch_id"]] = d
            evidence[str(path.relative_to(ROOT))] = sha(path)
        logs = []
        for model in MODELS:
            assert read(directory / "sources" / model / "complete.json") == {"passed": True, "evaluations": per_model}
            log = directory / "logs" / (model + ".log")
            text = log.read_text(errors="replace")
            logs.append(dict(path=str(log.relative_to(ROOT)), sha256=sha(log),
                             nfs_cleanup_error="Device or resource busy" in text,
                             traceback_count=text.count("Traceback (most recent call last)")))
        runs.append(dict(directory=name, expected=expected, observed=len(paths),
                         all_six_completion_markers_passed=True, logs=logs))
    def value(name, model, item):
        return metrics[name, model, item]["map50_pct"]
    ncq = {key: mean(value(NAMES[0], model, key) for model in MODELS)
           for key in ("advpatch", "p1", "p2", "ncq1", "ncq2")}
    retained, shuffled, shuffled_bc1 = [], [], []
    for seed in range(10):
        group, bc_group = NAMES[1 if seed < 5 else 2], NAMES[0 if seed < 5 else 2]
        retained.append(mean(value(group, m, f"retained_sum255_s{seed}") for m in MODELS))
        shuffled.append(mean(value(group, m, f"shuffled_sum255_s{seed}") for m in MODELS))
        shuffled_bc1.append(mean(value(bc_group, m, f"shuffle_bc1_s{seed}") for m in MODELS))
    factorial = dict(fixed_bc1=ncq["p1"], retained_random_mean=mean(retained), retained_random_sd=stdev(retained),
        shuffled_bc1_mean=mean(shuffled_bc1), shuffled_random_mean=mean(shuffled),
        shuffled_random_sd=stdev(shuffled), color_cost=mean(retained)-ncq["p1"],
        structure_cost_fixed=mean(shuffled_bc1)-ncq["p1"],
        structure_cost_random=mean(b-a for a,b in zip(retained,shuffled)))
    for group in NAMES[1:]:
        for model in MODELS:
            for key in ("map50", "conditional_asr_numerator", "conditional_asr_denominator"):
                assert metrics[group, model, "clean"][key] == metrics[NAMES[0], model, "clean"][key]
    write(RUN / "existing_experiments_audit.json", dict(checked_date="2026-11-02", passed=True,
          runs=runs, raw_metrics_checked=len(evidence), metrics_sha256=evidence, ncq=ncq, factorial=factorial))
    print({"passed": True, "metrics": len(evidence), "ncq": ncq, "factorial": factorial})


if __name__ == "__main__":
    main()
