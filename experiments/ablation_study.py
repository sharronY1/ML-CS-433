"""Reproduce the four historical GD feature-engineering comparisons."""

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from ablation.rules import prepare
from helpers import load_csv_data
from implementations import logistic_regression

SEED = 42
MAX_ITERS = 5000
GAMMA = 0.01
THRESHOLDS = [0.1, 0.15, 0.2, 0.25, 0.3]
PAIRS = [
    ("GENHLTH", "PHYSHLTH"),
    ("GENHLTH", "DIFFWALK"),
    ("PHYSHLTH", "DIFFWALK"),
    ("GENHLTH", "CVDSTRK3"),
    ("GENHLTH", "DIABETE3"),
    ("_BMI5", "DIFFWALK"),
    ("BPHIGH4", "DIABETE3"),
    ("CVDSTRK3", "DIFFWALK"),
]
SQUARES = ["GENHLTH", "PHYSHLTH", "_BMI5"]


def add_interactions(x_tr, x_val, names):
    """Append the historical eight products and three squared features."""

    # 原始特征已经标准化，第一列为截距，所以索引加 1。
    def extra(x):
        columns = [
            x[:, names.index(a) + 1] * x[:, names.index(b) + 1] for a, b in PAIRS
        ]
        columns += [x[:, names.index(name) + 1] ** 2 for name in SQUARES]
        return np.column_stack(columns)

    train_extra = extra(x_tr)
    val_extra = extra(x_val)
    means = train_extra.mean(axis=0)
    stds = train_extra.std(axis=0)
    stds = np.where(stds == 0, 1, stds)
    return (
        np.column_stack([x_tr, (train_extra - means) / stds]),
        np.column_stack([x_val, (val_extra - means) / stds]),
    )


def evaluate(x_val, y_val, w):
    """Evaluate the five original validation-threshold candidates."""
    p = np.exp(-np.logaddexp(0, -(x_val @ w)))
    scores = []
    for threshold in THRESHOLDS:
        pred = p >= threshold
        tp = int(np.sum(pred & (y_val == 1)))
        fp = int(np.sum(pred & (y_val == 0)))
        fn = int(np.sum(~pred & (y_val == 1)))
        denominator = 2 * tp + fp + fn
        scores.append(
            dict(
                threshold=threshold,
                f1=2 * tp / denominator if denominator else 0.0,
                tp=tp,
                fp=fp,
                fn=fn,
            )
        )
    return scores


def main():
    """Train four separate GD models and save newly computed results."""
    data_path = PROJECT / "data/dataset_to_release/dataset"
    with (data_path / "x_train.csv").open(encoding="utf-8-sig", newline="") as f:
        names = next(csv.reader(f))[1:]
    raw, unused_test, labels, _, _ = load_csv_data(str(data_path))
    del unused_test

    # 保留历史划分的随机数调用顺序：先负类，再正类。
    rng = np.random.default_rng(SEED)
    negative = rng.permutation(np.where(labels == -1)[0])
    positive = rng.permutation(np.where(labels == 1)[0])
    nn = int(len(negative) * 0.8)
    np_ = int(len(positive) * 0.8)
    train_idx = np.concatenate([negative[:nn], positive[:np_]])
    val_idx = np.concatenate([negative[nn:], positive[np_:]])

    records = []
    threshold_records = []
    for variant in ["baseline", "unknown_only", "units_log", "health_pairs"]:
        cleaning = "units_log" if variant == "health_pairs" else variant
        x_tr, y_tr, x_val, y_val, *_ = prepare(
            raw, labels, names, train_idx, val_idx, cleaning
        )
        if variant == "health_pairs":
            x_tr, x_val = add_interactions(x_tr, x_val, names)
        w = np.zeros(x_tr.shape[1])
        # 分段仅用于显示进度；每段继续使用上次的权重。
        for done in range(0, MAX_ITERS, 1000):
            w, loss = logistic_regression(
                y_tr, x_tr, w, min(1000, MAX_ITERS - done), GAMMA
            )
            print(variant, "Iterations:", done + 1000, flush=True)
        scores = evaluate(x_val, y_val, w)
        best = max(scores, key=lambda row: row["f1"])
        record = dict(
            variant=variant,
            seed=SEED,
            training_rows=len(train_idx),
            validation_rows=len(val_idx),
            features=x_tr.shape[1],
            max_iters=MAX_ITERS,
            gamma=GAMMA,
            loss=float(loss),
            **best,
        )
        records.append(record)
        threshold_records.extend(dict(variant=variant, **score) for score in scores)
        print(variant, "F1:", best["f1"], flush=True)
        del x_tr, x_val, y_tr, y_val

    # 新运行的结果与历史记录分开保存，避免覆盖报告数字的原始来源。
    output = PROJECT / "results/ablation"
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in [
        ("reproduced_results.csv", records),
        ("reproduced_thresholds.csv", threshold_records),
    ]:
        with (output / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return records


if __name__ == "__main__":
    main()
