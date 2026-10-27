# 8:2数据准备、GD训练与F1评估；不生成提交文件。
"""Prepare an 80/20 split, train GD, and evaluate F1 without generating a submission."""

from pathlib import Path
import sys
import json
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

SEED = 42


# 封装一下sigmond公式
def probabilities(x, w):
    # 稳定计算逻辑回归的正类概率。
    """Compute positive-class probabilities using a numerically stable sigmoid."""
    return np.exp(-np.logaddexp(0, -(x @ w)))


def split_indices(y_train):
    # 随机数
    rng = np.random.default_rng(SEED)
    # 索引打乱
    negative = rng.permutation(np.where(y_train == -1)[0])
    positive = rng.permutation(np.where(y_train == 1)[0])
    # 算80%的数量，每个类别
    n_negative = int(len(negative) * 0.8)
    n_positive = int(len(positive) * 0.8)
    # 去索引出来
    train_idx = np.concatenate([negative[:n_negative], positive[:n_positive]])
    val_idx = np.concatenate([negative[n_negative:], positive[n_positive:]])
    return train_idx, val_idx


def split_fit_calibration(y):
    # 沿用已记录的 15000 行分层校准集，其余训练行用于拟合。
    """Reuse the recorded stratified 15,000-row calibration split and fit on the remaining rows."""
    if len(y) != 262508:
        raise ValueError(
            "Validation requires the original 80% training portion from the seed=42 split."
        )
    rng = np.random.default_rng(20261025)
    positive = rng.permutation(np.where(y == 1)[0])
    negative = rng.permutation(np.where(y == 0)[0])
    calibration = np.concatenate([positive[3532:4857], negative[36468:50143]])
    fitting = np.setdiff1d(np.arange(len(y)), calibration)
    return fitting, calibration


def choose_threshold(p, y):
    # 仅根据校准集 F1 选择阈值，相同概率作为整体处理。
    """Select a threshold by calibration F1, keeping equal probabilities together."""
    order = np.argsort(-p, kind="stable")
    sorted_p = p[order]
    tp = np.cumsum(y[order] == 1)
    ends = np.r_[np.flatnonzero(sorted_p[:-1] != sorted_p[1:]), len(y) - 1]
    f1 = 2 * tp[ends] / (ends + 1 + np.sum(y == 1))
    best = int(ends[np.argmax(f1)])
    return float(sorted_p[best]), float(np.max(f1))


def metrics(p, y, threshold):
    # 计算独立于阈值选择步骤的 Precision、Recall 和 F1。
    """Compute precision, recall, and F1 without selecting a threshold."""
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    )


def prepare_validation_data(data_path):
    from run import load_data, prepare_features

    x_train, y_train, names = load_data(data_path)
    train_idx, val_idx = split_indices(y_train)
    # 取值
    x_tr = x_train[train_idx]
    y_tr = y_train[train_idx]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]
    fitting, _ = split_fit_calibration((y_tr + 1) / 2)
    return prepare_features(
        x_tr,
        y_tr,
        names,
        PROJECT / "cache/prepared_validation_data.npz",
        PROJECT / "models/validation",
        data_path,
        x_val=x_val,
        y_val=y_val,
        standardization_rows=fitting,
    )


def main():
    from model import GAMMA, MAX_ITERS
    from implementations import logistic_regression

    data_path = PROJECT / "data/dataset_to_release/dataset"
    prepare_validation_data(data_path)
    with np.load(PROJECT / "cache/prepared_validation_data.npz") as data:
        x_tr = data["x_tr"]
        y_tr = data["y_tr_01"]
        x_val = data["x_val"]
        y_val = data["y_val_01"]

    # 为复现已有结果，沿用80%训练部分内的15000行校准集。
    fitting, calibration = split_fit_calibration(y_tr)
    initial_w = np.zeros(x_tr.shape[1])
    w, loss = logistic_regression(
        y_tr[fitting], x_tr[fitting], initial_w, MAX_ITERS, GAMMA
    )
    # 阈值只用校准集选择，再到20%的验证集计算指标。
    threshold, calibration_f1 = choose_threshold(
        probabilities(x_tr[calibration], w), y_tr[calibration]
    )
    scores = metrics(probabilities(x_val, w), y_val, threshold)
    print("Training loss:", loss)
    print("Threshold:", threshold)
    print("Precision：", scores["precision"])
    print("Recall：", scores["recall"])
    print("F1：", scores["f1"])

    # 验证结果和模型单独保存，不覆盖全量模型。
    model_dir = PROJECT / "models/validation"
    np.savez(
        model_dir / "model.npz",
        w=w,
        threshold=threshold,
        gamma=GAMMA,
        max_iters=MAX_ITERS,
        training_rows=len(fitting),
        fitting_rows=fitting,
        calibration_rows=calibration,
    )
    result = dict(
        model="GD logistic regression",
        seed=SEED,
        gamma=GAMMA,
        max_iters=MAX_ITERS,
        threshold=threshold,
        training_rows=len(fitting),
        calibration_rows=len(calibration),
        validation_rows=len(y_val),
        training_loss=float(loss),
        calibration_f1=calibration_f1,
        validation=scores,
    )
    result_dir = PROJECT / "results/validation"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "gd_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    main()
