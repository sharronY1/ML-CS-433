# 独立复现旧333列六方法基线；不调用或修改最终训练流程。
"""Reproduce the six-method baseline with the frozen 333-feature protocol."""

from pathlib import Path
import csv
import sys
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import implementations as impl
from helpers import load_csv_data
from run import clean_columns

# 使用原校准集选出的设置，不重复超参数搜索，不读取旧F1作为本次结果。
SETTINGS = {
    "least_squares": {},
    "ridge_regression": {"lambda_": 0.0001},
    "mean_squared_error_gd": {"max_iters": 400, "gamma": 0.055181689913409604},
    "mean_squared_error_sgd": {"max_iters": 100000, "gamma": 1.3826195184992981e-05},
    "logistic_regression": {"max_iters": 400, "gamma": 0.22072675965363842},
    "reg_logistic_regression": {
        "lambda_": 0.001,
        "max_iters": 400,
        "gamma": 0.22061854539442863,
    },
}


def prepare_333(raw, labels, names):
    # 历史方案：321原始列，加11个交互/平方项，再加截距；无One-Hot、缺失标记或最后标准化。
    """Rebuild the historical features and select the fixed shared splits."""
    x_train = clean_columns(raw, names)
    y_train = labels

    # 随机数
    rng = np.random.default_rng(42)
    # 索引打乱
    negative = rng.permutation(np.where(y_train == -1)[0])
    positive = rng.permutation(np.where(y_train == 1)[0])
    # 算80%的数量，每个类别
    n_negative = int(len(negative) * 0.8)
    n_positive = int(len(positive) * 0.8)
    # 去索引出来
    train_idx = np.concatenate([negative[:n_negative], positive[:n_positive]])
    val_idx = np.concatenate([negative[n_negative:], positive[n_positive:]])
    # 取值
    x_tr = x_train[train_idx]
    y_tr = y_train[train_idx]

    x_val = x_train[val_idx]
    y_val = y_train[val_idx]

    # 算中位数填补缺失值
    medians = np.nanmedian(x_tr, axis=0)
    # 特殊情况一列全是nan，设为0
    medians = np.where(np.isnan(medians), 0, medians)
    # 补充进去
    x_tr = np.where(np.isnan(x_tr), medians, x_tr)
    x_val = np.where(np.isnan(x_val), medians, x_val)
    # 标准化
    means = np.mean(x_tr, axis=0)
    stds = np.std(x_tr, axis=0)
    # 防止某一列一样全是0
    stds = np.where(stds == 0, 1, stds)
    x_tr = (x_tr - means) / stds
    x_val = (x_val - means) / stds

    # ===== 新增：经验证的交互与非线性特征，保留所有原有特征 =====
    # 在原有标准化后生成；新增列的均值和标准差只用训练部分计算。

    # 新增空列
    _extra_tr = []
    _extra_val = []
    # 健康指标之间的交互项，两个原有主特征仍然保留。一共是8对新的，假设其中的两个特征会互相影响
    for _left, _right in [
        ["GENHLTH", "PHYSHLTH"],
        ["GENHLTH", "DIFFWALK"],
        ["PHYSHLTH", "DIFFWALK"],
        ["GENHLTH", "CVDSTRK3"],
        ["GENHLTH", "DIABETE3"],
        ["_BMI5", "DIFFWALK"],
        ["BPHIGH4", "DIABETE3"],
        ["CVDSTRK3", "DIFFWALK"],
    ]:
        _l = names.index(_left)
        _r = names.index(_right)
        _extra_tr.append(x_tr[:, _l] * x_tr[:, _r])
        _extra_val.append(x_val[:, _l] * x_val[:, _r])
    # 平方项允许单个指标的作用随取值发生弯曲。
    for _name in ["GENHLTH", "PHYSHLTH", "_BMI5"]:
        _j = names.index(_name)
        _extra_tr.append(x_tr[:, _j] ** 2)
        _extra_val.append(x_val[:, _j] ** 2)
    _extra_tr = np.column_stack(_extra_tr)
    _extra_val = np.column_stack(_extra_val)
    _extra_means = np.mean(_extra_tr, axis=0)
    _extra_stds = np.std(_extra_tr, axis=0)
    _extra_stds = np.where(_extra_stds == 0, 1, _extra_stds)
    x_tr = np.column_stack([x_tr, (_extra_tr - _extra_means) / _extra_stds])
    x_val = np.column_stack([x_val, (_extra_val - _extra_means) / _extra_stds])
    # ===== 年龄交互特征结束 =====

    # 加截距
    x_tr = np.column_stack([np.ones(len(x_tr)), x_tr])
    x_val = np.column_stack([np.ones(len(x_val)), x_val])
    # 转换成01
    y_tr_01 = (y_tr + 1) / 2
    y_val_01 = (y_val + 1) / 2

    # 在原80%训练部分内，按当时的顺序抽取40000行拟合和15000行校准。
    rng = np.random.default_rng(20261025)
    pos = rng.permutation(np.where(y_tr_01 == 1)[0])
    neg = rng.permutation(np.where(y_tr_01 == 0)[0])
    small = np.concatenate([pos[:3532], neg[:36468]])
    calibration = np.concatenate([pos[3532:4857], neg[36468:50143]])
    return dict(
        x_train=x_tr[small],
        y_train=y_tr_01[small],
        x_calib=x_tr[calibration],
        y_calib=y_tr_01[calibration],
        x_val=x_val,
        y_val=y_val_01,
    )


def choose_threshold(score, y):
    # 同概率样本整体处理，只根据校准集F1选阈值。
    """Select a threshold by calibration F1, keeping tied scores together."""
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    tp = np.cumsum(y[order] == 1)
    ends = np.r_[np.flatnonzero(sorted_score[:-1] != sorted_score[1:]), len(y) - 1]
    f1 = 2 * tp[ends] / (ends + 1 + np.sum(y == 1))
    best = ends[int(np.argmax(f1))]
    return float(sorted_score[best]), float(np.max(f1))


def metrics(score, y, threshold):
    """Compute validation precision, recall, and F1 at a fixed threshold."""
    pred = score >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return dict(
        precision=tp / (tp + fp) if tp + fp else 0.0,
        recall=tp / (tp + fn) if tp + fn else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    )


def train_method(name, data, settings):
    """Call the course implementation, then calibrate and evaluate it."""
    x, y = data["x_train"], data["y_train"]
    logistic = name in ["logistic_regression", "reg_logistic_regression"]
    target = y if logistic else 2 * y - 1
    initial_w = np.zeros(x.shape[1])
    record = dict(
        method=name,
        status="failed",
        gamma=settings.get("gamma", ""),
        max_iters=settings.get("max_iters", ""),
        lambda_=settings.get("lambda_", ""),
        seed=2026 if name == "mean_squared_error_sgd" else "",
        threshold="",
        calibration_f1="",
        precision="",
        recall="",
        f1="",
        loss="",
        error="",
    )
    try:
        # 不复制算法，直接调用根目录implementations.py中的对应函数。
        if name == "least_squares":
            w, loss = impl.least_squares(target, x)
        elif name == "ridge_regression":
            w, loss = impl.ridge_regression(target, x, settings["lambda_"])
        elif name == "reg_logistic_regression":
            w, loss = impl.reg_logistic_regression(
                target,
                x,
                settings["lambda_"],
                initial_w,
                settings["max_iters"],
                settings["gamma"],
            )
        else:
            if name == "mean_squared_error_sgd":
                np.random.seed(2026)
            w, loss = getattr(impl, name)(
                target, x, initial_w, settings["max_iters"], settings["gamma"]
            )
        cal_score, val_score = data["x_calib"] @ w, data["x_val"] @ w
        if logistic:
            cal_score = np.exp(-np.logaddexp(0, -cal_score))
            val_score = np.exp(-np.logaddexp(0, -val_score))
        threshold, cal_f1 = choose_threshold(cal_score, data["y_calib"])
        record.update(
            status="ok",
            threshold=threshold,
            calibration_f1=cal_f1,
            loss=float(loss),
            **metrics(val_score, data["y_val"], threshold),
        )
    except (
        np.linalg.LinAlgError,
        FloatingPointError,
        OverflowError,
        ValueError,
    ) as error:
        # 如实记录最小二乘等方法的失败，不用伪逆替换原函数。
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def main():
    """Read raw data, train all six methods, and write fresh comparison results."""
    data_path = PROJECT / "data/dataset_to_release/dataset"
    # 读取数据
    with (data_path / "x_train.csv").open(encoding="utf-8-sig", newline="") as f:
        names = next(csv.reader(f))[1:]
    raw, _, labels, _, _ = load_csv_data(str(data_path))
    data = prepare_333(raw, labels, names)
    del raw, labels
    # 运行六个模型
    records = []
    for name, settings in SETTINGS.items():
        record = train_method(name, data, settings)
        records.append(record)
        print(name, "F1:", record["f1"], "Status:", record["status"])
    # 保存实验结果
    output = PROJECT / "results/baseline_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


if __name__ == "__main__":
    main()
