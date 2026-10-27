import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
RULES = json.loads((ROOT / "candidate_rules.json").read_text(encoding="utf-8"))


def apply_rules(x, names, variant):
    x = x.copy()
    index = {name: i for i, name in enumerate(names)}
    j = index["PHYSHLTH"]
    x[np.isin(x[:, j], [77, 99]), j] = np.nan
    x[x[:, j] == 88, j] = 0
    if variant == "baseline":
        return x
    for name, codes in RULES["unknown"].items():
        if variant == "core" and index[name] > 106 and not name.startswith("_"):
            continue
        j = index[name]
        x[np.isin(x[:, j], codes), j] = np.nan
    # CSV 中部分派生列已去除隐含小数位，因此按 CSV 的实际单位处理哨兵值。
    scaled_missing = {
        "MAXVO2_": 999,
        "FC60_": 999,
        "PAFREQ1_": 99,
        "PAFREQ2_": 99,
        "STRFREQ_": 99,
    }
    for name, value in scaled_missing.items():
        if variant == "core" and not name.startswith("_"):
            continue
        j = index[name]
        x[x[:, j] == value, j] = np.nan
    if variant == "unknown_only":
        return x
    for name, codes in RULES["zero"].items():
        if variant == "core" and index[name] > 106:
            continue
        j = index[name]
        x[np.isin(x[:, j], codes), j] = 0
    if variant not in ("units", "units_log"):
        return x
    # 把原始体重、身高的单位前缀解码成统一的千克、厘米。
    j = index["WEIGHT2"]
    a = x[:, j].copy()
    x[:, j] = np.nan
    m = (a >= 50) & (a <= 999)
    x[m, j] = a[m] * 0.45359237
    m = (a >= 9000) & (a <= 9998)
    x[m, j] = a[m] - 9000
    j = index["HEIGHT3"]
    a = x[:, j].copy()
    x[:, j] = np.nan
    m = (a >= 200) & (a <= 711) & (a % 100 < 12)
    x[m, j] = (a[m] // 100 * 12 + a[m] % 100) * 2.54
    m = (a >= 9000) & (a <= 9998)
    x[m, j] = a[m] - 9000
    # 周/月频率统一到每天；零频率保持为零。
    for name in ["ALCDAY5", "EXEROFT1", "EXEROFT2", "STRENGTH"]:
        j = index[name]
        a = x[:, j].copy()
        out = np.full(len(a), np.nan)
        out[a == 0] = 0
        m = (a >= 101) & (a <= 199)
        out[m] = (a[m] - 100) / 7
        m = (a >= 201) & (a <= 299)
        out[m] = (a[m] - 200) / 30
        x[:, j] = out
    for name in ["FRUITJU1", "FRUIT1", "FVBEANS", "FVGREEN", "FVORANG", "VEGETAB1"]:
        j = index[name]
        a = x[:, j].copy()
        out = np.full(len(a), np.nan)
        out[a == 0] = 0
        m = (a >= 101) & (a <= 199)
        out[m] = a[m] - 100
        m = (a >= 201) & (a <= 299)
        out[m] = (a[m] - 200) / 7
        m = (a >= 301) & (a <= 399)
        out[m] = (a[m] - 300) / 30
        # 少于每月一次没有确切频率，不凭空指定一个数值，留待中位数填补。
        x[:, j] = out
    for name in ["EXERHMM1", "EXERHMM2"]:
        j = index[name]
        a = x[:, j].copy()
        x[:, j] = np.floor(a / 100) * 60 + a % 100
    if variant == "units_log":
        for name in [
            "WEIGHT2",
            "WTKG3",
            "_BMI5",
            "ALCDAY5",
            "DROCDY3_",
            "_DRNKWEK",
            "EXEROFT1",
            "EXEROFT2",
            "STRENGTH",
            "EXERHMM1",
            "EXERHMM2",
        ]:
            j = index[name]
            x[:, j] = np.log1p(x[:, j])
    return x


def prepare(raw, labels, names, train_idx, val_idx, variant):
    xtr = apply_rules(raw[train_idx], names, variant)
    xval = apply_rules(raw[val_idx], names, variant)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(xtr, axis=0)
    medians = np.where(np.isnan(medians), 0, medians)
    xtr = np.where(np.isnan(xtr), medians, xtr)
    xval = np.where(np.isnan(xval), medians, xval)
    means = np.mean(xtr, axis=0)
    stds = np.std(xtr, axis=0)
    stds = np.where(stds == 0, 1, stds)
    xtr = np.ascontiguousarray(
        np.column_stack([np.ones(len(xtr)), (xtr - means) / stds])
    )
    xval = np.ascontiguousarray(
        np.column_stack([np.ones(len(xval)), (xval - means) / stds])
    )
    return (
        xtr,
        (labels[train_idx] + 1) / 2,
        xval,
        (labels[val_idx] + 1) / 2,
        medians,
        means,
        stds,
    )
