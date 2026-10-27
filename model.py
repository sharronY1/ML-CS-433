"""Original GD training function shared by the final entry point and validation experiments; importing this module does not trigger training."""

from pathlib import Path
import numpy as np
from implementations import logistic_regression

# 固定参数
MAX_ITERS = 2000
GAMMA = 0.09377751905420602
FIXED_THRESHOLD = 0.2043741811970397


# 使用全部已预处理的训练数据拟合逻辑回归，保存权重和阈值。
def train_full(project_dir):
    """Fit logistic regression on the entire preprocessed training dataset and save the learned weights and classification threshold."""
    # 确定路径和创建文件夹什么的
    project = Path(project_dir)
    with np.load(project / "cache/prepared_full_data.npz", allow_pickle=False) as data:
        x, y = data["x_tr"], data["y_tr_01"]
    if x.shape != (328135, 584) or y.shape != (328135,) or not np.isin(y, [0, 1]).all():
        raise ValueError(
            "Full training requires 328,135 samples, 584 features, and binary labels (0/1)."
        )
    # 初始化权重
    initial_w = np.zeros(x.shape[1])
    # 开始训练模型
    w, loss = logistic_regression(y, x, initial_w, MAX_ITERS, GAMMA)
    # 保存权重
    model_path = project / "models/model.npz"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(model_path, w=w, threshold=FIXED_THRESHOLD)
    print("Training completed. Loss:", loss)
    return model_path
