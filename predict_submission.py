# 复用已训练逻辑回归模型生成提交CSV；不执行run.py或训练模型。
"""Generate a submission CSV using a trained logistic regression model without running training."""

import csv
from pathlib import Path
import numpy as np
from helpers import load_csv_data, create_csv_submission
from run import _process_features


# 传入已有的 params，让 _process_features() 使用这些参数转换数据
def transform_features(x, params):
    # 使用训练集保存的参数转换数据。
    """Transform data using preprocessing parameters saved from training."""
    names = params["names"].tolist()
    result, _ = _process_features(x, names, params=params)
    return result


# 读取之前的参数做个字典捏
def load_preprocessing(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


# 封装一个sigmond
def logistic_probabilities(x, w):
    # 稳定计算逻辑回归概率，不调用任何训练或优化器。
    """Compute stable logistic regression probabilities without running an optimizer."""
    return np.exp(-np.logaddexp(0, -(x @ w)))


def generate_submission(data_path, output, project_dir=None):
    # 读取测试集，应用训练参数，预测并保存提交CSV。
    """Load test data, apply training parameters, predict labels, and save the submission CSV."""
    # 确定文件路径
    project = (
        Path(project_dir)
        if project_dir is not None
        else Path(__file__).resolve().parent
    )
    output = Path(output).resolve()
    source = Path(data_path) / "x_test.csv"
    params_path = project / "models" / "preprocessing.npz"
    model_path = project / "models" / "model.npz"
    # 加载预处理参数
    params = load_preprocessing(params_path)
    with np.load(model_path, allow_pickle=False) as data:
        w = data["w"]
        threshold = float(data["threshold"])
    # 准备存储预测结果
    # 检查一下列的顺序
    with source.open(encoding="utf-8-sig", newline="") as f:
        header = next(csv.reader(f))
        if header[1:] != params["names"].tolist():
            raise ValueError("测试集特征名或顺序与训练参数不一致。")
    # 预测
    _, x_test, _, _, test_ids = load_csv_data(str(data_path))
    x_test = transform_features(x_test, params)
    probabilities = logistic_probabilities(x_test, w)
    predictions = np.where(probabilities >= threshold, 1, -1).astype(int)
    # 保存csv
    output.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(test_ids, predictions, str(output))
    print("已保存提交文件：", output, "，行数：", len(test_ids))
    return output


def main():
    project = Path(__file__).resolve().parent
    data_path = project / "data/dataset_to_release/dataset"
    output = project / "results/submission.csv"
    return generate_submission(data_path, output, project)


if __name__ == "__main__":
    main()
