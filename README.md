# CS-433 Project 1

最终提交版本为 **普通 Logistic Regression + 全批量 Gradient Descent**，直接调用 `implementations.py` 的 `logistic_regression`。采用全量训练数据、2000 轮、固定学习率和校准阈值。

这个上传文件夹不含数据、缓存、模型权重或提交 CSV。请把文件夹内的内容放到 GitHub 仓库根目录，下载课程数据后执行一次 `python run.py` 即可复现最终提交。最终报告尚需完成，当前 `reports/` 中是提纲和模板。

## 文件用途

正式流程需要根目录五份代码：

| 文件 | 用途 |
| --- | --- |
| `run.py` | 保留数据读取、清洗、特征处理与保存代码，末尾调用全量GD训练和预测 |
| `model.py` | 直接使用预处理好的矩阵调用课程GD训练，保存原始权重w |
| `predict_submission.py` | 读取预处理参数、转换测试集、加载GD权重并生成CSV，不训练 |
| `implementations.py` | 课程要求的六个函数；本次没有改动 |
| `helpers.py` | 原始 CSV 读取与官方格式写入 |


## 环境与数据

运行仅需 Python 3 与 NumPy，计算不依赖 Pandas、Scikit-learn、SciPy 或深度学习框架。当前确认版实验使用 Python 3.12.14、NumPy 2.3.5；不同环境可能有细微浮点差异。

```powershell
cd D:\433project1_GitHub
python -m pip install -r requirements.txt
```

安装与运行请使用同一个解释器；如使用 Miniconda，可统一把 python 替换为 `D:\Miniconda\python.exe`，不要混用 MSYS2 的 Python。

原始数据从课程指定渠道获取，放入：

```text
data/dataset_to_release/dataset/x_train.csv
data/dataset_to_release/dataset/y_train.csv
data/dataset_to_release/dataset/x_test.csv
```

数据为 328135 行带标签样本、109379 行测试样本、321 个原始特征。第一列为样本 ID，原始标签 -1/+1。程序内部训练标签转换为 0/1。

路径由脚本位置确定；数据放在 data/dataset_to_release/dataset，最终输出为 results/submission.csv。

## 一次运行生成最终全量 GD 提交

在项目根目录执行：

```powershell
python run.py
```

run.py 先执行文件内的预处理和保存代码，再调用训练与预测：

1. run.py 使用全部328135行训练样本拟合基础预处理，生成 cache/prepared_full_data.npz、models/preprocessing.npz；不创建8:2验证或校准划分。
2. run.py在特征构造末尾完成最后标准化；model.py从零初始化，直接调用课程GD函数连续更新2000次，输出 models/model.npz（仅保存w和threshold，不额外写全量训练JSON）。
3. predict_submission.py 处理 x_test 并生成 results/submission.csv。表头 Id,Prediction，标签-1/+1，保持测试ID顺序。

正式入口固定使用既定学习率和阈值，不进行模型比较、搜索参数或根据验证分数选阈值，也不读取验证实验的“最佳参数”自动改配置。

可以指定其他数据或输出目录：

```powershell
python run.py
```

GD一次调用完成2000轮，保留最终损失，不记录中间曲线。缓存约1.5 GB，运行时另需内存。更换数据或预处理后应从此入口完整重跑，保持模型与统计配套；全量模型没有新的独立验证分数。

## 已有模型时直接预测

已有以下配套文件时，可以只运行预测脚本，不再预处理或训练：

```text
models/model.npz
models/preprocessing.npz
```

```powershell
python predict_submission.py
```

本上传包没有附带权重，因此首次使用不能直接跳到预测。预测入口采用固定数据和输出路径；直接写入 results/submission.csv，不生成备份或检查JSON。

最后一层标准化在run.py中完成，参数final_means、final_stds保存到preprocessing.npz。model.py直接保存训练得到的w，不折算权重；预测脚本在构造相同顺序的584列后应用最后标准化，再乘w。截距列均值固定为0、标准差为1，保持为1。旧缓存、旧折算权重不能与新流程混用。

## 最终提交与验证实验

生成最终提交，使用全部训练数据：

```powershell
python run.py
```

复现GD验证实验，准备8:2数据、训练并打印Precision、Recall和F1：

```powershell
python experiments/validation.py
```

两个入口不使用命令行参数，数据目录固定为 data/dataset_to_release/dataset。
随机种子在 validation.py 的 SEED = 42 修改。prepare_data() 只处理全量数据，
8:2划分和验证数据准备完全在 validation.py 中完成。两边调用同一个 prepare_features()，其中不再进行划分或选择运行模式。

为对应已有GD验证记录，验证流程沿用分层8:2外层划分：262508行训练、65627行验证；
80%训练部分内再固定划出15000行校准，247508行拟合，校准集选择阈值，验证集评估。
修改种子后属于新实验，不承诺复现旧数值；验证集此前已反复用于探索。

两边共用run.py的特征处理，直接调用implementations.py中的logistic_regression，并使用model.py中相同的固定训练参数。
全量缓存为 cache/prepared_full_data.npz，预处理参数和权重在 models/；
验证缓存为 cache/prepared_validation_data.npz，参数和权重在 models/validation/，
指标保存到 results/validation/gd_results.json，不生成提交CSV，不覆盖全量模型。


## 当前方法与结果

基本处理包括特殊缺失编码、部分单位/数值转换、中位数填补、标准化、11个健康交互/平方项、两列类别编码、247个缺失标记和截距，最终584列。GD训练前再把所有非截距输入标准化，尤其包含后来添加的0/1列。

| 参数 | 最终配置 |
| --- | --- |
| 模型/训练函数 | 普通逻辑回归，`implementations.logistic_regression` |
| 目标 | 平均二元交叉熵，无正则化，不加类别权重 |
| 初始化 | 零向量 |
| 学习率 | 0.09377751905420602，固定，不在全量重训时重新选择 |
| 更新次数 | 2000，无额外早停或第二轮调参 |
| 全量分类阈值 | 0.2043741811970397，固定，来源于此前8:2 GD校准集 |
| 全量训练行数 | 328135 |

| 已记录的对照 | 轮数 | 验证 F1 |
| --- | ---: | ---: |
| GD，原尺度，gamma约0.03 | 5000 | 0.428458 |
| GD，原尺度，gamma=0.1 | 2000 | 0.430157 |
| GD，补齐标准化，gamma约0.09378 | 2000 | 0.438611 |


用户反馈此前8:2 GD提交成绩为0.447，属于用户报告、未独立核验；它不是本次全量模型的验证或线上成绩。全量确认版训练损失为0.21565471588982407，没有新的验证分数。

确认版提交 CSV 的 SHA256 是 `5d36a1d04ea2b9656da5d8cb78c7c100c4c54244eb869cc68f1540824fa619c6`。它保留在原本机试验目录，未放入GitHub包；本包通过一次 `python run.py` 重新生成。不同 NumPy/BLAS 环境重训可能有细微数值差异，不承诺所有环境逐位相同。

## 报告与实验材料

报告重点：数据清洗与缺失处理、特征构造、完整标准化、GD目标和更新、校准阈值、基线比较及消融。提纲见 `reports/报告提纲.md`。报告中的验证成绩来自模型选择阶段，最终全量拟合是提交阶段，二者要分开描述。




## 课程公开测试

```powershell
python -m pip install pytest gitpython black
python -m pytest tests/grading_tests --github_link .
```

课程六个函数保持原接口。数值、docstring 和当前 Black 格式检查通过；本地路径会使 GitHub 链接项失败。正式提交需使用真实公开仓库的具体提交地址：

```powershell
python -m pytest tests/grading_tests --github_link "https://github.com/OWNER/REPO/tree/COMMIT_HASH"
```

远程检查需要 Git 和网络，检查已上传的提交。当前 Black 为26.10.1，课程固定环境为22.6.0，需区分版本；公开测试通过不保证隐藏测试全部通过。
上传包仅保留当前GD流程和课程方法相关实验。校准划分、阈值选择和指标计算位于 experiments/validation.py；seed=42的已有GD验证F1为0.4386108832376265。其他优化器及其混合实验不在上传包中。


预处理参数仅保存预测所需的统计量、原始列名和特征定义，不再导出额外特征名称、完整输出列名或说明JSON。训练与预测的特征拼接顺序保持一致，最终为584列。

全量最后标准化使用全部328135行；验证阶段保留原协议，仅用247508行拟合样本计算最后统计，校准集和验证集复用。历史原尺度对照脚本在显式选择原尺度时先还原最后标准化，不会重复标准化新缓存。

训练与预测共用run.py中的_process_features()。训练调用fit_preprocessing()一次，同时得到处理后的训练数据和参数；验证集直接调用共用处理函数应用已有参数；测试集通过predict_submission.py中的transform_features()应用已有参数。原始数据在独立副本上清洗，避免重复log1p。
