# CS-433 Project 1


Perform binary classification using ordinary logistic regression and full-batch gradient descent. The final pipeline directly calls `logistic_regression()` in the course `implementations.py`, trains on the full training dataset, and then generates the test-set submission file.

## Files and environment

| File | Purpose |
| --- | --- |
| `run.py` | Read data, fit preprocessing parameters and process training data, then call full-data training and prediction |
| `model.py` | Initialize weights to zero, call the course GD function, and save weights and the threshold |
| `predict_submission.py` | Load the existing model and preprocessing parameters, process test data, and generate the CSV |
| `implementations.py` | The six machine learning functions required by the course |
| `helpers.py` | Course-provided functions for reading data and writing the submission CSV |
| `experiments/validation.py` | Split data, train with GD, select a threshold, and evaluate validation metrics |
| `experiments/baseline_comparison.py` | Compare the six course methods using the same data split |
| `experiments/ablation_study.py` | Reproduce four progressive feature-engineering ablation experiments |

Running requires Python 3 and NumPy. The verified full reproduction environment used Python 3.12.14 and NumPy 2.3.5. For reproduction, Python 3.12 and the NumPy version pinned in `requirements.txt` are recommended. Use the same Python interpreter for installation and execution.

```bash
python -m pip install -r requirements.txt
```

Course data structure:

```text
data/dataset_to_release/dataset/
    x_train.csv
    y_train.csv
    x_test.csv
```

The training set has 328135 rows, the test set has 109379 rows, and there are 321 original feature columns. The first CSV column contains sample IDs; the original labels are -1/+1 and are converted to 0/1 for training.


## Running and reproduction

Run from the project root:

```bash
python run.py
```

This command starts from the raw CSV files and performs full-data preprocessing, training from zero for 2000 iterations, and test-set prediction in sequence, without reusing old caches or weights.

| Generated file | Contents |
| --- | --- |
| `cache/prepared_full_data.npz` | Processed full training data and 0/1 labels |
| `models/preprocessing.npz` | Imputation values, standardization statistics, and feature definitions computed during training |
| `models/model.npz` | Weights `w` and the fixed classification threshold `threshold` |
| `results/submission.csv` | Header `Id,Prediction`, labels -1/+1, and the original test-set ID order |

When a matching model and preprocessing parameters already exist, prediction can be run on its own:

```bash
python predict_submission.py
```

Both entry points use fixed paths and have no command-line arguments. Prediction directly overwrites `results/submission.csv`. Prediction uses `load_csv_data()` to read the data at once; this function reads both the training and test sets, so it uses additional memory. The full-data cache is approximately 1.5 GB, and execution also requires working memory.

## Preprocessing and model

Preprocessing proceeds in order: clean special missing and zero codes, unify units, and apply the specified numerical transformations; record missing positions before imputation, fill missing values with medians, then perform base standardization and feature construction.

The final features include 8 interaction terms, 3 squared terms, 6 one-hot columns for `DIABETE3` and `BPHIGH4`, and 247 missingness indicators. The two original columns replaced by one-hot encoding are removed, giving 584 columns after adding the intercept:

```text
321 + 11 - 2 + 6 + 247 + 1 = 584
```

Finally, all non-intercept columns are standardized, while the intercept remains 1. `_process_features()` in `run.py` is the single shared feature-processing implementation for training, validation, and testing: training calls `fit_preprocessing()` once to obtain both the processed data and parameters; prediction uses the existing parameters through `transform_features()`, without refitting statistics on the test set.

`model.py` directly trains and saves `w`, without folding standardization into the weights. Prediction applies the full preprocessing rules, then computes probabilities and classifies using the saved threshold. The code retains the check of the original CSV feature names and their order.

| Parameter | Fixed configuration |
| --- | --- |
| Training function | `implementations.logistic_regression` |
| Objective | Mean binary cross-entropy, without regularization or class weighting |
| Initialization | Zero vector |
| Learning rate | `0.09377751905420602` |
| Iterations | `2000`, without early stopping |
| Final classification threshold | `0.2043741811970397`, determined by earlier calibration experiments and kept fixed during final full-data training. The official test set is not used for threshold selection. |
| Full training rows | `328135` |

## Validation and existing results

Reproduce the validation pipeline:

```bash
python experiments/validation.py
```

The outer split uses `SEED = 42`, shuffles the positive and negative classes separately, and creates an 8:2 split with 262508 training rows and 65627 validation rows. A fixed 15000-row calibration set is retained within the training portion, leaving 247508 rows for fitting the model.

To preserve the existing experimental protocol, base imputation and standardization statistics are computed using the outer training portion; the final standardization uses only the fitting rows. The calibration set selects the threshold, and the validation set reports Precision, Recall, and F1. The validation set has been used repeatedly for exploration and is not a fresh independent test set. Changing the seed in `validation.py` constitutes a new experiment, so the old numerical results cannot simply be reused.

The validation cache is saved to `cache/prepared_validation_data.npz`, models and parameters to `models/validation/`, and metrics to `results/validation/gd_results.json`. The validation pipeline neither generates a submission CSV nor overwrites the full-data model.

The final 584-feature configuration achieves an F1 of 0.438611 on the development validation set. The final submission model is retrained using all 328135 labeled samples, without separately computing a validation F1.

### Six-method baseline comparison (independent 333-feature protocol)

```bash
python experiments/baseline_comparison.py
```

The script rebuilds the historical 333-feature preprocessing from the raw CSV files, uses 40000 training rows, 15000 calibration rows, and 65627 validation rows, and directly calls the six functions in implementations.py in the root directory. Parameters are fixed to the configurations selected during the historical calibration stage.

Execution generates `results/baseline_comparison.csv`, recording parameters, thresholds, Precision, Recall, F1, and failure reasons. If Least Squares fails because of a singular matrix, the failure is recorded as it occurs, without substituting a pseudoinverse. This pipeline does not call the current 584-feature preprocessing or overwrite the final model. Preprocessing statistics follow the old protocol and are computed using the outer 80% training portion; the 40000 rows are the subset actually used to fit the model.

### Historical feature-engineering ablation

```bash
python experiments/ablation_study.py
```

This command retrains four logistic regression models: seed=42, a stratified 80/20 split, 262508 training rows and 65627 validation rows, 5000 GD iterations, gamma=0.01, and zero initial weights. It reproduces the historical 322/333-feature protocol, without using the current 584-feature cache or modifying the final model. `experiments/ablation/` separately retains the old cleaning rules to compare different cleaning stages; interaction terms are generated from already standardized original features and then standardized using training statistics.

`results/ablation/recorded_results.csv` contains historical results compiled from existing evidence. The first row comes from user terminal output and a historical input-consistency check, while the remaining three rows have saved machine-generated experimental results. `recorded_interaction_seeds.csv` contains the interaction comparisons for two seeds. These scores were not obtained by retraining during this organization step. Sources and limitations are described in [ablation evidence](results/records/Ablation_Evidence_20261027.md).

After execution, new results are written to `results/ablation/reproduced_results.csv`, and results for the five thresholds are written to `reproduced_thresholds.csv`, without overwriting the historical tables. All four models need to be trained, so execution will take some time; no training was started during this organization step. seed=43 is retained only as existing supplementary evidence; to reproduce it, change `SEED` in the script to 43 and compare units_log/health_pairs at threshold 0.2 in the threshold results table.

Thresholds were selected on the repeatedly used validation set, so these are development-stage results.



