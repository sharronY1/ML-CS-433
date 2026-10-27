# CS-433 Project 1



Binary classification using ordinary logistic regression and full-batch gradient descent. The final pipeline calls `logistic_regression()` from the course-required `implementations.py`, trains on the full training dataset, and generates test-set predictions.

## Files and environment

| File | Purpose |
| --- | --- |
| `run.py` | Load data, fit preprocessing parameters, process training data, and call full-data training and prediction |
| `model.py` | Initialize weights to zero, call the course GD implementation, and save weights and the threshold |
| `predict_submission.py` | Load the saved model and preprocessing parameters, transform test data, and generate the CSV |
| `implementations.py` | The six machine learning functions required by the course |
| `helpers.py` | Course-provided data loading and submission CSV writing functions |
| `experiments/validation.py` | Data splitting, GD training, threshold selection, and validation evaluation |

Execution requires Python 3 and NumPy. The verified full reproduction used Python 3.12.14 and NumPy 2.3.5. For reproduction, use Python 3.12 and the NumPy version pinned in `requirements.txt`. Use the same Python interpreter for installation and execution.

```bash
python -m pip install -r requirements.txt
```

Obtain the data through the course-designated channel and place it under the project directory:

```text
data/dataset_to_release/dataset/
    x_train.csv
    y_train.csv
    x_test.csv
```

There are 328135 labeled training rows, 109379 test rows, and 321 original features. The first CSV column contains sample IDs. Original labels are -1/+1 and are converted to 0/1 for training.

Data, caches, model weights, and submission CSV files are excluded by `.gitignore` and do not need to be uploaded to GitHub. They may remain locally; upload the prediction CSV to AIcrowd separately. The LaTeX template is in `latex-example-paper/`.

## Running and reproducing the pipeline

Run from the project root:

```bash
python run.py
```

This command starts from the raw CSV files, fits full-data preprocessing, trains from zero for 2000 iterations, and predicts the test set. It does not reuse an existing cache or model weights.

| Generated file | Contents |
| --- | --- |
| `cache/prepared_full_data.npz` | Processed full training data and 0/1 labels |
| `models/preprocessing.npz` | Training-derived imputation values, scaling statistics, and feature definitions |
| `models/model.npz` | Weights `w` and the fixed classification threshold `threshold` |
| `results/submission.csv` | Header `Id,Prediction`, labels -1/+1, and IDs in the original test-set order |

If a matching model and preprocessing parameter file already exist, run prediction only:

```bash
python predict_submission.py
```

Both entry points use fixed paths and accept no command-line arguments. Prediction directly overwrites `results/submission.csv` without creating backups or a check JSON. It uses the course-provided `load_csv_data()` to load data at once; this function reads both training and test data and therefore uses additional memory. The full training cache is approximately 1.5 GB, with further working memory needed during execution.

## Preprocessing and model

Preprocessing preserves the original operation order: handle special missing and zero codes, unify units, and apply the selected numerical transformations; record missingness before imputation, fill missing values with medians, and perform base standardization and feature construction.

The final features include 8 interaction terms, 3 squared terms, 6 one-hot columns for `DIABETE3` and `BPHIGH4`, and 247 missingness indicators. The two original columns replaced by one-hot encoding are removed. Including the intercept gives 584 columns:

```text
321 + 11 - 2 + 6 + 247 + 1 = 584
```

A final standardization is applied to all non-intercept columns; the intercept remains 1. `_process_features()` in `run.py` is the single shared feature-processing implementation for training, validation, and test data. Training calls `fit_preprocessing()` once to obtain both processed data and parameters. Prediction uses `transform_features()` with saved parameters and does not refit statistics on test data.

`model.py` directly trains and saves `w`; scaling is no longer folded into the weights. Prediction applies the full preprocessing sequence before computing probabilities and classifying with the saved threshold. The original CSV feature names and their order are checked.

| Parameter | Fixed configuration |
| --- | --- |
| Training function | `implementations.logistic_regression` |
| Objective | Mean binary cross-entropy, without regularization or class weighting |
| Initialization | Zero vector |
| Learning rate | `0.09377751905420602` |
| Iterations | `2000`, without early stopping |
| Final classification threshold | `0.2043741811970397`, not reselected during full-data training |
| Full training rows | `328135` |

## Validation and recorded results

Reproduce the validation pipeline:

```bash
python experiments/validation.py
```

The outer split uses `SEED = 42`, shuffles each class separately, and creates an 80/20 split with 262508 training rows and 65627 validation rows. A fixed 15000-row calibration subset is taken from the training portion, leaving 247508 rows for fitting the model.

To preserve the recorded experimental protocol, base imputation and scaling statistics use the outer training portion, while the final scaling statistics use only the fitting rows. The calibration set selects the threshold, and the validation set reports precision, recall, and F1. The validation set has been used repeatedly during exploration and is not a fresh independent test set. Changing the seed in `validation.py` creates a new experiment, so previous numerical results should not be assumed to hold.

The validation cache is saved to `cache/prepared_validation_data.npz`, models and preprocessing parameters to `models/validation/`, and metrics to `results/validation/gd_results.json`. Validation does not generate a submission CSV or overwrite the full-data model.

| Recorded GD experiment | Iterations | Validation F1 |
| --- | ---: | ---: |
| Feature scales before final standardization, gamma approximately 0.03 | 5000 | 0.428458 |
| Feature scales before final standardization, gamma = 0.1 | 2000 | 0.430157 |
| With final standardization, gamma approximately 0.09378 | 2000 | 0.438611 |

These experiments vary feature scaling, learning rate, or iteration count, so improvements cannot be attributed to a single change. Other historical experiment scripts may use an earlier 333-column protocol and cannot directly consume the current 584-column cache.

On 2026-10-27, the current algorithm was rerun from the raw data through the complete pipeline. Full-data training loss was `0.21565471588982407`, and the newly generated 109379-row CSV was byte-for-byte identical to the previously confirmed version. This rerun produced no new validation F1 or leaderboard score. The previously user-reported 80/20 GD submission score of 0.447 has not been independently verified and is not a new score for the full-data model.

SHA256 of the confirmed CSV:

```text
5d36a1d04ea2b9656da5d8cb78c7c100c4c54244eb869cc68f1540824fa619c6
```

The full-data reproduction settings and CSV hash are recorded above. Retraining in a different NumPy/BLAS environment may introduce small numerical differences.

### Six-method baseline comparison (independent 333-feature protocol)

```bash
python experiments/baseline_comparison.py
```

The script rebuilds the historical 333-feature preprocessing from raw CSV files, uses 40000 fitting rows, 15000 calibration rows and 65627 validation rows, and directly calls the six functions in the root implementations.py. It fixes the settings selected during the historical calibration stage; it does not repeat the original hyperparameter search or read old F1 values as new results.

Execution writes `results/baseline_comparison.csv`, including settings, thresholds, precision, recall, F1 and failure messages. If Least Squares fails because of a singular matrix, the failure is recorded without substituting a pseudoinverse. This pipeline neither calls the current 584-feature preprocessing nor overwrites the final model. Preprocessing statistics follow the old protocol and use the outer 80% training portion; the 40000 rows are the subset used to fit model weights. Iterative methods use finite budgets, without claiming strict convergence.

### Historical feature-engineering ablation

```bash
python experiments/ablation_study.py
```

This command trains four separate logistic regression models using the historical seed-42 stratified 80/20 split (262508 training and 65627 validation rows), 5000 GD iterations, gamma=0.01, and zero initial weights. It reproduces the 322/333-feature protocol, without using the current 584-feature cache or modifying the final model. The archived rules in `experiments/ablation/` are kept separately because this experiment must compare earlier cleaning stages. Eight products and three squares are constructed from standardized original features and then scaled using training statistics.

Historical values are in `results/ablation/recorded_results.csv`; the first row comes from user terminal output and a historical input-equivalence check. Later rows have saved machine-readable experiment results. The two-seed interaction comparison is in `recorded_interaction_seeds.csv`. These files were organized from existing evidence, not produced by a new training run. See [evidence and limitations](results/records/Ablation_Evidence_20261027.md).

Running the script writes fresh values to `results/ablation/reproduced_results.csv` and all five threshold candidates to `reproduced_thresholds.csv`, leaving the historical tables unchanged. It runs all four models, so expect a lengthy run. No retraining was performed while packaging this experiment. Seed 43 is supplemental recorded evidence; to reproduce it, set `SEED = 43` and compare the units_log/health_pairs results at threshold 0.2 in the threshold CSV.

Thresholds were selected on the repeatedly used validation set; these are development results. Do not attribute the combined zero-code/unit/log changes to log1p alone, or compare their differences directly with the separate six-method baseline or final-model protocol.

## Official public tests

```bash
python -m pip install pytest gitpython black
python -m pytest tests/grading_tests --github_link .
```

The current local result is 27 passed tests, with only the GitHub submission-link format check failing. Numerical, docstring, and Black formatting checks pass. Supplying a local path checks local files; it does not verify the remote repository.

After uploading the code, test the remote version using a real repository link pinned to a specific commit:

```bash
python -m pytest tests/grading_tests --github_link "https://github.com/OWNER/REPO/tree/COMMIT_HASH"
```

Remote testing requires Git and network access. The current local Black version is 26.10.1; the course test environment is defined in `tests/grading_tests/environment.yml`. Public tests do not cover hidden tests.
