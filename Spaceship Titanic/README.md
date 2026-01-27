# Spaceship Titanic — 项目说明 / Project README 🚀

## 中文说明（第一部分） 🇨🇳

### 概览 🔍
本目录包含一个基于 RandomForest 的简单 Kaggle 风格解法，用于 Spaceship Titanic 竞赛。核心脚本：`RandomForest_train.py`（训练并保存 pipeline）、`RandomForest_predict.py`（加载 pipeline 并对测试集生成提交文件）、`features.py`（所有特征工程与填充逻辑）。训练数据和测试数据位于 `./spaceship-titanic/train.csv` 与 `./spaceship-titanic/test.csv`。

### 文件及其作用 📂
- **`RandomForest_train.py`**：
  - 加载训练数据（默认 `./Spaceship Titanic/spaceship-titanic/train.csv`）。
  - 使用 `features.feature_engineering` 作为 `FunctionTransformer` 的第一步，接着做数值标准化与分类 One-Hot 编码，最后训练 `RandomForestClassifier`。
  - 将训练好的 `Pipeline` 使用 `joblib` 保存为 `random_forest_model.pkl`（脚本同级目录）。
  - 支持 `--seed` 参数来保证填充和特征生成的可复现性。

- **`RandomForest_predict.py`**：
  - 加载测试集（默认 `./Spaceship Titanic/spaceship-titanic/test.csv`）并读取 `PassengerId`。
  - 加载保存的 pipeline（默认 `random_forest_model.pkl`），直接将原始 DataFrame 传入以复用训练时的特征工程与预处理。
  - 保存提交文件，默认或通过 `--out` 指定输出路径（例如 `./Spaceship Titanic/newsubmission.csv`）。

- **`features.py`**：
  - 提供 `feature_engineering(df, seed=...)`：解析 `Cabin`（分为 `Deck/Num/Side`）、解析 `PassengerId`（生成 `Group`、`GroupSize`）并删除 `PassengerId` 与 `Name`，填充缺失值（类别按分布随机填充，数值按中位数填充），计算 `TotalExpense` 等。
  - 辅助函数：`fill_categorical_by_distribution`、`fill_numerical_by_median`，它们可通过 `seed` 保持填充可复现性。

### 使用方法（快速） ⚡️
训练：
```bash
python3 ./Spaceship\ Titanic/RandomForest_train.py --seed 114514
```
预测并保存提交：
```bash
python3 ./Spaceship\ Titanic/RandomForest_predict.py --model random_forest_model.pkl --test "./Spaceship Titanic/spaceship-titanic/test.csv" --out "./Spaceship Titanic/newsubmission.csv"
```

### 参数说明与注意事项 ⚠️
- `--seed`：传入训练脚本可复现特征填充（`features.feature_engineering` 中使用）。
- 预测脚本假设 `model` 是一个包含完整 `feature_engineering` 步骤的 `Pipeline`（默认由训练脚本保存）。
- 提交文件会包含 `PassengerId` 与布尔类型的 `Transported` 列（True / False）。

### 依赖 / 环境 🧰
- Python 3.8+
- pandas, numpy, scikit-learn, joblib
- 可通过 `pip install -r requirements.txt`（项目根）或手动安装上述包。

---

## English README (second section) 🇬🇧

### Overview 🔍
This folder contains a simple Kaggle-style solution using a RandomForest pipeline for the Spaceship Titanic problem. Main scripts: `RandomForest_train.py` (train and save a pipeline), `RandomForest_predict.py` (load pipeline and produce submission), `features.py` (feature engineering and imputation). Default dataset paths are `./spaceship-titanic/train.csv` and `./spaceship-titanic/test.csv`.

### Files & Roles 📂
- **`RandomForest_train.py`**:
  - Loads training data (default `./Spaceship Titanic/spaceship-titanic/train.csv`).
  - Uses `features.feature_engineering` as the first step via `FunctionTransformer`, then applies numeric scaling and one-hot encoding, and trains a `RandomForestClassifier`.
  - Saves the trained `Pipeline` with `joblib` as `random_forest_model.pkl` in the script directory.
  - Accepts `--seed` to make imputation and feature generation reproducible.

- **`RandomForest_predict.py`**:
  - Loads the test set (default `./Spaceship Titanic/spaceship-titanic/test.csv`) and reads `PassengerId`.
  - Loads a saved pipeline (default `random_forest_model.pkl`) and passes the raw test DataFrame into it so the same feature engineering and preprocessing are applied.
  - Saves a submission CSV (default or specified with `--out`).

- **`features.py`**:
  - `feature_engineering(df, seed=...)`: parses `Cabin` into `Deck/Num/Side`, splits `PassengerId` into `Group` and `GroupSize`, drops `PassengerId` and `Name`, fills missing values (categorical by distribution, numeric by median), computes `TotalExpense`, and more.
  - Helpers: `fill_categorical_by_distribution`, `fill_numerical_by_median`, both support reproducibility via `seed`.

### Quick Usage ⚡️
Train:
```bash
python3 ./Spaceship\ Titanic/RandomForest_train.py --seed 114514
```
Predict/submit:
```bash
python3 ./Spaceship\ Titanic/RandomForest_predict.py --model random_forest_model.pkl --test "./Spaceship Titanic/spaceship-titanic/test.csv" --out "./Spaceship Titanic/newsubmission.csv"
```

### Notes & Tips ⚠️
- `--seed` ensures reproducible imputations and feature operations inside `feature_engineering`.
- Prediction expects a saved pipeline that includes the feature engineering step; otherwise, pre-processing must be applied before prediction.
- Output submission contains `PassengerId` and boolean `Transported` column (True/False).

### Dependencies 🧰
- Python 3.8+
- pandas, numpy, scikit-learn, joblib

---
