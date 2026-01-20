import pandas as pd
import numpy as np
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

"""
joblib 是一个 Python 库，主要用于高效地序列化和反序列化 Python 对象。
它特别适合保存和加载机器学习模型（如 scikit-learn 的模型），因为它比标准的 pickle 更快、更节省内存，常用于模型持久化（例如保存训练好的模型到文件，然后加载使用）。
"""

def load_data(path: str):
    return pd.read_csv(path)
    raise FileNotFoundError(
        f"train.csv 未找到。尝试的路径: {tried}.\n请确认数据文件位置，或在调用 load_data 时传入正确的路径（例如 load_data(path='/full/path/to/train.csv')）。"
    )


# 注意：完整的 feature_engineering 和填充逻辑已被包含在训练时保存的 pipeline 中，
# 因此预测脚本不再需要实现这些函数。直接加载 pipeline 并把原始 test DataFrame 传入即可。



"""
df = df.copy() 的作用
`df = df.copy()` 的作用是创建一个 DataFrame 的副本（拷贝），并将变量 `df` 重新指向这个新对象。这样做的好处是**避免原地修改原始数据**，即使前后变量名都是 `df`。

为什么前后变量名一样但能避免原地修改？
- **变量名是引用**：在 Python 中，`df` 只是一个指向 DataFrame 对象的引用（类似指针）。`df = df.copy()` 先调用 `copy()` 创建一个全新的 DataFrame 对象（内存中独立的新副本），然后让 `df` 指向这个新对象。
- **前后对比**：
  - 执行前：`df` 指向原始 DataFrame。
  - 执行后：`df` 指向新拷贝的 DataFrame（原始对象不变，如果没有其他引用，会被垃圾回收）。
- **如果不拷贝**：直接修改 `df`（如 `df['new_col'] = ...`），会改变原始对象，导致数据污染。
- **拷贝类型**：`copy()` 默认是深拷贝（deep copy），确保数据完全独立；如果用 `copy(deep=False)`，则是浅拷贝（只拷贝结构，不拷贝数据）。

这样可以安全地处理数据，避免意外修改源数据。
"""


def predict(model_path: str = 'random_forest_model.pkl', test_path: str | None = None, out_path: str = 'submission.csv'):
    """对测试集进行预测并保存提交文件。

    参数：
    - model_path：训练好的模型文件路径（joblib 文件）
    - test_path：测试集路径（若为 None 使用默认常见路径）
    - out_path：输出提交文件路径
    返回：保存的 submission DataFrame
    """
    # 默认测试集路径
    if test_path is None:
        test_path = "./Spaceship Titanic/spaceship-titanic/test.csv"

    # 加载测试数据
    test = load_data(test_path)

    # 记录 PassengerId（测试集需要提交）
    if 'PassengerId' in test.columns:
        ids = test['PassengerId']
    else:
        raise ValueError("测试集找不到 'PassengerId' 列，请确认文件格式")

    # 加载模型（优先使用给定路径；若找不到，尝试脚本同级目录）
    model_file = Path(model_path)
    if not model_file.exists():
        alt = Path(__file__).resolve().parent / model_path
        if alt.exists():
            model_file = alt
            print(f"Loading model from script directory: {model_file}")
        else:
            raise FileNotFoundError(f"模型文件未找到：{model_path}，也未在脚本同级目录找到：{alt}. 请先训练并保存模型（train()）。")
    model = joblib.load(model_file)

    # 预测（直接传入原始 test DataFrame，若模型是完整 pipeline，则会先执行 feature_engineering 和预处理）
    preds = model.predict(test)
    # 将预测转为布尔（比赛格式一般为 True/False）
    preds_bool = pd.Series(preds).astype(bool)

    submission = pd.DataFrame({'PassengerId': ids, 'Transported': preds_bool})
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path}")
    return submission

if __name__ == "__main__":
    predict()