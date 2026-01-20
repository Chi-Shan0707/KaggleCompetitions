import pandas as pd
import numpy as np
from pathlib import Path
import features
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





def predict(model_path: str = 'random_forest_model.pkl', test_path: str | None = None, out_path="./Spaceship Titanic/newsubmission.csv"):
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

    # 默认输出路径（相对于脚本目录）
    if out_path is None:
        out_path = 'submission#002.csv'

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

    # 处理输出路径（相对于脚本目录如果相对路径）
    out_file = Path(out_path)
    
    submission.to_csv(str(out_file), index=False)
    print(f"Saved submission to {out_file}")
    return submission

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Load trained pipeline and predict on test set')
    parser.add_argument('--model', type=str, default='random_forest_model.pkl', help='Path to trained model (joblib)')
    parser.add_argument('--test', type=str, default=None, help='Path to test CSV file')
    parser.add_argument('--out', type=str, default='submission#002.csv', help='Output submission CSV path (relative to script directory if not absolute)')
    args = parser.parse_args()
    predict(model_path=args.model, test_path=args.test, out_path=args.out)