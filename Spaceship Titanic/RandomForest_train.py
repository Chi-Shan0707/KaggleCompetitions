import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split,cross_val_predict  # 用于划分训练和测试数据集，以及进行交叉验证预测
from sklearn.pipeline import Pipeline  # 用于构建机器学习管道，将预处理和模型串联起来
from sklearn.compose import ColumnTransformer  # 用于对不同类型的特征列应用不同的变换器
from sklearn.impute import SimpleImputer  # 用于填充缺失值
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer  # OneHotEncoder 用于独热编码类别特征，StandardScaler 用于标准化数值特征，FunctionTransformer 用于把自定义函数加入 pipeline

from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器模型
from sklearn.metrics import accuracy_score, classification_report  # 用于评估模型性能的准确率和分类报告

import joblib

import features
"""
joblib 是一个 Python 库，主要用于高效地序列化和反序列化 Python 对象。
它特别适合保存和加载机器学习模型（如 scikit-learn 的模型），因为它比标准的 pickle 更快、更节省内存，常用于模型持久化（例如保存训练好的模型到文件，然后加载使用）。
"""

def load_data(path: str):
    return pd.read_csv(path)
    raise FileNotFoundError(
        f"train.csv 未找到。尝试的路径: {tried}.\n请确认数据文件位置，或在调用 load_data 时传入正确的路径（例如 load_data(path='/full/path/to/train.csv')）。"
    )


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

def build_pipeline(seed: int = 2026):
    """构建预处理管道并将特征工程作为第一步放入管道

    参数:
    - seed: 传给 feature_engineering 和内部填充函数以保证可复现
    """
    # 列定义（feature_engineering 生成这些列）
    categorical_cols = ['HomePlanet', 'Destination', 'Deck', 'Side','CryoSleep', 'VIP','Num']
    numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalExpense', 'GroupSize']

    # 数值特征预处理：标准化
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # 类别特征预处理：独热编码
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # 完整管道：先自定义特征工程，再预处理，最后模型
    pipeline = Pipeline(steps=[
        ('feature_engineer', FunctionTransformer(features.feature_engineering, validate=False, kw_args={'seed': seed})),
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=2026))
    ])

    return pipeline

def train(seed: int = 42):
    """主函数：加载数据、预处理、训练模型

    参数：
    - seed: 随机种子（用于 pipeline 内的 feature_engineering/填充，可保证可复现）
    """
    # 加载数据
    df = load_data("./Spaceship Titanic/spaceship-titanic/train.csv")
    
    # 转换目标变量
    df['Transported'] = df['Transported'].astype(int)
    
    # 注意：feature_engineering 已经被包含到 pipeline 的第一步（FunctionTransformer），
    # 因此这里不需要再对 df 做整体的特征工程以避免重复处理。
    # 简单验证：确保目标列存在
    if 'Transported' not in df.columns:
        raise ValueError("输入数据缺少 'Transported' 列，请确认训练数据")

    # 分离特征和目标
    X = df.drop('Transported', axis=1) # axis=1表示按列删除
    y = df['Transported']
    
    # 划分训练和测试集（可控随机种子）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    
    # 构建管道（传入 seed）
    pipeline = build_pipeline(seed=seed)
    
    # 训练模型
    pipeline.fit(X_train, y_train)
    
    # 预测
    y_pred = pipeline.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # 保存模型到脚本同级目录
    model_path = Path(__file__).resolve().parent / 'random_forest_model.pkl'
    joblib.dump(pipeline, model_path)
    print(f"Model saved as {model_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train RandomForest pipeline')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (InputSeed) to use for feature filling and pipeline')
    args = parser.parse_args()
    print(f"Running training with seed={args.seed}")
    train(seed=args.seed)
    