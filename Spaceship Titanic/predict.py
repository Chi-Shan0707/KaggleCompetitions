import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split,cross_val_predict  # 用于划分训练和测试数据集，以及进行交叉验证预测
from sklearn.pipeline import Pipeline  # 用于构建机器学习管道，将预处理和模型串联起来
from sklearn.compose import ColumnTransformer  # 用于对不同类型的特征列应用不同的变换器
from sklearn.impute import SimpleImputer  # 用于填充缺失值
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # OneHotEncoder 用于独热编码类别特征，StandardScaler 用于标准化数值特征
from sklearn.ensemble import RandomForestClassifier  # 随机森林分类器模型
from sklearn.metrics import accuracy_score, classification_report  # 用于评估模型性能的准确率和分类报告

import joblib
"""
joblib 是一个 Python 库，主要用于高效地序列化和反序列化 Python 对象。
它特别适合保存和加载机器学习模型（如 scikit-learn 的模型），因为它比标准的 pickle 更快、更节省内存，常用于模型持久化（例如保存训练好的模型到文件，然后加载使用）。
"""

def load_data(path: str | None = None):
    """加载数据集。

    逻辑：
    - 如果传入 `path` 且文件存在，则直接读取；
    - 否则按常见位置尝试查找 `train.csv`（含带空格的文件夹名）；
    - 如果仍未找到，抛出包含尝试路径列表的清晰错误，方便排查。
    """
    

    candidates = []
    if path:
        candidates.append(Path(path))

    # 常见的相对位置（按工作区结构列出）
    candidates.extend([
        Path("./Spaceship Titanic/spaceship-titanic/train.csv"),
        Path("./Spaceship Titanic/train.csv"),
        Path("./spaceship-titanic/train.csv"),
        Path("./spaceship_titanic/train.csv"),
        Path("./train.csv"),
    ])

    for p in candidates:
        if p.exists():
            print(f"Loading data from: {p}")
            return pd.read_csv(p)

    # 更友好的错误信息，列出尝试过的路径
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"train.csv 未找到。尝试的路径: {tried}.\n请确认数据文件位置，或在调用 load_data 时传入正确的路径（例如 load_data(path='/full/path/to/train.csv')）。"
    )


def fill_categorical_by_distribution(df: pd.DataFrame, column: str, candidates=None, seed: int | None = None) -> pd.DataFrame:
    """按已有分布按比例随机填充分类列的缺失值或指定占位符（如 'Unknown'）。

    参数：
    - df: DataFrame，原表（就地修改并返回）。
    - column: 要填充的列名。
    - candidates: 列表，将视作缺失需填充的占位符（默认 ['Unknown']）。
    - seed: 随机种子（可选），保证可复现。
    """
    if candidates is None:
        candidates = ['Unknown']
    # mask：需要填充的位置（NaN 或者在 candidates 中的值）
    mask = df[column].isna() | df[column].isin(candidates)
    # pool：用于抽样的“真实”值（非 NaN 且不在 candidates 中）
    pool = df.loc[~df[column].isna() & ~df[column].isin(candidates), column]
    if mask.any() and not pool.empty:
        rng = np.random.default_rng(seed)
        probs = pool.value_counts(normalize=True)
        choices = rng.choice(probs.index, size=mask.sum(), p=probs.values)
        df.loc[mask, column] = choices
    return df


def fill_numerical_by_median(df: pd.DataFrame, column: str, candidates=None, seed: int | None = None) -> pd.DataFrame:
    """按中位数填充单个数值列的缺失或占位符值（签名类似于 fill_categorical_by_distribution）。

    参数：
    - df: DataFrame（就地修改并返回）
    - column: 要填充的单列名
    - candidates: 将视作缺失需填充的占位符（默认 ['Unknown']）
    - seed: 保持签名一致（未使用）
    """
    if candidates is None:
        candidates = ['Unknown']

    # 将候选占位符视作缺失，尝试把剩下的值解析为数值以计算中位数
    nums = pd.to_numeric(df[column].where(~df[column].isin(candidates) & df[column].notna(), np.nan), errors='coerce')
    median = nums.median()
    if pd.isna(median):
        # 如果没有可用数值，跳过
        return df

    # 需要填充的位置：原本就是 NaN / 在 candidates 中，或无法解析为数值的条目
    mask = nums.isna()
    if mask.any():
        df.loc[mask, column] = median

    # 尝试把整列转为数值（保留填充值）
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame: #类型注解，告诉函数的输入输出类型
    """进行特征工程处理"""
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    # 先根据已有分布填充部分分类列（HomePlanet, Destination）以便后续使用
    fill_categorical_by_distribution(df, 'HomePlanet', candidates=['Unknown'], seed=2026)
    fill_categorical_by_distribution(df, 'Deck', candidates=['Unknown'], seed=2026)
    fill_categorical_by_distribution(df, 'Num', candidates=['Unknown'], seed=2026)
    fill_categorical_by_distribution(df, 'Side', candidates=['Unknown'], seed=2026)
    fill_categorical_by_distribution(df, 'CryoSleep', candidates=['Unknown'], seed=2026)
    fill_categorical_by_distribution(df, 'Destination', candidates=['Unknown'], seed=2026)
    fill_numerical_by_median(df,'Age',candidates=['Unknown'], seed=2026)
    fill_categorical_by_distribution(df, 'VIP', candidates=['Unknown'], seed=2026)
    fill_numerical_by_median(df,'RoomService',candidates=['Unknown'], seed=2026)
    fill_numerical_by_median(df,'FoodCourt',candidates=['Unknown'], seed=2026)
    fill_numerical_by_median(df,'ShoppingMall',candidates=['Unknown'], seed=2026)
    fill_numerical_by_median(df,'Spa',candidates=['Unknown'], seed=2026)
    fill_numerical_by_median(df,'VRDeck',candidates=['Unknown'], seed=2026)
    # 解析 HomePlanet 为独热编码（将在 fill_missing 中处理）

    # 解析 PassengerId 为 Group 和 GroupSize
    df[['Group', 'GroupSize']] = df['PassengerId'].str.split('_', expand=True).astype(int)
    
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df.drop(['PassengerId'],axis=1,inplace=True)
    # 解析来自于哪
    # 无需额外操作

    # 解析是否休眠



    # 解析 Cabin 为 Deck, Num, Side
    
    df.drop('Cabin', axis=1, inplace=True)
    # inplace = True: 就地修改这个副本

    # 解析目的地


    # 解析年龄

    # 解析vip
    
   
    # 解析花费

   
    # 计算 TotalSpent
    expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalSpent'] = df[expense_cols].sum(axis=1)
    
    
    # 删除不需要的列
    df.drop(['Name'], axis=1, inplace=True)
    
    return df


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

def build_pipeline():
    """构建预处理管道"""
    # 定义列类型
    categorical_cols = ['HomePlanet', 'Destination', 'Deck','Num', 'Side']
    numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalSpent', 'GroupSize']
    
    # 数值特征预处理：标准化（缺失值已在特征工程中处理）
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # 类别特征预处理：独热编码（缺失值已在特征工程中处理）
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # 组合预处理器
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # 完整管道：预处理 + 模型
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return pipeline

def main():
    """主函数：加载数据、预处理、训练模型"""
    # 加载数据
    df = load_data()
    
    # 转换目标变量
    df['Transported'] = df['Transported'].astype(int)
    
    # 特征工程
    df = feature_engineering(df)

   
    # 检查 NaN 并打印出现 NaN 的列和示例行（如果仍有的话）
    na_counts = df.isna().sum()
    na_cols = na_counts[na_counts > 0]
    if not na_cols.empty:
        print("WARNING: Detected NaN values after fill_missing:")
        print(na_cols)
        print("Example rows with NaN (first 5):")
        print(df[df.isna().any(axis=1)].head())
    else:
        print("No NaN values found after feature_engineering and fill_missing.")

    # 分离特征和目标
    X = df.drop('Transported', axis=1) # axis=1表示按列删除
    y = df['Transported']
    
    # 划分训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 构建管道
    pipeline = build_pipeline()
    
    # 训练模型
    pipeline.fit(X_train, y_train)
    
    # 预测
    y_pred = pipeline.predict(X_test)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    
    # 保存模型
    joblib.dump(pipeline, 'random_forest_model.pkl')
    print("Model saved as random_forest_model.pkl")

if __name__ == "__main__":
    main()
    