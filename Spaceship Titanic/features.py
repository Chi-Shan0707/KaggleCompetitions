import pandas as pd
import numpy as np


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
    mask = df[column].isna() | df[column].isin(candidates)
    pool = df.loc[~df[column].isna() & ~df[column].isin(candidates), column]
    if mask.any() and not pool.empty:
        rng = np.random.default_rng(seed)
        probs = pool.value_counts(normalize=True)
        choices = rng.choice(probs.index, size=mask.sum(), p=probs.values)
        df.loc[mask, column] = choices
    return df


def fill_numerical_by_median(df: pd.DataFrame, column: str, candidates=None, seed: int | None = None) -> pd.DataFrame:
    """按中位数填充单个数值列的缺失或占位符值。

    参数：
    - df: DataFrame（就地修改并返回）
    - column: 要填充的单列名
    - candidates: 将视作缺失需填充的占位符（默认 ['Unknown']）
    - seed: 保持签名一致（未使用）
    """
    if candidates is None:
        candidates = ['Unknown']

    nums = pd.to_numeric(df[column].where(~df[column].isin(candidates) & df[column].notna(), np.nan), errors='coerce')
    median = nums.median()
    if pd.isna(median):
        return df
    mask = nums.isna()
    if mask.any():
        df.loc[mask, column] = median
    df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def feature_engineering(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """进行特征工程处理（可复现通过 seed）。

    - 解析 PassengerId, Cabin 等；
    - 填充缺失（调用上面的辅助函数）；
    - 计算 TotalExpense 等派生特征；
    - 打印 NaN 情况（仅在存在 NaN 时）。
    """
    df = df.copy()
    df[['Deck', 'Num', 'Side']] = df['Cabin'].str.split('/', expand=True)

    # PassengerId -> Group, GroupSize
    df[['Group', 'GroupSize']] = df['PassengerId'].str.split('_', expand=True).astype(int)
    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df.drop(['PassengerId'], axis=1, inplace=True)

    # HomePlanet
    fill_categorical_by_distribution(df, 'HomePlanet', candidates=['Unknown'], seed=seed)

    # CryoSleep 缺失根据花费判断
    for row in df.itertuples(index=True, name='Row'):
        if pd.isna(row.CryoSleep) or row.CryoSleep == 'Unknown':
            spend = 0
            flag = 1
            for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
                val = getattr(row, col)
                if not (pd.isna(val) or val == 'Unknown'):
                    num = pd.to_numeric(val, errors='coerce')
                    spend += 0 if pd.isna(num) else num
                else :
                    flag =0
            if flag==1 :
                df.at[row.Index, 'CryoSleep'] = False if spend > 0 else True
    fill_categorical_by_distribution(df, 'CryoSleep', candidates=['Unknown'], seed=seed)


    # Cabin
    fill_categorical_by_distribution(df, 'Deck', candidates=['Unknown'], seed=seed)
    fill_categorical_by_distribution(df, 'Num', candidates=['Unknown'], seed=seed)
    fill_categorical_by_distribution(df, 'Side', candidates=['Unknown'], seed=seed)
    df.drop('Cabin', axis=1, inplace=True)

    # Destination
    fill_categorical_by_distribution(df, 'Destination', candidates=['Unknown'], seed=seed)

    # Age
    fill_numerical_by_median(df, 'Age', candidates=['Unknown'], seed=seed)

    # VIP 缺失根据花费判断
    for row in df.itertuples(index=True, name='Row'):
        if pd.isna(row.VIP) or row.VIP == 'Unknown':
            rich = 0
            flag = 1
            for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
                val = getattr(row, col)
                if not (pd.isna(val) or val == 'Unknown'):
                    num = pd.to_numeric(val, errors='coerce')
                    if num >= df[col].median():
                        rich += 3
                    else :
                        rich -= 2
                else :
                    flag =0
            if flag== 1 :
                df.at[row.Index, 'VIP'] = False if rich >= 0 else True
    fill_categorical_by_distribution(df, 'VIP', candidates=['Unknown'], seed=seed)

    # Expense
    for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']:
        fill_numerical_by_median(df, col, candidates=['Unknown'], seed=seed)
    expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['TotalExpense'] = df[expense_cols].sum(axis=1)


    # VIP
    fill_categorical_by_distribution(df, 'VIP', candidates=['Unknown'], seed=seed)

    # Name
    df.drop(['Name'], axis=1, inplace=True)

    # 仅在存在 NaN 时打印
    na_counts = df.isna().sum()
    na_cols = na_counts[na_counts > 0]
    if not na_cols.empty:
        print("feature_engineering: Detected NaN values:")
        print(na_cols)
        print("Example rows with NaN (first 5):")
        print(df[df.isna().any(axis=1)].head())

    return df
