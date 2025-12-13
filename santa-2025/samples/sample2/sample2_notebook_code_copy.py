# -*- coding: utf-8 -*-
"""
English (step-by-step):
This script is a faithful copy of the sample2 notebook logic, with clear annotations on collision validation and scoring.

1) Data loading and parsing:
   - Reads an optimized submission CSV, parses `id,x,y,deg` into numeric fields and helper columns: n_trees, tree_idx, x_val, y_val, deg_val.

2) Exact tree geometry (Kaggle-style):
   - The `ChristmasTree` class builds a Shapely `Polygon` with fixed dimensions (trunk + tiers), scaled by `SCALE_FACTOR` to preserve precision, then rotated and translated.
   - Rotation and translation are done via `shapely.affinity.rotate/translate`.

3) Collision detection (Shapely, not Python built-in):
   - Uses `STRtree` (spatial index) to query potential intersecting pairs efficiently.
   - For each polygon, checks `poly.intersects(other)` and excludes mere boundary touching via `not poly.touches(other)`.
   - Returns True if any actual area overlap exists.

4) Kaggle score computation:
   - Concatenates all polygon exterior coordinates, rescales by `SCALE_FACTOR`, takes axis-aligned bounding square side = max(width, height).
   - Score = side^2 / n, accumulated across configurations.

5) Visualization and submission:
   - Produces multiple matplotlib dashboards (score curves, histograms, heatmaps, sample configs with bounding boxes).
   - Writes final `submission.csv` if no overlaps found.

Chinese (逐步说明):
1）数据加载与解析：读取优化后的提交 CSV，解析 `id,x,y,deg` 为数值，派生 `n_trees`（树数量）、`tree_idx`（树索引）与 `x_val/y_val/deg_val`。
2）精确树几何（Kaggle 风格）：`ChristmasTree` 构建 Shapely 多边形，使用固定尺寸（树干+树冠层），通过 `SCALE_FACTOR` 保精度；随后旋转与平移到指定位置。
3）碰撞检测（依赖 Shapely，非 Python 内置）：通过 `STRtree` 空间索引高效查找潜在相交对；对每个多边形检查 `intersects`，并用 `not touches` 排除仅边界接触；若存在面积交叠则判定重叠。
4）Kaggle 评分：合并所有外轮廓坐标，按比例还原；外接正方形边长取 `max(width,height)`；每组评分为 `side^2 / n`，总分为累加。
5）可视化与提交：生成得分趋势、直方图、热图与样例配置图；若无重叠，则输出 `submission.csv` 以供提交。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from decimal import Decimal, getcontext
from shapely.geometry import Polygon
from shapely import affinity
from shapely.strtree import STRtree
import warnings
warnings.filterwarnings('ignore')

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# Precision
getcontext().prec = 25
SCALE_FACTOR = Decimal("1e15")

class ChristmasTree:
    """Exact Kaggle tree geometry"""
    def __init__(self, center_x="0", center_y="0", angle="0"):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = Decimal(str(angle))
        
        trunk_w = Decimal("0.15")
        trunk_h = Decimal("0.2")
        base_w = Decimal("0.7")
        mid_w = Decimal("0.4")
        top_w = Decimal("0.25")
        tip_y = Decimal("0.8")
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h
        
        initial_polygon = Polygon([
            (Decimal("0.0") * SCALE_FACTOR, tip_y * SCALE_FACTOR),
            (top_w / Decimal("2") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            (top_w / Decimal("4") * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            (mid_w / Decimal("2") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            (mid_w / Decimal("4") * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            (base_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
            (trunk_w / Decimal("2") * SCALE_FACTOR, base_y * SCALE_FACTOR),
            (trunk_w / Decimal("2") * SCALE_FACTOR, trunk_bottom_y * SCALE_FACTOR),
            (-(trunk_w / Decimal("2")) * SCALE_FACTOR, trunk_bottom_y * SCALE_FACTOR),
            (-(trunk_w / Decimal("2")) * SCALE_FACTOR, base_y * SCALE_FACTOR),
            (-(base_w / Decimal("2")) * SCALE_FACTOR, base_y * SCALE_FACTOR),
            (-(mid_w / Decimal("4")) * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            (-(mid_w / Decimal("2")) * SCALE_FACTOR, tier_2_y * SCALE_FACTOR),
            (-(top_w / Decimal("4")) * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
            (-(top_w / Decimal("2")) * SCALE_FACTOR, tier_1_y * SCALE_FACTOR),
        ])
        
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                         xoff=float(self.center_x * SCALE_FACTOR),
                                         yoff=float(self.center_y * SCALE_FACTOR))


def get_kaggle_score(trees, n):
    """Calculate exact Kaggle score"""
    if not trees:
        return 0.0
    xys = np.concatenate([np.asarray(t.polygon.exterior.xy).T / float(SCALE_FACTOR) for t in trees])
    min_x, min_y = xys.min(axis=0)
    max_x, max_y = xys.max(axis=0)
    side_length = max(max_x - min_x, max_y - min_y)
    return side_length**2 / n


def has_overlap(trees):
    """Check overlaps using Shapely (intersects && !touches => true overlap)"""
    if len(trees) <= 1:
        return False
    polygons = [t.polygon for t in trees]
    tree_index = STRtree(polygons)
    for i, poly in enumerate(polygons):
        indices = tree_index.query(poly)
        for idx in indices:
            if idx == i:
                continue
            if poly.intersects(polygons[idx]) and not poly.touches(polygons[idx]):
                return True
    return False


def main(input_csv: str = '/mnt/d/CS/Kaggle/santa-2025/samples/sample2/input/sample_submission.csv'):
    # Load optimized submission
    df = pd.read_csv(input_csv)

    # Parse
    df['n_trees'] = df['id'].str[:3].astype(int)
    df['tree_idx'] = df['id'].str.split('_').str[1].astype(int)
    df['x_val'] = df['x'].str.replace('s', '').astype(float)
    df['y_val'] = df['y'].str.replace('s', '').astype(float)
    df['deg_val'] = df['deg'].str.replace('s', '').astype(float)

    total_score = 0.0
    config_scores = []
    failed_configs = []

    # Validate and score
    for n in range(1, 201):
        config_df = df[df['n_trees'] == n]

        # Build trees for this n
        trees = []
        for _, row in config_df.iterrows():
            x = str(row['x'])[1:]  # strip 's'
            y = str(row['y'])[1:]
            deg = str(row['deg'])[1:]
            if x and y and deg:
                trees.append(ChristmasTree(x, y, deg))

        if not trees:
            continue

        # Overlap check
        overlaps = has_overlap(trees)

        # Score
        score = get_kaggle_score(trees, n)
        total_score += score

        # Side for display
        x_min, x_max = config_df['x_val'].min(), config_df['x_val'].max()
        y_min, y_max = config_df['y_val'].min(), config_df['y_val'].max()
        side = max(x_max - x_min, y_max - y_min)

        config_scores.append({'n': n, 'score': score, 'side': side, 'area': side**2, 'overlaps': overlaps})
        if overlaps:
            failed_configs.append(n)

    scores_df = pd.DataFrame(config_scores)

    # Visualization examples (minimal)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(scores_df['n'], scores_df['score'], marker='o')
    ax.set_title('Score per configuration')
    ax.set_xlabel('n')
    ax.set_ylabel('s^2/n')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(scores_df['score'], bins=30)
    ax2.set_title('Score distribution')
    ax2.set_xlabel('s^2/n')
    ax2.set_ylabel('freq')

    plt.tight_layout()
    plt.savefig('/mnt/d/CS/Kaggle/santa-2025/samples/sample2/output/sample2_dashboard.png', bbox_inches='tight', facecolor='white')

    # Submission
    if len(failed_configs) == 0:
        df[['id', 'x', 'y', 'deg']].to_csv('/mnt/d/CS/Kaggle/santa-2025/samples/sample2/output/submission.csv', index=False)
    else:
        print(f'Overlaps found in {failed_configs}; fix before submission.')


if __name__ == '__main__':
    main()
