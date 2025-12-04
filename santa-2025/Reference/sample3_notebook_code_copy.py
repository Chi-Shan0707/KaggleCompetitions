# -*- coding: utf-8 -*-
"""
English (step-by-step):
This script is a faithful copy of sample3's core logic, with annotations on collision detection and packing.

1) Imports and seeds:
   - Uses numpy, matplotlib, shapely (Polygon, affinity) and Decimal.
   - Sets random seeds for reproducibility.

2) Tree geometry:
   - `ChristmasTree` constructs a Shapely `Polygon` with fixed tier widths and y-levels.
   - It rotates around origin, then translates to `(center_x, center_y)`.
   - Here `scale_factor = Decimal('1.0')` (no scaling), but competition setups often use larger scale to stabilize precision.

3) Visualization helper:
   - `plot_packing(trees)` renders polygons, sets equal aspect.

4) Baseline packing:
   - `simple_grid_packing(num_trees, spacing)` lays trees on an axis-aligned grid using a spacing buffer (e.g., 0.8 for width≈0.7).

5) Collision check:
   - `check_overlaps(trees)` uses Shapely's `polygon.intersects(other)` to count pairwise overlaps.
   - Note: `intersects` returns True for boundary touching as well; competition-grade validation often excludes pure `touches`.

6) Submission output:
   - `save_submission(trees)` writes `id,x,y,angle` (note: competition expects `id,x,y,deg` and 's'-prefixed strings for precision).

Chinese (逐步说明):
1）导入与随机种子：使用 numpy、matplotlib、shapely 与 Decimal，并设定随机种子以复现结果。
2）树几何：`ChristmasTree` 构造固定尺寸的多边形，先原点旋转再平移到指定位置；当前未放大比例，赛中常用更大比例保证精度。
3）可视化：`plot_packing` 绘制树形外轮廓，设置等比例坐标。
4）基线摆放：`simple_grid_packing` 将树按照网格、给定间距放置，间距略大于最大宽度作为缓冲。
5）碰撞检查：`check_overlaps` 调用 `intersects` 统计两两相交；严格校验可改为 `intersects and not touches` 避免把纯边界接触算作碰撞。
6）提交输出：`save_submission` 输出 `id,x,y,angle`；正式比赛需改为 `id,x,y,deg` 且值前加 's' 字符串以保精度。
"""

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely import affinity
from decimal import Decimal
import random
import math
import pandas as pd

# Seeds
random.seed(42)
np.random.seed(42)

scale_factor = Decimal('1.0')

class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon([
            (Decimal('0.0') * scale_factor, tip_y * scale_factor),
            (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
            (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
            (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
            (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
            (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
            (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
            (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
            (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
            (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
            (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
            (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
        ])
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))


def plot_packing(trees, title="Christmas Tree Packing"):
    fig, ax = plt.subplots(figsize=(8, 8))
    for tree in trees:
        x, y = tree.polygon.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='green', ec='black')
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.grid(True)
    plt.show()


def simple_grid_packing(num_trees, spacing=1.0):
    trees = []
    cols = int(np.ceil(np.sqrt(num_trees)))
    for i in range(num_trees):
        row = i // cols
        col = i % cols
        x = col * spacing
        y = row * spacing
        tree = ChristmasTree(x, y, 0)
        trees.append(tree)
    return trees


def check_overlaps(trees):
    overlaps = 0
    for i in range(len(trees)):
        for j in range(i + 1, len(trees)):
            if trees[i].polygon.intersects(trees[j].polygon):
                overlaps += 1
    return overlaps


def save_submission(trees, filename="submission.csv"):
    data = []
    for i, tree in enumerate(trees):
        data.append({
            'id': i,
            'x': float(tree.center_x),
            'y': float(tree.center_y),
            'angle': float(tree.angle)
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}")
    return df


if __name__ == '__main__':
    # Visualize a single tree
    plot_packing([ChristmasTree(0, 0, 0)], "Single Tree Visualization")

    # Generate a sample packing of 25 trees
    packed_trees = simple_grid_packing(25, spacing=0.8)
    plot_packing(packed_trees, "Grid Packing (25 Trees)")

    # Check overlaps
    num_overlaps = check_overlaps(packed_trees)
    print(f"Total Overlaps: {num_overlaps}")
    print("Packing is VALID." if num_overlaps == 0 else "Packing is INVALID.")

    # Save submission (note: adapt columns to Kaggle format for actual submission)
    submission_df = save_submission(packed_trees)
