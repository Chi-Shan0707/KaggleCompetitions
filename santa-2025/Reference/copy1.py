# -*- coding: utf-8 -*-
"""
English:
Step-by-step explanation of how this notebook implements collision detection and packing optimization:

1) Geometry and precision setup:
   - Uses Decimal with high precision to avoid floating point drift when converting to/from Kaggle's 's' string format.
   - Defines a global scale_factor (1e18) so all coordinates are scaled to integers before building Shapely polygons.

2) ChristmasTree shape modeling (collision basis):
   - Each tree is represented by a single Shapely Polygon approximating trunk + tiered canopy.
   - The initial polygon is built around origin using fixed dimensions, then rotated by 'deg' and translated to (x,y).
   - Collision detection is implicit: any unary_union of overlapping polygons would create invalid geometry or overlaps can be tested via pairwise intersects.
   - The code relies on constraint: submissions with any overlapping trees will throw an error, so later packing avoids overlaps by preserving the already-feasible solution and only removing trees.

3) Bounding box and size metric:
   - get_tree_list_side_lenght() computes the axis-aligned bounding box of the union of all trees, then takes the max side length as the square side.
   - Score aggregates s^2/n per group_id (number of trees), matching Kaggle's evaluation metric.

4) Loading and parsing a solution:
   - parse_csv() reads a Kaggle submission-like CSV and strips leading 's' from x,y,deg (precision-safe strings).
   - Groups rows by group_id (n-tree configuration), builds ChristmasTree list per group, and computes each group's square side.

5) Packing optimization heuristic:
   - get_bbox_touching_tree_indices(): finds trees whose boundaries touch the overall bounding box boundary; these are candidates for removal to reduce the bounding box.
   - For groups from high n downwards, try removing one touching tree to see if the n-1 configuration side improves vs previously stored best.
   - If improved, propagate this configuration down to the n-1 group; this is the 'downward improvement propagation'.
   - This approach never moves trees, it only deletes one, ensuring no new overlaps are introduced; it aims to tighten bounding boxes for smaller groups.

Chinese:
逐步解释该 notebook 如何实现碰撞检测与包装优化：

1）几何与精度设置：
   - 使用高精度 Decimal，避免从 Kaggle “s” 字符串格式转换时的浮点误差。
   - 定义全局 scale_factor(1e18)，在构建 Shapely 多边形前把坐标放大到整数域。

2）圣诞树形状建模（碰撞基础）：
   - 每棵树用一个 Shapely Polygon 表示，近似包含树干与分层树冠。
   - 初始多边形围绕原点构造，随后依据 deg 旋转、依据 (x,y) 平移到最终位置。
   - 碰撞检测是“隐式”的：
     · 可用多边形 pairwise intersects/overlaps 检测；
     · 或者对所有树做 unary_union 后看几何是否异常。
   - 代码依赖比赛约束：若提交存在重叠会报错；因此优化阶段只删除树而不移动树，保持无重叠的可行性。

3）外接正方形与度量：
   - get_tree_list_side_lenght() 计算所有树的并集的轴对齐包围盒，取较长边作为所需正方形边长。
   - 评分按每组 s^2/n 累加，符合比赛评价指标。

4）读取与解析方案：
   - parse_csv() 读取 Kaggle 提交格式 CSV，去掉 x,y,deg 前缀 's'（字符串保精度）。
   - 按 group_id（树的数量）分组，构造每组的 ChristmasTree 列表，并计算该组的正方形边长。

5）包装优化启发式：
   - get_bbox_touching_tree_indices()：找到触碰总体包围盒边界的树，这些树是删除候选，可减少包围盒。
   - 从较大的 n 组向下迭代，尝试删除一个触边树，看删后 n-1 的边长是否优于之前的最优记录。
   - 若改进，则把该配置“向下传播”到 n-1 组：这就是向下改进传播机制。
   - 该方法不挪动树，仅删除一个，确保不会引入新重叠；目标是为更小的树数配置收紧外接正方形。
"""

import pandas as pd
from decimal import Decimal, getcontext
from shapely import affinity, touches
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from pathlib import Path

getcontext().prec = 25
scale_factor = Decimal('1e18')

class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0'):
        """Initializes the Christmas tree with a specific position and rotation."""
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
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )

    def clone(self) -> "ChristmasTree":
        return ChristmasTree(
            center_x=str(self.center_x),
            center_y=str(self.center_y),
            angle=str(self.angle),
        )


def get_tree_list_side_lenght(tree_list: list[ChristmasTree]) -> Decimal:
    all_polygons = [t.polygon for t in tree_list]
    bounds = unary_union(all_polygons).bounds
    return Decimal(max(bounds[2] - bounds[0], bounds[3] - bounds[1])) / scale_factor


def get_total_score(dict_of_side_length: dict[str, Decimal]):
    score = 0
    for k, v in dict_of_side_length.items():
        score += v ** 2 / Decimal(k)
    return score


def parse_csv(csv_path) -> tuple[dict[str, list[ChristmasTree]], dict[str, Decimal]]:
    print(f'parse_csv: {csv_path=}')

    result = pd.read_csv(csv_path)
    result['x'] = result['x'].str.strip('s')
    result['y'] = result['y'].str.strip('s')
    result['deg'] = result['deg'].str.strip('s')
    result[['group_id', 'item_id']] = result['id'].str.split('_', n=2, expand=True)

    dict_of_tree_list: dict[str, list[ChristmasTree]] = {}
    dict_of_side_length: dict[str, Decimal] = {}
    for group_id, group_data in result.groupby('group_id'):
        tree_list = [
            ChristmasTree(center_x=row['x'], center_y=row['y'], angle=row['deg'])
            for _, row in group_data.iterrows()
        ]
        dict_of_tree_list[group_id] = tree_list
        dict_of_side_length[group_id] = get_tree_list_side_lenght(tree_list)

    return dict_of_tree_list, dict_of_side_length


def get_bbox_touching_tree_indices(tree_list: list[ChristmasTree]) -> list[int]:
    """
    Given a list of trees, this function:

      1. Computes the minimal axis-aligned bounding box around all trees.
      2. Returns the list of indices of trees whose boundaries touch
         the boundary of that bounding box.

    Returns:
        touching_indices: list[int]  -- indices in tree_list
    """

    if not tree_list:
        return []

    polys = [t.polygon for t in tree_list]

    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)

    bbox = box(minx, miny, maxx, maxy)

    touching_indices = [
        i
        for i, poly in enumerate(polys)
        if poly.boundary.intersects(bbox.boundary)
    ]

    return touching_indices


if __name__ == '__main__':
    # Load current best solution
    current_solution_path = list(Path('/kaggle/input/').glob('*/*.csv'))[1]
    dict_of_tree_list, dict_of_side_length = parse_csv(current_solution_path)

    # Calculate current total score
    current_score = get_total_score(dict_of_side_length)
    print(f'\n{current_score=:0.8f}')

    # Build report with possible optimizations
    report = pd.DataFrame(pd.Series(dict_of_side_length), columns=['side_length'])
    report['side_length_prev'] = report['side_length'].shift(1)
    report['side_length_increase'] = report['side_length'] - report['side_length_prev']
    report = report[report['side_length_increase'] <= 0].sort_index(ascending=False)
    print('Solutions with easy optimization')
    print(report)

    # Downward propagation optimization
    for group_id_main in range(200, 2, -1):
        group_id_main = f'{int(group_id_main):03n}'
        print(f'\nCurrent box: {group_id_main}')

        candidate_tree_list = [tree.clone() for tree in dict_of_tree_list[group_id_main]]
        candidate_tree_list = sorted(candidate_tree_list, key=lambda a: -a.center_y)

        while len(candidate_tree_list) > 1:
            group_id_prev = f'{len(candidate_tree_list) - 1:03n}'
            best_side_length = dict_of_side_length[group_id_prev]
            best_side_length_temp = Decimal('100')
            best_tree_idx_to_delete: int | None = None

            tree_idx_list = get_bbox_touching_tree_indices(candidate_tree_list)
            for tree_idx_to_delete in tree_idx_list:
                candidate_tree_list_short = [tree.clone() for tree in candidate_tree_list]
                del candidate_tree_list_short[tree_idx_to_delete]

                candidate_side_length = get_tree_list_side_lenght(candidate_tree_list_short)

                if candidate_side_length < best_side_length_temp:
                    best_side_length_temp = candidate_side_length
                    best_tree_idx_to_delete = tree_idx_to_delete

            if best_tree_idx_to_delete is not None:
                del candidate_tree_list[best_tree_idx_to_delete]
                print(len(candidate_tree_list), end=' ')

                if candidate_side_length < best_side_length:
                    print(f'\nimprovement {best_side_length:0.8f} -> {candidate_side_length:0.8f}')

                    dict_of_tree_list[group_id_prev] = [tree.clone() for tree in candidate_tree_list]
                    dict_of_side_length[group_id_prev] = get_tree_list_side_lenght(
                        dict_of_tree_list[group_id_prev]
                    )

            break

    new_score = get_total_score(dict_of_side_length)
    print(f'\n{current_score=:0.8f} {new_score=:0.8f} ({current_score - new_score:0.8f})')

    # Save results
    tree_data = []
    for group_name, tree_list in dict_of_tree_list.items():
        for item_id, tree in enumerate(tree_list):
            tree_data.append({
                'id': f'{group_name}_{item_id}',
                'x': f's{tree.center_x}',
                'y': f's{tree.center_y}',
                'deg': f's{tree.angle}'
            })
    tree_data = pd.DataFrame(tree_data)
    tree_data.to_csv('results.csv', index=False)
