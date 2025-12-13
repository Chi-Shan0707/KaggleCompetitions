"""
English (Overview):
    Minimal Shapely-based environment for the Santa 2025 packing task.
    - State: normalized placements of placed trees + progress ratio.
    - Action: 3D continuous (x, y, deg) in [-1,1], mapped to coordinates and rotation.
    - Reward: negative normalized square bounding area; hard penalty (reject) on overlap.
    - Goal: place exactly n trees with no overlap, compactly.

中文（概览）：
    基于 Shapely 的极简环境，用于 Santa 2025 打包任务。
    - 状态：已放树的归一化 (x,y,deg) 序列 + 进度比例。
    - 动作：连续 3 维 (x, y, deg)∈[-1,1]，映射到坐标与角度。
    - 回报：包围正方形面积的负值；若重叠则拒绝当前放置（等价于强惩罚）。
    - 目标：无重叠地恰好放置 n 棵树，并尽可能紧凑。
"""
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from shapely.geometry import Polygon
from shapely import affinity


def build_christmas_tree_polygon(cx: float, cy: float, angle_deg: float, scale: float = 1.0) -> Polygon:
    """Build a single Christmas-tree polygon at (cx, cy) rotated by angle_deg.

    English: The points follow the same shape definition as the competition prompt.
    中文：点序列遵循赛题提供的树轮廓定义。
    """
    trunk_w = 0.15
    trunk_h = 0.2
    base_w = 0.7
    mid_w = 0.4
    top_w = 0.25
    tip_y = 0.8
    tier_1_y = 0.5
    tier_2_y = 0.25
    base_y = 0.0
    trunk_bottom_y = -trunk_h

    pts = [
        (0.0 * scale, tip_y * scale),
        (top_w / 2 * scale, tier_1_y * scale),
        (top_w / 4 * scale, tier_1_y * scale),
        (mid_w / 2 * scale, tier_2_y * scale),
        (mid_w / 4 * scale, tier_2_y * scale),
        (base_w / 2 * scale, base_y * scale),
        (trunk_w / 2 * scale, base_y * scale),
        (trunk_w / 2 * scale, trunk_bottom_y * scale),
        (-(trunk_w / 2) * scale, trunk_bottom_y * scale),
        (-(trunk_w / 2) * scale, base_y * scale),
        (-(base_w / 2) * scale, base_y * scale),
        (-(mid_w / 4) * scale, tier_2_y * scale),
        (-(mid_w / 2) * scale, tier_2_y * scale),
        (-(top_w / 4) * scale, tier_1_y * scale),
        (-(top_w / 2) * scale, tier_1_y * scale),
    ]

    poly = Polygon(pts)
    rotated = affinity.rotate(poly, angle_deg, origin=(0, 0))
    moved = affinity.translate(rotated, xoff=cx * scale, yoff=cy * scale)
    return moved


@dataclass
class SimpleEnvConfig:
    n_trees: int = 10
    max_coord: float = 50.0
    scale: float = 1.0
    overlap_eps: float = 1e-9


class SimplePackingEnv:
    """English: Tiny environment with overlap checks and bounding-square reward.

    中文：包含重叠检测与包围正方形回报的简易环境。
    """
    def __init__(self, cfg: SimpleEnvConfig):
        self.cfg = cfg
        self.placed: List[Tuple[float, float, float]] = []
        self.polys: List[Polygon] = []
        self.index: int = 0

    def reset(self):
        """English: Clear state and start a new episode.

        中文：清空状态并开始新的回合。
        """
        self.placed.clear()
        self.polys.clear()
        self.index = 0
        return self._state()

    def load_initial(self, placements: List[Tuple[float, float, float]]):
        """English: Load seed placements (used for input.csv seeding).

        中文：加载种子解（用于从 input.csv 预置放置）。
        - 会将坐标钳制到 |x|,|y|<=100，并跳过与已放置树重叠的条目。
        """
        self.reset()
        for x, y, deg in placements:
            # Clamp seed coordinates to |x|,|y| <= 100 per competition constraint
            x = max(-100.0, min(100.0, float(x)))
            y = max(-100.0, min(100.0, float(y)))
            deg = float(deg)
            poly = build_christmas_tree_polygon(x, y, deg, self.cfg.scale)
            # Skip overlapping seeds defensively
            if self.any_overlap_with(poly):
                continue
            self.polys.append(poly)
            self.placed.append((x, y, deg))
            self.index += 1
        return self._state()

    def _state(self) -> List[float]:
        """English: Build normalized flat state vector.

        中文：构造归一化的一维状态向量。
        - 先是按顺序的 (x/max_coord, y/max_coord, deg/360)；
        - 然后用 0 补足到 n_trees；
        - 末尾附加进度（index/n_trees）。
        """
        flat = []
        for x, y, d in self.placed:
            flat.extend([x / self.cfg.max_coord, y / self.cfg.max_coord, d / 360.0])
        needed = self.cfg.n_trees - len(self.placed)
        flat.extend([0.0, 0.0, 0.0] * needed)
        flat.append(self.index / max(1, self.cfg.n_trees))
        return flat

    def any_overlap_with(self, new_poly: Polygon) -> bool:
        """English: Return True if new_poly overlaps any existing polygon.

        中文：若 new_poly 与任一已放多边形重叠则返回 True。
        """
        for p in self.polys:
            inter = p.intersection(new_poly)
            if not inter.is_empty and inter.area > self.cfg.overlap_eps:
                return True
        return False

    def bounding_square_side(self) -> float:
        """English: Side length of the minimal axis-aligned square covering all trees.

        中文：覆盖全部树的最小轴对齐正方形的边长。
        """
        if not self.polys:
            return 0.0
        minx = min(p.bounds[0] for p in self.polys)
        miny = min(p.bounds[1] for p in self.polys)
        maxx = max(p.bounds[2] for p in self.polys)
        maxy = max(p.bounds[3] for p in self.polys)
        w = maxx - minx
        h = maxy - miny
        return max(w, h)

    def step(self, action: List[float]):
        """English: Place one tree according to action; reject overlaps; give reward.

        中文：根据动作放置一棵树；若重叠则拒绝；给出回报。
        - 超过目标数量时立即 done=True，避免超额输出。
        """
        # Disallow placing beyond target count
        if self.index >= self.cfg.n_trees:
            return self._state(), 0.0, True, {"reason": "max_placed"}

        ax, ay, adeg = action
        x = max(-1.0, min(1.0, ax)) * self.cfg.max_coord
        y = max(-1.0, min(1.0, ay)) * self.cfg.max_coord
        deg = max(-1.0, min(1.0, adeg)) * 180.0

        new_poly = build_christmas_tree_polygon(x, y, deg, self.cfg.scale)
        if self.any_overlap_with(new_poly):
            return self._state(), -10, False, {}
        # 放错的惩罚

        self.polys.append(new_poly)
        self.placed.append((x, y, deg))
        self.index += 1

        side = self.bounding_square_side()
      #  score_component = (side * side) / len(self.polys)
        score_component = (side * side)/ self.index
        reward = -score_component
        done = (self.index >= self.cfg.n_trees)
        return self._state(), reward, done, {"side": side, "score_component": score_component}

    def output_rows(self, n: int) -> List[Tuple[str, str, str, str]]:
        """English: Format placements as Kaggle-required CSV rows with 's' prefix.

        中文：按 Kaggle 要求格式化输出行，数值以字符串并加前缀 's'。
        """
        rows = []
        for i, (x, y, deg) in enumerate(self.placed):
            sid = f"{n:03d}_{i}"
            rows.append((sid, f"s{float(x):.6f}", f"s{float(y):.6f}", f"s{float(deg):.6f}"))
        return rows
