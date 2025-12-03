"""
English Summary:
 Simple feasibility environment for Santa-2025 packing. We sequentially place regular polygons.
 State: normalized list of (x,y,deg) for placed polygons + progress index.
 Action: 3 continuous values in [-1,1] mapped to (x,y,deg) of next polygon.
 Reward: negative bounding-square density, large penalty for overlap.
 Episode ends when all n_trees placed.

中文摘要：
 针对 Santa-2025 的可行性环境原型，按顺序放置正多边形。
 状态：已放置多边形 (x,y,deg) 归一化列表 + 进度索引。
 动作：3 个连续值（范围 [-1,1]）映射为下一个多边形的 (x,y,deg)。
 回报：负的包围正方形密度（越紧凑越高），重叠给予大惩罚。
 当全部 n_trees 放置完成时结束。
"""
import math    # EN: Might be used for future extensions. CN：为后续扩展保留。
import random  # EN: Random exploratory actions. CN：随机探索动作。
from typing import List, Tuple  # EN: Type hints for clarity. CN：类型提示提升可读性。
import satgeom  # EN: pybind11 module with geometry ops. CN：几何操作的 pybind11 模块。

class PackingEnv:
    def __init__(self, n_trees:int=10, radius:float=1.0, sides:int=3, rotation_step:float=5.0, max_coord:float=50.0):
        # EN: Number of polygons to place in one episode.
        # CN：单个回合要放置的多边形数量。
        self.n_trees = n_trees
        # EN: Radius of regular polygons. CN：正多边形半径。
        self.radius = radius
        # EN: Number of sides (shape complexity). CN：边数（形状复杂度）。
        self.sides = sides
        # EN: Intended discrete rotation step (not strictly used yet). CN：计划的旋转步长（当前未强制）。
        self.rotation_step = rotation_step
        # EN: Maximum coordinate absolute value for placement scaling. CN：坐标放置的最大幅度，用于缩放动作。
        self.max_coord = max_coord
        # EN: List storing placed polygon transforms. CN：已放置多边形的变换信息列表。
        self.placed: List[Tuple[float,float,float]] = []  # (x,y,deg)
        # EN: Actual geometry objects for collision checks. CN：真实几何对象用于碰撞检测。
        self.polys: List[satgeom.ConvexPolygon] = []
        # EN: Index of next polygon to place. CN：下一个待放置多边形的索引。
        self.index = 0

    def reset(self):
        # EN: Clear episode data and start from scratch. CN：清空回合数据重新开始。
        self.placed.clear(); self.polys.clear(); self.index = 0
        return self._get_state()

    def _get_state(self):
        # EN: Build flattened normalized state vector.
        # CN：构造归一化的状态向量。
        flat = []
        for x,y,d in self.placed:
            # EN: Normalize position by max_coord, rotation by 360.
            # CN：位置除以 max_coord 归一；角度除以 360。
            flat.extend([x/ self.max_coord, y/ self.max_coord, d/360.0])
        # EN: Pad remaining slots with zeros so state length stays constant.
        # CN：用 0 填充剩余未放置位置，保证状态长度恒定。
        needed = self.n_trees - len(self.placed)
        flat.extend([0.0,0.0,0.0]*needed)
        # EN: Append progress (fraction completed). CN：附加进度比例。
        flat.append(self.index / self.n_trees)
        return flat

    def step(self, action):
        # EN: Apply action to produce placement candidate.
        # CN：应用动作生成待放置位置。
        ax, ay, adeg = action  # EN: Unpack 3 raw action components. CN：拆解动作三个分量。
        x = ax * self.max_coord  # EN: Scale normalized x. CN：缩放 x。
        y = ay * self.max_coord  # EN: Scale normalized y. CN：缩放 y。
        deg = adeg * 180.0       # EN: Map [-1,1] to roughly [-180,180]. CN：角度映射。
        poly = satgeom.make_regular_polygon(self.sides, self.radius, satgeom.Vec2(x,y), deg)  # EN: Build geometry. CN：构建几何对象。
        test_polys = self.polys + [poly]  # EN: Candidate list with new polygon. CN：包含新多边形的检测列表。
        if satgeom.any_overlap(test_polys):  # EN: Reject if overlaps existing. CN：若与已有重叠则拒绝。
            reward = -10.0       # EN: Penalty for invalid placement. CN：非法放置惩罚。
            done = False         # EN: Episode continues. CN：回合继续。
            return self._get_state(), reward, done, {}  # EN: Retry same index. CN：不推进索引，允许重试。
        # EN: Accept placement.
        # CN：接受放置。
        self.polys.append(poly)
        self.placed.append((x,y,deg))
        self.index += 1
        side = satgeom.bounding_square_side(self.polys)  # EN: Current bounding square side. CN：当前包围正方形边长。
        score_component = (side*side)/ len(self.polys)   # EN: Normalized area contribution. CN：归一面积指标。
        reward = -score_component  # EN: Lower is better (minimize area). CN：越小越好。
        done = (self.index >= self.n_trees)  # EN: Episode completion check. CN：检查是否结束。
        return self._get_state(), reward, done, {"side": side, "score_component": score_component}

if __name__ == '__main__':
    # EN: Quick smoke test. CN：快速冒烟测试。
    env = PackingEnv(n_trees=3)
    s = env.reset()
    for _ in range(10):
        a = [random.uniform(-1,1) for _ in range(3)]  # EN: Random exploratory action. CN：随机探索动作。
        ns,r,d,info = env.step(a)
        print(r,d,info)
        if d: break  # EN: Stop once all placed. CN：全部放置后退出。
