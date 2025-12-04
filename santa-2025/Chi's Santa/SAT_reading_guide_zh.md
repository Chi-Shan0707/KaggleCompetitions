# collision_detection.cpp 阅读指南：SAT 分离轴定理实现

本指南按“先概念→再定位→再验证”的顺序，带你高效读懂 `santa-2025/collision_detection.cpp` 中的 SAT（Separating Axis Theorem）实现与凹多边形处理策略。

## 阅读顺序
- Vec2 与基础运算：先看 `geom::Vec2<T>` 及 `dot/cross/perp/rotate`，理解投影与法向的来源。
- 统一接口 `Shape`：理解多态入口 `overlaps` 的职责与意图。
- `PolygonBase`：看 `translate/rotateAround` 与 `projectOnAxis`，掌握“投影区间”的实现细节。
- `ConvexPolygon::overlaps`：SAT 的核心逻辑，逐边法向→投影→区间判定。
- `ConcavePolygon`：阅读“剪耳法三角化”，理解“用多个三角形近似凹多边形做对碰”的策略。
- `makeRegularPolygon` 与 `main`：结合示例改动参数，验证你对重叠判定的理解。

## SAT 概念速览
- 基本思想：
  - 对每条边取法向轴 `a`，把两个多边形分别投影到 `a` 上。
  - 投影是一个实数区间：`[p_min, p_max]`，其中 `p = v ⋅ a`（点积）。
  - 若存在任意轴使区间不相交（`aMax < bMin` 或 `bMax < aMin`），则两形状分离（不重叠）。
  - 所有轴均相交 ⇒ 不存在分离轴 ⇒ 重叠。
- 轴的来源：
  - 凸多边形只需测试双方所有边的法向（`perp(edge)`）。轴不需要单位化，区间比较对比例不敏感。

## 关键符号与位置（按文件内出现顺序）
- `geom::Vec2<T>`：二维向量与基本运算（加减、数乘）。
- `dot/cross/perp/rotate`：
  - `dot(a,b)` 投影的标量；`cross(a,b)` 判断转角方向；
  - `perp(v)` 生成边的候选分离轴；
  - `rotate(v, deg)` 以“度”为单位的旋转，便于可视化测试。
- `Shape<T>`：抽象接口 `translate/rotateAround/overlaps`，为多态对碰提供统一入口。
- `PolygonBase<T>::projectOnAxis`：将顶点集投影到给定轴，返回 `(min, max)` 区间。
- `ConvexPolygon<T>::overlaps`（核心）：
  - 生成轴：对 A、B 的每条边取 `axis = perp(v[i+1]-v[i])`；
  - 投影：`projectOnAxis(A, axis)` 与 `projectOnAxis(B, axis)`；
  - 判定：若 `aMax < bMin || bMax < aMin`，立即返回“不重叠”（早退出）；
  - 若所有轴都通过，则返回“重叠”。
- `ConcavePolygon<T>`：
  - `triangulate`：剪耳法把凹多边形拆成若干三角形；
  - `ensureConvexParts`：惰性三角化与缓存；
  - `overlaps`：与对方（凸/凹）逐三角形对碰，任一对重叠即整体重叠。
- `makeRegularPolygon`：快速构造规则多边形（用于验证/测试）。
- `main`：构造方形与三角形、平移对碰；再用“箭头”凹多边形对碰小方块。

## 代码走读要点
- `perp(v){-v.y, v.x}` 即边的候选分离轴来源。
- `projectOnAxis` 的正确性：对每个顶点 `v_i` 计算 `p_i = dot(v_i, axis)`，取全体最小/最大即为区间；轴是否单位化不影响“是否相交”。
- SAT 必须测试双方所有边的法向：见 `ConvexPolygon::overlaps` 对 `A`、`B` 各自 `testAxes`；任意一轴分离即可判“不重叠”。
- 凹多边形对碰：先三角化为凸部件，然后与对方逐部件使用凸多边形的 SAT；凹 vs 凹 则双方都拆再两两尝试。

## 数值与鲁棒性提示
- 精度选择：文件示例使用 `double`，更稳健。若替换为 `float`，建议在区间比较引入容差 `eps`：
  - 例如把 `aMax < bMin` 改为 `aMax < bMin - eps`（`eps ≈ 1e-9 ~ 1e-6`）。
- 退化与清洗：重复点/零长度边会影响法向与三角化；工程中通常在构造阶段做顶点去重/简化。
- 三角化是极简版：遇到自交或强退化多边形可能失败；原型/竞赛常够用，工程中可替换更鲁棒的算法（如 Hertel–Mehlhorn 等）。

## 如何动手验证
- 在 Windows 下用 MinGW g++ 编译运行（确保当前目录为 `santa-2025`）：

```cmd
cd santa-2025
g++ -std=c++17 -O2 collision_detection.cpp -o collision_detection.exe
collision_detection.exe
```

- 若使用 MSVC，需要把 `#include <bits/stdc++.h>` 替换为常规头（`<vector> <array> <numeric> <algorithm> <iostream> <cmath>` 等）。
- 小实验建议：
  - 改变 `makeRegularPolygon` 的 `rotationDeg` 与 `center`，观察重叠输出变化；
  - 把 `triangle` 改为自定义 `ConcavePolygon`，观察与方形的结果；
  - 在 `ConvexPolygon::overlaps` 中临时打印每个轴的 `(aMin,aMax)` 与 `(bMin,bMax)`，直观看区间关系（调试时用，完成后移除）。

## 进阶改造方向（按需）
- 碰撞响应：在 SAT 循环中记录最小重叠量与对应轴，输出最小穿透向量（MTV）。
- 性能加速：加入 AABB 预筛（轴对齐包围盒）来快速剔除明显不重叠的情况。
- 轴去重：共线边只需测试一次法向；近零长度边可跳过。
- 代码复用：若要与 `geom.hpp` 融合，可抽离通用几何组件，统一数值类型与容差策略。

---
如需，我可以：
- 增加 MTV 版本的 `overlaps`，用于后续解碰或排序摆放；
- 为凹多边形加入更鲁棒的剖分/预清洗；
- 附带一个最小单元测试脚本，覆盖凸/凹、旋转/平移与边界情形。
