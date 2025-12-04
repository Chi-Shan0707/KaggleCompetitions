# Santa-2025 Packing Prototype Manual / 圣诞老人2025打包原型手册

English first, then Chinese equivalent below each bullet. 先英文后中文翻译，保证双语学习体验。

---
## 1. Theory Overview / 理论概览
**Goal:** Place gift polygons to minimize sum of (bounding square side^2 / n). 目标: 放置礼物多边形, 最小化 (最小包围正方形边长^2 / 多边形数量) 总和。
**Convex Collision (SAT):** Two convex polygons overlap if all projections onto every edge normal overlap; if any axis separates, no collision. 凸多边形碰撞(SAT): 若投影到每条边法向量上都重叠则相交; 若存在分离轴则不相交。
**Concave Handling (Demo):** Triangulate (ear clipping) then test convex triangles pairwise. 凹多边形处理: 耳切法三角剖分后两两凸三角形测试。
**Numerical Robustness:** Floating error near touching edges—use epsilon (e.g., 1e-9). 数值鲁棒性: 边缘接触时浮点误差; 可使用 epsilon (如1e-9)。
**Rotation + Translation Order:** Rotate about centroid, then translate. 旋转平移顺序: 先绕质心旋转, 再平移。
**Performance Hooks:** Precompute normals, early exit on first separating axis, spatial partitioning (grid / BVH) for many polygons. 性能策略: 法向量预计算, 分离轴立即退出, 使用网格或层次包围体加速多物体检测。
**RL Framing:** PPO learns placement actions (x,y,θ) minimizing post-placement penalty. 强化学习框架: PPO 学习 (x,y,角度θ) 放置操作以降低惩罚。
**Reward Proxy:** Negative of bounding square metric approximates score improvement. 奖励代理: 使用包围正方形度量的负值近似得分提升。

---
## 2. File Roles / 文件作用
`collision_detection.cpp`: Full SAT + concave demo with standalone main. 完整SAT及凹多边形演示含main。
`geom.hpp`: Header-only convex geometry utilities for binding. 仅头文件凸几何工具用于绑定。
`sat_bindings.cpp`: Pybind11 module exposing core C++ types/functions. Pybind11 模块暴露核心C++类型/函数。
`setup.py`: Build script for Python extension (`satgeom`). 构建Python扩展的脚本(`satgeom`)。
`ppo_env.py`: Simplified packing environment wrapping geometry metrics. 简化打包环境封装几何度量。
`ppo_agent.py`: Minimal PPO implementation (Actor-Critic + GAE). 最小PPO实现 (Actor-Critic + GAE)。
`GPT's guide.md`: High-level overview (superseded by this manual). 总览(本手册更全面)。
`MANUAL.md`: Comprehensive bilingual guide. 全面双语指南。

---
## 3. Build & Install / 构建与安装
Prerequisites: Python >=3.9, C++17 compiler (g++/clang), PyTorch, pybind11 headers. 先决条件: Python>=3.9, C++17编译器, PyTorch, pybind11头文件。
Create env: `conda create -n santa_env python=3.9 -y && conda activate santa_env` 创建环境。
Install deps: `pip install pybind11 torch` 安装依赖。
Build extension: In project root run `pip install -e santa-2025`. 构建扩展: 在项目根执行上述命令。
Verify: `python -c "import satgeom; print(satgeom.bounding_square_side([]))"` 验证。
Potential flags: Adjust `-O2` to `-O3 -march=native` for speed. 可能优化: 将`-O2`改为`-O3 -march=native`以提速。

---
## 4. Usage Workflow / 使用流程
1. Prepare polygon list (convex regular prototypes). 准备多边形列表(规则凸原型)。
2. Reset env -> get initial state. 重置环境获取初始状态。
3. PPO selects (dx, dy, dθ) ∈ [-1,1]^3 then scaled. PPO 输出归一化动作后再缩放。
4. Env applies transform, checks overlap via SAT; invalid => penalty. 环境应用变换, SAT检测, 无效则惩罚。
5. Repeat rollout for horizon then optimize PPO. 完成时间步后优化 PPO。
Run training: `python santa-2025/ppo_agent.py`. 运行训练命令。
Monitor loss: Printouts indicate policy/value update dynamics. 监控损失: 输出显示策略/价值更新趋势。

---
## 5. Debugging Guide / 调试指南
Build errors: Missing pybind11 -> `pip install pybind11`. 构建错误: 缺少pybind11则安装。
Undefined symbols: C++ name mismatch; ensure identical signature. 未定义符号: C++名称不匹配需核对。
ImportError satgeom: Re-run editable install or check `PYTHONPATH`. 无法导入: 重新运行安装或检查路径。
Overlap false positives: Inspect edge normal generation; add epsilon separation check. 误报碰撞: 检查法向量生成并使用epsilon。
Reward exploding: Large negative returns—scale action space smaller. 奖励发散: 大负回报则缩小动作尺度。
NaN in training: Check learning rate, normalize advantages. 训练出现NaN: 调整学习率并归一化优势。
Slow loop: Vectorize or move more logic into C++. 速度慢: 向量化或更多逻辑移入C++。

---
## 6. PPO Training & Tuning / PPO训练与调优
Key Hyperparameters: 学习率(lr), γ (discount), λ (GAE), clip_range, entropy_coef, value_coef, batch_size, epochs.
Start Defaults: lr=3e-4, γ=0.99, λ=0.95, clip=0.2. 初始默认值。
Entropy Coef: Increase to encourage exploration early. 增大熵系数促进早期探索。
Value Loss Weight: Raise if value underfits (high prediction error). 价值损失权重不足则提高。
Advantage Normalization: Keeps gradient stable; always apply. 优势归一化保持梯度稳定。
Action Scaling: Map tanh output to feasible spatial range. 动作缩放: 将tanh结果映射到合理空间范围。
Curriculum: Start with fewer polygons then scale complexity. 课程学习: 先少多边形再增复杂度。
Parallel Envs: Use vectorized envs for better sample efficiency. 并行环境: 提高样本效率。
Checkpointing: Save model every N updates; name with score. 检查点: 定期保存并附加得分。

---
## 7. Submission Generation / 提交生成
Format (example assumption): CSV with columns id,x,y,deg; id prefixed by 's'. 格式假设: CSV列 id,x,y,deg; id 前缀 's'。
Procedure: 程序步骤:
- Collect placed polygons final transforms. 收集最终多边形变换。
- Convert rotation radians -> degrees. 弧度转度数。
- Validate non-overlap (SAT). 验证无重叠。
- Write CSV `submission.csv`. 写出CSV。
Command sketch: `python make_submission.py --model ckpt.pth --out submission.csv`. 命令示例。
Quality check: Re-run quick overlap scan before upload. 质量检查: 上传前重新扫碰撞。

---
## 8. Extension Roadmap / 扩展路线图
Concave Binding: Expose concave decomposition to Python. 凹多边形绑定: 暴露剖分结果。
Spatial Index: Uniform grid or BVH for O(N log N) checks. 空间索引: 网格或层次包围体降低复杂度。
Advanced Reward: Include packing density & overlap penalty shaping. 高级奖励: 加入密度与重叠惩罚塑造。
Multi-Agent: Cooperative placement strategies. 多智能体: 协同放置策略。
GPU Geometry: Port SAT to CUDA for large batches. GPU几何: 将SAT移植CUDA。
Auto Curriculum: Difficulty scheduler. 自动课程: 难度调度器。

---
## 9. Quick Reference / 快速参考
Build: `pip install -e santa-2025` 构建命令。
Train: `python santa-2025/ppo_agent.py` 训练命令。
Inspect: `python -m pip show satgeom` 查看扩展。
List Symbols: `nm -C build/temp*/satgeom*.so | grep ConvexPolygon` 符号列表。
Profile: `python -m cProfile -o prof.out ppo_agent.py` 性能分析。

---
## 10. FAQ / 常见问题
Q: Why convex only in Python? A: Simplicity & speed first; extend later. 问: 为什么Python只支持凸? 答: 先保证简单与速度, 之后扩展。
Q: Can I use irregular polygons? A: Yes if convex; update loader. 问: 能用不规则多边形吗? 答: 凸即可, 需更新加载器。
Q: Training too slow? A: Parallelize envs or move more logic to C++. 问: 训练慢? 答: 并行环境或迁移逻辑到C++。
Q: Overlap mis-detected? A: Add epsilon & review normals. 问: 碰撞误判? 答: 加epsilon并检查法向量。

---
End of Manual / 手册结束
