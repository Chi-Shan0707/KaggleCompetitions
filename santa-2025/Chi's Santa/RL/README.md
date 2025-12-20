Basic RL Baseline (REINFORCE) · 双语说明

English
- This folder provides a minimal REINFORCE baseline using a Shapely-based local env (no pybind/Torch/NumPy).
- Files:
	- reinforce_agent.py: Standard-library Gaussian linear policy + REINFORCE updates (no NumPy/Torch).
	- env_simple.py: Shapely-based `SimplePackingEnv` (tree polygon, overlap checks, bounding square).
	- train_basic_rl.py: Training script saving policy parameters to JSON.
	- make_submission.py: Deterministic inference and CSV writer, supports optional input.csv seed.

中文
- 本目录提供一个基于 Shapely 的最小 REINFORCE 基线（无需 pybind/Torch/NumPy）。
- 文件说明：
	- reinforce_agent.py：纯标准库实现的高斯线性策略 + REINFORCE 更新。
	- env_simple.py：Shapely 环境（树的多边形轮廓、重叠检测、包围正方形评分）。
	- train_basic_rl.py：训练脚本，参数以 JSON 格式保存。
	- make_submission.py：确定性推理与 CSV 写出，支持 `input.csv` 作为可行解种子。

Requirements · 依赖
```bash
pip install -r "santa-2025/Chi's Santa/RL/requirements.txt"
```

Train · 训练
```bash
python "santa-2025/Chi's Santa/RL/train_basic_rl.py" \
	--n_trees 10 --epochs 5 \
	--save_path "santa-2025/Chi's Santa/RL/reinforce_policy.json"
```

Submit · 生成提交
```bash
python "santa-2025/Chi's Santa/RL/make_submission.py" \
	--policy "santa-2025/Chi's Santa/RL/reinforce_policy.json" \
	--input "santa-2025/Chi's Santa/RL/input.csv" \
	--out "submission.csv"
```

Make it different next time · 让结果“不同”起来
- Deterministic (default) always outputs the same given the same policy and seeds.
- To diversify submissions, either retrain with a different seed or sample at inference:
	- English:
		```bash
		# Retrain with a different seed
		python "santa-2025/Chi's Santa/RL/train_basic_rl.py" --n_trees 12 --epochs 500 --seed 123 \
			--save_path "santa-2025/Chi's Santa/RL/reinforce_policy.json"

		# Or sample during submission (stochastic) with a chosen RNG seed
		python "santa-2025/Chi's Santa/RL/make_submission.py" \
			--policy "santa-2025/Chi's Santa/RL/reinforce_policy.json" \
			--mode sample --seed 20251213 \
			--input "santa-2025/Chi's Santa/RL/input.csv" \
			--out "submission.csv"
		```
	- 中文：
		```bash
		# 使用不同随机种子重新训练
		python "santa-2025/Chi's Santa/RL/train_basic_rl.py" --n_trees 10 --epochs 500 --seed 123 \
			--save_path "santa-2025/Chi's Santa/RL/reinforce_policy_seed123.json"

		# 或者在提交时开启采样（随机），并设定 RNG 种子
		python "santa-2025/Chi's Santa/RL/make_submission.py" \
			--policy "santa-2025/Chi's Santa/RL/reinforce_policy.json" \
			--mode sample --seed 20251213 \
			--input "santa-2025/Chi's Santa/RL/input.csv" \
			--out "submission.csv"
		```

Submission Rules · 提交规则
- Values must be strings with an `s` prefix (e.g., `s-0.541068`).
- No overlaps; placements outside bounds are rejected.
- abs(x) < 100 and abs(y) < 100.
- Exactly 20100 data rows (sum for n=1..200), header is `id,x,y,deg`.

Notes · 说明
- Actions are 3D continuous: (x, y, deg) squashed by tanh to [-1,1], mapped to coordinates and rotation.
- This baseline is intentionally simple; for stronger results, consider improved exploration, curriculum, or a richer policy.


```
python make_submission.py \
--policy "reinforce_policy.json" \
--input "input.csv" \
--out "submission.csv"
```