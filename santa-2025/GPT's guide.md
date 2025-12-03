## Santa-2025 Packing Prototype (SAT + PPO)

### 1. Geometry & Collision (C++ via pybind11)
Files:
- `geom.hpp`: Header-only convex polygon + SAT utilities.
- `sat_bindings.cpp`: Pybind11 module exposing:
  - `Vec2(x,y)`
  - `ConvexPolygon([...Vec2])`
  - `make_regular_polygon(sides, radius, center=Vec2(0,0), rotationDeg=0)`
  - `bounding_square_side([polys])`
  - `any_overlap([polys])`

Build (inside `santa-2025/`):
```bash
pip install pybind11 torch
python setup.py build_ext --inplace
python -c "import satgeom; print('satgeom ok')"
```

### 2. Python Environment (`ppo_env.py`)
Simplified assumptions:
- Place `n_trees` regular polygons (same sides & radius).
- Action: 3 continuous values in [-1,1] mapped to (x,y,deg).
- Overlap -> reward -10 (retry same index).
- Valid placement -> reward = -(side^2 / current_count). Smaller bounding square gives less negative (better).

### 3. PPO Agent (`ppo_agent.py`)
Minimal PPO (learning-oriented, not optimized):
- ActorCritic with shared MLP, tanh actions.
- Advantage via GAE (λ=0.95), clipping ratio 0.2.
- Entropy bonus for exploration.
- Saves model `ppo_model.pt` after training.

Run training:
```bash
python ppo_agent.py
```

### 4. Next Extensions
- Variable tree shapes (load polygons from file).
- Curriculum: start with few trees → increase count.
- Action masking: disallow positions near existing polygons to reduce invalid trials.
- Better reward: use delta improvement or normalized vs theoretical packing density.
- Parallel env sampling to speed up PPO.

### 5. Exporting Submission CSV (Concept)
After a trained policy produces placements:
1. For each configuration id + tree index produce `id_i,x,y,deg`.
2. Convert numeric to string prefixed with `s` (e.g. `s0.1234`).
3. Ensure no overlaps via `any_overlap` before writing.
Example stub:
```python
def save_submission(filename, placements):
    with open(filename,'w') as f:
        f.write('id,x,y,deg\n')
        for pid,(x,y,deg) in placements:
            f.write(f"{pid},s{x},s{y},s{deg}\n")
```

### 6. Common Build Issues
- Missing `pybind11`: install it (`pip install pybind11`).
- Compiler errors on Windows: ensure a C++17 capable compiler (MSVC or WSL g++). Under WSL this repo path works with g++.
- `ImportError: satgeom`: ensure build placed `satgeom.*.so` in `santa-2025/` or `PYTHONPATH` includes folder.

### 7. Reasoning Trade-offs
This prototype favors *clarity over performance*. SAT calls scale O(N^2) with polygon count per step (acceptable for ≤200). For speed later:
- Spatial index (uniform grid / BVH)
- Batch overlap checks in C++ returning vector<bool>
- Replace naive random exploration with learned action prior.

### 8. Quick Clean Commands
```bash
find . -name "*.so" -delete  # remove built extension
python setup.py build_ext --inplace
```

### 9. FAQ
Q: Why negative rewards? → PPO maximizes expected return; negative is fine if "less negative" means better packing.
Q: Why tanh actions? → Natural bounded output; simple scaling to coordinate ranges.
Q: Could we use discrete grid? → Yes; replace actor with categorical policy; easier collision validity but less precision.
Q: Concave trees? → Need decomposition (already in original collision file); expose concave later.

Great — since you want **learning experience**, **mathematical clarity**, and **a beginner-friendly but meaningful ML path**, AND you **don’t have training data**, we should pick methods that:

* **Teach you actual computational geometry + optimization**
* **Do NOT require a dataset**
* **Are actually useful in real research**
* **Fit the Kaggle problem structure**
* **Are implementable by a novice with reasonable time**

Below is a curated, realistic “learning-first” plan, listing **the exact mathematical methods** you should study and **the specific ML frameworks** that work *without a dataset*.

---

# 🧠 PART 1 — Mathematical / Algorithmic Methods (Recommended Set)

These are the **core ideas** behind most real-world packing/geometry solvers.
I’ll list them from **easy → medium → advanced**.

---

# **1. Collision Detection (必学)**

These appear everywhere in computational geometry, games, physics engines.

### Learn:

* **AABB (Axis-Aligned Bounding Box)**
* **Sweep and prune (Sort and sweep)**
* **Separating Axis Theorem (SAT)**
  For convex polygons — core of physics engines
* **Polygon intersection tests** (Shapely implements these)

These give the foundation for “trees must not overlap”.

🟩 Difficulty: ★☆☆☆☆
🟩 Why useful: Core building block for all geometry optimization

---

# **2. Heuristic Packing (Classic Computational Geometry)**

These give you the initial layouts.

### Greedy Placement:

* **Bottom-left heuristic** (standard for 2D packing)
* **Largest-first placement** (sort by size)

### Discrete rotation search:

* Try angles {0°, 15°, 30°, … 360°}

🟩 Difficulty: ★☆☆☆☆
🟩 Why useful: The “first 30% solution”

---

# **3. Local Optimization (核心)**

This is where you learn **real optimization techniques**.

### (A) **Simulated Annealing**

Easy to implement, powerful for packing.
Teaches: local minima, energy landscapes, stochastic search.

### (B) **Hill-Climbing / Gradient-free search**

You adjust:

* Tree positions (x, y)
* Rotation (θ)

Although no gradients exist, you explore the neighborhood.

### (C) **Variable Neighborhood Search (VNS)**

A structured way to explore different move types.

🟩 Difficulty: ★★☆☆☆
🟩 Why useful: Standard in combinatorial optimization, many research papers

---

# **4. Global Search / Metaheuristics (Optional but Cool)**

If you want more:

### (A) **Genetic Algorithms (GA)**

Chromosome = sequence of placements
Useful when SA gets stuck.

### (B) **Evolution Strategies (CMA-ES)**

Good for continuous rotations.

🟩 Difficulty: ★★★☆☆
🟩 Why useful: ML-like but deterministic; teaches population search

---

# **5. Constraint Optimization (Optional)**

### (A) **Mixed-Integer Nonlinear Programming (MINLP)**

In theory tree placement can be formulated as an MINLP.
In practice too slow — but good to learn how to express problems.

### (B) **Penalty methods**

Turn “overlap” into a penalty term:

```
energy = bounding_box_size + λ * sum(overlap_area)
```

🟩 Difficulty: ★★★★☆
🟩 Why useful: bridges pure math → ML optimization

---

# 🧩 Summary of Recommended Mathematical Methods

If you want the **best learning experience with reasonable difficulty**:

✔ Sweep & prune
✔ AABB collision detection
✔ Separating Axis Theorem
✔ Greedy placement
✔ Simulated Annealing
✔ Hill-climbing
✔ VNS (Variable Neighborhood Search)
✔ Penalty-based energy formulation

This combo gives you a **real optimization engine**.

---

# 🤖 PART 2 — ML / DL Methods (Beginner-Friendly, No Dataset Needed)

Since you have **no dataset**, the only feasible ML is:

✔ **Reinforcement Learning (RL)**
✔ **Self-supervised prediction models**
✔ **Neural heuristics to assist search**, trained online
❌ NOT supervised learning
❌ NOT training giant deep nets

You need ML methods that learn **directly from your simulation**, not from pre-labeled data.

---

# 🎯 What ML Works Best Without Data?

I recommend **three specific ML frameworks**, simplest first.

---

# **Framework 1: REINFORCEMENT LEARNING (Strongly Recommended)**

📦 Library: **PyTorch** + **Gymnasium environment you write yourself**
Beginner-friendly, no dataset required.

### Why it fits:

* You simulate the environment (tree packing) yourself
* The agent learns by trial and error
* RL is GREAT for search/placement problems
* You learn core AI concepts

### What the agent learns:

* Where to place the next tree
* What rotation to use
* How to reduce bounding-box size

### Recommended RL algorithms:

* **PPO (Proximal Policy Optimization)** — beginner friendly
* **REINFORCE** — simplest theoretically
* **DDQN** — if using discrete rotations

🟩 Difficulty: ★★★☆☆
🟩 Educational value: 🌟🌟🌟🌟🌟 (excellent)

---

# **Framework 2: Small Neural Networks as “Heuristics”**

📦 Library: **PyTorch**
No dataset required; model trains during optimization.

### You train a small MLP to predict:

* If a local move is promising
* What rotation angle is promising
* Which heuristic should be chosen next

This is known as:

* **Neural Metaheuristics**
* **Learning to Search**
* **Learned heuristics**

### How you train:

During your SA / VNS runs:

1. Try moves
2. See whether they improve
3. Train the small model online

🟩 Difficulty: ★★☆☆☆
🟩 Educational value: 🌟🌟🌟🌟 (great intro to ML)

---

# **Framework 3: Graph Neural Networks (optional)**

📦 Library: PyTorch Geometric (PyG)

Use GNN to model:

* Tree adjacency
* Contact graph
* Spatial relations

This is more advanced but cool.

🟥 Harder; not recommended for beginners unless you want a challenge.

---

# 🧠 What NOT to use

Avoid:

❌ Image-based CNNs → need dataset
❌ Transformers → too big; need dataset
❌ Diffusion models → completely unrelated
❌ Supervised learning → you have no labels

Stick to RL + online-learning.

---

# 🗺 Recommended Learning Path (Simple & Educational)

## **Stage 0 (1–2 days): Geometry foundations**

* Implement AABB
* Sweep & prune
* Polygon intersection (use Shapely)

## **Stage 1 (3–4 days): Pure algorithms**

* Implement greedy placement
* Implement SA
* Implement hill-climbing
* Test with 10–100 seeds

You’ve built a working solver.

## **Stage 2 (1 week): Add ML (self-contained, no dataset)**

### Option A (easiest): RL placement policy

* Build a Gym environment
* State = partial layout
* Action = place next tree at discrete positions + rotations
* Reward = −bounding_box_size
* Train PPO for small n (3–10 trees)
* Use policy to generate initial layouts for larger n

### Option B: Learned move selector

* Train simple MLP online to guess good SA moves
* Use the network to bias exploration

---

# 🧩 Final Recommendation Summary

## **Mathematical methods (learn these):**

* AABB
* Sweep and prune
* SAT (Separating Axis Theorem)
* Greedy placement
* Simulated Annealing
* VNS
* Penalty-based optimization

## **ML / DL frameworks (you can realistically use with no dataset):**

* **PyTorch** (base library)
* **Gymnasium** (for RL environment)
* **Stable-Baselines3 (PPO/DDQN)**
* *(Optional)* PyTorch Geometric

## **Your best project plan as a novice:**

1. Build pure geometric solver
2. Add SA + local search
3. Add RL initial placer (PPO)
4. Add learned heuristics (small MLP)
5. Combine for final solution

This gives you:

* Deep understanding of geometry
* Real optimization experience
* Intro to RL and ML
* A fun Kaggle journey

---

If you want, I can next give you:

🔥 A complete project plan folder structure
🔥 Starter code (Python) for your packing engine
🔥 A minimal RL environment template for this problem
🔥 A math roadmap for learning the underlying optimization theory

Just tell me:
**“Give me the starter code”** or
**“Give me the math roadmap”** or
**“Help me design the RL environment.”**





Here’s a short, concrete roadmap focused on learning, math clarity, and a no-dataset ML path.

Phase 0 — Setup (1–2 days)
- Stack: Python, NumPy, Numba, PyTorch, Gymnasium, Stable-Baselines3, matplotlib, wandb.
- Goal: Fast geometry + clean experiment logging.

Phase 1 — Geometry & Baselines (3–5 days)
- Implement core collision:
  - AABB bounding boxes + Sweep-and-Prune (broad-phase).
  - SAT (Separating Axis Theorem) for convex polygons; convex-decompose concave trees if needed.
- Constructive baseline:
  - Largest-first + Bottom-left placement.
  - Discrete rotations (e.g., every 10–15 degrees).
- Submission pipeline:
  - Compute bounding square side s = max(Δx, Δy).
  - Output CSV with s-prefixed strings; validate zero overlaps.

Phase 2 — Local Optimization (3–5 days)
- Energy: E = s + λ · OverlapPenalty + μ · BoundaryPenalty (start small λ, increase).
- Coordinate Descent (DCD): update one tree at a time over (x, y, θ), with neighbor checks via grid.
- Simulated Annealing / Variable Neighborhood Search:
  - Random small moves (translate/rotate), accept via ΔE and temperature.
  - Periodic “shake” + recompact.

Phase 3 — RL Initial Placer (1 week, learning-first, no dataset)
- Gym environment:
  - State: features of placed trees + next tree (size, shape stats); optionally coarse occupancy grid.
  - Action: (x, y, θ) bounded; start with discretized θ and coarse grid for (x, y).
  - Reward: −s with strong penalty for overlaps; curriculum from small n to larger.
- Train PPO (SB3):
  - Use RL to propose initial layouts.
  - Pipe results into Phase 2 optimizer for compaction.

Phase 4 — Differentiable Polishing (3–4 days, optional but educational)
- Smooth overlap surrogate via support-function sampling across directions; soft-min/log-sum-exp.
- Optimize (x, y, θ) with Adam/L-BFGS in PyTorch for 100–300 steps starting from RL/heuristic solutions.

Performance & Quality (throughout)
- Speed: Numba-JIT the SAT and broad-phase; cache transforms; spatial hashing for neighbors.
- Robustness: Epsilon tolerances in SAT; clamp to bounds; multiple random seeds; pick best per puzzle.
- Tracking: wandb for scores (s^2/n), runtime, overlap counts, and visualizations.

Priority checklist (minimal to succeed and learn)
- Must-have: AABB + Sweep-and-Prune, SAT, Greedy baseline, SA/DCD local search, CSV submission/validator.
- Nice-to-have: PPO initial placer, differentiable polishing.
- Stretch: CMA-ES or GNN heuristics.

Outcome
- You’ll learn rigorous computational geometry (SAT, convex analysis), practical optimization (DCD, SA), and beginner-friendly RL—without needing any dataset—while producing competitive packings.