"""
Scipy_packer.py (Optimized Version)
Dependencies: scipy, numpy, shapely
Key optimizations: reduced DE params, simplified objective, faster binary search, cached geometry checks
"""
from decimal import Decimal
import math
import random
import csv
import time
import functools

# Dependencies declaration
import numpy as np
from scipy.optimize import differential_evolution



# Reuse functions from rl_packer.py
from RL_packer import tree_polygon, single_tree_size, fits_in_box, no_overlap

# Cache single tree size (global)
_SINGLE_TREE_CACHE = single_tree_size()
# Cache for polygon checks (reduce repeated tree_polygon calls)
@functools.lru_cache(maxsize=10000)
def _cached_tree_check(x, y, a, side):
    """Cache polygon checks to avoid redundant computations"""
    a = a % 360.0
    poly = tree_polygon(x, y, a)
    # Early filter: coordinate out of bounds (no need to check polygon)
    if x < 0 or x > side or y < 0 or y > side:
        return False, poly
    # Check if fits in box (simplified)
    fits = fits_in_box(poly, side)
    return fits, poly


def _objective(vars_flat, n, side):
    """
    Optimized objective: minimal penalty (only count placed trees, no redundant distance calc)
    Big penalty only for unplaced trees (fast, no extra computation)
    """
    big_penalty = 1e3
    placed = 0
    placed_polys = []

    for i in range(n):
        x = float(vars_flat[3 * i + 0])
        y = float(vars_flat[3 * i + 1])
        a = float(vars_flat[3 * i + 2])

        # Early exit for obvious out-of-bounds (skip polygon generation)
        if x < 0 or x > side or y < 0 or y > side:
            continue

        # Use cached polygon check
        fits, poly = _cached_tree_check(x, y, a, side)
        if not fits:
            continue

        # Simplified overlap check (relax precision to 1e-6)
        overlap = False
        for p in placed_polys:
            if poly.intersects(p) and poly.intersection(p).area > 1e-6:
                overlap = True
                break
        if overlap:
            continue

        placed_polys.append(poly)
        placed += 1

    # Only penalty for unplaced trees (no small penalty, faster)
    return big_penalty * (n - placed)


def optimize_tree_placement(n, side, popsize=5, maxiter=5, prev_result=None):
    """
    Optimized DE parameters: minimal popsize/maxiter for speed
    """
    # Bounds: x,y in [0,side], angle in [0,360]
    bounds = []
    for i in range(n):
        bounds.append((0.0, float(side)))
        bounds.append((0.0, float(side)))
        bounds.append((0.0, 360.0))

    try:
        result = differential_evolution(
            lambda v: _objective(v, n, side),
            bounds,
            strategy='best1bin',  # Fastest strategy
            maxiter=maxiter,
            popsize=popsize,
            polish=False,  # Disable polish (big speedup, no need for layout)
            tol=1e-2,      # Relax tolerance (speed up convergence)
            disp=False,
            updating='deferred',
            workers=1,     # Single thread (faster for small n)
            seed=42        # Fixed seed for reproducibility
        )
    except Exception:
        return False, ([], [], [])

    # Reconstruct valid placements
    best = result.x
    centers = []
    angles = []
    placed_polys = []
    for i in range(n):
        x = float(best[3 * i + 0])
        y = float(best[3 * i + 1])
        a = float(best[3 * i + 2]) % 360.0

        # Early filter
        if x < 0 or x > side or y < 0 or y > side:
            break

        fits, poly = _cached_tree_check(x, y, a, side)
        if not fits or not no_overlap(poly, placed_polys):
            break

        centers.append((x, y))
        angles.append(a)
        placed_polys.append(poly)

    success = (len(centers) == n)
    return success, (centers, angles, placed_polys)


def find_min_side_for_n(n, timeout=30, result_dict=None):  # Reduced timeout from 210→10 for small n
    """
    Optimized binary search with reuse/fallback:
    - If `result_dict` provided, try reusing result_dict[k] for k = n-1..0
      as a starting state and attempt to place the remaining trees.
    - Falls back progressively to smaller k until success or exhaustion.

    带复用/回退的二分搜索：如果提供了 `result_dict`，则依次尝试使用
    result_dict[n-1], result_dict[n-2], ... , result_dict[0] 作为起始状态。
    """
    if result_dict is None:
        result_dict = {}

    max_dim, w, h = _SINGLE_TREE_CACHE
    # tighten bounds using previous result for n-1 when available to reduce search
    prev_side = result_dict.get(n-1, (0.0, [], [], []))[0] if n-1 in result_dict else 0.0
    lo = max(max_dim * math.sqrt(n) * 0.8, prev_side * 0.9 if prev_side > 0 else 0.0)
    hi = max(max_dim * max(1.5, math.sqrt(n) * 1.2), prev_side * 1.5 if prev_side > 0 else 0.0)
    start_time = time.time()
    best_found = None

    # Adaptive precision: relax for small n (1e-1 vs 1e-2)
    precision = 1e-1 if n <= 20 else 1e-2
    while hi - lo > precision and time.time() - start_time < timeout:
        mid = (lo + hi) / 2.0

        # pick small DE params based on n (kept small for speed)
       
        popsize, maxiter = 15  ,  15

        # Try fallbacks: reuse states from k=n-1 down to 0 (empty)
        success_any = False
        found_res = None
        for k in range(n-1, -1, -1):
            prev_entry = result_dict.get(k)
            prev_result = None
            if prev_entry is not None:
                # prev_entry: (side_k, centers_k, angles_k, placed_polys_k)
                _, c_k, a_k, p_k = prev_entry
                prev_result = (c_k, a_k, p_k)

            success, res = optimize_tree_placement(n, mid, popsize=popsize, maxiter=maxiter, prev_result=prev_result)
            if success:
                success_any = True
                found_res = res
                break

        if success_any:
            hi = mid
            best_found = (mid, found_res)
            if n <= 20:
                break
        else:
            lo = mid
    return best_found


def generate_submission(max_n=200, outpath='submission.csv'):
    # Maintain a global result dict similar to RL_packer for reuse and incremental packing
    # result[k] = (side_k, centers_k, angles_k, placed_polys_k)
    # 全局 result 字典用于复用和增量放置
    result = {}
    result[0] = (0.0, [], [], [])

    for n in range(1, max_n + 1):
        print(f"Packing n={n}")

        # use find_min_side_for_n with result dict to attempt reuse from result[n-1], n-2, ...
        found = find_min_side_for_n(n, result_dict=result)
        if found is None:
            print(f"  Failed to find packing for n={n}, will use grid fallback")
            base = _SINGLE_TREE_CACHE[0]
            per_row = math.ceil(math.sqrt(n))
            side = per_row * base * 1.5
            centers = []
            angles = []
            for i in range(n):
                rx = (i % per_row) * base * 1.5 + base / 2.0
                ry = (i // per_row) * base * 1.5 + base / 2.0
                centers.append((rx, ry))
                angles.append(0.0)
            placed_polys = []
            result[n] = (side, centers, angles, placed_polys)
        else:
            side, res = found
            centers, angles, placed_polys = res
            result[n] = (side, centers, angles, placed_polys)

    # Write CSV directly from global result dict
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'deg'])
        for n in range(1, max_n + 1):
            side, centers, angles, placed_polys = result.get(n, (0.0, [], [], []))
            for idx, (x, y) in enumerate(centers):
                deg = angles[idx] if idx < len(angles) else 0.0
                id_str = f"{n:03d}_{idx}"
                writer.writerow([id_str, f's{float(x-80):.6f}', f's{float(y-80):.6f}', f's{float(deg):.6f}'])


if __name__ == '__main__':
    generate_submission(max_n=200, outpath='Scipy_submission.csv')