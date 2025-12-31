"""
Scipy_packer.py
Dependencies: scipy, numpy, shapely

Uses `scipy.optimize.differential_evolution` to optimize tree placements.
Reuses `tree_polygon`, `single_tree_size`, `fits_in_box`, and `no_overlap`
from `rl_packer.py` as required.

This script preserves logging, CSV format, and execution flow consistent with
`rl_packer.py` while replacing the sampling/optimizer with SciPy's DE.
"""
from decimal import Decimal
import math
import random
import csv
import time

# Dependencies declaration (for readers): scipy, numpy, shapely
import numpy as np
from scipy.optimize import differential_evolution

import shapely.speedups
shapely.speedups.enable()

# Reuse functions from rl_packer.py per requirements
from RL_packer import tree_polygon, single_tree_size, fits_in_box, no_overlap

# Cache single tree size to avoid repeated instantiation
_SINGLE_TREE_CACHE = single_tree_size()


def _objective(vars_flat, n, side):
    """
    Objective for differential evolution.
    We try to maximize the number of successfully placed trees by minimizing a penalty:
    penalty = big_penalty * (n - placed_count) + small out-of-box distance penalty.

    vars_flat: flat array of length 3*n => [x1,y1,a1, x2,y2,a2, ...]
    """
    big_penalty = 1e3
    placed_polys = []
    placed = 0
    small_penalty = 0.0
    for i in range(n):
        x = float(vars_flat[3 * i + 0])
        y = float(vars_flat[3 * i + 1])
        a = float(vars_flat[3 * i + 2]) % 360.0
        poly = tree_polygon(x, y, a)
        if not fits_in_box(poly, side):
            # compute rough distance outside box as small penalty
            minx, miny, maxx, maxy = poly.bounds
            # convert to original units
            minx /= float(Decimal('1e15'))
            miny /= float(Decimal('1e15'))
            maxx /= float(Decimal('1e15'))
            maxy /= float(Decimal('1e15'))
            dist = 0.0
            if minx < 0.0:
                dist += abs(minx)
            if miny < 0.0:
                dist += abs(miny)
            if maxx > side:
                dist += (maxx - side)
            if maxy > side:
                dist += (maxy - side)
            small_penalty += dist
            continue
        # check overlap with previous placed polygons
        bad = False
        for p in placed_polys:
            if poly.intersects(p) and poly.intersection(p).area > 1e-12:
                bad = True
                break
        if bad:
            # heavy penalty for overlaps
            small_penalty += 0.0
            continue
        placed_polys.append(poly)
        placed += 1

    penalty = big_penalty * (n - placed) + small_penalty
    return penalty


def optimize_tree_placement(n, side, popsize=15, maxiter=30):
    """
    Use scipy's differential_evolution to search for placements for `n` trees
    inside a square of size `side`.

    Returns (success_bool, (centers, angles, placed_polys)).

    Chinese/English notes:
    - 对于每个候选解，DE 优化器会尝试将 3*n 个变量（x,y,angle）最小化目标函数。
    - 目标是让所有树都无重叠且都在方盒内；若成功，返回 centers/angles 列表。
    """
    # bounds: x,y in [0,side], angle in [0,360]
    bounds = []
    for i in range(n):
        bounds.append((0.0, float(side)))
        bounds.append((0.0, float(side)))
        bounds.append((0.0, 360.0))

    # differential evolution
    try:
        result = differential_evolution(
            lambda v: _objective(v, n, side),
            bounds,
            strategy='best1bin',
            maxiter=maxiter,
            popsize=popsize,
            polish=True,
            tol=1e-3,
            disp=False,
            updating='deferred'
        )
    except Exception:
        return False, ([], [], [])

    best = result.x
    # reconstruct placements and check feasibility
    centers = []
    angles = []
    placed_polys = []
    for i in range(n):
        x = float(best[3 * i + 0])
        y = float(best[3 * i + 1])
        a = float(best[3 * i + 2]) % 360.0
        poly = tree_polygon(x, y, a)
        if not fits_in_box(poly, side):
            break
        if not no_overlap(poly, placed_polys):
            break
        centers.append((x, y))
        angles.append(a)
        placed_polys.append(poly)

    success = (len(centers) == n)
    return success, (centers, angles, placed_polys)


def find_min_side_for_n(n, timeout=210):
    # Binary search for minimal side that fits n trees using the SciPy optimizer
    max_dim, w, h = _SINGLE_TREE_CACHE
    lo = max_dim * 0.9
    hi = max_dim * max(1.5, math.sqrt(n) * 1.2)
    start_time = time.time()
    best_found = None
    while hi - lo > 1e-2 and time.time() - start_time < timeout:
        mid = (lo + hi) / 2.0
        # adapt DE parameters for small n
        if n <= 10:
            popsize = 10
            maxiter = 20
        else:
            popsize = 20
            maxiter = 50
        success, result = optimize_tree_placement(n, mid, popsize=popsize, maxiter=maxiter)
        if success:
            hi = mid
            best_found = (mid, result)
        else:
            lo = mid
    return best_found


def generate_submission(max_n=200, outpath='submission.csv'):
    # Mirrors RL_packer.generate_submission behavior and I/O
    results = {}
    for n in range(1, max_n + 1):
        print(f"Packing n={n}")
        found = find_min_side_for_n(n, timeout=30)
        if found is None:
            print(f"  Failed to find packing for n={n}, will try generous placement")
            # fallback: increase side and re-optimize (same logic as RL_packer)
            base = _SINGLE_TREE_CACHE[0]
            side = base * n
            # adapt parameters
            if n <= 10:
                popsize = 10
                maxiter = 20
            else:
                popsize = 20
                maxiter = 50
            success, result = optimize_tree_placement(n, side, popsize=popsize, maxiter=maxiter)
            if not success:
                # as last resort: grid placement
                centers = []
                angles = []
                per_row = math.ceil(math.sqrt(n))
                side = per_row * base * 1.5
                for i in range(n):
                    rx = (i % per_row) * base * 1.5 + base / 2.0
                    ry = (i // per_row) * base * 1.5 + base / 2.0
                    centers.append((rx, ry))
                    angles.append(0.0)
            else:
                centers, angles, _ = result
        else:
            side, result = found
            centers, angles, _ = result
        results[n] = (side, centers, angles)

    # write CSV
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'deg'])
        for n in range(1, max_n + 1):
            side, centers, angles = results[n]
            for idx, (x, y) in enumerate(centers):
                deg = angles[idx] if idx < len(angles) else 0.0
                id_str = f"{n:03d}_{idx}"
                writer.writerow([id_str, f's{float(x):.6f}', f's{float(y):.6f}', f's{float(deg):.6f}'])


if __name__ == '__main__':
    generate_submission(max_n=200, outpath='submission.csv')
