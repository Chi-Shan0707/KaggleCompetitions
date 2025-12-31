"""
Simple CEM-based RL packer for the Santa 2025 challenge.

Creates placements using the exact polygon in Christmastree.py and uses
shapely for all geometry checks. Outputs submission CSV with id,x,y,deg
values prefixed by 's' as required.

This is a lightweight, simple RL-like (CEM) solver meant as a starting point.
"""
from decimal import Decimal
import math
import random
import csv
import time

from shapely.geometry import box
from shapely.ops import unary_union

from Christmastree import ChristmasTree, scale_factor


def tree_polygon(x, y, angle_deg):
    # ChristmasTree expects Decimal-like strings for coordinates and angle
    return ChristmasTree(center_x=str(Decimal(x)), center_y=str(Decimal(y)), angle=str(Decimal(angle_deg))).polygon


TREE_SIZE_CACHE = None
def single_tree_size():
    global TREE_SIZE_CACHE
    if TREE_SIZE_CACHE is not None:
        return TREE_SIZE_CACHE  # 直接返回缓存结果
    # 仅第一次调用时计算
    t = ChristmasTree(center_x='0', center_y='0', angle='0')
    minx, miny, maxx, maxy = t.polygon.bounds
    w = (maxx - minx) / float(scale_factor)
    h = (maxy - miny) / float(scale_factor)
    TREE_SIZE_CACHE = (max(w, h), w, h)  # 缓存结果
    return TREE_SIZE_CACHE

def fits_in_box(poly, side):
    # Create a shapely box in the scaled coordinate system
    S = float(side) * float(scale_factor)
    b = box(0.0, 0.0, S, S)
    return poly.within(b)


def no_overlap(poly, placed_polys):
    # Ensure polygon does not intersect any previously placed polygon with positive area
    for p in placed_polys:
        if poly.intersects(p):
            # check real area of intersection
            if poly.intersection(p).area > 1e-6:
                return False
    return True


def attempt_pack_cem(n, side, iterations=60, population=200, elite_frac=0.15, max_attempts_per_tree=200):
    """
    Attempt to place n trees inside a square of given side.

    This function samples sequential placements for each candidate in the population.
    For each tree placement: with 75% probability we sample a point on the perimeter
    (boundary) of the union of already-placed trees (converted to original units),
    otherwise we sample uniformly inside the square. Angles are uniformly random.

    Returns (success_bool, (centers_list, angles_list, placed_polys_list)) where
    centers_list is [(x,y),...], angles_list is [deg,...], and placed_polys_list is
    kept for internal checking (but NOT stored in final results).
    """

    side = float(side)

    best_solution = None
    best_placed = -1

    for it in range(iterations):
        for s_idx in range(population):
            placed_polys = []  # shapely polygons in scaled coords for collision checks
            centers = []
            angles = []
            placed = 0
            for i in range(n):
                placed_ok = False
                for attempt in range(max_attempts_per_tree):
                    # decide perimeter (75%) or random (25%)
                    if placed_polys and random.random() < 0.75:
                        # sample a point on the boundary of the union of placed polygons
                        union = unary_union(placed_polys)
                        boundary = union.boundary
                        if boundary.length <= 0:
                            # fallback to random
                            x = random.random() * side
                            y = random.random() * side
                        else:
                            # pick a random distance along the boundary and interpolate
                            d = random.random() * boundary.length
                            pt = boundary.interpolate(d)
                            # boundary coordinates are in scaled units
                            x0 = pt.x / float(scale_factor)
                            y0 = pt.y / float(scale_factor)
                            # try small outward jitter to move slightly outside existing union
                            jitter = single_tree_size()[0] * 0.1
                            theta = random.random() * 2.0 * math.pi
                            dx = math.cos(theta) * jitter * random.random()
                            dy = math.sin(theta) * jitter * random.random()
                            x = x0 + dx
                            y = y0 + dy
                    else:
                        # uniform random inside box
                        x = random.random() * side
                        y = random.random() * side

                    a = random.random() * 360.0
                    poly = tree_polygon(x, y, a)
                    if not fits_in_box(poly, side):
                        continue
                    if not no_overlap(poly, placed_polys):
                        continue
                    # placement OK
                    placed_polys.append(poly)
                    centers.append((x, y))
                    angles.append(a)
                    placed += 1
                    placed_ok = True
                    break
                if not placed_ok:
                    break

            # record candidate
            if placed > best_placed:
                best_placed = placed
                best_solution = (centers.copy(), angles.copy(), placed_polys.copy())
            if placed == n:
                return True, (centers, angles, placed_polys)

    return (best_placed == n), (best_solution if best_solution is not None else ([], [], []))


def find_min_side_for_n(n, timeout=210):
    # Binary search for minimal side that fits n trees using the CEM packer
    # Start bounds
    if n <= 10:
        iterations = 5       # 原60 → 5
        population = 10      # 原400 → 10
        max_attempts = 10    # 原200 → 10
    else:
        iterations = 60
        population = 400
        max_attempts = 200
    max_dim, w, h = single_tree_size()
    lo = max_dim * 0.9
    hi = max_dim * max(1.5, math.sqrt(n) * 1.2)
    start_time = time.time()
    best_found = None
    while hi - lo > 1e-2 and time.time() - start_time < timeout:
        mid = (lo + hi) / 2.0
        success, result = attempt_pack_cem(n, mid, iterations, population, max_attempts)
        if success:
            hi = mid
            # result = (centers, angles, placed_polys)
            best_found = (mid, result)
        else:
            lo = mid
    return best_found


def generate_submission(max_n=200, outpath='submission.csv'):
    # Generates placements for n in 1..max_n and writes CSV
    results = {}
    for n in range(1, max_n + 1):
        print(f"Packing n={n}")
        found = find_min_side_for_n(n, timeout=210)
        if found is None:
            print(f"  Failed to find packing for n={n}, will try generous placement")
            # fallback: place greedily into a square with side = n * single tree size
            side = single_tree_size()[0] * n
            success, result = attempt_pack_cem(n, side, iterations=80, population=600)
            if not success:
                # as last resort place trees on grid without overlap
                centers = []
                angles = []
                base = single_tree_size()[0]
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
    # default run for first 200 configurations
    generate_submission(max_n=200, outpath='submission.csv')
