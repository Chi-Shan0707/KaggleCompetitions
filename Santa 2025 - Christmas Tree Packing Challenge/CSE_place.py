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

from matplotlib.patches import Polygon
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree
from Christmastree import ChristmasTree, scale_factor
import statistics

single_tree_size = [1.0, 0.7, 1.0]
# max_dim , width , height
# 初始化全局 result 字典，result[0] 表示空状态
result = {}

def tree_polygon(x, y, angle_deg):
    # ChristmasTree expects Decimal-like strings for coordinates and angle
    return ChristmasTree(center_x=str(Decimal(x)), center_y=str(Decimal(y)), angle=str(Decimal(angle_deg))).polygon


def fits_in_box(poly, side):
    # Faster bounds-based check instead of constructing a box and calling within()
    S = float(side) * float(scale_factor)
    minx, miny, maxx, maxy = poly.bounds
    return (minx >= 0.0) and (miny >= 0.0) and (maxx <= S) and (maxy <= S)


def all_within_box(placed_polys, side, tree=None):
    """Return True if all polygons in `placed_polys` are fully inside the box [0,S]x[0,S].

    This attempts to use `STRtree` for a fast early-reject: query the box to fetch
    candidates whose envelopes intersect the box, then verify containment. The
    function accepts both `list` of geometry objects and `STRtree.query` results
    that may be indices or geometry objects depending on Shapely version.
    """
    if not placed_polys:
        return True
    
    else:
        S = float(side) * float(scale_factor)

        box_geom = box(0.0, 0.0, S, S)
        try:
            candidates = tree.query(box_geom)
            for idx in candidates:
                p = placed_polys[int(idx)]
                minx, miny, maxx, maxy = p.bounds
                if (minx < 0.0) or (miny < 0.0) or (maxx > S) or (maxy > S):
                    return False
            return True
        except:
            for p in placed_polys:
                minx, miny, maxx, maxy = p.bounds
                if (minx < 0.0) or (miny < 0.0) or (maxx > S) or (maxy > S):
                    return False
    return True


def no_overlap(poly, placed_polys, tree=None):

    if not placed_polys:
        return True
    else:
        try:
            candidates = tree.query(poly)
            for cand in candidates:
                p = placed_polys[cand]
                if p is poly:
                    continue
                if poly.intersects(p) and poly.intersection(p).area > 1e-6:
                    return False
            return True
        except:
            for p in placed_polys:
                if poly.intersects(p) and poly.intersection(p).area > 1e-6:
                    return False
    return True


def choose_pref_front(n_candidates, alpha=0.49):
    """Choose an index in [0, n_candidates-1] biased toward the front.

    Probability proportional to alpha**i (i=0..n-1). alpha in (0,1).
    Smaller alpha => stronger bias toward index 0. Default alpha=0.7 is a moderate bias.
    """
    if n_candidates <= 1:
        return 0
    # compute decay weights
    weights = [alpha ** i for i in range(n_candidates)]
    total = sum(weights)
    r = random.random() * total
    cum = 0.0
    for i, w in enumerate(weights):
        cum += w
        if r <= cum:
            return i
    return n_candidates - 1

from shapely.ops import unary_union


def calculate_min_side(placed_polys, base=None, n=None, safety_pad=0.0):
    """Compute side as the maximal span (width/height) of the union of polygons.

    This matches the definition used in `checker.py` (side = max(width, height)).
    Returns a floating-point side in the same units as centers (not scaled).
    Optionally adds a small `safety_pad` (in same units) to ensure containment margins.
    """
    if not placed_polys:
        return 0.0

    bounds = unary_union(placed_polys).bounds
    minx, miny, maxx, maxy = bounds
    width = (maxx - minx) / float(scale_factor)
    height = (maxy - miny) / float(scale_factor)
    side = max(width, height)
    # Add optional safety pad
    return side + float(safety_pad)
def place_one_from_prev(n,prev_result,iterations=40, population=40,deadline=None):
    if n == 1:
        side = 1
        centers = []
        angles = []
        placed_polys = []
    else:
        side = prev_result[0]
        centers, angles, placed_polys = list(prev_result[1]), list(prev_result[2]), list(prev_result[3])

    tree = STRtree(placed_polys) if placed_polys else None
    time_over = False
    mu_jitter = 0.1  
    sigma_jitter = 0.05 
    mu_boundary = 0.6
    sigma_boundary = 0.08

    best_place=(0.0,[],[],[])

    for it_id in range(iterations):
        population_results = []
        if time_over == True:
            break
        for population_round in range(population):
            if time_over == True:
                break
            jitter_mult = random.gauss(mu_jitter, sigma_jitter)
            if jitter_mult < 0.0:
                jitter_mult = abs(jitter_mult)
            angle_center = random.random() * 360.0

            placed_polys_local = list(placed_polys)
            centers_local = list(centers)
            angles_local = list(angles)
            tree_local = STRtree(placed_polys_local) if placed_polys_local else None
            br=0

            for attempt in range(55):
                if deadline is not None and time.time() > deadline:
                    time_over = True
                    break
                if placed_polys_local and random.random() < 0.95:
                    union = unary_union(placed_polys_local)
                    boundary = union.boundary
                    if not hasattr(boundary, 'length') or boundary.length <= 0:
                        x = random.random() * side
                        y = random.random() * side
                    else:
                        br = max(0.0, min(1.0, random.gauss(mu_boundary, sigma_boundary)))
                        d = br * boundary.length
                        pt = boundary.interpolate(d)
                        x0 = pt.x / float(scale_factor)
                        y0 = pt.y / float(scale_factor)
                        jitter = single_tree_size[0] * jitter_mult
                        theta = random.random() * 2.0 * math.pi
                        dx = math.cos(theta) * jitter
                        dy = math.sin(theta) * jitter
                        x = x0 + dx
                        y = y0 + dy
                else:
                    x = random.random() * side
                    y = random.random() * side

                a = random.gauss(angle_center, 90.0) % 360.0
                poly = tree_polygon(x, y, a)
                if no_overlap(poly, placed_polys_local, tree=tree_local):
                    placed_polys_local.append(poly)
                    centers_local.append((x, y))
                    angles_local.append(a)
                    # use the union-bounds-based side calculation (matches checker)
                    min_side = calculate_min_side(placed_polys_local)
                    population_results.append((min_side, jitter_mult, br, (centers_local, angles_local, placed_polys_local)))
                    break
                a = random.gauss(angle_center, 90.0) % 360.0
                poly = tree_polygon(x, y, a)
                if no_overlap(poly, placed_polys_local, tree=tree_local):
                    placed_polys_local.append(poly)
                    centers_local.append((x, y))
                    angles_local.append(a)
                    # use the union-bounds-based side calculation (matches checker)
                    min_side = calculate_min_side(placed_polys_local)
                    population_results.append((min_side, jitter_mult, br, (centers_local, angles_local, placed_polys_local)))
                    break
                
            ##end of attempts
        if not population_results:
            # nothing found this iteration
            continue

        # sort ascending by side (smaller is better)
        population_results.sort(key=lambda x: x[0])
        k = max(len(population_results) // 4, 10)
        k = min(k, len(population_results))
        elites = population_results[:k]
        elite_jitters = [e[1] for e in elites]

        
        try:
            mu_jitter = statistics.mean(elite_jitters) * 0.8 + mu_jitter * 0.2
        except Exception:
            mu_jitter = mu_jitter
        try:
            sigma_jitter = statistics.stdev(elite_jitters) * 0.7 + sigma_jitter * 0.3
        except Exception:
            sigma_jitter = max(sigma_jitter, 1e-3)

        elite_boundaries = [e[2] for e in elites]
        try:
            mu_boundary = statistics.mean(elite_boundaries) * 0.8 + mu_boundary * 0.2
        except Exception:
            mu_boundary = mu_boundary

        try:
            sigma_boundary = statistics.stdev(elite_boundaries) * 0.6 + sigma_boundary * 0.4
        except Exception:
            sigma_boundary = max(sigma_boundary, 1e-3)

        if best_place[0] ==0.0 or best_place[0] > population_results[0][0]:
            best_place = population_results[0]


    if best_place[0] > 0.0:
        return True, best_place
    return False, (None,[], [], [])

def generate_submission(max_n=200, outpath='#CSE_Place#001.csv', top_k=10):
    # Incremental top-k placement pipeline
    
    candidates_num=0
    candidates = [(-1, [], [], [])]
    for n in range(1, max_n + 1):
        print(f"Packing n={n}")
        feasible_list = []
        max_dim = single_tree_size[0]
        DDL = time.time() + 200  # 5 minutes per n
        for _ in range(123):
            if n == 1:
                prev_id = 0
            else:
                prev_id = choose_pref_front(candidates_num, alpha=0.40)
            place_res = place_one_from_prev(n, candidates[prev_id], iterations=80, population=60,deadline=DDL)
            if place_res[0] == True:
                feasible_side,_j,_b,(feasible_centers,feasible_angles,feasible_polys)=place_res[1]
                feasible_list.append((feasible_side,feasible_centers, feasible_angles, feasible_polys))
                print(f"  Found feasible placement option  with side={place_res[1][0]:.6f}")

        if not feasible_list:
            print(f"  No feasible expansions at n={n}; stopping")
            break

        # keep top_k best by side
        feasible_list.sort(key=lambda x: x[0])
        candidates = feasible_list[:top_k]
        candidates_num=len(candidates)
        # store best candidate in result for compatibility
        side_best, centers_best, angles_best, polys_best= candidates[0]
        result[n] = (side_best, centers_best, angles_best, polys_best)

    # write CSV directly from the global result dict
    with open(outpath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'deg'])
        for n in range(1, max_n + 1):
            side, centers, angles, placed_polys = result.get(n, (0.0, [], [], []))
            for idx, (x, y) in enumerate(centers):
                deg = angles[idx] if idx < len(angles) else 0.0
                id_str = f"{n:03d}_{idx}"
                writer.writerow([id_str, f's{float(x-90):.6f}', f's{float(y-90):.6f}', f's{float(deg):.6f}'])


if __name__ == '__main__':
    # default run for first 200 configurations
    generate_submission(max_n=200, outpath='CSE_Place_submission#002.csv')