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
    # Create a shapely box in the scaled coordinate system
    S = float(side) * float(scale_factor)
    b = box(0.0, 0.0, S, S)
    return poly.within(b)


def no_overlap(poly, placed_polys, tree=None):
    """Ensure polygon does not intersect any previously placed polygon with positive area.
    If an STRtree `tree` is provided, use it to query candidates for speed.
    Returns True when there is NO overlap (same semantics as the old `no_overlap`).
    """
    if not placed_polys:
        return True

    if tree is not None:
        try:
            candidates = tree.query(poly)
        except Exception:
            candidates = range(len(placed_polys))

        for cand in candidates:
            p = placed_polys[cand]
            if p is poly:
                continue
            if poly.intersects(p) and poly.intersection(p).area > 1e-6:
                return False
        return True

    # fallback brute-force
    for p in placed_polys:
        if poly.intersects(p) and poly.intersection(p).area > 1e-6:
            return False
    return True


def attempt_pack_cem(n, side, iterations, population, elite_frac=0.10, max_attempts_per_tree=10, prev_result=None):
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

    if prev_result is not None:
        try:
            prev_centers, prev_angles, prev_placed_polys = prev_result
        except Exception:
            prev_centers = prev_angles = prev_placed_polys = None
    else:
        prev_centers = prev_angles = prev_placed_polys = None



    # --- CEM parameter: we only adapt the outward jitter multiplier ---
    # Keep the original placement logic unchanged; CEM searches over the
    # jitter multiplier used when sampling just-off-the-boundary points.
    # 我们只对外部扰动（jitter）倍数做CEM搜索。
    base_jitter = single_tree_size[0] * 0.1
    mu = 0.1  
    sigma = 0.05 
    n_elite = max(1, int(population * elite_frac))

    if prev_result != None :
        if side < result[len(list ( prev_centers ))][0] * 0.95:
            return False, ([], [], [])
        elif side < result[len(list ( prev_centers ))][0] :
            for p in prev_placed_polys:
                    if not fits_in_box(p, side):
                        return False, ([], [], [])
    for it in range(iterations):
        population_results = [] 
        for s_idx in range(population):
            jitter_mult = random.gauss(mu, sigma)
            if jitter_mult < 0.0:
                jitter_mult = abs(jitter_mult)
            centers = list(prev_centers)
            angles = list(prev_angles)
            placed_polys = list(prev_placed_polys)
            tree = STRtree(placed_polys) if placed_polys else None
            placed = len(centers)
            rest = n - placed
            max_x = 0
            max_y = 0
            for i in range(rest):
                placed_ok = False
                for attempt in range(max_attempts_per_tree):
                    if placed_polys and random.random() < 0.98:
                        union = unary_union(placed_polys)
                        boundary = union.boundary
                        if boundary.length <= 0:
                            x = random.random() * side
                            y = random.random() * side
                        else:
                            d = random.random() * boundary.length
                            pt = boundary.interpolate(d)
                            x0 = pt.x / float(scale_factor)
                            y0 = pt.y / float(scale_factor)
                            jitter = single_tree_size[0] * jitter_mult
                            theta = random.random() * 2.0 * math.pi
                            dx = math.cos(theta) * jitter * random.random()
                            dy = math.sin(theta) * jitter * random.random()
                            x = x0 + dx
                            y = y0 + dy
                    else:
                        x = random.random() * side
                        y = random.random() * side
                    a = random.random() * 360.0
                    poly = tree_polygon(x, y, a)
                    if not no_overlap(poly, placed_polys, tree=tree):
                        continue
                    
                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    if fits_in_box(poly, side):
                        placed_polys.append(poly)
                        tree = STRtree(placed_polys)
                        centers.append((x, y))
                        angles.append(a)
                        placed += 1
                        placed_ok = True
                        break
                if not placed_ok:
                    break
            #
            #
            if placed == n:
                return True, (centers, angles, placed_polys)
            #
            if max_x * max_y ==0 :
                reward = -100
            else :
                reward = placed - 0.5 * max_x*max_y
            population_results.append((reward,  jitter_mult))
        population_results.sort(key=lambda x: x[0], reverse=True)
        elites = population_results[:n_elite]
        elite_jitters = [e[1] for e in elites]
        try:
            new_mu = statistics.mean(elite_jitters)
            new_sigma = statistics.pstdev(elite_jitters)
        except Exception:
            new_mu = mu
            new_sigma = sigma
        mu = 0.75 * mu + 0.25 * new_mu
        sigma = max(1e-3, 0.8 * sigma + 0.2 * new_sigma)

    return False,([], [], [])


def find_min_side_for_n(n, timeout=15, result_dict=None):
    # Binary search for minimal side that fits n trees using the CEM packer.
    # This version will attempt fallback starts based on entries in result_dict
    # (i.e., result_dict[k] holds a previously successful placement for k trees).
    # 在二分搜索内，基于 result_dict 中已有的放置结果做逐级回退尝试。
    if result_dict is None:
        result_dict = {}

    # Start bounds and hyperparams (kept as original heuristics)
    iterations = 20
    population = 20
    max_attempts = 20

    max_dim, w, h = single_tree_size
    if n == 1:
        lo  = 0.8
        hi  = 0.88
    else :
        lo = max ( max_dim * math.sqrt(n) * 0.7 , result_dict.get(n-1, (0.0, [], [], []))[0] * (1.0 - 0.6/n )  )
        hi = max ( max_dim * math.sqrt(n) * 0.9 , result_dict.get(n-1, (0.0, [], [], []))[0] * (1.0 + 1.0/n )  )
    
    best_found = None
    if n <= 10:
        lmt = -1
        eps = 1e-8
        timeout = 40
    elif n <= 20 :
        lmt = n - 8
        eps = 1e-7
        timeout = 35
    elif n <= 50:
        lmt = n - 6
        eps = 1e-6
        timeout = 30
    elif n <= 100:
        lmt = n - 4
        eps = 1e-5
        timeout = 25
    else :
        lmt = n - 2
        eps = 1e-4
        timeout = 20
    # For each candidate mid side during binary search, try fallbacks from
    # using result[n-1], result[n-2], ..., result[0] (empty start).
    # 对于每个二分候选 mid，依次尝试基于 result[n-1], result[n-2], ..., result[0] 的回退。
    DDL = time.time() + timeout
    while hi - lo > eps and time.time()  < DDL:
        mid = (lo + hi) / 2.0
        success_any = False
        found_result = None

        # try reusing previously computed placements: prefer larger k (closer to n)
       
        for k in range(n - 1, lmt, -1):
            prev_entry = result_dict.get(k)
            # prev_entry format: (side_k, centers_k, angles_k, placed_polys_k)
            prev_result = None
            if prev_entry is not None:
                # pass only (centers, angles, placed_polys) to attempt_pack_cem
                _, c_k, a_k, p_k = prev_entry
                prev_result = (c_k, a_k, p_k)

            # call attempt_pack_cem with this prev_result; it will reuse the
            # k-tree state and attempt to place the remaining trees.
            # 将 prev_result 传入 attempt_pack_cem，函数会基于 k 棵树的状态尝试放置剩余树。
            success, result = attempt_pack_cem(n, mid, iterations=iterations, population=population, max_attempts_per_tree=max_attempts, prev_result=prev_result)
            if success:
                success_any = True
                found_result = result
                break

        if success_any:
            hi = mid
            best_found = (mid, found_result)
            print("log : n=", n, " side=", mid, f" time={time.time() - (DDL-timeout):.1f}s")
        else:
            lo = mid

    return best_found


def generate_submission(max_n=200, outpath='submission.csv'):
    # Generates placements for n in 1..max_n and writes CSV
    # global result dict: result[k] = (side_k, centers_k, angles_k, placed_polys_k)
    
    result[0] = (0.0, [], [], [])

    for n in range(1, max_n + 1):
        print(f"Packing n={n}")

        # attempt to find minimal side while trying fallbacks based on
        # previously stored placements in `result` (result[n-1], result[n-2],...)
        # 在二分搜索中，基于已存的 result[n-1], result[n-2], ... 做回退尝试
        found = find_min_side_for_n(n, result_dict=result)
        if found is None:
            found = find_min_side_for_n(n, result_dict=result)
            if found is None:
                break
        side, res = found
        centers, angles, placed_polys = res
        result[n] = (side, centers, angles, placed_polys)

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
    generate_submission(max_n=200, outpath='CSE_submission#008.csv')