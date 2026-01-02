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

from Christmastree import ChristmasTree, scale_factor
import statistics

single_tree_size = [1.0, 0.7, 1.0]
# max_dim , width , height
# 初始化全局 result 字典，result[0] 表示空状态

base_jitter =  0.1
jitter_mu = 0.1  # initial mean multiplier (relative to single tree size)
jitter_sigma = 0.05  # initial std dev


mu_boundary = 0.5    # 选点位置的均值（0~1）
sigma_boundary = 0.2 # 选点位置的标准差

result = {}

def tree_polygon(x, y, angle_deg):
    # ChristmasTree expects Decimal-like strings for coordinates and angle
    return ChristmasTree(center_x=str(Decimal(x)), center_y=str(Decimal(y)), angle=str(Decimal(angle_deg))).polygon


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

    # prev_result: optional tuple (centers, angles, placed_polys) from a previous
    # successful placement (e.g., for n-1 trees). If provided and compatible,
    # the algorithm will reuse these placed polygons as a starting state.
    # prev_result: 可选参数，形如 (centers, angles, placed_polys)，表示之前成功的放置结果（例如 n-1）。
    # 如果提供且兼容，算法将在该放置状态上继续放置新的树。
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
    
    n_elite = max(1, int(population * elite_frac))

    global jitter_mu , jitter_sigma , mu_boundary , sigma_boundary 

    if prev_result != None :
        if side < result[len(list ( prev_centers ))][0] :
            for p in prev_placed_polys:
                    if not fits_in_box(p, side):
                        return False, ([], [], [])
    # 依据已算得的结果进行快速判断 prev_result 是否放的进当前盒子
    
    for it in range(iterations):
        population_results = []  # list of tuples (reward, placed, centers, angles, placed_polys, jitter_mult , boundary_ratio)

        # sample a population of jitter multipliers and evaluate each using the
        # original placement core logic (kept intact). 保持原始放置逻辑不变。
        for s_idx in range(population):
            # sample jitter multiplier for this candidate
            jitter_mult = random.gauss(jitter_mu, jitter_sigma)
            if jitter_mult < 0.0:
                jitter_mult = abs(jitter_mult)

           

            boundary_ratio = random.gauss(mu_boundary, sigma_boundary)
            boundary_ratio = max(0.0, min(1.0, boundary_ratio))  # 限制在0~1之间

            # start placement attempt
            centers = list(prev_centers) if prev_centers else []
            angles = list(prev_angles) if prev_angles else []
            placed_polys = list(prev_placed_polys) if prev_placed_polys else []
            placed = len(centers)

            max_x = 0
            max_y = 0
            # iterate exactly for remaining trees we need to place
            for i in range(n - len(centers)):
                placed_ok = False
                for attempt in range(max_attempts_per_tree):
                    # with 80% probability sample near the perimeter of the
                    # union of placed polygons; otherwise sample uniformly.
                    if placed_polys and random.random() < 0.80:
                        union = unary_union(placed_polys)
                        boundary = union.boundary
                        if boundary.length <= 0:
                            x = random.random() * side
                            y = random.random() * side
                        else:
                            d = boundary_ratio * boundary.length
                            pt = boundary.interpolate(d)
                            x0 = pt.x / float(scale_factor)
                            y0 = pt.y / float(scale_factor)
                            # outward jitter uses the sampled jitter_mult
                            jitter = 1.0 * jitter_mult
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
                    if not fits_in_box(poly, side):
                        continue
                    if not no_overlap(poly, placed_polys):
                        continue
                    placed_polys.append(poly)
                    centers.append((x, y))
                    angles.append(a)
                    placed += 1
                    placed_ok = True
                    max_x = max(x, max_x)
                    max_y = max(y, max_y)
                    break
                if not placed_ok:
                    break

            # Reward: consider both number placed and approximate box size
            # 奖励同时考虑放置数量和方箱尺寸（越小越好）。
            # normalize side by expected minimal length ~ single_size*sqrt(n)

            if placed ==n :
               print("log:",n,side)
            # reward = placed trees minus a penalty proportional to approx_scale
            reward = placed - (max_x + max_y ) * 0.5
            population_results.append((reward, placed, centers, angles, placed_polys, jitter_mult, boundary_ratio))

            # track global best like before
            if placed == n:
                return True, (centers, angles, placed_polys)
            # 我们只是证明这个给定的side是可行的，并返回可以一种可行的方案，不需要继续优化

        # sort by reward and select elites
        population_results.sort(key=lambda x: x[0], reverse=True)
        elites = population_results[:n_elite]

        # update mu and sigma based on elites (CEM update)
        elite_jitters = [e[5] for e in elites]
        elite_boundary_ratios = [e[6] for e in elites]
        # use statistics module; guard against zero variance
        try:
            new_mu = statistics.mean(elite_jitters)
            new_sigma = statistics.pstdev(elite_jitters)
        except Exception:
            new_mu = jitter_mu
            new_sigma = jitter_sigma

        # small smoothing to avoid collapse
        jitter_mu = 0.8 * jitter_mu + 0.2 * new_mu
        jitter_sigma = max(1e-3, 0.8 * jitter_sigma + 0.2 * new_sigma)


        try:
            new_mu_boundary = statistics.mean(elite_boundary_ratios)
            new_sigma_boundary = statistics.pstdev(elite_boundary_ratios) if len(elite_boundary_ratios) > 1 else sigma_boundary
        except Exception:
            new_mu_boundary, new_sigma_boundary = mu_boundary, sigma_boundary
        mu_boundary = 0.8 * mu_boundary + 0.2 * new_mu_boundary
        sigma_boundary = max(1e-3, 0.8 * sigma_boundary + 0.2 * new_sigma_boundary)

    return False,([], [], [])


def find_min_side_for_n(n, timeout=30, result_dict=None):
    # Binary search for minimal side that fits n trees using the CEM packer.
    # This version will attempt fallback starts based on entries in result_dict
    # (i.e., result_dict[k] holds a previously successful placement for k trees).
    # 在二分搜索内，基于 result_dict 中已有的放置结果做逐级回退尝试。
    if result_dict is None:
        result_dict = {}

    # Start bounds and hyperparams (kept as original heuristics)
    iterations = 20
    population = 50
    max_attempts = 10

    max_dim, w, h = single_tree_size
    lo = max ( max_dim * math.sqrt(n) * 0.7 , result_dict.get(n-1, (0.0, [], [], []))[0] * 0.95  if n-1 in result_dict else 0.0 )
    hi = max ( max_dim * math.sqrt(n) * 1.2 , result_dict.get(n-1, (0.0, [], [], []))[0] * 1.08  if n-1 in result_dict else 0.0 )
    start_time = time.time()
    best_found = None
    if n <= 10:
        lmt = -1
    elif n <= 20 :
        lmt = n - 5
    elif n <= 50:
        lmt = n - 20
    elif n <= 100:
        lmt = n - 15
    else :
        lmt = n - 10
    # For each candidate mid side during binary search, try fallbacks from
    # using result[n-1], result[n-2], ..., result[0] (empty start).
    # 对于每个二分候选 mid，依次尝试基于 result[n-1], result[n-2], ..., result[0] 的回退。
    while hi - lo > 1e-2 and time.time() - start_time < timeout:
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
            print(f"  Failed to find packing for n={n}, will try grid fallback") 
            # final fallback: place trees on a simple grid without overlap
            centers = []
            angles = []
            base = single_tree_size[0]
            per_row = math.ceil(math.sqrt(n))
            side = per_row * base * 1.5
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
            # store successful packing into global result dict
            result[n] = (side, centers, angles, placed_polys)

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
    generate_submission(max_n=200, outpath='CSE_submission#004.csv')
