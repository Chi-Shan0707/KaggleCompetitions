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
import numpy as np
from matplotlib.patches import Polygon
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.strtree import STRtree

from Christmastree import ChristmasTree, scale_factor
import statistics

single_tree_size = [1.0, 0.7, 1.0]
# max_dim , width , height
# 初始化全局 result 字典，result[0] 表示空状态



mu_angle = 135
sigma_angle = 90



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
    If an STRtree `tree` is provided, query it for candidates to avoid full scan.
    """
    # use spatial index if available
    if not placed_polys:
        return True
    
    if tree is not None:
        try:
            candidates = tree.query(poly)
        except Exception:
            for i in range(len(placed_polys)):
                candidates.append(i)
            
        for cand in candidates:
            if placed_polys[cand] is poly:
                continue
            intersects = poly.intersects(placed_polys[cand])
            if intersects:
                inter_area = poly.intersection(placed_polys[cand]).area
                if inter_area > 1e-6:
                    return False

        return True

    # fallback: brute-force
    for p in placed_polys:
        if poly.intersects(p):
            if poly.intersection(p).area > 1e-6:
                return False
    return True


def attempt_pack_cem(n, side, iterations, population, elite_frac=0.12, max_attempts_per_tree=10, prev_result=None,deadline=None):
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
    
    # ensure at least a couple elites but not more than population
    n_elite = min(population, max(2, int(population * elite_frac)))

    global jitter_mu , jitter_sigma
    global mu_angle , sigma_angle

    mu_boundary = 0.6    # 选点位置的均值（0~1） — nudged outward a bit
    sigma_boundary = 0.08 # 选点位置的标准差 — tighter
    # angle distribution (local to this attempt function) - will be adapted per-iteration
    

    if prev_result != None :
        if side < result[len(list ( prev_centers ))][0] :
            for p in prev_placed_polys:
                    if not fits_in_box(p, side):
                        return False, ([], [], [])
    # 依据已算得的结果进行快速判断 prev_result 是否放的进当前盒子
    
    for it in range(iterations):
        population_results = []  
        for s_idx in range(population):
            if deadline is not None and time.time() > deadline:
                return False, ([], [], [])
            # sample jitter multiplier for this candidate
            jitter_mult = random.gauss(jitter_mu, jitter_sigma)
            if jitter_mult < 0.0:
                jitter_mult = abs(jitter_mult)

            # candidate-level boundary_ratio (kept for CEM bookkeeping)
            boundary_ratio = random.gauss(mu_boundary, sigma_boundary)
            boundary_ratio = max(0.0, min(1.0, boundary_ratio))  # 限制在0~1之间

            # candidate-level angle center sampled from current angle distribution
            angle_center = random.gauss(mu_angle, sigma_angle) % 360.0

            # start placement attempt
            centers = list(prev_centers) if prev_centers else []
            angles = list(prev_angles) if prev_angles else []
            placed_polys = list(prev_placed_polys) if prev_placed_polys else []
            # build spatial index for faster overlap queries

            tree = STRtree(placed_polys) if placed_polys else None
            max_x = 0
            max_y = 0
            rem = n - len(placed_polys)
            outside = False
            placed_num =len(placed_polys)
            for need_to_place in range(rem) :
                for attempt in range(max_attempts_per_tree):
                    if deadline is not None and time.time() > deadline:
                        return False, ([], [], [])
                    if not placed_polys:
                        # no placed trees yet, sample uniformly
                        x = random.random() * side
                        y = random.random() * side
                    else:
                        union = unary_union(placed_polys)
                        boundary = union.boundary
                        br = random.gauss(mu_boundary, sigma_boundary)
                        br = max(0.0, min(1.0, br))
                        d = br * boundary.length
                        pt = boundary.interpolate(d)
                        x0 = pt.x / float(scale_factor)
                        y0 = pt.y / float(scale_factor)
                        # outward jitter uses the sampled jitter_mult
                        jitter = single_tree_size[0] * jitter_mult
                        theta = random.random() * 2.0 * math.pi
                        dx = math.cos(theta) * jitter * random.random()
                        dy = math.sin(theta) * jitter * random.random()
                        x = x0 + dx
                        y = y0 + dy
                    a = random.uniform(0.0, 360.0)
                    poly = tree_polygon(x, y, a)
                    if not no_overlap(poly, placed_polys, tree=tree):
                        continue
                    
                    centers.append((x, y))
                    angles.append(a)
                    placed_polys.append(poly)
                    placed_num += 1
                    if not fits_in_box(poly, side):
                        outside = True
                    if centers:
                        xs = [c[0] for c in centers]
                        ys = [c[1] for c in centers]
                        max_x = max(xs)
                        max_y = max(ys)
                    else:
                        max_x = x
                        max_y = y
                    break

            if placed_num == n and not outside:
                # full placement success
                return True, (centers, angles, placed_polys)
            reward = math.sqrt(placed_num) - (max_x + max_y)/2
            population_results.append((reward,jitter_mult, boundary_ratio, angle_center))
            
            # Reward: consider approximate box size
            
        population_results.sort(key=lambda x: x[0], reverse=True)
        elites = population_results[:n_elite]
        elite_jitters = [e[1] for e in elites]
        elite_boundary_ratios = [e[2] for e in elites]
        elite_angles = [e[3] for e in elites]
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
            new_sigma_boundary = statistics.pstdev(elite_boundary_ratios) 
        except Exception:
            new_mu_boundary, new_sigma_boundary = mu_boundary, sigma_boundary
        

        mu_boundary = 0.8 * mu_boundary + 0.2 * new_mu_boundary
        sigma_boundary = max(1e-3, 0.8 * sigma_boundary + 0.2 * new_sigma_boundary)
        
        
        try:
            new_mu_angle = statistics.mean(elite_angles)
            new_sigma_angle = statistics.pstdev(elite_angles) if len(elite_angles) > 1 else sigma_angle
        except Exception:
            new_mu_angle, new_sigma_angle = mu_angle, sigma_angle
        mu_angle = 0.5 * mu_angle + 0.5 * new_mu_angle
        sigma_angle = max(1e-3, 0.5 * sigma_angle + 0.5 * new_sigma_angle)
       

    return False,([], [], [])


def find_min_side_for_n(n, timeout=30, result_dict=None, state=False):
    # Binary search for minimal side that fits n trees using the CEM packer.
    # This version will attempt fallback starts based on entries in result_dict
    # (i.e., result_dict[k] holds a previously successful placement for k trees).
    # 在二分搜索内，基于 result_dict 中已有的放置结果做逐级回退尝试。
    if result_dict is None:
        result_dict = {}

    # Start bounds and hyperparams (kept as original heuristics)
    
    time_rate = 0.80
    
    eps =1e-4
    if n ==1 :
        iterations = 100
        population = 30
        max_attempts = 40
        lo =0.80
        hi =1.20
        timeout = 120
        eps=1e-6
        time_rate = 0.80
    elif n== 2:
        iterations =90
        population = 75
        max_attempts = 20
        lo =0.95
        hi =1.45
        timeout = 100
        eps=1e-6
        time_rate = 0.70
    elif n==3 :
        iterations = 80
        population = 50
        max_attempts = 40
        lo =1.25
        hi =1.60
        timeout = 100
        eps=1e-6
        time_rate = 0.70
    elif n== 4:
        iterations = 75
        population = 75
        max_attempts = 40
        lo =1.40
        timeout = 100
        eps=1e-6
        time_rate = 0.70
    elif n<= 9:
        iterations = 55
        population = 50
        max_attempts = 53
        lo = math.sqrt(n) * 0.70
        hi = math.sqrt(n) * 1.05
        timeout = 120
        eps=1e-5
        time_rate = 0.65
    elif n <= 16 :
        iterations = 55
        population = 30
        max_attempts = 50
        lo =  result_dict.get(n-1, (0.0, [], [], []))[0] * 0.92
        hi =  math.sqrt(n) * 1.02
        timeout = 120
        eps=1e-5
        time_rate = 0.60
    elif n<=49 :
        iterations = 50
        population = 50
        max_attempts = 30
        lo = result_dict.get(n-1, (0.0, [], [], []))[0] * 0.92
        hi = result_dict.get(n-1, (0.0, [], [], []))[0] * (1 + 1 / n)
        timeout = 75
        time_rate = 0.55
    else :
        time_rate = 0.50
        lo = max ( math.sqrt(n) * 0.85 , result_dict.get(n-1, (0.0, [], [], []))[0] * 0.975  )
        hi = max ( math.sqrt(n) * 0.9 , result_dict.get(n-1, (0.0, [], [], []))[0] * (1+ 0.8 /n) )
        iterations = 40
        population = 50
        max_attempts = 30
        if n <= 100 :
            timeout = 60
        else :
           timeout = 45
   
   

    # For each candidate mid side during binary search, try fallbacks from
    # using result[n-1], result[n-2], ..., result[0] (empty start).
    # 对于每个二分候选 mid，依次尝试基于 result[n-1], result[n-2], ..., result[0] 的回退。

    
    best_found = None
    

    global mu_angle , sigma_angle
    mu_angle = 180.0
    sigma_angle = 200
    
    global jitter_mu , jitter_sigma
    jitter_mu = 0.15  # initial mean multiplier (relative to single tree size)
    jitter_sigma = 0.10  # initial std dev

    for k in range(n-1, -1, -1):
        rest_time = timeout
        start_time = time.time()
        DDL= start_time + timeout
        if n==1:
            prev_entry = None
        else :
            prev_entry = result_dict.get(k, None)
        while hi - lo > eps and time.time() < DDL:
            mid = (lo + hi) / 2.0
            success_any = False
            found_result = None
            _, c_k, a_k, p_k = prev_entry if prev_entry is not None else (0.0, [], [], [])
            prev_result = (c_k, a_k, p_k)
            success, res = attempt_pack_cem(n, mid, iterations=iterations, population=population, max_attempts_per_tree=max_attempts, prev_result=prev_result,deadline=DDL- rest_time *time_rate)
            rest_time = rest_time * time_rate
            
            if success:
                hi = mid
                print(f" n={n} success at side={mid:.6f} ")
                if mid < (best_found[0] if best_found is not None else float('inf')):
                    best_found = (mid, res)
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
        found = find_min_side_for_n(n, result_dict=result,state=False)
        if found is None:
            break
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
    generate_submission(max_n=200, outpath='CSE_submission#007.csv')
