#!/usr/bin/env python3
"""
checker.py
Validate submission CSV placements using ChristmasTree geometry.
检查提交文件的放置是否合法：中心坐标绝对值<=100，且多边形不重叠且完全位于 [-100,100] 区域内。

Usage:
    python3 checker.py submission.csv

Output: prints per-configuration report and exits with non-zero code if any invalid.
"""
from decimal import Decimal
import sys
import csv
from collections import defaultdict
import math

from Christmastree import ChristmasTree, scale_factor

# tolerance (in shapely geometric units after scaling)
AREA_EPS = 1e-6
COORD_EPS = 1e-9

# Limits for center coordinates (absolute value <= 100)
# 中心坐标允许的绝对值上限
CENTER_LIMIT = Decimal('100')


def parse_coord(s):
    # submission values are like 's12.345678' per generator; strip leading 's' if present
    if isinstance(s, str) and s.startswith('s'):
        s = s[1:]
    return Decimal(s)


def make_poly(x_dec, y_dec, deg_dec):
    # Build ChristmasTree polygon from Decimal coordinates/angle
    return ChristmasTree(center_x=str(x_dec), center_y=str(y_dec), angle=str(deg_dec)).polygon


def validate_group(rows):
    """Validate one configuration group. rows: list of (id, xstr, ystr, degstr)
    返回 (valid_bool, info_dict)
    info contains 'n', 'side', 'minx','miny','maxx','maxy','overlaps' list
    """
    polys = []
    coords = []
    angles = []
    for rid, xs, ys, degs in rows:
        try:
            x = parse_coord(xs)
            y = parse_coord(ys)
            a = parse_coord(degs)
        except Exception as e:
            return False, {'error': f'bad number parse for {rid}: {e}'}
        # check center coordinate absolute limit
        if abs(x) > CENTER_LIMIT or abs(y) > CENTER_LIMIT:
            return False, {'error': f'center coordinate out of allowed range for {rid}: ({x},{y})'}
        poly = make_poly(x, y, a)
        polys.append(poly)
        coords.append((x, y))
        angles.append(a)

    # compute overall bounds in scaled coordinates (polygons already in scaled units)
    minx = min(p.bounds[0] for p in polys)
    miny = min(p.bounds[1] for p in polys)
    maxx = max(p.bounds[2] for p in polys)
    maxy = max(p.bounds[3] for p in polys)

    # check that entire polygons lie within [-100, 100] in original units
    Smin = -float(CENTER_LIMIT) * float(scale_factor)
    Smax = float(CENTER_LIMIT) * float(scale_factor)
    within = True
    if minx < Smin - COORD_EPS or miny < Smin - COORD_EPS or maxx > Smax + COORD_EPS or maxy > Smax + COORD_EPS:
        within = False

    # overlap checks
    overlaps = []
    n = len(polys)
    for i in range(n):
        for j in range(i + 1, n):
            inter = polys[i].intersection(polys[j])
            try:
                area = inter.area
            except Exception:
                area = 0.0
            if area > AREA_EPS:
                overlaps.append((i, j, area))

    valid = within and (len(overlaps) == 0)

    info = {
        'n': n,
        'minx': float(minx/float(scale_factor)),
        'miny': float(miny/float(scale_factor)),
        'maxx': float(maxx/float(scale_factor)),
        'maxy': float(maxy/float(scale_factor)),
        'within_bounds': within,
        'overlaps': overlaps,
    }
    return valid, info


def main(path):
    groups = defaultdict(list)
    # read CSV
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # expect columns id,x,y,deg
            rid = row.get('id')
            xs = row.get('x')
            ys = row.get('y')
            degs = row.get('deg')
            if rid is None or xs is None or ys is None or degs is None:
                print('CSV missing required columns id,x,y,deg')
                return 2
            # group by prefix before underscore if present (e.g., '005_0')
            prefix = rid.split('_')[0]
            groups[prefix].append((rid, xs, ys, degs))

    any_bad = False
    for prefix in sorted(groups.keys(), key=lambda x: int(x) if x.isdigit() else x):
        rows = groups[prefix]
        valid, info = validate_group(rows)
        if 'error' in info:
            print(f'Group {prefix}: parse/limit error: {info["error"]}')
            any_bad = True
            continue
        if valid:
            print(f'Group {prefix}: VALID; n={info["n"]}; bounds=({info["minx"]:.6f},{info["miny"]:.6f})-({info["maxx"]:.6f},{info["maxy"]:.6f})')
        else:
            print(f'Group {prefix}: INVALID; n={info["n"]}; bounds=({info["minx"]:.6f},{info["miny"]:.6f})-({info["maxx"]:.6f},{info["maxy"]:.6f})')
            if not info['within_bounds']:
                print('  Reason: some polygons exceed coordinate limit +/-100 (based on full polygon bounds)')
            if info['overlaps']:
                print(f'  Reason: {len(info["overlaps"])} overlapping pairs (i,j,area)')
                for i,j,area in info['overlaps'][:10]:
                    print(f'    overlap {i} vs {j}: area={area:.6e}')
            any_bad = True

    return 1 if any_bad else 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 checker.py submission.csv')
        sys.exit(2)
    path = sys.argv[1]
    sys.exit(main(path))
