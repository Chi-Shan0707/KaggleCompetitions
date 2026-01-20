#!/usr/bin/env python3
"""
checker.py
Validate submission CSV placements using ChristmasTree geometry. Requires shapely and matplotlib for plotting.
检查提交文件的放置是否合法：中心坐标绝对值<=100，且多边形不重叠且完全位于 [-100,100] 区域内。

Usage:
    python3 checker.py submission.csv [--plot] [--saveplots DIR]
    Use --plot to display per-group visualizations; use --saveplots DIR to save plots to the specified directory.

Output: prints per-configuration report and exits with non-zero code if any invalid.
"""
from decimal import Decimal
import sys
import csv
from collections import defaultdict
import math

# Attempt to import the project's ChristmasTree helper (may require shapely)
try:
    from Christmastree import ChristmasTree, scale_factor
    HAS_CHRISTMASTREE = True
except Exception as _e:
    ChristmasTree = None
    scale_factor = Decimal('1')
    HAS_CHRISTMASTREE = False
    Christmastree_import_error = _e

# Additional geometry/plotting utilities
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import os

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


def compute_bounding_square(polys):
    """Compute minimal square bounding the union of polys.
    Returns (side_length, square_x, square_y, minx, miny, maxx, maxy) in original units (Decimal).
    """
    if not polys:
        return Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0'), Decimal('0')
    bounds = unary_union(polys).bounds
    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor

    width = maxx - minx
    height = maxy - miny
    side_length = max(width, height)

    square_x = minx if width >= height else minx - (side_length - width) / Decimal('2')
    square_y = miny if height >= width else miny - (side_length - height) / Decimal('2')
    return side_length, square_x, square_y, minx, miny, maxx, maxy


def plot_polygons(polys, side_length=None, square_x=None, square_y=None, title=None, savepath=None, show=True, highlight_overlaps=True):
    """Plot polygons (expects shapely polygons in scaled units). Optionally highlights overlaps.

    This implementation uses Decimal internally (returned by compute_bounding_square) then
    converts to floats only when passing to matplotlib, avoiding float/Decimal mixing errors.
    """
    try:
        import matplotlib.pyplot as plt  # local import to avoid top-level requirement if not plotting
        from matplotlib.patches import Rectangle
        import numpy as np
    except Exception as e:
        raise RuntimeError('matplotlib/numpy required for plotting: ' + str(e))

    n = len(polys)
    _, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, max(1, n)))

    # plot polygons
    for i, poly in enumerate(polys):
        x_scaled, y_scaled = poly.exterior.xy
        x = [float(Decimal(val) / scale_factor) for val in x_scaled]
        y = [float(Decimal(val) / scale_factor) for val in y_scaled]
        ax.plot(x, y, color=colors[i % len(colors)])
        ax.fill(x, y, alpha=0.5, color=colors[i % len(colors)])

    # obtain Decimal-based square parameters
    if side_length is None or square_x is None or square_y is None:
        s_dec, sx_dec, sy_dec, _, _, _, _ = compute_bounding_square(polys)
    else:
        # normalize provided values to Decimal for consistent arithmetic
        s_dec = Decimal(str(side_length)) if not isinstance(side_length, Decimal) else side_length
        sx_dec = Decimal(str(square_x)) if not isinstance(square_x, Decimal) else square_x
        sy_dec = Decimal(str(square_y)) if not isinstance(square_y, Decimal) else square_y

    side_length_f = float(s_dec)
    square_x_f = float(sx_dec)
    square_y_f = float(sy_dec)

    bounding_square = Rectangle(
        (square_x_f, square_y_f),
        side_length_f,
        side_length_f,
        fill=False,
        edgecolor='red',
        linewidth=2,
        linestyle='--',
    )
    ax.add_patch(bounding_square)

    # highlight overlaps
    if highlight_overlaps:
        for i in range(n):
            for j in range(i + 1, n):
                inter = polys[i].intersection(polys[j])
                try:
                    area = inter.area
                except Exception:
                    area = 0.0
                if area > AREA_EPS:
                    # plot intersection region in red
                    if not inter.is_empty and hasattr(inter, 'exterior'):
                        x_i, y_i = inter.exterior.xy
                        x_i = [float(Decimal(val) / scale_factor) for val in x_i]
                        y_i = [float(Decimal(val) / scale_factor) for val in y_i]
                        ax.fill(x_i, y_i, color='red', alpha=0.7)

    padding = Decimal('0.5')
    ax.set_xlim(float(Decimal(square_x_f) - padding), float(Decimal(square_x_f) + Decimal(str(side_length_f)) + padding))
    ax.set_ylim(float(Decimal(square_y_f) - padding), float(Decimal(square_y_f) + Decimal(str(side_length_f)) + padding))
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    if title:
        ax.set_title(title)
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    plt.close()


# --- Square-number plotting helpers ---

def is_perfect_square(n):
    try:
        n = int(n)
    except Exception:
        return False
    if n <= 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n


def plot_square_grid(n, cell_size=1.0, savepath=None, show=True, title=None):
    """Plot a grid representing the perfect square n (e.g., n=9 -> 3x3 grid)."""
    if not is_perfect_square(n):
        raise ValueError(f'{n} is not a perfect square')
    r = int(math.isqrt(n))

    fig, ax = plt.subplots(figsize=(max(2, r * 0.6), max(2, r * 0.6)))
    for i in range(r):
        for j in range(r):
            rect = Rectangle((i * cell_size, j * cell_size), cell_size, cell_size, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    ax.set_xlim(0, r * cell_size)
    ax.set_ylim(0, r * cell_size)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title)
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    plt.close()


def generate_square_plots(max_n=25, save_dir=None, show=False):
    """Generate plots for square numbers up to max_n (inclusive when it is a perfect square)."""
    max_n = int(max_n)
    max_r = int(math.isqrt(max_n))
    square_values = [i * i for i in range(1, max_r + 1)]
    saved = []
    for s in square_values:
        name = f'square_{s}.png'
        savepath = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            savepath = os.path.join(save_dir, name)
        title = f'{s} = {int(math.isqrt(s))}^2'
        plot_square_grid(s, savepath=savepath, show=show, title=title)
        saved.append(savepath or name)
    return saved


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

    # compute bounding square and bounds in original units
    side_length_dec, square_x_dec, square_y_dec, minx_dec, miny_dec, maxx_dec, maxy_dec = compute_bounding_square(polys)
    minx = float(minx_dec)
    miny = float(miny_dec)
    maxx = float(maxx_dec)
    maxy = float(maxy_dec)
    side_length = float(side_length_dec)

    # check that entire polygons lie within [-100, 100] in original units
    within = True
    if minx < -float(CENTER_LIMIT) - COORD_EPS or miny < -float(CENTER_LIMIT) - COORD_EPS or maxx > float(CENTER_LIMIT) + COORD_EPS or maxy > float(CENTER_LIMIT) + COORD_EPS:
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
        'minx': float(minx),
        'miny': float(miny),
        'maxx': float(maxx),
        'maxy': float(maxy),
        'within_bounds': within,
        'overlaps': overlaps,
        'side': float(side_length),
        'square_x': float(square_x_dec),
        'square_y': float(square_y_dec),
    }
    return valid, info


def main(path, plot=False, plot_all=False, save_dir=None, squares_up_to=None, save_squares_dir=None, show_squares=False):
    # Optionally generate square-number plots first
    if squares_up_to is not None:
        saved = generate_square_plots(squares_up_to, save_dir=save_squares_dir, show=show_squares)
        if save_squares_dir:
            print(f'Square-number plots saved to {save_squares_dir}:')
            for s in saved:
                print('  ', s)
        else:
            print('Square-number plots generated:')
            for s in saved:
                print('  ', s)

    # If Christmastree helper is missing, abort when we attempt to validate groups
    if not HAS_CHRISTMASTREE:
        print('Error: cannot validate submissions because Christmastree import failed: ', Christmastree_import_error)
        print('You can still generate square-number plots with --squares-up-to')
        return 2

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

    DEFAULT_SQUARE_SET = {1, 4, 9, 16}

    def should_plot_group(prefix, plot=False, plot_all=False, save_dir=None):
        """Decide whether to plot/save a group's figure.

        Rules:
        - If save_dir is provided, save EVERY group's plot (user requested automatic saves).
        - If plot_all is True, plot every group.
        - If plot is True (but not plot_all), plot only default squares and multiples of 10.
        """
        # save_dir implies user wants all plots saved
        if save_dir:
            return True
        if plot_all:
            return True
        if not plot:
            return False
        try:
            pnum = int(prefix)
        except Exception:
            return False
        if pnum in DEFAULT_SQUARE_SET or (pnum % 10 == 0):
            return True
        return False

    for prefix in sorted(groups.keys(), key=lambda x: int(x) if x.isdigit() else x):
        rows = groups[prefix]
        valid, info = validate_group(rows)
        if 'error' in info:
            print(f'Group {prefix}: parse/limit error: {info["error"]}')
            any_bad = True
            continue
        side_str = f"; side={info.get('side'):.6f}" if 'side' in info else ''
        if valid:
            print(f'Group {prefix}: VALID; n={info["n"]}; bounds=({info["minx"]:.6f},{info["miny"]:.6f})-({info["maxx"]:.6f},{info["maxy"]:.6f}){side_str}')
        else:
            print(f'Group {prefix}: INVALID; n={info["n"]}; bounds=({info["minx"]:.6f},{info["miny"]:.6f})-({info["maxx"]:.6f},{info["maxy"]:.6f}){side_str}')
            if not info['within_bounds']:
                print('  Reason: some polygons exceed coordinate limit +/-100 (based on full polygon bounds)')
            if info['overlaps']:
                print(f'  Reason: {len(info["overlaps"])} overlapping pairs (i,j,area)')
                for i,j,area in info['overlaps'][:10]:
                    print(f'    overlap {i} vs {j}: area={area:.6e}')
            any_bad = True

        # Decide whether to plot this group
        if should_plot_group(prefix, plot=plot, plot_all=plot_all, save_dir=save_dir):
            polys = [make_poly(parse_coord(xs), parse_coord(ys), parse_coord(degs)) for _, xs, ys, degs in rows]
            title = f'Group {prefix} n={len(polys)} side={info.get("side"):.6f}'
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                savepath = os.path.join(save_dir, f'{prefix}.png')
                plot_polygons(polys, show=False, savepath=savepath, title=title)
                print(f'  Plot saved to {savepath}')
            else:
                plot_polygons(polys, title=title)

    return 1 if any_bad else 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 checker.py submission.csv [--plot] [--plot-all] [--saveplots DIR] [--squares-up-to N] [--save-squares DIR]')
        sys.exit(2)
    path = sys.argv[1]
    plot = '--plot' in sys.argv
    plot_all = '--plot-all' in sys.argv
    save_dir = None
    if '--saveplots' in sys.argv:
        idx = sys.argv.index('--saveplots')
        if idx + 1 < len(sys.argv):
            save_dir = sys.argv[idx + 1]
        else:
            print('Missing directory after --saveplots')
            sys.exit(2)
    sys.exit(main(path, plot=plot, plot_all=plot_all, save_dir=save_dir))
