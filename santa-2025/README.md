# Santa 2025 - Simple CEM RL Packer

This repository contains a lightweight CEM-based RL-style packer that uses the exact
`ChristmasTree` polygon from `Christmastree.py` and `shapely` for all geometry checks.

Files:
- `Christmastree.py` : tree polygon definition (provided)
- `rl_packer.py` : main solver that generates `submission.csv`
- `requirements.txt` : Python dependencies

Usage:

Install dependencies (recommended in a venv):

```bash
pip install -r requirements.txt
```

Run the packer (default: generate placements for n=1..20):

```bash
python rl_packer.py
```

Output:
- `submission.csv` in the same folder. Format: header `id,x,y,deg` with each value prefixed by `s`.

Notes:
- This is a simple, initial solver using a cross-entropy method to sample placements.
- It strictly uses `shapely` for containment and overlap checks and the polygon from
  `Christmastree.py` as requested.
