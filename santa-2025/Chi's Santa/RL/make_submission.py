"""
English (Overview):
    Deterministic inference + CSV writer for Kaggle submission.
    - Loads JSON policy params (W, b), adapts input obs length, and outputs s-prefixed values.
    - Uses fallback (jitter/random) when a placement is rejected due to overlap.

中文（概览）：
    用于 Kaggle 提交的确定性推理与 CSV 写出。
    - 加载 JSON 参数（W, b），自适应观测长度，并输出带 s 前缀的数值字符串。
    - 若因重叠被拒绝，使用抖动/随机的回退策略继续尝试。
"""
import argparse
import csv
import json
import os
import sys
import random
from typing import List, Dict, Tuple

import math
import torch

# Flexible imports for both package and script execution
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
try:
    from .env_simple import SimplePackingEnv, SimpleEnvConfig  # type: ignore
except Exception:
    from env_simple import SimplePackingEnv, SimpleEnvConfig  # type: ignore


def _expected_obs_dim(W) -> int:
    # infer expected obs dim from policy weight matrix shape (act_dim x obs_dim)
    return len(W[0]) if W and isinstance(W[0], list) else 0


def _adapt_obs(obs: List[float], expected_dim: int) -> List[float]:
    if len(obs) >= expected_dim:
        return obs[:expected_dim]
    # pad with zeros if current obs shorter than expected
    return obs + [0.0] * (expected_dim - len(obs))


def policy_mean_action(W, b, obs: List[float]):
    # W: list[list[float]] act_dim x obs_dim; b: list[float]
    exp_dim = _expected_obs_dim(W)
    o = _adapt_obs(obs, exp_dim)
    out = []
    for j in range(len(W)):
        mu = sum(W[j][i] * o[i] for i in range(exp_dim)) + b[j]
        out.append(math.tanh(mu))
    return out


def policy_sample_action(W, b, log_std, obs: List[float]):
    """English: Sample action ~ tanh(N(mu, std^2)) using saved log_std.

    中文：使用保存的 log_std 按 tanh(N(mu, std^2)) 采样动作。
    """
    exp_dim = _expected_obs_dim(W)
    o = _adapt_obs(obs, exp_dim)
    act = []
    for j in range(len(W)):
        mu = sum(W[j][i] * o[i] for i in range(exp_dim)) + b[j]
        ls = log_std[j] if j < len(log_std) else 0.0
        std = math.exp(ls)
        raw = mu + std * random.gauss(0.0, 1.0)
        act.append(math.tanh(raw))
    return act


def _pad_obs(obs: List[float], expected_dim: int) -> List[float]:
    if len(obs) >= expected_dim:
        return obs[:expected_dim]
    return obs + [0.0] * (expected_dim - len(obs))


def policy_mean_action_torch(state_dict_path: str, obs: List[float], device: torch.device):
    # load state_dict and run deterministic forward on device
    sd = torch.load(state_dict_path, map_location=device)
    w = sd["net.weight"]
    act_dim, obs_dim = w.shape
    from reinforce_agent import GaussianLinearPolicy  # local import
    model = GaussianLinearPolicy(obs_dim, act_dim)
    model.load_state_dict(sd)
    model.to(device)
    o = _pad_obs(obs, obs_dim)
    xt = torch.tensor(o, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu = model.net(xt)
        act = torch.tanh(mu)
    return act.cpu().tolist()


def policy_sample_action_torch(state_dict_path: str, obs: List[float], device: torch.device):
    sd = torch.load(state_dict_path, map_location=device)
    w = sd["net.weight"]
    act_dim, obs_dim = w.shape
    from reinforce_agent import GaussianLinearPolicy  # local import
    model = GaussianLinearPolicy(obs_dim, act_dim)
    model.load_state_dict(sd)
    model.to(device)
    o = _pad_obs(obs, obs_dim)
    xt = torch.tensor(o, dtype=torch.float32, device=device)
    with torch.no_grad():
        mu = model.net(xt)
        std = torch.exp(model.log_std)
        raw = mu + std * torch.randn_like(mu)
        act = torch.tanh(raw)
    return act.cpu().tolist()


def try_step_with_fallback(env: SimplePackingEnv, W, b, obs: List[float], max_tries: int = 10,
                           act_fn=None):
    """English: Try mean action, then jittered actions, finally random actions.

    中文：先尝试均值动作，再尝试抖动动作，最后尝试随机动作。
    """
    # Choose action generator (default: mean)
    if act_fn is None:
        act_fn = lambda o: policy_mean_action(W, b, o)

    # Try primary action; if overlap penalty, jitter and try random
    act = act_fn(obs)
    nobs, rew, done, info = env.step(act)
    if rew > -10.0:  # accepted
        return nobs, rew, done, info

    # fallback: small jitters
    for scale in [0.05, 0.1, 0.2, 0.5, 1.0]:
        for _ in range(2):
            noise = [random.gauss(0.0, scale) for _ in range(3)]
            jittered = [max(-1.0, min(1.0, act[k] + noise[k])) for k in range(3)]
            nobs, rew, done, info = env.step(jittered)
            if rew > -10.0:
                return nobs, rew, done, info

    # final random tries
    tries = 0
    while tries < max_tries:
        rnd = [random.uniform(-1, 1) for _ in range(3)]
        nobs, rew, done, info = env.step(rnd)
        if rew > -10.0:
            return nobs, rew, done, info
        tries += 1
    # give up: return last
    return nobs, rew, done, info


def parse_input_csv(path: str) -> Dict[int, List[Tuple[float, float, float]]]:
    """Reads an optional feasible arrangement CSV: id,x,y,deg with s-prefixed numbers.
    Returns a dict: n -> list[(x,y,deg)] sorted by index.
    """
    if not os.path.exists(path):
        return {}
    data: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row["id"]
            if "_" not in rid:
                continue
            n_str, idx_str = rid.split("_", 1)
            try:
                n = int(n_str)
                i = int(idx_str)
            except ValueError:
                continue
            def parse_s(val: str) -> float:
                s = val.strip()
                if s.startswith("s"):
                    s = s[1:]
                return float(s)
            x = parse_s(row["x"]) ; y = parse_s(row["y"]) ; deg = parse_s(row["deg"]) 
            data.setdefault(n, {})[i] = (x, y, deg)
    # convert to ordered lists
    out: Dict[int, List[Tuple[float, float, float]]] = {}
    for n, m in data.items():
        lst = [m[i] for i in sorted(m.keys())]
        out[n] = lst
    return out


def write_submission(pi_path: str, out_csv: str, input_csv: str = "", n_max: int = 200,
                     mode: str = "deterministic", seed: int | None = None):
    """English: Generate exactly n lines for each n=1..n_max, respecting constraints.

    中文：为每个 n=1..n_max 精确生成 n 行，满足提交格式与约束。
    """
    # Optional RNG seed for reproducibility/diversity
    if seed is not None:
        random.seed(seed)
    # Resolve policy path relative to this script
    if not os.path.isabs(pi_path):
        pi_path = os.path.join(CURRENT_DIR, pi_path)

    # Prefer a .pt state_dict if present (and torch available), else load JSON params
    pt_candidate = None
    if pi_path.endswith('.json'):
        pt_candidate = pi_path[:-5] + '.pt'
    else:
        pt_candidate = pi_path + '.pt'

    use_torch_pt = False
    if os.path.exists(pt_candidate):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # quick check load
            _ = torch.load(pt_candidate, map_location=device)
            use_torch_pt = True
            pt_path = pt_candidate
        except Exception:
            use_torch_pt = False

    if not use_torch_pt:
        # Load JSON fallback
        with open(pi_path, "r") as f:
            data = json.load(f)
        W = data["W"]
        b = data["b"]
        log_std = data.get("log_std", [0.0, 0.0, 0.0])

    seeds = parse_input_csv(input_csv) if input_csv else {}

    # Resolve output path to RL folder when a relative filename is provided
    out_path = out_csv
    if not os.path.isabs(out_path):
        out_path = os.path.join(CURRENT_DIR, out_csv)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x", "y", "deg"])

        # Choose action function per mode and available policy backend
        if use_torch_pt:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if mode == "sample":
                act_fn = lambda o: policy_sample_action_torch(pt_path, o, device)
            else:
                act_fn = lambda o: policy_mean_action_torch(pt_path, o, device)
        else:
            if mode == "sample":
                act_fn = lambda o: policy_sample_action(W, b, log_std, o)
            else:
                act_fn = lambda o: policy_mean_action(W, b, o)

        for n in range(1, n_max + 1):
            env = SimplePackingEnv(SimpleEnvConfig(n_trees=n, max_coord=50.0, scale=1.0))
            if n in seeds:
                # seed exactly n placements; if不足则只放已有的
                env.load_initial(seeds[n][:n])
            # If已经放满，直接输出
            if env.index >= n:
                for sid, sx, sy, sd in env.output_rows(n):
                    w.writerow([sid, sx, sy, sd])
                continue
            obs = env._state()
            # 放置直至达到 n
            while env.index < n:
                obs, rew, done, info = try_step_with_fallback(env, W, b, obs, act_fn=act_fn)
                # 防御：若 done 但未放满，重置 obs 继续尝试（避免死循环）
                if done and env.index < n:
                    obs = env._state()
            # Emit placements for this n (恰好 n 行)
            for sid, sx, sy, sd in env.output_rows(n):
                w.writerow([sid, sx, sy, sd])

    print(f"Wrote submission to {out_path}")


if __name__ == "__main__":
    import math
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=str, default="RL/reinforce_policy.json")
    ap.add_argument("--out", type=str, default="submission.csv")
    ap.add_argument("--input", type=str, default="RL/input.csv", help="Optional feasible arrangement CSV to seed placements")
    ap.add_argument("--n_max", type=int, default=200)
    ap.add_argument("--mode", type=str, choices=["deterministic","sample"], default="deterministic",
                    help="Use mean action (deterministic) or sample from policy (sample)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Optional RNG seed for reproducibility/diversity in submission")
    args = ap.parse_args()

    write_submission(args.policy, args.out, input_csv=args.input, n_max=args.n_max,
                     mode=args.mode, seed=args.seed)
