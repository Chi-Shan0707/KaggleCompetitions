"""
English (Overview):
    Training script for the minimal REINFORCE baseline on the local Shapely env.
    Saves parameters to JSON (no NumPy/Torch dependency).

中文（概览）：
    在本地 Shapely 环境上训练最小 REINFORCE，并将参数保存为 JSON。
    不依赖 NumPy/Torch，便于快速运行与部署。
"""
import argparse
import json
import os
import sys
import random
from statistics import mean
import torch


# Flexible imports for both package and script execution
CURRENT_DIR = os.path.dirname(__file__)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
try:
    from .env_simple import SimplePackingEnv, SimpleEnvConfig  # type: ignore
    from .reinforce_agent import ReinforceAgent, ReinforceConfig  # type: ignore
except Exception:
    from env_simple import SimplePackingEnv, SimpleEnvConfig  # type: ignore
    from reinforce_agent import ReinforceAgent, ReinforceConfig  # type: ignore


def train(n_trees: int = 10, epochs: int = 50,
          steps_per_episode: int = 512, batch_episodes: int = 8, seed: int = 42,
          save_path: str = "reinforce_policy.json"):
    """English: Train the agent and save parameters to JSON.

    中文：训练智能体并将参数保存为 JSON 文件。
    - n_trees 影响状态维度（3*n + 1），建议与推理时尽量一致或采用自适应输入。
    - epochs、batch_episodes、steps_per_episode 控制采样与更新规模。
    """
    random.seed(seed)

    env = SimplePackingEnv(SimpleEnvConfig(n_trees=n_trees, max_coord=50.0, scale=1.0))
    obs_dim = len(env.reset())
    act_dim = 3

    cfg = ReinforceConfig(epochs=epochs, batch_episodes=batch_episodes, max_episode_steps=steps_per_episode)
    agent = ReinforceAgent(obs_dim, act_dim, cfg)
    # move policy to device (GPU if available)
 #   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device='cpu'
    agent.pi.to(device)

    # Detailed plain-language notes (English / 中文)
    # ------------------------------------------------
    # English:
    # - Where the agent learns: the core training loop below (for epoch in range(cfg.epochs))
    #   collects `cfg.batch_episodes` episodes, stores trajectories, and calls `agent.update()` to adjust W and b.
    # - State / observation range: the environment state is a flat vector of length 3*n_trees + 1,
    #   containing normalized (x/max_coord, y/max_coord, deg/360) values for already-placed trees and a progress scalar index/n.
    # - Action range and mapping: agent outputs `act = (ax, ay, adeg)` in [-1,1]^3. The environment maps these to real placements as:
    #       x = ax * max_coord   (so x in [-max_coord, +max_coord])
    #       y = ay * max_coord   (so y in [-max_coord, +max_coord])
    #       deg = adeg * 180.0   (so deg roughly in [-180, 180])
    #   See `env_simple.py` step() for the exact mapping and overlap checks.
    # - Exploration during training: `agent.select_action()` uses `pi.sample()` which draws raw ~ N(mu, std^2) and then tanh(raw).
    #   Coverage of the position/degree space depends on mu (W·obs+b) and std. Larger std -> wider sampling.
    # - During inference, `make_submission.py` has two modes:
    #     * deterministic: uses tanh(mu) (no sampling)
    #     * sample: samples from tanh(N(mu,std^2)) (introduces stochasticity)
    #   Additionally, the submission script will jitter or random-search on rejection (fallback), extending local exploration.
    #
    # 中文：
    # - 学习发生在下方的训练循环（for epoch ...）：每个 epoch 收集若干回合（cfg.batch_episodes），将轨迹传入 `agent.update()` 更新策略参数。
    # - 状态/观测范围：状态向量长度为 3*n_trees + 1，包含已放树的 (x/max_coord, y/max_coord, deg/360) 与进度 index/n。
    # - 动作范围与映射：智能体输出 act=(ax,ay,adeg) ∈ [-1,1]^3，环境映射为真实值：
    #       x = ax * max_coord   (x 在 [-max_coord,+max_coord])
    #       y = ay * max_coord   (y 在 [-max_coord,+max_coord])
    #       deg = adeg * 180.0   (deg 约在 [-180,180])
    #   详见 `env_simple.py` 的 step() 实现及重叠检测逻辑。
    # - 训练时的探索：`agent.select_action()` 内部从 N(mu,std^2) 采样 raw，再用 tanh 得到动作。是否能覆盖整个坐标/角度空间，受 mu 与 std 的影响。
    # - 推理时：`deterministic` 使用均值（不探索），`sample` 使用采样（有探索）。当动作被拒绝（重叠）时，提交脚本会尝试抖动和随机动作作为回退。
    #
    # Small, practical tricks to teach the agent / 教你教 agent 的小技巧：
    # - Reward shaping: keep per-step reward r_t = - (s_t^2 / t) to align with final score; add a larger overlap penalty (e.g. -50) to strongly discourage overlaps.
    # - Curriculum: train first on small n_trees (e.g. 3,5) to learn compact placement, then gradually increase to target n.
    # - Increase exploration early: set a larger `init_std` (e.g. 1.0) or make `log_std` learnable so agent explores more initially.
    # - Demonstrations / seeds: provide good partial placements via `input.csv` or programmatic heuristics (load_initial) to teach patterns.
    # - Hand-crafted features: augment the observation with simple heuristic features (e.g. nearest free region, local density) to simplify learning.
    # - Local search during inference: keep fallback jitter/random tries to escape local rejections (already implemented in make_submission.py).
    # - Small policy upgrade: replace linear policy with a small MLP for richer mu(x) mapping if linear is insufficient.
    #
    # Where to change code for the above:
    # - Reward: edit `env_simple.py` step() to adjust reward and overlap penalty.
    # - Exploration/std: edit `reinforce_agent.py` to change `init_std` or make `log_std` learnable.
    # - Curriculum/seeds: call train() with smaller n_trees first or provide `input.csv` seed placements for make_submission.
    # ------------------------------------------------
    for epoch in range(cfg.epochs):
        trajectories = []
        ep_returns = []
        for _ in range(cfg.batch_episodes):
            obs = env.reset()
            # English: Per-episode trajectory storage.
            # 中文：按回合存放的轨迹数据。
            #
            # Fields (解释):
            # - obs: observation / state vector at each timestep.
            #        It is a flat list: for each already-placed tree the tuple (x/max_coord, y/max_coord, deg/360),
            #        padded with zeros to length 3*n_trees, plus a final progress scalar index/n.
            #        用法：作为策略的输入，取值大致在 [-1,1]（坐标归一化）和 [0,1]（进度）。
            # - act: action executed in the environment (post-tanh), a list of 3 floats (ax,ay,adeg) in [-1,1].
            #        映射为真实值后： x = ax * max_coord, y = ay * max_coord, deg = adeg * 180.
            # - logp: log-probability of the sampled pre-tanh raw action under the current policy (useful for diagnostics).
            # - rew: scalar reward received after executing the action (per-step reward).
            #        In this env: typically - (s^2 / t) for a successful placement, or a hard penalty (e.g. -10) on overlap.
            # - raw: the pre-tanh sampled action (raw = mu + std * noise). Stored because learning uses gradients w.r.t. this Gaussian.
            # - mu: the deterministic mean action produced by the policy (mu = W @ obs + b) before sampling/noise.
            #
            # 中文要点总结：
            # - `obs` 是输入状态（归一化坐标与进度），`act` 是实际执行的动作（tanh 后），
            #   `raw` 与 `mu` 用于计算 REINFORCE 的梯度，`rew` 是一步的回报，`logp` 是动作的对数概率（诊断用）。
            traj = {"obs": [], "act": [], "logp": [], "rew": [], "raw": [], "mu": []}
            steps = 0
            done = False
            while steps < cfg.max_episode_steps:
                act, logp, policy_info = agent.select_action(obs)
                nobs, rew, done, step_info = env.step(act)
                traj["obs"].append(obs)
                traj["act"].append(act)
                traj["logp"].append(logp)
                traj["rew"].append(rew)
                # store raw action (pre-tanh) and mean for REINFORCE gradient
                traj["raw"].append(policy_info["raw"]) 
                traj["mu"].append(policy_info["mu"]) 
                obs = nobs
                steps += 1
                if done:
                    break
            trajectories.append(traj)
            ep_returns.append(sum(traj["rew"]))

        loss = agent.update(trajectories)
        print(f"Epoch {epoch+1}/{cfg.epochs} | Avg return: {mean(ep_returns):.3f} | GradNorm: {loss:.4f}")

    # Save parameters to JSON and PyTorch state_dict.
    # Ensure save_path is placed in the script directory unless absolute path provided.
    if not os.path.isabs(save_path):
        save_path = os.path.join(CURRENT_DIR, save_path)

    params = agent.pi.params()
    # determine json/pt paths
    if save_path.endswith('.pt'):
        pt_path = save_path
        json_path = save_path[:-3] + '.json'
    elif save_path.endswith('.json'):
        json_path = save_path
        pt_path = save_path[:-5] + '.pt'
    else:
        json_path = save_path + '.json'
        pt_path = save_path + '.pt'

    # ensure parent directory for json exists before writing
    json_dir = os.path.dirname(json_path)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(params, f)
    # Save torch state_dict (best-effort)
    try:
        pt_dir = os.path.dirname(pt_path)
        if pt_dir:
            os.makedirs(pt_dir, exist_ok=True)
        torch.save(agent.pi.state_dict(), pt_path)
    except Exception:
        pass
    print(f"Saved policy params to {json_path} and state_dict to {pt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # English: Main hyperparameters for reproducible training.
    # 中文：主要超参数，便于重复实验。
    parser.add_argument("--n_trees", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps_per_episode", type=int, default=512)
    parser.add_argument("--batch_episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--save_path", type=str, default="reinforce_policy.json")
    args = parser.parse_args()
#  python "train_basic_rl.py" --epochs 5 --batch_episodes 4 --steps_per_episode 256
    train(
        n_trees=args.n_trees,
        epochs=args.epochs,
        steps_per_episode=args.steps_per_episode,
        batch_episodes=args.batch_episodes,
        seed=args.seed,
        save_path=args.save_path,
    )
