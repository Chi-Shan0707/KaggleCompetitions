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


def train(n_trees: int = 10, sides: int = 3, radius: float = 1.0, epochs: int = 50,
          steps_per_episode: int = 512, batch_episodes: int = 8, seed: int = 42,
          save_path: str = "RL/reinforce_policy.pt"):
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

    for epoch in range(cfg.epochs):
        trajectories = []
        ep_returns = []
        for _ in range(cfg.batch_episodes):
            obs = env.reset()
            # English: Per-episode trajectory storage.
            # 中文：按回合存放的轨迹数据。
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

    # Save parameters to JSON for portability
    params = agent.pi.params()
    with open(save_path, "w") as f:
        json.dump(params, f)
    print(f"Saved policy params to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # English: Main hyperparameters for reproducible training.
    # 中文：主要超参数，便于重复实验。
    parser.add_argument("--n_trees", type=int, default=10)
    parser.add_argument("--sides", type=int, default=3)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_episode", type=int, default=512)
    parser.add_argument("--batch_episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_path", type=str, default="RL/reinforce_policy.json")
    args = parser.parse_args()

    train(
        n_trees=args.n_trees,
        sides=args.sides,
        radius=args.radius,
        epochs=args.epochs,
        steps_per_episode=args.steps_per_episode,
        batch_episodes=args.batch_episodes,
        seed=args.seed,
        save_path=args.save_path,
    )
