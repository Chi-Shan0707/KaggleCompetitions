"""
English (Overview):
    Minimal REINFORCE agent using a Gaussian linear policy (no NumPy/Torch).
    - Policy: a ~ tanh(N(Wx+b, std^2))
    - Update: score function (REINFORCE) with normalized returns as baseline.

中文（概览）：
    使用高斯线性策略的最小 REINFORCE（不依赖 NumPy/Torch）。
    - 策略：a ~ tanh(N(Wx+b, std^2))
    - 更新：得分函数（REINFORCE），回报标准化作为基线。

Plain-language summary / 朴素说明：
    This agent is intentionally minimal and educational. It learns by trial-and-error:
    - Observe state -> compute a linear mean action (Wx+b) -> sample noisy action -> apply action.
    - After collecting episodes, compute returns and use the REINFORCE score-function gradient
      to nudge parameters so that actions that led to higher returns become more probable.

    该智能体为教学示例，尽量精简。学习流程：
    - 观察状态 -> 线性计算动作均值 (Wx+b) -> 从高斯分布采样带噪动作 -> 执行动作。
    - 收集若干回合后，计算回报并用 REINFORCE 的得分函数梯度更新参数，使得高回报对应的动作被更频繁采样。
"""
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim

@dataclass
class ReinforceConfig:
    lr: float = 1e-2
    gamma: float = 0.99
    max_episode_steps: int = 2000
    batch_episodes: int = 8
    epochs: int = 50
    init_std: float = 0.5


class GaussianLinearPolicy(nn.Module):
    """Minimal PyTorch Gaussian linear policy with tanh squashing.

    - mu = Linear(obs)
    - raw ~ N(mu, std^2)
    - act = tanh(raw)
    """

    def __init__(self, obs_dim: int, act_dim: int, init_std: float = 0.5, lr: float = 1e-2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.net = nn.Linear(obs_dim, act_dim)
        # initialize weights similar to original random.uniform(-0.1,0.1)
        nn.init.uniform_(self.net.weight, -0.1, 0.1)
        nn.init.constant_(self.net.bias, 0.0)
        # log_std as a learnable parameter (kept small by default)
        self.log_std = nn.Parameter(torch.ones(act_dim) * math.log(init_std))
        self.lr = lr

    def params(self) -> Dict[str, List]:
        """Export parameters as plain lists for JSON saving."""
        W = self.net.weight.detach().cpu().tolist()
        b = self.net.bias.detach().cpu().tolist()
        log_std = self.log_std.detach().cpu().tolist()
        return {"W": W, "b": b, "log_std": log_std}

    def mean(self, obs: List[float]) -> List[float]:
        device = next(self.parameters()).device
        x = torch.tensor(obs, dtype=torch.float32, device=device)
        mu = self.net(x)
        return mu.detach().cpu().tolist()

    def sample(self, obs: List[float]) -> Tuple[List[float], List[float], List[float]]:
        device = next(self.parameters()).device
        x = torch.tensor(obs, dtype=torch.float32, device=device)
        mu = self.net(x)
        std = torch.exp(self.log_std)
        raw = mu + std * torch.randn_like(mu)
        act = torch.tanh(raw)
        return act.detach().cpu().tolist(), raw.detach().cpu().tolist(), mu.detach().cpu().tolist()

    def log_prob_raw(self, raw: List[float], mu: List[float]) -> float:
        device = next(self.parameters()).device
        raw_t = torch.tensor(raw, dtype=torch.float32, device=device)
        mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
        std = torch.exp(self.log_std)
        var = std * std
        # gaussian log prob for diagonal covariance
        out = -0.5 * (((raw_t - mu_t) ** 2) / var + 2 * self.log_std + math.log(2 * math.pi))
        return float(out.sum().detach().cpu().item())

    def to_device(self, device: torch.device):
        self.to(device)


class ReinforceAgent:
    """
    English: The main agent class that interacts with the environment and learns.

    中文：主智能体类，负责与环境交互并学习。
    """
    def __init__(self, obs_dim: int, act_dim: int, cfg: ReinforceConfig):
        self.cfg = cfg
        self.pi = GaussianLinearPolicy(obs_dim, act_dim, init_std=cfg.init_std, lr=cfg.lr)
        self.optimizer = optim.Adam(self.pi.parameters(), lr=cfg.lr)

    def select_action(self, obs: List[float]):
        """
        English: Return (action, logp, info) with raw and mu for learning.

        中文：返回 (action, logp, info)，其中包含学习所需的 raw 与 mu。

        Plain explanation:
        - The agent samples an action using the policy (with exploration noise).
        - It computes the log-probability of the sampled raw action under the current policy.
        - Returns the action (post-tanh), the log-prob (for logging if desired), and extra info used for learning.

        朴素解释：
        - 智能体用策略（含探索噪声）采样动作。
        - 计算该 raw 动作在当前策略下的对数概率。
        - 返回动作（tanh 后）、对数概率以及用于学习的额外信息。
        """
        act, raw, mu = self.pi.sample(obs)
        logp = self.pi.log_prob_raw(raw, mu)
        return act, float(logp), {"raw": raw, "mu": mu}

    def update(self, trajectories: List[Dict]):
        """
        English: REINFORCE update with normalized returns as baseline.

        中文：使用回报标准化作为基线的 REINFORCE 参数更新。

        Step-by-step (plain language):
        1. For each trajectory (episode), compute the return G_t for each timestep (discounted sum of future rewards).
        2. Collect all returns across the batch and compute mean/std to standardize them. Standardization reduces gradient variance.
        3. For each timestep, compute the gradient of the log-probability of the sampled raw action w.r.t mu (g_mu).
           Using chain rule, d logp / d W = g_mu * obs, and d logp / d b = g_mu * 1.
        4. Scale these gradients by the standardized return (higher return -> larger positive weight), and accumulate.
        5. After processing the batch, apply the accumulated gradients to update W and b (gradient ascent).

        逐步说明（通俗）：
        1. 对每条轨迹计算每步的回报 G_t（折扣和）。
        2. 将本批次所有回报收集并计算均值/标准差以标准化，减少方差。
        3. 对每一步，计算采样到的 raw 在均值 mu 下的对数概率关于 mu 的梯度 g_mu。
           根据链式法则，d logp / d W = g_mu * obs，d logp / d b = g_mu。
        4. 用标准化回报作为权重放大或缩小这些梯度（回报越高，梯度影响越大），并累加。
        5. 处理完整个批次后，把累积梯度应用到参数上（梯度上升）。

        Implementation details:
        - The standard deviation (std) is kept fixed for stability; we only update W and b here.
        - The function returns a simple gradient-norm scalar for logging.

        细节说明：
        - 为了稳定，只固定 std，不更新它；这里只更新 W 和 b。
        - 函数返回一个梯度范数的标量，便于日志记录。
        """
        # Compute episode returns (discounted) - same as before
        all_returns = []
        for traj in trajectories:
            rews = traj["rew"]
            G = 0.0
            ret = []
            for r in reversed(rews):
                G = r + self.cfg.gamma * G
                ret.append(G)
            ret.reverse()
            traj["ret"] = ret
            all_returns.extend(ret)
        # Normalize returns as baseline
        mean_R = sum(all_returns)/max(1, len(all_returns))
        var_R = sum((r-mean_R)**2 for r in all_returns)/max(1, len(all_returns))
        std_R = math.sqrt(var_R + 1e-8)

        # Build loss using torch (negative log-prob weighted by standardized returns)
        device = next(self.pi.parameters()).device
        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        idx = 0
        for traj in trajectories:
            for t in range(len(traj["obs"])):
                obs = traj["obs"][t]
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                # recompute mu from current policy network to get proper gradients
                mu_t = self.pi.net(obs_t)
                raw_t = torch.tensor(traj["raw"][t], dtype=torch.float32, device=device)
                std = torch.exp(self.pi.log_std)
                var = std * std
                out = -0.5 * (((raw_t - mu_t) ** 2) / var + 2 * self.pi.log_std + math.log(2 * math.pi))
                logp = out.sum()
                Rn = (all_returns[idx] - mean_R) / std_R
                Rn_t = torch.tensor(Rn, dtype=torch.float32, device=device)
                loss = loss + (-logp * Rn_t)
                idx += 1

        if idx > 0:
            loss = loss / float(idx)

        self.optimizer.zero_grad()
        loss.backward()
        # compute simple grad-norm for logging
        gn = 0.0
        for p in self.pi.parameters():
            if p.grad is not None:
                gn += float(p.grad.abs().sum().detach().cpu().item())
        self.optimizer.step()
        return float(gn)

