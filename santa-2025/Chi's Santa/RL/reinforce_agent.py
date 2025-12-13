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


@dataclass
class ReinforceConfig:
    lr: float = 1e-2
    gamma: float = 0.99
    max_episode_steps: int = 2000
    batch_episodes: int = 8
    epochs: int = 50
    init_std: float = 0.5


class GaussianLinearPolicy:
    """
    English: Minimal Gaussian policy, tanh-squashed to [-1,1]. Stdlib only.

    中文：最小高斯策略，tanh 压缩到 [-1,1]，仅用标准库实现。

    How the policy maps observations to actions (plain language):
    - Compute mean action mu = W x + b (a simple linear mapping).
    - Add Gaussian noise (std fixed) to mu to produce a raw action (exploration).
    - Squash the raw action with tanh so outputs lie in [-1,1].

    策略如何将观测映射为动作（通俗）：
    - 计算动作为 mu = W x + b（线性映射）。
    - 对 mu 加高斯噪声（固定方差）得到 raw（引入探索）。
    - 用 tanh 压缩 raw，使动作落在 [-1,1]。
    """

    def __init__(self, obs_dim: int, act_dim: int, init_std: float = 0.5, lr: float = 1e-2):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        # Initialize small random weights
        self.W = [[random.uniform(-0.1, 0.1) for _ in range(obs_dim)] for _ in range(act_dim)]
        self.b = [0.0 for _ in range(act_dim)]
        self.log_std = [math.log(init_std) for _ in range(act_dim)]
        self.lr = lr

    def params(self) -> Dict[str, List]:
        """English: Export parameters as plain lists for JSON saving.

        中文：将参数导出为普通列表，便于保存为 JSON。
        """
        return {"W": self.W, "b": self.b, "log_std": self.log_std}

    def _dot(self, a: List[float], b: List[float]) -> float:
        return sum(x*y for x, y in zip(a, b))

    def mean(self, obs: List[float]) -> List[float]:
        """
        English: Compute action mean mu = W x + b.

        中文：计算动作均值 mu = W x + b。
        - This produces the 'typical' action for the current observation before adding noise.
        - 该函数在加入噪声前给出当前观测下的“典型”动作。
        """
        return [self._dot(w_row, obs) + self.b[j] for j, w_row in enumerate(self.W)]

    def sample(self, obs: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        English: Sample raw ~ N(mu, std^2) then act = tanh(raw).

        中文：先采样 raw ~ N(mu, std^2)，再取 act = tanh(raw)。
        - Returns: (act, raw, mu) where raw is pre-tanh, mu is deterministic mean.
        - raw is used in learning because tanh is non-linear and we compute gradients w.r.t. the Gaussian.
        """
        mu = self.mean(obs)
        std = [math.exp(ls) for ls in self.log_std]
        raw = [mu[j] + std[j] * random.gauss(0.0, 1.0) for j in range(self.act_dim)]
        act = [math.tanh(v) for v in raw]
        return act, raw, mu

    def log_prob_raw(self, raw: List[float], mu: List[float]) -> float:
        """
        English: Log-prob of raw under diagonal Gaussian N(mu, std^2).

        中文：对角高斯 N(mu, std^2) 下 raw 的对数概率。
        - This is used to evaluate how (a) likely an observed raw sample was under the current policy.
        - 在学习时用来评估在当前策略下采样到该 raw 的可能性。
        """
        std = [math.exp(ls) for ls in self.log_std]
        out = 0.0
        for j in range(self.act_dim):
            var = std[j] * std[j]
            out += -0.5 * (((raw[j] - mu[j]) ** 2) / var + 2 * self.log_std[j] + math.log(2 * math.pi))
        return out

    def grad_logprob_wrt_mu(self, raw: List[float], mu: List[float]) -> List[float]:
        """
        English: d log N(raw; mu, std^2) / d mu = (raw - mu) / std^2.

        中文：d log N(raw; mu, std^2) / d mu = (raw - mu) / std^2。
        - This gradient tells how to change mu to increase the log-probability of the sampled raw.
        - 该梯度表示如何改变 mu 以提高采样到 raw 的对数概率。
        """
        std = [math.exp(ls) for ls in self.log_std]
        return [ (raw[j] - mu[j]) / (std[j]*std[j]) for j in range(self.act_dim) ]

    def update(self, grads: Dict[str, List[List[float]]] ):
        """
        English: Gradient ascent on W and b.

        中文：对 W 与 b 做梯度上升更新。
        - Apply a simple gradient ascent step: param += lr * grad.
        - 直接使用梯度上升更新参数：param += lr * grad。
        """
        # Gradient ascent
        for j in range(self.act_dim):
            for i in range(self.obs_dim):
                self.W[j][i] += self.lr * grads["W"][j][i]
            self.b[j] += self.lr * grads["b"][j]


class ReinforceAgent:
    """
    English: The main agent class that interacts with the environment and learns.

    中文：主智能体类，负责与环境交互并学习。
    """
    def __init__(self, obs_dim: int, act_dim: int, cfg: ReinforceConfig):
        self.cfg = cfg
        self.pi = GaussianLinearPolicy(obs_dim, act_dim, init_std=cfg.init_std, lr=cfg.lr)

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
        # Compute episode returns and policy gradients (score function) for W,b only (keep std fixed for stability)
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

        # Initialize grads
        W_grad = [[0.0 for _ in range(self.pi.obs_dim)] for _ in range(self.pi.act_dim)]
        b_grad = [0.0 for _ in range(self.pi.act_dim)]

        idx = 0
        for traj in trajectories:
            for t in range(len(traj["obs"])):
                obs = traj["obs"][t]
                raw = traj["raw"][t]
                mu = traj["mu"][t]
                g_mu = self.pi.grad_logprob_wrt_mu(raw, mu)  # d logp / d mu
                # Chain rule: d mu / d W = obs, d mu / d b = 1
                Rn = (all_returns[idx] - mean_R) / std_R
                for j in range(self.pi.act_dim):
                    for i in range(self.pi.obs_dim):
                        W_grad[j][i] += g_mu[j] * obs[i] * Rn
                    b_grad[j] += g_mu[j] * Rn
                idx += 1

        self.pi.update({"W": W_grad, "b": b_grad})
        # Return simple scalar for logging (gradient norm)
        gn = sum(abs(x) for row in W_grad for x in row) + sum(abs(x) for x in b_grad)
        return float(gn)

