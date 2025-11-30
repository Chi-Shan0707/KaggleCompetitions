"""
English Summary:
 Minimal PPO for the packing environment. Emphasis on readability over performance.
 - Actor-Critic network: shared trunk + separate policy (mu, log_std) and value head.
 - Continuous actions in [-1,1]^3 enforced by tanh.
 - Uses GAE for advantage, clipping for stable policy updates, and entropy bonus.
Chinese Summary（中文）：
 针对打包环境的最小 PPO，实现重点在可读性而非性能。
 - 使用共享干路 + 策略/价值分支的 Actor-Critic 网络。
 - 连续动作范围通过 tanh 限制在 [-1,1]^3。
 - 使用 GAE 优势估计、剪切稳定策略更新，并加入熵奖励。
"""
import math    # EN: Reserved for potential math ops. CN：为可能的数学操作保留。
import time    # EN: Can be used for profiling epochs. CN：可用于周期性能分析。
from dataclasses import dataclass  # EN: Lightweight config container. CN：轻量级配置容器。
from typing import List, Tuple     # EN: Type hints improve clarity. CN：类型提示提高可读性。

import torch                    # EN: Core tensor library. CN：核心张量库。
import torch.nn as nn           # EN: Neural network components. CN：神经网络组件。
import torch.optim as optim     # EN: Optimizers for training. CN：训练优化器。

from ppo_env import PackingEnv   # EN: Our environment definition. CN：环境定义导入。

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # EN: Use GPU if available. CN：优先使用 GPU。

@dataclass
class PPOConfig:  # EN: Configuration hyperparameters for PPO. CN：PPO 的超参数配置类。
    steps_per_epoch: int = 1024   # EN: Environment steps collected each epoch. CN：每轮收集的环境步数。
    epochs: int = 20              # EN: Total training epochs. CN：训练轮数。
    gamma: float = 0.99           # EN: Discount factor. CN：折扣因子。
    lam: float = 0.95             # EN: GAE lambda for bias/variance trade-off. CN：GAE λ 平衡偏差与方差。
    clip_ratio: float = 0.2       # EN: PPO clipping threshold. CN：PPO 剪切阈值。
    pi_lr: float = 3e-4           # EN: Policy learning rate. CN：策略学习率。
    v_lr: float = 1e-3            # EN: Value function learning rate. CN：价值网络学习率。
    train_iters: int = 10         # EN: Policy/value update passes per epoch. CN：每轮策略/价值更新次数。
    target_kl: float = 0.02       # EN: Early stop threshold on KL divergence. CN：KL 触发提前停止阈值。
    entropy_coef: float = 0.01    # EN: Encourage exploration. CN：熵系数鼓励探索。
    value_coef: float = 0.5       # EN: Scale value loss contribution. CN：价值损失权重。
    batch_size: int = 256         # EN: Mini-batch size for SGD. CN：小批量大小。

class ActorCritic(nn.Module):  # EN: Combined policy/value network. CN：联合策略/价值网络。
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        hidden = 128  # EN: Hidden layer width. CN：隐藏层宽度。
        self.shared = nn.Sequential(  # EN: Shared feature extractor. CN：共享特征提取层。
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, act_dim)           # EN: Outputs mean of Gaussian. CN：高斯分布均值。
        self.log_std = nn.Parameter(torch.zeros(act_dim))   # EN: Trainable log std deviation. CN：可训练对数标准差。
        self.v_head = nn.Linear(hidden, 1)                  # EN: Value prediction. CN：价值预测。

    def forward(self, obs):  # EN: Compute policy parameters & value. CN：计算策略参数与价值。
        h = self.shared(obs)      # EN: Shared features. CN：共享特征。
        mu = self.mu_head(h)      # EN: Action mean. CN：动作均值。
        std = torch.exp(self.log_std)  # EN: Convert log_std to std. CN：对数标准差取指数。
        v = self.v_head(h)        # EN: State value. CN：状态价值。
        return mu, std, v

    def act(self, obs):  # EN: Sample action + return log prob & value. CN：采样动作并返回对数概率与价值。
        with torch.no_grad():
            mu, std, v = self.forward(obs)
            dist = torch.distributions.Normal(mu, std)  # EN: Factorized Normal distribution. CN：分量独立正态分布。
            a = dist.sample()                          # EN: Sample raw action. CN：采样原始动作。
            logp = dist.log_prob(a).sum(-1)            # EN: Sum log probs across dims. CN：各维度对数概率求和。
        return torch.tanh(a), logp, v.squeeze(-1)      # EN: Squash to [-1,1]. CN：tanh 压缩到 [-1,1]。

    def evaluate(self, obs, act):  # EN: Evaluate log prob & value for given actions. CN：评估给定动作的对数概率与价值。
        mu, std, v = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        # EN: Invert tanh (approx) to map squashed action back before log prob.
        # CN：近似反转 tanh 以获取原始未压缩动作再计算对数概率。
        raw_act = torch.atanh(torch.clamp(act, -0.999, 0.999))
        logp = dist.log_prob(raw_act).sum(-1)          # EN: New log probs. CN：新对数概率。
        entropy = dist.entropy().sum(-1)               # EN: Entropy encourages exploration. CN：熵用于鼓励探索。
        return logp, v.squeeze(-1), entropy

class RolloutBuffer:  # EN: Stores trajectory data for PPO updates. CN：存储轨迹数据用于 PPO 更新。
    def __init__(self):
        self.obs = []   # EN: Observations. CN：观察值。
        self.act = []   # EN: Actions taken. CN：已执行动作。
        self.rew = []   # EN: Rewards received. CN：获得的奖励。
        self.val = []   # EN: Value predictions. CN：价值预测。
        self.logp = []  # EN: Log probabilities of actions. CN：动作的对数概率。
        self.done = []  # EN: Episode termination flags. CN：终止标记。

    def add(self, obs, act, rew, val, logp, done):  # EN: Append one timestep. CN：添加单步数据。
        self.obs.append(obs)
        self.act.append(act)
        self.rew.append(rew)
        self.val.append(val)
        self.logp.append(logp)
        self.done.append(done)

    def compute_advantages(self, gamma, lam):  # EN: GAE advantage computation. CN：GAE 优势计算。
        adv = []
        gae = 0
        values = self.val + [0]  # EN: Bootstrap with zero at end. CN：终止位置引导值设为0。
        for t in reversed(range(len(self.rew))):  # EN: Reverse-time loop for GAE. CN：反向迭代计算 GAE。
            delta = self.rew[t] + gamma * (0 if self.done[t] else values[t+1]) - values[t]  # EN: TD error. CN：时序差分误差。
            gae = delta + gamma * lam * (0 if self.done[t] else gae)  # EN: Recursive accumulation. CN：递归累积。
            adv.insert(0, gae)  # EN: Prepend to keep order. CN：插入前端保持时间顺序。
        returns = [a + v for a, v in zip(adv, self.val)]  # EN: Advantage + value = return. CN：优势 + 价值 = 回报。
        adv_tensor = torch.tensor(adv, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)  # EN: Normalize advantages. CN：归一化优势。
        return adv_tensor, returns_tensor


def ppo_train(env: PackingEnv, cfg: PPOConfig):  # EN: Main training loop. CN：主训练循环。
    sample_obs = env.reset()              # EN: Obtain initial observation. CN：获取初始观察。
    obs_dim = len(sample_obs)             # EN: Observation dimension. CN：状态维度。
    act_dim = 3                           # EN: Action dimension fixed to 3. CN：动作维度固定为 3。
    ac = ActorCritic(obs_dim, act_dim).to(device)  # EN: Initialize network. CN：初始化网络。
    pi_opt = optim.Adam(ac.parameters(), lr=cfg.pi_lr)  # EN: Policy optimizer. CN：策略优化器。
    v_opt = optim.Adam(ac.parameters(), lr=cfg.v_lr)    # EN: Value optimizer (shared params included). CN：价值优化器。

    for epoch in range(cfg.epochs):      # EN: Epoch loop. CN：轮次循环。
        buf = RolloutBuffer()            # EN: Fresh buffer. CN：新建采样缓冲。
        obs = torch.tensor(env.reset(), dtype=torch.float32).to(device)  # EN: Reset environment. CN：重置环境。
        ep_rewards = 0.0
        for step in range(cfg.steps_per_epoch):  # EN: Collect trajectories. CN：收集轨迹。
            act, logp, val = ac.act(obs.unsqueeze(0))  # EN: Sample action. CN：采样动作。
            act_np = act.squeeze(0).cpu().numpy()      # EN: Detach to numpy. CN：转为 numpy。
            nobs_list, reward, done, info = env.step(act_np)  # EN: Environment transition. CN：环境步进。
            ep_rewards += reward
            buf.add(obs.cpu().numpy(), act_np, reward, val.item(), logp.item(), done)  # EN: Store timestep. CN：存储步数据。
            obs = torch.tensor(nobs_list, dtype=torch.float32).to(device)  # EN: Move to next state. CN：进入下一状态。
            if done:  # EN: If episode ends, reset to continue filling batch. CN：若回合结束，重置继续采样。
                obs = torch.tensor(env.reset(), dtype=torch.float32).to(device)
        adv, ret = buf.compute_advantages(cfg.gamma, cfg.lam)  # EN: Compute advantages/returns. CN：计算优势与回报。
        obs_batch = torch.tensor(buf.obs, dtype=torch.float32).to(device)
        act_batch = torch.tensor(buf.act, dtype=torch.float32).to(device)
        logp_old = torch.tensor(buf.logp, dtype=torch.float32).to(device)
        ret_batch = ret.to(device)
        adv_batch = adv.to(device)

        for _ in range(cfg.train_iters):  # EN: Multiple optimization passes. CN：多次优化迭代。
            idx = torch.randperm(len(obs_batch))  # EN: Shuffle indices. CN：打乱索引。
            for i in range(0, len(idx), cfg.batch_size):  # EN: Mini-batch loop. CN：小批量循环。
                batch_idx = idx[i:i+cfg.batch_size]
                ob = obs_batch[batch_idx]
                act_b = act_batch[batch_idx]
                logp_b_old = logp_old[batch_idx]
                adv_b = adv_batch[batch_idx]
                ret_b = ret_batch[batch_idx]

                logp_new, v_pred, entropy = ac.evaluate(ob, act_b)  # EN: Evaluate batch. CN：评估当前批。
                ratio = torch.exp(logp_new - logp_b_old)            # EN: Probability ratio. CN：概率比率。
                clip_adv = torch.clamp(ratio, 1-cfg.clip_ratio, 1+cfg.clip_ratio) * adv_b  # EN: Clipped advantage. CN：剪切后的优势。
                pi_loss = -(torch.min(ratio * adv_b, clip_adv) + cfg.entropy_coef * entropy).mean()  # EN: Policy loss with entropy. CN：策略损失含熵项。
                v_loss = ((v_pred - ret_b)**2).mean() * cfg.value_coef  # EN: Scaled value MSE. CN：价值均方误差（加权）。

                pi_opt.zero_grad(); pi_loss.backward(); pi_opt.step()  # EN: Update policy. CN：更新策略。
                v_opt.zero_grad(); v_loss.backward(); v_opt.step()     # EN: Update value. CN：更新价值。

            approx_kl = (logp_b_old - logp_new).mean().item()  # EN: KL estimate. CN：KL 近似值。
            if approx_kl > cfg.target_kl: break                # EN: Early stop if divergence too high. CN：KL 超阈值提前停止。

        print(f"Epoch {epoch} | Mean reward per batch: {ep_rewards/cfg.steps_per_epoch:.3f}")  # EN: Training progress. CN：训练进度输出。

    return ac  # EN: Return trained model. CN：返回训练后的模型。

if __name__ == '__main__':
    # EN: Basic training invocation for quick test. CN：快速测试的基本训练调用。
    env = PackingEnv(n_trees=5, radius=1.0, sides=3)
    cfg = PPOConfig(steps_per_epoch=512, epochs=5)
    model = ppo_train(env, cfg)
    torch.save(model.state_dict(), 'ppo_model.pt')  # EN: Persist weights. CN：保存模型权重。
    print('Saved model to ppo_model.pt')
