"""
PPO (Proximal Policy Optimization) 近端策略优化
多模态决策优化核心模块，将内容生成/推荐视为决策任务
"""
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


@dataclass
class PPOExperience:
    """PPO经验数据"""
    states: torch.Tensor = None           # 多模态状态特征
    actions: torch.Tensor = None          # 决策动作
    log_probs: torch.Tensor = None        # 动作log概率
    rewards: torch.Tensor = None          # 奖励信号
    values: torch.Tensor = None           # 价值估计
    advantages: torch.Tensor = None       # 优势函数
    returns: torch.Tensor = None          # 回报
    attention_mask: torch.Tensor = None   # 掩码


class ValueNetwork(nn.Module):
    """
    价值网络 (Critic)
    估计多模态状态的价值
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states).squeeze(-1)


class PolicyNetwork(nn.Module):
    """
    策略网络 (Actor)
    基于多模态状态输出动作分布
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        action_dim: int = 256,
        continuous: bool = False,
    ):
        super().__init__()
        self.continuous = continuous

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        if continuous:
            self.mean_head = nn.Linear(hidden_dim // 2, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.action_head = nn.Linear(hidden_dim // 2, action_dim)

    def forward(self, states: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(states)

        if self.continuous:
            mean = self.mean_head(features)
            std = self.log_std.exp().expand_as(mean)
            return {"mean": mean, "std": std}
        else:
            logits = self.action_head(features)
            return {"logits": logits}

    def get_action(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作并返回log概率"""
        out = self.forward(states)

        if self.continuous:
            dist = torch.distributions.Normal(out["mean"], out["std"])
            if deterministic:
                action = out["mean"]
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=out["logits"])
            if deterministic:
                action = out["logits"].argmax(dim=-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob

    def evaluate_actions(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估给定状态-动作对的log概率和熵"""
        out = self.forward(states)

        if self.continuous:
            dist = torch.distributions.Normal(out["mean"], out["std"])
            log_prob = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=out["logits"])
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

        return log_prob, entropy


class PPOTrainer:
    """
    工业级PPO训练器
    特点：
    - 多模态奖励融合
    - GAE优势估计
    - 自适应KL惩罚
    - 经验回放机制
    - 梯度裁剪 + 混合精度
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reward_model: nn.Module,
        policy_network: Optional[PolicyNetwork] = None,
        value_network: Optional[ValueNetwork] = None,
        state_dim: int = 512,
        action_dim: int = 256,
        learning_rate: float = 1e-6,
        clip_range: float = 0.2,
        clip_range_value: float = 0.2,
        vf_coef: float = 0.1,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        gamma: float = 0.99,
        lam: float = 0.95,
        ppo_epochs: int = 4,
        mini_batch_size: int = 4,
        target_kl: float = 0.01,
        adap_kl_ctrl: bool = True,
        kl_coef: float = 0.2,
        device: str = "cuda",
    ):
        self.device = device
        self.clip_range = clip_range
        self.clip_range_value = clip_range_value
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.target_kl = target_kl
        self.adap_kl_ctrl = adap_kl_ctrl
        self.kl_coef = kl_coef
        self.global_step = 0

        # 多模态基座 (用于特征提取)
        self.policy_model = policy_model.to(device)
        self.reward_model = reward_model.to(device)
        self.reward_model.eval()

        # 参考策略 (KL约束)
        self.ref_policy = copy.deepcopy(policy_model).to(device)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad = False

        # Actor-Critic网络
        self.policy_net = (policy_network or PolicyNetwork(state_dim, action_dim=action_dim)).to(device)
        self.value_net = (value_network or ValueNetwork(state_dim)).to(device)

        # 优化器
        params = list(self.policy_net.parameters()) + list(self.value_net.parameters())
        trainable_policy_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        params.extend(trainable_policy_params)

        self.optimizer = torch.optim.AdamW(params, lr=learning_rate, eps=1e-5)

        # 经验缓冲
        self.experience_buffer = deque(maxlen=10000)

        logger.info(
            f"🎮 PPO训练器初始化: clip={clip_range}, gamma={gamma}, "
            f"lam={lam}, epochs={ppo_epochs}, target_kl={target_kl}"
        )

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GAE (Generalized Advantage Estimation) 优势函数估计
        A_t = Σ (γλ)^l * δ_{t+l}
        δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        if dones is None:
            dones = torch.zeros_like(rewards)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def collect_experience(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        user_features: Optional[torch.Tensor] = None,
        business_features: Optional[torch.Tensor] = None,
    ) -> PPOExperience:
        """采集经验数据"""
        self.policy_model.eval()

        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)

            # 获取多模态状态特征
            outputs = self.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                task="matching",
            )
            states = outputs.get("fused_features", outputs.get("text_features"))

            # 策略网络采样动作
            actions, log_probs = self.policy_net.get_action(states)

            # 价值估计
            values = self.value_net(states)

            # 奖励计算 (多模态奖励模型)
            reward_out = self.reward_model(
                states, user_features, business_features
            )
            rewards = reward_out["total_reward"]

        # GAE
        advantages, returns = self.compute_gae(rewards, values)

        experience = PPOExperience(
            states=states,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            values=values,
            advantages=advantages,
            returns=returns,
            attention_mask=attention_mask,
        )

        return experience

    def train_step(self, experience: PPOExperience) -> Dict[str, float]:
        """PPO单步训练（多个epoch的minibatch更新）"""
        self.policy_model.train()
        self.policy_net.train()
        self.value_net.train()

        states = experience.states
        actions = experience.actions
        old_log_probs = experience.log_probs
        advantages = experience.advantages
        returns = experience.returns
        old_values = experience.values

        # 标准化优势
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_metrics = {
            "policy_loss": 0, "value_loss": 0, "entropy": 0,
            "kl_div": 0, "clip_fraction": 0, "total_loss": 0,
        }
        num_updates = 0

        B = states.shape[0]
        for epoch in range(self.ppo_epochs):
            # Mini-batch迭代
            indices = torch.randperm(B, device=self.device)

            for start in range(0, B, self.mini_batch_size):
                end = min(start + self.mini_batch_size, B)
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]
                mb_old_values = old_values[mb_idx]

                # 评估当前策略
                new_log_probs, entropy = self.policy_net.evaluate_actions(mb_states, mb_actions)
                new_values = self.value_net(mb_states)

                # 策略损失 (Clipped PPO)
                ratio = torch.exp(new_log_probs - mb_old_logp)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值损失 (Clipped Value)
                value_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values,
                    -self.clip_range_value,
                    self.clip_range_value,
                )
                value_loss1 = F.mse_loss(new_values, mb_returns)
                value_loss2 = F.mse_loss(value_clipped, mb_returns)
                value_loss = torch.max(value_loss1, value_loss2)

                # 熵正则
                entropy_loss = -entropy.mean()

                # KL散度
                kl_div = (mb_old_logp - new_log_probs).mean()

                # 总损失
                total_loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    + self.entropy_coef * entropy_loss
                    + self.kl_coef * kl_div
                )

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # 统计
                clip_fraction = ((ratio - 1).abs() > self.clip_range).float().mean()
                total_metrics["policy_loss"] += policy_loss.item()
                total_metrics["value_loss"] += value_loss.item()
                total_metrics["entropy"] += entropy.mean().item()
                total_metrics["kl_div"] += kl_div.item()
                total_metrics["clip_fraction"] += clip_fraction.item()
                total_metrics["total_loss"] += total_loss.item()
                num_updates += 1

            # 提前停止: KL散度过大
            if self.adap_kl_ctrl:
                avg_kl = total_metrics["kl_div"] / max(num_updates, 1)
                if avg_kl > self.target_kl * 1.5:
                    logger.warning(f"⚠️ KL散度过大 ({avg_kl:.4f}), 提前停止epoch {epoch}")
                    break

        # 自适应KL系数调整
        if self.adap_kl_ctrl:
            avg_kl = total_metrics["kl_div"] / max(num_updates, 1)
            if avg_kl > self.target_kl * 2:
                self.kl_coef *= 1.5
            elif avg_kl < self.target_kl / 2:
                self.kl_coef *= 0.5
            self.kl_coef = max(0.01, min(10.0, self.kl_coef))

        self.global_step += 1

        return {
            k: v / max(num_updates, 1) for k, v in total_metrics.items()
        } | {
            "kl_coef": self.kl_coef,
            "global_step": self.global_step,
        }

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "kl_coef": self.kl_coef,
            "global_step": self.global_step,
        }, path)
        logger.info(f"💾 PPO检查点已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.value_net.load_state_dict(ckpt["value_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.kl_coef = ckpt["kl_coef"]
        self.global_step = ckpt["global_step"]
        logger.info(f"📂 PPO检查点已加载: {path}, step={self.global_step}")
