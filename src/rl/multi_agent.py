"""
多智能体强化学习扩展 (MARL)
生成智能体、推荐智能体、排序智能体协同决策
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional
from loguru import logger


class ContentAgent(nn.Module):
    """内容生成智能体"""

    def __init__(self, state_dim: int = 512, action_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.name = "content_agent"

        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        action_logits = self.policy(state)
        value = self.value(state).squeeze(-1)
        return {"action_logits": action_logits, "value": value}

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        out = self.forward(state)
        dist = torch.distributions.Categorical(logits=out["action_logits"])
        action = out["action_logits"].argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), out["value"]


class RecommendationAgent(nn.Module):
    """推荐排序智能体"""

    def __init__(self, state_dim: int = 512, num_items: int = 100, hidden_dim: int = 512):
        super().__init__()
        self.name = "recommendation_agent"

        self.scorer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_items),
        )
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        scores = self.scorer(state)
        value = self.value(state).squeeze(-1)
        return {"scores": scores, "value": value}

    def get_ranking(self, state: torch.Tensor, top_k: int = 10) -> torch.Tensor:
        out = self.forward(state)
        _, top_indices = out["scores"].topk(top_k, dim=-1)
        return top_indices


class RankingAgent(nn.Module):
    """精排智能体"""

    def __init__(self, state_dim: int = 512, item_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.name = "ranking_agent"

        self.cross_scorer = nn.Sequential(
            nn.Linear(state_dim + item_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user_state: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_state: [B, state_dim]
            item_features: [B, N, item_dim] N个候选物品
        """
        B, N, D = item_features.shape
        user_expanded = user_state.unsqueeze(1).expand(-1, N, -1)  # [B, N, state_dim]
        combined = torch.cat([user_expanded, item_features], dim=-1)  # [B, N, state+item]
        scores = self.cross_scorer(combined).squeeze(-1)  # [B, N]
        return scores


class MultiAgentSystem(nn.Module):
    """
    多智能体协同决策系统
    生成 -> 推荐 -> 精排 三级联动
    """

    def __init__(
        self,
        state_dim: int = 512,
        action_dim: int = 256,
        num_items: int = 100,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.content_agent = ContentAgent(state_dim, action_dim, hidden_dim)
        self.rec_agent = RecommendationAgent(state_dim, num_items, hidden_dim)
        self.rank_agent = RankingAgent(state_dim, action_dim, hidden_dim)

        # 智能体间通信模块
        self.communication = AgentCommunication(state_dim, hidden_dim)

        logger.info(f"🤖 多智能体系统: 3 agents, state_dim={state_dim}")

    def forward(
        self,
        user_state: torch.Tensor,
        item_features: Optional[torch.Tensor] = None,
        mode: str = "full_pipeline",
    ) -> Dict[str, Any]:
        """
        多智能体协同决策
        mode: full_pipeline / generate_only / recommend_only
        """
        results = {}

        if mode in ("full_pipeline", "generate_only"):
            # 生成智能体
            content_out = self.content_agent(user_state)
            results["content_actions"] = content_out["action_logits"]
            results["content_value"] = content_out["value"]

        if mode in ("full_pipeline", "recommend_only"):
            # 推荐智能体（融入生成智能体的信号）
            if "content_actions" in results:
                enhanced_state = self.communication(
                    user_state,
                    content_out["action_logits"],
                    source="content",
                )
            else:
                enhanced_state = user_state

            rec_out = self.rec_agent(enhanced_state)
            results["rec_scores"] = rec_out["scores"]
            results["rec_value"] = rec_out["value"]
            results["top_items"] = self.rec_agent.get_ranking(enhanced_state)

        if mode == "full_pipeline" and item_features is not None:
            # 精排智能体
            rank_scores = self.rank_agent(user_state, item_features)
            results["rank_scores"] = rank_scores

        return results


class AgentCommunication(nn.Module):
    """智能体间通信模块"""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.transform = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
        )

    def forward(
        self,
        receiver_state: torch.Tensor,
        sender_signal: torch.Tensor,
        source: str = "content",
    ) -> torch.Tensor:
        # 将sender_signal投影到与receiver_state相同维度
        if sender_signal.shape[-1] != receiver_state.shape[-1]:
            # 简单均值池化对齐
            sender_signal = F.adaptive_avg_pool1d(
                sender_signal.unsqueeze(1), receiver_state.shape[-1]
            ).squeeze(1)

        combined = torch.cat([receiver_state, sender_signal], dim=-1)
        gate = self.gate(combined)
        message = self.transform(combined)
        return receiver_state + gate * message
