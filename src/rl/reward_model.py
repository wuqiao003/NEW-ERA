"""
多模态奖励模型 (Reward Model)
融合文本语义、图像质量、用户反馈、业务指标的多维奖励信号
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


class MultiDimensionalRewardHead(nn.Module):
    """
    多维奖励评分头
    输出多个维度的奖励分数:
      - 内容质量分 (content_quality)
      - 用户偏好分 (user_preference)
      - 业务合规分 (business_compliance)
      - 相关性分 (relevance)
    """

    def __init__(self, input_dim: int, num_heads: int = 4, hidden_dim: int = 512):
        super().__init__()
        self.num_heads = num_heads

        # 共享特征提取
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_dim),
        )

        # 每个维度独立的评分头
        self.reward_heads = nn.ModuleDict({
            "content_quality": self._build_head(hidden_dim),
            "user_preference": self._build_head(hidden_dim),
            "business_compliance": self._build_head(hidden_dim),
            "relevance": self._build_head(hidden_dim),
        })

    def _build_head(self, hidden_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, D] 多模态融合特征
        Returns:
            各维度奖励分数和加权总分
        """
        shared = self.shared_layer(features)

        scores = {}
        for name, head in self.reward_heads.items():
            scores[name] = head(shared).squeeze(-1)  # [B]

        return scores


class MultimodalRewardModel(nn.Module):
    """
    工业级多模态奖励模型
    输入: 多模态内容表征 + 用户反馈特征 + 业务指标特征
    输出: 多维奖励分数 + 加权综合奖励
    """

    def __init__(
        self,
        multimodal_dim: int = 512,
        user_feature_dim: int = 64,
        business_feature_dim: int = 32,
        hidden_size: int = 1024,
        num_reward_heads: int = 4,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.multimodal_dim = multimodal_dim
        self.hidden_size = hidden_size
        self.user_feature_dim = user_feature_dim
        self.business_feature_dim = business_feature_dim

        # 默认奖励权重
        self.reward_weights = reward_weights or {
            "content_quality": 0.3,
            "user_preference": 0.3,
            "business_compliance": 0.2,
            "relevance": 0.2,
        }

        # 特征融合：将多模态特征、用户特征、业务特征合并
        total_input_dim = multimodal_dim + user_feature_dim + business_feature_dim
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # 多维奖励头
        self.reward_head = MultiDimensionalRewardHead(
            input_dim=hidden_size,
            num_heads=num_reward_heads,
        )

        # 可学习的奖励权重（动态调整）
        self.learnable_weights = nn.Parameter(
            torch.tensor([self.reward_weights.get(k, 0.25) for k in [
                "content_quality", "user_preference", "business_compliance", "relevance"
            ]])
        )

        # 用户特征编码器
        self.user_encoder = nn.Sequential(
            nn.Linear(user_feature_dim, user_feature_dim),
            nn.GELU(),
            nn.LayerNorm(user_feature_dim),
        )

        # 业务特征编码器
        self.business_encoder = nn.Sequential(
            nn.Linear(business_feature_dim, business_feature_dim),
            nn.GELU(),
            nn.LayerNorm(business_feature_dim),
        )

        logger.info(
            f"🎯 多模态奖励模型初始化: mm_dim={multimodal_dim}, "
            f"hidden={hidden_size}, heads={num_reward_heads}"
        )

    def forward(
        self,
        multimodal_features: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        business_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            multimodal_features: [B, mm_dim] 多模态融合特征
            user_features: [B, user_dim] 用户行为特征
            business_features: [B, biz_dim] 业务指标特征
        Returns:
            reward_scores: 各维度奖励分数
            total_reward: 加权综合奖励
        """
        B = multimodal_features.shape[0]
        device = multimodal_features.device

        # 编码用户特征（不存在则用零向量）
        if user_features is None:
            user_features = torch.zeros(B, self.user_feature_dim, device=device)
        user_encoded = self.user_encoder(user_features)

        # 编码业务特征
        if business_features is None:
            business_features = torch.zeros(B, self.business_feature_dim, device=device)
        business_encoded = self.business_encoder(business_features)

        # 特征融合
        combined = torch.cat([multimodal_features, user_encoded, business_encoded], dim=-1)
        fused = self.feature_fusion(combined)

        # 多维奖励评分
        reward_scores = self.reward_head(fused)

        # 加权综合奖励
        weights = F.softmax(self.learnable_weights, dim=0)
        score_list = [
            reward_scores["content_quality"],
            reward_scores["user_preference"],
            reward_scores["business_compliance"],
            reward_scores["relevance"],
        ]
        total_reward = sum(w * s for w, s in zip(weights, score_list))

        return {
            **reward_scores,
            "total_reward": total_reward,
            "reward_weights": weights.detach(),
        }

    def compute_preference_loss(
        self,
        chosen_features: torch.Tensor,
        rejected_features: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        business_features: Optional[torch.Tensor] = None,
        margin: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        偏好对损失计算（奖励模型训练核心）
        chosen应获得更高奖励，rejected应获得更低奖励
        """
        chosen_out = self.forward(chosen_features, user_features, business_features)
        rejected_out = self.forward(rejected_features, user_features, business_features)

        chosen_reward = chosen_out["total_reward"]
        rejected_reward = rejected_out["total_reward"]

        # 偏好损失: -log(sigmoid(r_chosen - r_rejected))
        reward_diff = chosen_reward - rejected_reward - margin
        loss = -F.logsigmoid(reward_diff).mean()

        # 准确率
        accuracy = (chosen_reward > rejected_reward).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "chosen_reward": chosen_reward.mean(),
            "rejected_reward": rejected_reward.mean(),
            "reward_diff": reward_diff.mean(),
        }


class RewardModelTrainer:
    """奖励模型训练器"""

    def __init__(
        self,
        model: MultimodalRewardModel,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000, eta_min=1e-7
        )

        logger.info(f"🏋️ 奖励模型训练器: lr={learning_rate}, device={device}")

    def train_step(
        self,
        chosen_features: torch.Tensor,
        rejected_features: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        business_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """单步训练"""
        self.model.train()

        chosen_features = chosen_features.to(self.device)
        rejected_features = rejected_features.to(self.device)
        if user_features is not None:
            user_features = user_features.to(self.device)
        if business_features is not None:
            business_features = business_features.to(self.device)

        # 前向传播
        result = self.model.compute_preference_loss(
            chosen_features, rejected_features, user_features, business_features
        )

        # 反向传播
        self.optimizer.zero_grad()
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": result["loss"].item(),
            "accuracy": result["accuracy"].item(),
            "chosen_reward": result["chosen_reward"].item(),
            "rejected_reward": result["rejected_reward"].item(),
            "reward_diff": result["reward_diff"].item(),
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    @torch.no_grad()
    def evaluate(
        self,
        eval_dataloader,
    ) -> Dict[str, float]:
        """评估奖励模型"""
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_steps = 0

        for batch in eval_dataloader:
            chosen = batch["chosen_features"].to(self.device)
            rejected = batch["rejected_features"].to(self.device)
            user_feat = batch.get("user_features")
            biz_feat = batch.get("business_features")

            if user_feat is not None:
                user_feat = user_feat.to(self.device)
            if biz_feat is not None:
                biz_feat = biz_feat.to(self.device)

            result = self.model.compute_preference_loss(chosen, rejected, user_feat, biz_feat)
            total_loss += result["loss"].item()
            total_acc += result["accuracy"].item()
            total_steps += 1

        return {
            "eval_loss": total_loss / max(total_steps, 1),
            "eval_accuracy": total_acc / max(total_steps, 1),
        }
