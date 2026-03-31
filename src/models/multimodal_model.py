"""
多模态基座模型
整合视觉编码器、文本编码器、跨模态融合模块
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from src.models.vision_encoder import VisionEncoder
from src.models.text_encoder import TextEncoder
from src.models.fusion import MultimodalFusionModule


class MultimodalBaseModel(nn.Module):
    """
    多模态基座模型
    架构: Vision Encoder + Text Encoder + Cross-Modal Fusion + Task Heads
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config

        model_cfg = config.get("model", {})
        vision_cfg = model_cfg.get("vision_encoder", {})
        text_cfg = model_cfg.get("text_encoder", {})
        fusion_cfg = model_cfg.get("fusion", {})
        proj_cfg = model_cfg.get("projection", {})

        shared_dim = proj_cfg.get("shared_dim", 512)

        # 视觉编码器
        self.vision_encoder = VisionEncoder(
            model_name=vision_cfg.get("name", "openai/clip-vit-large-patch14"),
            hidden_size=vision_cfg.get("hidden_size", 1024),
            projection_dim=shared_dim,
            freeze=vision_cfg.get("freeze", True),
        )

        # 文本编码器
        self.text_encoder = TextEncoder(
            model_name=text_cfg.get("name", "Qwen/Qwen2-7B-Instruct"),
            hidden_size=text_cfg.get("hidden_size", 4096),
            projection_dim=shared_dim,
            max_length=text_cfg.get("max_length", 512),
        )

        # 跨模态融合
        self.fusion_module = MultimodalFusionModule(
            fusion_type=fusion_cfg.get("type", "cross_attention"),
            hidden_size=shared_dim,
            num_heads=fusion_cfg.get("num_heads", 16),
            num_layers=fusion_cfg.get("num_layers", 4),
            dropout=fusion_cfg.get("dropout", 0.1),
            use_gate=fusion_cfg.get("use_gate", True),
        )

        # 任务头
        self.content_head = ContentGenerationHead(shared_dim, text_cfg.get("hidden_size", 4096))
        self.matching_head = MatchingHead(shared_dim)
        self.recommendation_head = RecommendationHead(shared_dim)

        logger.info("🏗️  多模态基座模型构建完成")

    def encode_vision(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """编码视觉特征"""
        return self.vision_encoder(pixel_values)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """编码文本特征"""
        return self.text_encoder(input_ids, attention_mask)

    def fuse(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """跨模态融合"""
        return self.fusion_module(text_features, vision_features, text_mask)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        task: str = "matching",
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        统一前向传播
        Args:
            task: "matching" | "generation" | "recommendation"
        """
        outputs = {}

        # 编码
        if pixel_values is not None:
            vision_out = self.encode_vision(pixel_values)
            outputs["vision_features"] = vision_out["projected"]

        if input_ids is not None:
            text_out = self.encode_text(input_ids, attention_mask)
            outputs["text_features"] = text_out["projected"]

        # 融合
        if pixel_values is not None and input_ids is not None:
            text_feat = text_out.get("token_features", text_out["projected"])
            vision_feat = vision_out.get("patch_features", vision_out["projected"])

            # 维度对齐
            if text_feat.shape[-1] != self.fusion_module.hidden_size:
                text_feat = text_out["projected"]
            if vision_feat.shape[-1] != self.fusion_module.hidden_size:
                vision_feat = vision_out["projected"]

            fusion_out = self.fuse(text_feat, vision_feat, text_mask=attention_mask)
            outputs["fused_features"] = fusion_out["fused_features"]
        else:
            # 单模态
            if "vision_features" in outputs:
                outputs["fused_features"] = outputs["vision_features"]
            elif "text_features" in outputs:
                outputs["fused_features"] = outputs["text_features"]

        # 任务头
        fused = outputs.get("fused_features")
        if fused is not None:
            if task == "matching":
                task_out = self.matching_head(
                    outputs.get("text_features"),
                    outputs.get("vision_features"),
                )
                outputs.update(task_out)
            elif task == "generation":
                task_out = self.content_head(fused)
                outputs.update(task_out)
            elif task == "recommendation":
                task_out = self.recommendation_head(fused)
                outputs.update(task_out)

        # 计算损失
        if labels is not None:
            loss = self._compute_loss(outputs, labels, task)
            outputs["loss"] = loss

        return outputs

    def _compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        """计算任务损失"""
        if task == "matching":
            # 对比学习损失
            text_feat = outputs.get("text_features")
            vision_feat = outputs.get("vision_features")
            if text_feat is not None and vision_feat is not None:
                return self._contrastive_loss(text_feat, vision_feat)

        elif task == "generation":
            logits = outputs.get("logits")
            if logits is not None:
                return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        elif task == "recommendation":
            scores = outputs.get("scores")
            if scores is not None:
                return F.binary_cross_entropy_with_logits(scores, labels.float())

        return torch.tensor(0.0)

    def _contrastive_loss(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """CLIP风格对比学习损失"""
        # 归一化
        text_features = F.normalize(text_features, dim=-1)
        vision_features = F.normalize(vision_features, dim=-1)

        # 相似度矩阵
        logits = torch.matmul(text_features, vision_features.T) / temperature
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)

        # 双向对比损失
        loss_t2v = F.cross_entropy(logits, labels)
        loss_v2t = F.cross_entropy(logits.T, labels)

        return (loss_t2v + loss_v2t) / 2

    def get_multimodal_features(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """获取多模态融合特征（用于下游任务）"""
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                task="matching",
            )
        return outputs.get("fused_features", outputs.get("text_features", outputs.get("vision_features")))


class ContentGenerationHead(nn.Module):
    """内容生成任务头"""

    def __init__(self, input_dim: int, vocab_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Linear(input_dim * 2, vocab_size),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.head(features)
        return {"logits": logits}


class MatchingHead(nn.Module):
    """图文匹配任务头"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(
        self,
        text_features: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if text_features is None or vision_features is None:
            return {"similarity": torch.tensor(0.0)}

        text_norm = F.normalize(text_features, dim=-1)
        vision_norm = F.normalize(vision_features, dim=-1)
        similarity = torch.matmul(text_norm, vision_norm.T) / self.temperature.exp()
        return {"similarity": similarity}


class RecommendationHead(nn.Module):
    """推荐排序任务头"""

    def __init__(self, input_dim: int, num_classes: int = 1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        scores = self.scorer(features).squeeze(-1)
        return {"scores": scores}
