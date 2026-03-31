"""
视觉编码器模块
支持 CLIP-ViT、InternVL 等工业级视觉模型
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from loguru import logger


class VisionEncoder(nn.Module):
    """
    工业级视觉编码器
    支持CLIP-ViT-L/14、自定义CNN编码器
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        hidden_size: int = 1024,
        projection_dim: int = 512,
        freeze: bool = True,
        freeze_layers: int = -1,
        use_pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim

        # 初始化视觉骨干网络
        self.backbone = self._build_backbone(model_name, use_pretrained)

        # 投影层：将视觉特征映射到共享空间
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # 冻结策略
        if freeze:
            self._freeze_backbone(freeze_layers)

        logger.info(f"🖼️  视觉编码器初始化: {model_name}, hidden={hidden_size}, proj={projection_dim}")

    def _build_backbone(self, model_name: str, use_pretrained: bool) -> nn.Module:
        """构建视觉骨干网络"""
        try:
            if "clip" in model_name.lower():
                return self._build_clip_backbone(model_name)
            else:
                return self._build_default_backbone()
        except Exception as e:
            logger.warning(f"⚠️ 加载预训练模型失败: {e}, 使用默认编码器")
            return self._build_default_backbone()

    def _build_clip_backbone(self, model_name: str) -> nn.Module:
        """构建CLIP视觉骨干"""
        try:
            from transformers import CLIPVisionModel
            model = CLIPVisionModel.from_pretrained(model_name)
            return model
        except Exception:
            logger.warning("⚠️ CLIP模型加载失败，使用替代ViT编码器")
            return self._build_default_backbone()

    def _build_default_backbone(self) -> nn.Module:
        """默认轻量级ViT编码器（用于开发测试）"""
        return LightweightViT(
            image_size=224,
            patch_size=16,
            hidden_size=self.hidden_size,
            num_layers=6,
            num_heads=8,
        )

    def _freeze_backbone(self, freeze_layers: int = -1):
        """冻结视觉编码器参数"""
        if freeze_layers == -1:
            # 冻结全部
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("🔒 视觉编码器: 全部冻结")
        elif freeze_layers > 0:
            # 部分冻结
            frozen = 0
            for name, param in self.backbone.named_parameters():
                if frozen < freeze_layers:
                    param.requires_grad = False
                    frozen += 1
            logger.info(f"🔒 视觉编码器: 冻结前 {freeze_layers} 层")

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_all_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            pixel_values: [B, C, H, W] 图像张量
            return_all_hidden: 是否返回所有隐藏层
        Returns:
            Dict包含:
                - features: [B, hidden_size] 全局视觉特征
                - projected: [B, projection_dim] 投影后特征
                - patch_features: [B, N, hidden_size] patch级特征 (可选)
        """
        outputs = {}

        # 骨干网络提取特征
        if hasattr(self.backbone, "forward"):
            backbone_out = self.backbone(pixel_values)

            if hasattr(backbone_out, "last_hidden_state"):
                # Transformers模型输出
                hidden_state = backbone_out.last_hidden_state
                features = hidden_state[:, 0, :]  # CLS token
                outputs["patch_features"] = hidden_state[:, 1:, :]
            elif hasattr(backbone_out, "pooler_output"):
                features = backbone_out.pooler_output
            elif isinstance(backbone_out, dict):
                features = backbone_out.get("features", backbone_out.get("cls_token"))
                if "patch_features" in backbone_out:
                    outputs["patch_features"] = backbone_out["patch_features"]
            else:
                features = backbone_out
        else:
            features = self.backbone(pixel_values)

        outputs["features"] = features
        outputs["projected"] = self.projection(features)

        return outputs

    def get_output_dim(self) -> int:
        return self.projection_dim


class LightweightViT(nn.Module):
    """轻量级Vision Transformer，用于开发测试"""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_size = hidden_size

        # Patch嵌入
        self.patch_embed = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        )

        # 位置编码 + CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, hidden_size))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = pixel_values.shape[0]

        # Patch嵌入
        x = self.patch_embed(pixel_values)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # 添加CLS token + 位置编码
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        # Transformer编码
        x = self.encoder(x)
        x = self.norm(x)

        return {
            "features": x[:, 0, :],  # CLS token
            "patch_features": x[:, 1:, :],  # Patch features
        }
