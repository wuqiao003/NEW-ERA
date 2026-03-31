"""
跨模态融合模块
实现 Cross-Attention、MLP融合、门控融合 等多模态深度对齐方案
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from loguru import logger


class CrossAttentionFusion(nn.Module):
    """
    跨模态交叉注意力融合
    Text attend to Vision / Vision attend to Text
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Text-to-Vision Cross-Attention 层
        self.t2v_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Vision-to-Text Cross-Attention 层
        self.v2t_layers = nn.ModuleList([
            CrossAttentionBlock(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 融合输出投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        logger.info(f"🔗 CrossAttention融合: heads={num_heads}, layers={num_layers}")

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_features: [B, Lt, D] 文本token特征
            vision_features: [B, Lv, D] 视觉patch特征
        Returns:
            fused: [B, D] 融合后的多模态特征
        """
        # 确保维度兼容
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        if vision_features.dim() == 2:
            vision_features = vision_features.unsqueeze(1)

        # Text attend to Vision
        t2v_out = text_features
        for layer in self.t2v_layers:
            t2v_out = layer(t2v_out, vision_features, kv_mask=vision_mask)

        # Vision attend to Text
        v2t_out = vision_features
        for layer in self.v2t_layers:
            v2t_out = layer(v2t_out, text_features, kv_mask=text_mask)

        # 池化
        t2v_pooled = t2v_out.mean(dim=1)
        v2t_pooled = v2t_out.mean(dim=1)

        # 融合
        fused = self.fusion_proj(torch.cat([t2v_pooled, v2t_pooled], dim=-1))

        return {
            "fused_features": fused,
            "text_attended": t2v_out,
            "vision_attended": v2t_out,
        }


class CrossAttentionBlock(nn.Module):
    """单层跨模态注意力块"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-Attention + Residual
        residual = query
        query = self.norm1(query)
        key_padding_mask = (kv_mask == 0) if kv_mask is not None else None
        attn_out, _ = self.cross_attn(query, key_value, key_value, key_padding_mask=key_padding_mask)
        query = residual + attn_out

        # FFN + Residual
        residual = query
        query = self.norm2(query)
        query = residual + self.ffn(query)

        return query


class GatedFusion(nn.Module):
    """
    门控融合模块
    自适应控制各模态信息的融合比例，解决模态信息失衡问题
    """

    def __init__(self, hidden_size: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size

        # 门控网络
        self.gate_text = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )
        self.gate_vision = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        # 融合输出
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
        )

        logger.info(f"🚪 门控融合模块初始化: hidden_size={hidden_size}")

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_features: [B, D] 文本特征
            vision_features: [B, D] 视觉特征
        """
        # 拼接特征用于门控计算
        combined = torch.cat([text_features, vision_features], dim=-1)

        # 计算门控权重
        alpha_text = self.gate_text(combined)
        alpha_vision = self.gate_vision(combined)

        # 加权融合
        fused = alpha_text * text_features + alpha_vision * vision_features
        fused = self.output_proj(fused)

        return {
            "fused_features": fused,
            "gate_text": alpha_text.mean().item(),
            "gate_vision": alpha_vision.mean().item(),
        }


class MultimodalFusionModule(nn.Module):
    """
    多模态融合总模块
    支持Cross-Attention / MLP / Gated 多种融合策略
    """

    def __init__(
        self,
        fusion_type: str = "cross_attention",
        hidden_size: int = 1024,
        num_heads: int = 16,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_gate: bool = True,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.hidden_size = hidden_size

        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )
        elif fusion_type == "gated":
            self.fusion = GatedFusion(hidden_size=hidden_size, dropout=dropout)
        elif fusion_type == "mlp":
            self.fusion = MLPFusion(hidden_size=hidden_size, dropout=dropout)
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")

        # 可选门控增强
        self.use_gate = use_gate and fusion_type != "gated"
        if self.use_gate:
            self.gate = GatedFusion(hidden_size=hidden_size, dropout=dropout)

        logger.info(f"🔗 多模态融合模块: type={fusion_type}, gate={use_gate}")

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """统一融合接口"""
        if self.fusion_type == "cross_attention":
            result = self.fusion(text_features, vision_features, text_mask, vision_mask)
        else:
            # MLP和Gated需要[B, D]维度
            if text_features.dim() == 3:
                text_features = text_features.mean(dim=1)
            if vision_features.dim() == 3:
                vision_features = vision_features.mean(dim=1)
            result = self.fusion(text_features, vision_features)

        # 门控增强
        if self.use_gate:
            fused = result["fused_features"]
            if fused.dim() == 3:
                fused = fused.mean(dim=1)
            if text_features.dim() == 3:
                text_pooled = text_features.mean(dim=1)
            else:
                text_pooled = text_features
            gate_result = self.gate(fused, text_pooled)
            result["fused_features"] = gate_result["fused_features"]
            result.update({f"gate_{k}": v for k, v in gate_result.items() if k.startswith("gate_")})

        return result


class MLPFusion(nn.Module):
    """MLP融合（简单拼接+MLP）"""

    def __init__(self, hidden_size: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

    def forward(
        self,
        text_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        combined = torch.cat([text_features, vision_features], dim=-1)
        fused = self.fusion(combined)
        return {"fused_features": fused}
