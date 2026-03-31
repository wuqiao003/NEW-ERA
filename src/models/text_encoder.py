"""
文本编码器模块
支持 Qwen、Llama 等大语言模型作为文本编码器
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
from loguru import logger


class TextEncoder(nn.Module):
    """
    工业级文本编码器
    支持Qwen-7B、Llama-3等模型
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        hidden_size: int = 4096,
        projection_dim: int = 512,
        max_length: int = 512,
        use_pretrained: bool = True,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.max_length = max_length

        # 初始化文本骨干
        self.backbone = self._build_backbone(model_name, use_pretrained)

        # 投影层：将文本特征映射到共享空间
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, projection_dim),
            nn.LayerNorm(projection_dim),
        )

        # LoRA配置
        if use_lora and use_pretrained:
            self._apply_lora(lora_r, lora_alpha)

        logger.info(f"📝 文本编码器初始化: {model_name}, hidden={hidden_size}, proj={projection_dim}")

    def _build_backbone(self, model_name: str, use_pretrained: bool) -> nn.Module:
        """构建文本骨干网络"""
        try:
            if use_pretrained:
                return self._build_pretrained_backbone(model_name)
            else:
                return self._build_default_backbone()
        except Exception as e:
            logger.warning(f"⚠️ 加载预训练文本模型失败: {e}, 使用默认编码器")
            return self._build_default_backbone()

    def _build_pretrained_backbone(self, model_name: str) -> nn.Module:
        """构建预训练文本模型"""
        try:
            from transformers import AutoModel, BitsAndBytesConfig

            # 尝试使用4bit量化加载
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            model = AutoModel.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
            return model
        except Exception as e:
            logger.warning(f"⚠️ 量化加载失败: {e}")
            return self._build_default_backbone()

    def _build_default_backbone(self) -> nn.Module:
        """默认轻量级文本编码器"""
        return LightweightTextEncoder(
            vocab_size=32000,
            hidden_size=self.hidden_size,
            num_layers=6,
            num_heads=8,
            max_length=self.max_length,
        )

    def _apply_lora(self, lora_r: int, lora_alpha: int):
        """应用LoRA微调"""
        try:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
                task_type="FEATURE_EXTRACTION",
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
            logger.info(f"🔧 LoRA已应用: r={lora_r}, alpha={lora_alpha}")
        except Exception as e:
            logger.warning(f"⚠️ LoRA应用失败: {e}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_all_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        Args:
            input_ids: [B, L] 文本token ids
            attention_mask: [B, L] 注意力掩码
        Returns:
            Dict:
                - features: [B, hidden_size] 句子级特征
                - projected: [B, projection_dim] 投影后特征
                - token_features: [B, L, hidden_size] token级特征
        """
        outputs = {}

        # 骨干网络
        backbone_out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(backbone_out, "last_hidden_state"):
            hidden_state = backbone_out.last_hidden_state
        elif isinstance(backbone_out, dict):
            hidden_state = backbone_out.get("last_hidden_state", backbone_out.get("hidden_states"))
        elif isinstance(backbone_out, torch.Tensor):
            hidden_state = backbone_out
        else:
            hidden_state = backbone_out[0]

        # 池化获取句子级特征（均值池化 + attention mask）
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_state.size()).float()
            sum_hidden = torch.sum(hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            features = sum_hidden / sum_mask
        else:
            features = hidden_state.mean(dim=1)

        outputs["features"] = features
        outputs["projected"] = self.projection(features)
        outputs["token_features"] = hidden_state

        return outputs

    def get_output_dim(self) -> int:
        return self.projection_dim


class LightweightTextEncoder(nn.Module):
    """轻量级文本编码器，用于开发测试"""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 1024,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # 词嵌入
        self.token_embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(max_length, hidden_size)
        self.embed_drop = nn.Dropout(dropout)

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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape

        # 嵌入
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.embed_drop(x)

        # 注意力掩码
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)

        # 编码
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        return {"last_hidden_state": x}
