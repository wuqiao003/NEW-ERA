"""
模型模块单元测试
覆盖: VisionEncoder、TextEncoder、MultimodalFusion、MultimodalBaseModel
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.vision_encoder import VisionEncoder, LightweightViT
from src.models.text_encoder import TextEncoder, LightweightTextEncoder
from src.models.fusion import (
    CrossAttentionFusion,
    GatedFusion,
    MLPFusion,
    MultimodalFusionModule,
)
from src.models.multimodal_model import (
    MultimodalBaseModel,
    ContentGenerationHead,
    MatchingHead,
    RecommendationHead,
)


# ============ 视觉编码器测试 ============

class TestLightweightViT:
    """轻量级ViT测试"""

    def setup_method(self):
        self.vit = LightweightViT(
            image_size=224, patch_size=16,
            hidden_size=256, num_layers=2, num_heads=4,
        )

    def test_output_shape(self):
        x = torch.randn(2, 3, 224, 224)
        out = self.vit(x)
        assert "features" in out
        assert "patch_features" in out
        assert out["features"].shape == (2, 256)
        num_patches = (224 // 16) ** 2
        assert out["patch_features"].shape == (2, num_patches, 256)

    def test_single_sample(self):
        x = torch.randn(1, 3, 224, 224)
        out = self.vit(x)
        assert out["features"].shape == (1, 256)

    def test_gradient_flow(self):
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        out = self.vit(x)
        loss = out["features"].sum()
        loss.backward()
        assert x.grad is not None


class TestVisionEncoder:
    """视觉编码器测试"""

    def setup_method(self):
        self.encoder = VisionEncoder(
            model_name="default",
            hidden_size=256,
            projection_dim=128,
            freeze=False,
            use_pretrained=False,
        )

    def test_forward(self):
        x = torch.randn(2, 3, 224, 224)
        out = self.encoder(x)
        assert "features" in out
        assert "projected" in out
        assert out["projected"].shape == (2, 128)

    def test_output_dim(self):
        assert self.encoder.get_output_dim() == 128


# ============ 文本编码器测试 ============

class TestLightweightTextEncoder:
    """轻量级文本编码器测试"""

    def setup_method(self):
        self.encoder = LightweightTextEncoder(
            vocab_size=1000, hidden_size=256,
            num_layers=2, num_heads=4, max_length=128,
        )

    def test_output_shape(self):
        ids = torch.randint(0, 1000, (2, 64))
        mask = torch.ones(2, 64, dtype=torch.long)
        out = self.encoder(ids, mask)
        assert "last_hidden_state" in out
        assert out["last_hidden_state"].shape == (2, 64, 256)

    def test_with_masking(self):
        ids = torch.randint(0, 1000, (2, 64))
        mask = torch.ones(2, 64, dtype=torch.long)
        mask[0, 32:] = 0  # 第一个样本截半
        out = self.encoder(ids, mask)
        assert out["last_hidden_state"].shape == (2, 64, 256)


class TestTextEncoder:
    """文本编码器测试"""

    def setup_method(self):
        self.encoder = TextEncoder(
            model_name="default",
            hidden_size=256,
            projection_dim=128,
            max_length=128,
            use_pretrained=False,
            use_lora=False,
        )

    def test_forward(self):
        ids = torch.randint(0, 32000, (2, 64))
        mask = torch.ones(2, 64, dtype=torch.long)
        out = self.encoder(ids, mask)
        assert "features" in out
        assert "projected" in out
        assert "token_features" in out
        assert out["projected"].shape == (2, 128)

    def test_output_dim(self):
        assert self.encoder.get_output_dim() == 128


# ============ 融合模块测试 ============

class TestCrossAttentionFusion:
    """跨模态注意力融合测试"""

    def setup_method(self):
        self.fusion = CrossAttentionFusion(
            hidden_size=128, num_heads=4, num_layers=2, dropout=0.1,
        )

    def test_forward_3d(self):
        text = torch.randn(2, 16, 128)
        vision = torch.randn(2, 49, 128)
        out = self.fusion(text, vision)
        assert "fused_features" in out
        assert out["fused_features"].shape == (2, 128)

    def test_forward_2d(self):
        text = torch.randn(2, 128)
        vision = torch.randn(2, 128)
        out = self.fusion(text, vision)
        assert out["fused_features"].shape == (2, 128)

    def test_gradient_flow(self):
        text = torch.randn(2, 8, 128, requires_grad=True)
        vision = torch.randn(2, 8, 128, requires_grad=True)
        out = self.fusion(text, vision)
        loss = out["fused_features"].sum()
        loss.backward()
        assert text.grad is not None
        assert vision.grad is not None


class TestGatedFusion:
    """门控融合测试"""

    def setup_method(self):
        self.fusion = GatedFusion(hidden_size=128)

    def test_forward(self):
        text = torch.randn(4, 128)
        vision = torch.randn(4, 128)
        out = self.fusion(text, vision)
        assert out["fused_features"].shape == (4, 128)
        assert "gate_text" in out
        assert "gate_vision" in out

    def test_gate_values(self):
        text = torch.randn(4, 128)
        vision = torch.randn(4, 128)
        out = self.fusion(text, vision)
        assert 0 <= out["gate_text"] <= 1
        assert 0 <= out["gate_vision"] <= 1


class TestMLPFusion:
    """MLP融合测试"""

    def setup_method(self):
        self.fusion = MLPFusion(hidden_size=128)

    def test_forward(self):
        text = torch.randn(4, 128)
        vision = torch.randn(4, 128)
        out = self.fusion(text, vision)
        assert out["fused_features"].shape == (4, 128)


class TestMultimodalFusionModule:
    """融合总模块测试"""

    @pytest.mark.parametrize("fusion_type", ["cross_attention", "gated", "mlp"])
    def test_fusion_types(self, fusion_type):
        fusion = MultimodalFusionModule(
            fusion_type=fusion_type,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            use_gate=(fusion_type != "gated"),
        )
        text = torch.randn(2, 128)
        vision = torch.randn(2, 128)
        out = fusion(text, vision)
        assert "fused_features" in out
        assert out["fused_features"].shape[0] == 2


# ============ 任务头测试 ============

class TestTaskHeads:
    """任务头测试"""

    def test_content_generation_head(self):
        head = ContentGenerationHead(128, 32000)
        features = torch.randn(2, 128)
        out = head(features)
        assert out["logits"].shape == (2, 32000)

    def test_matching_head(self):
        head = MatchingHead(128)
        text = torch.randn(4, 128)
        vision = torch.randn(4, 128)
        out = head(text, vision)
        assert "similarity" in out
        assert out["similarity"].shape == (4, 4)

    def test_recommendation_head(self):
        head = RecommendationHead(128)
        features = torch.randn(4, 128)
        out = head(features)
        assert "scores" in out
        assert out["scores"].shape == (4,)


# ============ 基座模型测试 ============

class TestMultimodalBaseModel:
    """多模态基座模型集成测试"""

    def setup_method(self):
        self.config = {
            "model": {
                "vision_encoder": {"name": "default", "hidden_size": 256},
                "text_encoder": {"name": "default", "hidden_size": 256, "max_length": 128},
                "fusion": {
                    "type": "cross_attention",
                    "hidden_size": 128, "num_heads": 4,
                    "num_layers": 2, "dropout": 0.1, "use_gate": True,
                },
                "projection": {"shared_dim": 128},
            }
        }
        self.model = MultimodalBaseModel(self.config)

    def test_forward_matching(self):
        ids = torch.randint(0, 32000, (2, 64))
        mask = torch.ones(2, 64, dtype=torch.long)
        pixels = torch.randn(2, 3, 224, 224)

        out = self.model(
            input_ids=ids, attention_mask=mask,
            pixel_values=pixels, task="matching",
        )
        assert "text_features" in out
        assert "vision_features" in out
        assert "fused_features" in out

    def test_forward_recommendation(self):
        ids = torch.randint(0, 32000, (2, 64))
        mask = torch.ones(2, 64, dtype=torch.long)
        pixels = torch.randn(2, 3, 224, 224)

        out = self.model(
            input_ids=ids, attention_mask=mask,
            pixel_values=pixels, task="recommendation",
        )
        assert "scores" in out

    def test_forward_generation(self):
        ids = torch.randint(0, 32000, (2, 64))
        mask = torch.ones(2, 64, dtype=torch.long)
        pixels = torch.randn(2, 3, 224, 224)

        out = self.model(
            input_ids=ids, attention_mask=mask,
            pixel_values=pixels, task="generation",
        )
        assert "logits" in out

    def test_contrastive_loss(self):
        ids = torch.randint(0, 32000, (4, 64))
        mask = torch.ones(4, 64, dtype=torch.long)
        pixels = torch.randn(4, 3, 224, 224)
        labels = torch.arange(4)

        out = self.model(
            input_ids=ids, attention_mask=mask,
            pixel_values=pixels, task="matching", labels=labels,
        )
        assert "loss" in out
        assert out["loss"].requires_grad

    def test_vision_only(self):
        pixels = torch.randn(2, 3, 224, 224)
        out = self.model(pixel_values=pixels, task="matching")
        assert "vision_features" in out

    def test_text_only(self):
        ids = torch.randint(0, 32000, (2, 64))
        mask = torch.ones(2, 64, dtype=torch.long)
        out = self.model(input_ids=ids, attention_mask=mask, task="matching")
        assert "text_features" in out

    def test_get_multimodal_features(self):
        ids = torch.randint(0, 32000, (2, 64))
        mask = torch.ones(2, 64, dtype=torch.long)
        pixels = torch.randn(2, 3, 224, 224)
        feat = self.model.get_multimodal_features(ids, mask, pixels)
        assert feat is not None
        assert feat.shape[0] == 2
