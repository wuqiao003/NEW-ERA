"""
模型模块单元测试
覆盖: VisionEncoder、TextEncoder、MultimodalFusion、MultimodalBaseModel
     + CopyGenerationHead、MarketingCopyGenerator、CopyQualityEvaluator
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
from src.models.copy_generator import (
    CopyGenerationHead,
    MarketingCopyGenerator,
    CopyQualityEvaluator,
    COPY_STYLES,
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


# ============ 文案生成头测试 ============

class TestCopyGenerationHead:
    """文案生成头测试"""

    def setup_method(self):
        self.head = CopyGenerationHead(
            input_dim=128,
            hidden_dim=256,
            vocab_size=32000,
            max_length=256,
            num_styles=5,
        )

    def test_output_shape(self):
        features = torch.randn(4, 128)
        out = self.head(features)
        assert "logits" in out
        assert "quality_scores" in out
        assert "enhanced_features" in out
        assert out["logits"].shape == (4, 32000)
        assert out["quality_scores"].shape == (4,)

    def test_with_style_ids(self):
        """测试指定风格 ID"""
        features = torch.randn(4, 128)
        style_ids = torch.tensor([0, 1, 2, 3])  # 种草/促销/情感/专业
        out = self.head(features, style_ids=style_ids)
        assert out["logits"].shape == (4, 32000)

    def test_default_style(self):
        """默认风格为种草(0)"""
        features = torch.randn(2, 128)
        out = self.head(features)  # 不指定 style_ids
        assert out["logits"].shape == (2, 32000)

    def test_quality_scores_range(self):
        """质量分应在 [0, 1] 范围内（Sigmoid 输出）"""
        features = torch.randn(8, 128)
        out = self.head(features)
        assert (out["quality_scores"] >= 0).all()
        assert (out["quality_scores"] <= 1).all()

    def test_gradient_flow(self):
        """梯度流通测试"""
        features = torch.randn(2, 128, requires_grad=True)
        out = self.head(features)
        loss = out["logits"].sum() + out["quality_scores"].sum()
        loss.backward()
        assert features.grad is not None

    def test_different_styles_different_output(self):
        """不同风格应产生不同输出"""
        features = torch.randn(1, 128)
        style_0 = torch.tensor([0])
        style_4 = torch.tensor([4])
        out_0 = self.head(features, style_ids=style_0)
        out_4 = self.head(features, style_ids=style_4)
        # 不同风格 embedding → 不同 logits
        assert not torch.allclose(out_0["logits"], out_4["logits"])


# ============ 营销文案生成器测试 ============

class TestMarketingCopyGenerator:
    """营销文案生成器测试"""

    def setup_method(self):
        config = {
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
        self.model = MultimodalBaseModel(config)
        self.copy_head = CopyGenerationHead(
            input_dim=128, hidden_dim=256, vocab_size=32000,
        )
        self.generator = MarketingCopyGenerator(
            multimodal_model=self.model,
            copy_head=self.copy_head,
            device="cpu",
        )

    def test_generate_basic(self):
        """基本生成测试"""
        result = self.generator.generate(
            product_title="防晒霜SPF50+",
            pixel_values=torch.randn(1, 3, 224, 224),
        )
        assert "product_title" in result
        assert "copies" in result
        assert "best_copy" in result
        assert "latency_ms" in result
        assert len(result["copies"]) > 0

    def test_generate_specific_styles(self):
        """指定风格生成"""
        result = self.generator.generate(
            product_title="蓝牙耳机",
            pixel_values=torch.randn(1, 3, 224, 224),
            styles=["种草", "促销"],
        )
        # 只请求了2种风格
        styles_generated = set(c["style"] for c in result["copies"])
        assert "种草" in styles_generated or "促销" in styles_generated

    def test_generate_with_text_input(self):
        """使用文本输入"""
        result = self.generator.generate(
            product_title="测试商品",
            input_ids=torch.randint(0, 32000, (1, 64)),
            attention_mask=torch.ones(1, 64, dtype=torch.long),
        )
        assert len(result["copies"]) > 0

    def test_generate_multiple_variants(self):
        """每种风格生成多个变体"""
        result = self.generator.generate(
            product_title="测试商品",
            pixel_values=torch.randn(1, 3, 224, 224),
            styles=["种草"],
            num_variants=3,
        )
        assert len(result["copies"]) == 3

    def test_batch_generate(self):
        """批量生成"""
        products = [
            {"title": "商品A", "pixel_values": torch.randn(1, 3, 224, 224)},
            {"title": "商品B", "pixel_values": torch.randn(1, 3, 224, 224)},
        ]
        results = self.generator.batch_generate(products, styles=["种草"], num_variants=1)
        assert len(results) == 2
        assert results[0]["product_title"] == "商品A"
        assert results[1]["product_title"] == "商品B"


# ============ 文案质量评估器测试 ============

class TestCopyQualityEvaluator:
    """文案质量评估器测试"""

    def setup_method(self):
        self.evaluator = CopyQualityEvaluator()

    def test_evaluate_basic(self):
        """基本评估"""
        scores = self.evaluator.evaluate_copy(
            copy_text="姐妹们这个防晒霜真的绝了！轻薄不油腻，用了一周皮肤状态肉眼可见变好了～",
            style="种草",
        )
        assert "length_score" in scores
        assert "readability_score" in scores
        assert "style_match_score" in scores
        assert "overall" in scores
        assert 0.0 <= scores["overall"] <= 1.0

    def test_evaluate_different_styles(self):
        """不同风格的评估"""
        zhongcao_text = "姐妹们这个推荐必入！太好用了绝了，安利给所有人。性价比超高，闭眼入不会后悔的！已经回购三次了！"
        cuxiao_text = "🔥限时秒杀！原价199今日仅需99！手慢无，库存不多！"
        jianyue_text = "防晒霜SPF50+ | 清爽不油腻 | ¥89"

        score_zc = self.evaluator.evaluate_copy(zhongcao_text, "种草")
        score_cx = self.evaluator.evaluate_copy(cuxiao_text, "促销")
        score_jy = self.evaluator.evaluate_copy(jianyue_text, "简约")

        # 每种风格的文案对应风格评分应较高
        assert score_zc["style_match_score"] > 0.3
        assert score_jy["overall"] > 0  # 简约文案评估不应为0

    def test_evaluate_empty_text(self):
        """空文本评估"""
        scores = self.evaluator.evaluate_copy("", "种草")
        assert scores["readability_score"] == 0.0

    def test_score_length(self):
        """长度评分"""
        # 简约风格短文本应得高分
        score = CopyQualityEvaluator._score_length("简短卖点文案", "简约")
        assert score > 0.3

    def test_score_readability(self):
        """可读性评分"""
        good_text = "这款产品非常好用，质量很棒。推荐给大家！"
        score = CopyQualityEvaluator._score_readability(good_text)
        assert score > 0.3

    def test_copy_styles_definition(self):
        """验证文案风格定义完整"""
        expected_styles = ["种草", "促销", "情感", "专业", "简约"]
        for style in expected_styles:
            assert style in COPY_STYLES
            assert "temperature" in COPY_STYLES[style]
            assert "prompt_prefix" in COPY_STYLES[style]
