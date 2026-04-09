"""
数据模块单元测试
覆盖: MultimodalDataset、PreferenceDataset、数据预处理Pipeline
     + 电商数据集: EcommerceProductDataset、CopyPreferenceDataset、UserBehaviorDataset
"""
import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import (
    MultimodalDataset,
    PreferenceDataset,
    create_dataloader,
    _multimodal_collate_fn,
)
from src.data.preprocessing import (
    TextPreprocessor,
    ImagePreprocessor,
    DataAugmentor,
    MultimodalPipeline,
)
from src.data.ecommerce_dataset import (
    EcommerceProductDataset,
    CopyPreferenceDataset,
    UserBehaviorDataset,
    create_ecommerce_dataloader,
    _ecommerce_collate_fn,
    ProductItem,
    MarketingCopy,
    CopyPreferencePair,
)


class TestTextPreprocessor:
    """文本预处理测试"""

    def setup_method(self):
        self.processor = TextPreprocessor(max_length=256)

    def test_clean_normal_text(self):
        text = "这是一段正常的中文文本"
        result = self.processor.clean(text)
        assert result == "这是一段正常的中文文本"

    def test_clean_removes_urls(self):
        text = "访问 https://example.com 了解更多"
        result = self.processor.clean(text)
        assert "https" not in result

    def test_clean_removes_html(self):
        text = "<p>段落内容</p>"
        result = self.processor.clean(text)
        assert "<p>" not in result
        assert "段落内容" in result

    def test_clean_removes_whitespace(self):
        text = "  多余   空格   文本  "
        result = self.processor.clean(text)
        assert "  " not in result

    def test_clean_truncates(self):
        text = "x" * 1000
        result = self.processor.clean(text)
        assert len(result) <= 256

    def test_clean_empty_string(self):
        assert self.processor.clean("") == ""
        assert self.processor.clean(None) == ""

    def test_is_valid(self):
        assert self.processor.is_valid("正常文本内容") is True
        assert self.processor.is_valid("") is False
        assert self.processor.is_valid("ab") is False
        assert self.processor.is_valid("....") is False

    def test_clean_control_chars(self):
        text = "文本\x00包含\x1f控制字符"
        result = self.processor.clean(text)
        assert "\x00" not in result
        assert "\x1f" not in result


class TestImagePreprocessor:
    """图像预处理测试"""

    def setup_method(self):
        self.processor = ImagePreprocessor(image_size=224)

    def test_process_returns_correct_shape(self):
        from PIL import Image
        img = Image.new("RGB", (640, 480), color="red")
        tensor = self.processor.process(img)
        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_process_grayscale_conversion(self):
        from PIL import Image
        img = Image.new("L", (100, 100), color=128)
        tensor = self.processor.process(img)
        assert tensor.shape == (3, 224, 224)

    def test_is_valid(self):
        from PIL import Image
        valid_img = Image.new("RGB", (256, 256))
        small_img = Image.new("RGB", (32, 32))
        assert self.processor.is_valid(valid_img) is True
        assert self.processor.is_valid(small_img) is False
        assert self.processor.is_valid(None) is False

    def test_normalization(self):
        from PIL import Image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        tensor = self.processor.process(img)
        # 标准化后不应全为正值
        assert tensor.min() < 0.5


class TestDataAugmentor:
    """数据增强测试"""

    def setup_method(self):
        self.augmentor = DataAugmentor(augment_prob=1.0)

    def test_augment_image(self):
        from PIL import Image
        img = Image.new("RGB", (256, 256), color="blue")
        augmented = self.augmentor.augment_image(img)
        assert isinstance(augmented, Image.Image)
        assert augmented.size[0] > 0

    def test_augment_text(self):
        text = "这是一段 测试 文本 用于 增强"
        augmented = self.augmentor.augment_text(text)
        assert isinstance(augmented, str)
        assert len(augmented) > 0

    def test_augment_short_text_safe(self):
        text = "短"
        augmented = self.augmentor.augment_text(text)
        assert len(augmented) > 0


class TestMultimodalPipeline:
    """多模态Pipeline测试"""

    def setup_method(self):
        self.pipeline = MultimodalPipeline(
            max_text_length=256,
            image_size=224,
            augment=True,
            augment_prob=0.5,
        )

    def test_process_sample_text_only(self):
        result = self.pipeline.process_sample("测试文本内容")
        assert "text" in result
        assert "pixel_values" in result
        assert result["text_valid"] is True
        assert result["image_valid"] is False

    def test_process_sample_with_image(self):
        from PIL import Image
        img = Image.new("RGB", (300, 300), color="green")
        result = self.pipeline.process_sample("带图片的测试文本内容", image=img)
        assert result["text_valid"] is True
        assert result["image_valid"] is True
        assert result["pixel_values"].shape == (3, 224, 224)

    def test_process_batch(self):
        from PIL import Image
        texts = ["文本1", "文本2", "文本3"]
        images = [Image.new("RGB", (200, 200), color=c) for c in ["red", "green", "blue"]]
        result = self.pipeline.process_batch(texts, images)
        assert result["pixel_values"].shape == (3, 3, 224, 224)
        assert len(result["texts"]) == 3


class TestMultimodalDataset:
    """多模态数据集测试"""

    def test_dataset_creation(self):
        dataset = MultimodalDataset(
            data_path="data/processed/train",
            max_text_length=128,
            image_size=224,
            split="train",
        )
        assert len(dataset) > 0

    def test_dataset_getitem(self):
        dataset = MultimodalDataset(
            data_path="data/processed/train",
            max_text_length=128,
            image_size=224,
            split="train",
        )
        sample = dataset[0]
        assert "id" in sample
        assert "pixel_values" in sample
        assert sample["pixel_values"].shape == (3, 224, 224)

    def test_dataset_iteration(self):
        dataset = MultimodalDataset(
            data_path="data/processed/train",
            max_text_length=128,
            image_size=224,
            split="train",
        )
        count = 0
        for sample in dataset:
            count += 1
            if count >= 5:
                break
        assert count == 5


class TestPreferenceDataset:
    """偏好数据集测试"""

    def test_preference_dataset_creation(self):
        dataset = PreferenceDataset(
            data_path="data/preference",
            max_length=128,
            split="train",
        )
        assert len(dataset) > 0

    def test_preference_getitem(self):
        dataset = PreferenceDataset(
            data_path="data/preference",
            max_length=128,
            split="train",
        )
        pair = dataset[0]
        assert "id" in pair
        assert "prompt" in pair
        assert "chosen" in pair
        assert "rejected" in pair


class TestDataLoader:
    """DataLoader测试"""

    def test_create_dataloader(self):
        dataset = MultimodalDataset(
            data_path="data/processed/train",
            max_text_length=128,
            image_size=224,
            split="train",
        )
        loader = create_dataloader(dataset, batch_size=4, shuffle=True, num_workers=0)
        batch = next(iter(loader))
        assert "pixel_values" in batch
        assert batch["pixel_values"].shape[0] == 4

    def test_collate_fn(self):
        samples = [
            {"id": "a", "feat": torch.randn(10), "text": "hello"},
            {"id": "b", "feat": torch.randn(10), "text": "world"},
        ]
        collated = _multimodal_collate_fn(samples)
        assert collated["feat"].shape == (2, 10)
        assert collated["text"] == ["hello", "world"]


# ============ 电商数据结构测试 ============

class TestProductItem:
    """商品数据结构测试"""

    def test_creation(self):
        item = ProductItem(product_id="P001", title="测试商品")
        assert item.product_id == "P001"
        assert item.title == "测试商品"
        assert item.category == "通用"
        assert item.tags == []
        assert item.price == 0.0

    def test_full_creation(self):
        item = ProductItem(
            product_id="P002",
            title="防晒霜",
            description="SPF50+",
            category="美妆",
            price=89.0,
            tags=["防晒", "清爽"],
            attributes={"SPF": "50+"},
        )
        assert item.category == "美妆"
        assert len(item.tags) == 2
        assert item.attributes["SPF"] == "50+"


class TestMarketingCopy:
    """营销文案数据结构测试"""

    def test_creation(self):
        copy = MarketingCopy(
            copy_id="C001",
            product_id="P001",
            style="种草",
            content="这个产品真的绝了！",
        )
        assert copy.style == "种草"
        assert copy.quality_score == 0.0


class TestCopyPreferencePair:
    """偏好对数据结构测试"""

    def test_creation(self):
        pair = CopyPreferencePair(
            product_id="P001",
            product_title="测试商品",
            chosen_copy="好文案",
            rejected_copy="差文案",
            chosen_score=0.9,
            rejected_score=0.3,
        )
        assert pair.chosen_score > pair.rejected_score


# ============ 电商商品数据集测试 ============

class TestEcommerceProductDataset:
    """电商商品图文数据集测试"""

    def test_dataset_creation_demo(self):
        """使用示例数据创建数据集"""
        dataset = EcommerceProductDataset(
            data_path="data/ecommerce_test",
            max_text_length=128,
            image_size=224,
            split="train",
        )
        assert len(dataset) > 0
        assert len(dataset.products) > 0

    def test_dataset_getitem(self):
        """获取单个样本"""
        dataset = EcommerceProductDataset(
            data_path="data/ecommerce_test",
            max_text_length=128,
            image_size=224,
            split="train",
        )
        sample = dataset[0]
        assert "id" in sample
        assert "style" in sample
        assert "category" in sample
        assert "pixel_values" in sample
        assert "text" in sample or "input_ids" in sample
        assert "quality_score" in sample
        assert sample["pixel_values"].shape == (3, 224, 224)

    def test_dataset_target_text(self):
        """验证目标文案文本存在"""
        dataset = EcommerceProductDataset(
            data_path="data/ecommerce_test",
            max_text_length=128,
            image_size=224,
            split="train",
        )
        sample = dataset[0]
        assert "target_text" in sample or "labels" in sample

    def test_dataset_multiple_styles(self):
        """验证多风格样本展开"""
        dataset = EcommerceProductDataset(
            data_path="data/ecommerce_test",
            max_text_length=128,
            image_size=224,
            split="train",
            include_copies=True,
        )
        # 每个商品应展开为多个风格样本
        styles = set()
        for i in range(min(50, len(dataset))):
            styles.add(dataset[i]["style"])
        assert len(styles) >= 3  # 至少有3种风格

    def test_dataset_target_style_filter(self):
        """测试按目标风格过滤"""
        dataset = EcommerceProductDataset(
            data_path="data/ecommerce_test",
            max_text_length=128,
            image_size=224,
            split="train",
            target_style="种草",
        )
        for i in range(min(20, len(dataset))):
            assert dataset[i]["style"] == "种草"

    def test_quality_score_range(self):
        """验证质量分在合理范围内"""
        dataset = EcommerceProductDataset(
            data_path="data/ecommerce_test",
            max_text_length=128,
            image_size=224,
            split="train",
        )
        for i in range(min(30, len(dataset))):
            score = dataset[i]["quality_score"]
            assert 0.0 <= score <= 1.0

    def test_build_prompt(self):
        """验证提示词构建"""
        dataset = EcommerceProductDataset(
            data_path="data/ecommerce_test",
            max_text_length=128,
            image_size=224,
            split="train",
        )
        sample = dataset.samples[0]
        prompt = dataset._build_prompt(sample)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert sample["title"] in prompt

    def test_compute_quality_score_with_feedback(self):
        """测试基于用户反馈的质量分计算"""
        feedback = {
            "click_count": 1000,
            "favorite_count": 200,
            "purchase_count": 80,
            "avg_stay_time": 25.0,
        }
        score = EcommerceProductDataset._compute_quality_score(feedback)
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # 有正向反馈应大于0

    def test_compute_quality_score_empty(self):
        """空反馈时质量分为默认值"""
        score = EcommerceProductDataset._compute_quality_score({})
        assert score == 0.5

    def test_iteration(self):
        """数据集迭代"""
        dataset = EcommerceProductDataset(
            data_path="data/ecommerce_test",
            max_text_length=128,
            image_size=224,
            split="train",
        )
        count = 0
        for sample in dataset:
            count += 1
            if count >= 10:
                break
        assert count == 10


# ============ 文案偏好数据集测试 ============

class TestCopyPreferenceDataset:
    """文案偏好对数据集测试"""

    def test_dataset_creation_demo(self):
        """使用示例偏好数据创建"""
        dataset = CopyPreferenceDataset(
            data_path="data/ecommerce_test",
            max_length=128,
            image_size=224,
            split="train",
        )
        assert len(dataset) > 0

    def test_preference_getitem(self):
        """获取偏好对"""
        dataset = CopyPreferenceDataset(
            data_path="data/ecommerce_test",
            max_length=128,
            split="train",
        )
        pair = dataset[0]
        assert "id" in pair
        assert "style" in pair
        assert "prompt" in pair or "prompt_ids" in pair
        assert "chosen" in pair or "chosen_ids" in pair
        assert "rejected" in pair or "rejected_ids" in pair
        assert "chosen_score" in pair
        assert "rejected_score" in pair

    def test_chosen_score_higher(self):
        """验证 chosen 分数高于 rejected"""
        dataset = CopyPreferenceDataset(
            data_path="data/ecommerce_test",
            max_length=128,
            split="train",
        )
        higher_count = 0
        total = min(50, len(dataset))
        for i in range(total):
            pair = dataset[i]
            if pair["chosen_score"] > pair["rejected_score"]:
                higher_count += 1
        # 绝大多数偏好对应满足 chosen > rejected
        assert higher_count / total > 0.8

    def test_pixel_values_shape(self):
        """验证图像张量形状"""
        dataset = CopyPreferenceDataset(
            data_path="data/ecommerce_test",
            max_length=128,
            image_size=224,
            split="train",
        )
        pair = dataset[0]
        assert "pixel_values" in pair
        assert pair["pixel_values"].shape == (3, 224, 224)

    def test_styles_coverage(self):
        """验证多风格覆盖"""
        dataset = CopyPreferenceDataset(
            data_path="data/ecommerce_test",
            max_length=128,
            split="train",
        )
        styles = set()
        for i in range(min(100, len(dataset))):
            styles.add(dataset[i]["style"])
        assert len(styles) >= 3


# ============ 用户行为数据集测试 ============

class TestUserBehaviorDataset:
    """用户行为数据集测试"""

    def test_dataset_creation(self):
        dataset = UserBehaviorDataset(
            data_path="data/ecommerce_test",
            num_users=100,
            num_products=50,
            split="train",
        )
        assert len(dataset) > 0

    def test_getitem(self):
        dataset = UserBehaviorDataset(
            data_path="data/ecommerce_test",
            num_users=100,
            num_products=50,
            user_feature_dim=32,
            product_feature_dim=32,
            split="train",
        )
        item = dataset[0]
        assert "user_features" in item
        assert "product_features" in item
        assert "action_label" in item
        assert "positive_label" in item
        assert item["user_features"].shape == (32,)
        assert item["product_features"].shape == (32,)

    def test_action_label_range(self):
        """行为标签应在 [0, 3] 范围内"""
        dataset = UserBehaviorDataset(
            data_path="data/ecommerce_test",
            num_users=100,
            num_products=50,
            split="train",
        )
        for i in range(min(50, len(dataset))):
            item = dataset[i]
            assert 0 <= item["action_label"] <= 3

    def test_positive_label_binary(self):
        """正样本标签应为 0 或 1"""
        dataset = UserBehaviorDataset(
            data_path="data/ecommerce_test",
            num_users=100,
            num_products=50,
            split="train",
        )
        for i in range(min(50, len(dataset))):
            item = dataset[i]
            assert item["positive_label"] in (0.0, 1.0)


# ============ 电商 DataLoader 测试 ============

class TestEcommerceDataLoader:
    """电商数据 DataLoader 测试"""

    def test_create_ecommerce_dataloader(self):
        dataset = EcommerceProductDataset(
            data_path="data/ecommerce_test",
            max_text_length=128,
            image_size=224,
            split="train",
        )
        loader = create_ecommerce_dataloader(
            dataset, batch_size=4, shuffle=False, num_workers=0,
        )
        batch = next(iter(loader))
        assert "pixel_values" in batch
        assert batch["pixel_values"].shape[0] == 4
        assert batch["pixel_values"].shape[1:] == (3, 224, 224)

    def test_ecommerce_collate_fn(self):
        """电商 collate 函数测试"""
        samples = [
            {
                "id": "P001",
                "pixel_values": torch.randn(3, 224, 224),
                "quality_score": 0.85,
                "text": "文案1",
            },
            {
                "id": "P002",
                "pixel_values": torch.randn(3, 224, 224),
                "quality_score": 0.72,
                "text": "文案2",
            },
        ]
        collated = _ecommerce_collate_fn(samples)
        assert collated["pixel_values"].shape == (2, 3, 224, 224)
        assert collated["text"] == ["文案1", "文案2"]
        assert collated["quality_score"].shape == (2,)

    def test_preference_dataloader(self):
        """偏好数据集 DataLoader"""
        dataset = CopyPreferenceDataset(
            data_path="data/ecommerce_test",
            max_length=128,
            split="train",
        )
        loader = create_ecommerce_dataloader(
            dataset, batch_size=4, shuffle=False, num_workers=0,
        )
        batch = next(iter(loader))
        assert "pixel_values" in batch
        assert batch["pixel_values"].shape[0] == 4

    def test_behavior_dataloader(self):
        """行为数据集 DataLoader"""
        dataset = UserBehaviorDataset(
            data_path="data/ecommerce_test",
            num_users=50,
            num_products=20,
            split="train",
        )
        loader = create_ecommerce_dataloader(
            dataset, batch_size=8, shuffle=False, num_workers=0,
        )
        batch = next(iter(loader))
        assert "user_features" in batch
        assert "product_features" in batch
        assert batch["user_features"].shape[0] == 8
