"""
数据模块单元测试
覆盖: MultimodalDataset、PreferenceDataset、数据预处理Pipeline
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
