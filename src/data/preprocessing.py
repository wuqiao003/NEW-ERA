"""
多模态数据预处理Pipeline
数据清洗、增强、特征工程
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image, ImageFilter, ImageEnhance


class TextPreprocessor:
    """文本预处理器"""

    def __init__(self, max_length: int = 512, remove_urls: bool = True, remove_emojis: bool = False):
        self.max_length = max_length
        self.remove_urls = remove_urls
        self.remove_emojis = remove_emojis

    def clean(self, text: str) -> str:
        """文本清洗Pipeline"""
        if not text or not isinstance(text, str):
            return ""

        # 去除多余空白
        text = re.sub(r"\s+", " ", text).strip()

        # 去除URL
        if self.remove_urls:
            text = re.sub(r"http[s]?://\S+", "", text)

        # 去除HTML标签
        text = re.sub(r"<[^>]+>", "", text)

        # 去除特殊控制字符
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

        # 截断
        if len(text) > self.max_length:
            text = text[: self.max_length]

        return text.strip()

    def is_valid(self, text: str, min_length: int = 5) -> bool:
        """文本质量验证"""
        if not text or len(text.strip()) < min_length:
            return False
        # 检查是否全为标点或数字
        cleaned = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
        return len(cleaned) > 0


class ImagePreprocessor:
    """图像预处理器"""

    def __init__(self, image_size: int = 224, normalize: bool = True):
        self.image_size = image_size
        self.normalize = normalize
        self.mean = [0.48145466, 0.4578275, 0.40821073]  # CLIP标准化参数
        self.std = [0.26862954, 0.26130258, 0.27577711]

    def process(self, image: Image.Image) -> torch.Tensor:
        """图像预处理"""
        # 转RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 调整尺寸
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)

        # 转tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW

        # 标准化
        if self.normalize:
            for c in range(3):
                img_tensor[c] = (img_tensor[c] - self.mean[c]) / self.std[c]

        return img_tensor

    def is_valid(self, image: Image.Image, min_size: int = 64) -> bool:
        """图像质量验证"""
        if image is None:
            return False
        w, h = image.size
        if w < min_size or h < min_size:
            return False
        return True


class DataAugmentor:
    """多模态数据增强"""

    def __init__(self, augment_prob: float = 0.5):
        self.augment_prob = augment_prob

    def augment_image(self, image: Image.Image) -> Image.Image:
        """图像增强"""
        augmentations = [
            self._random_horizontal_flip,
            self._random_color_jitter,
            self._random_blur,
            self._random_crop_resize,
        ]

        for aug_fn in augmentations:
            if np.random.random() < self.augment_prob:
                image = aug_fn(image)

        return image

    def augment_text(self, text: str) -> str:
        """文本增强（简单版本，工业级可接入回译API）"""
        augmentations = [
            self._random_word_swap,
            self._random_char_insert,
        ]

        for aug_fn in augmentations:
            if np.random.random() < self.augment_prob * 0.3:  # 文本增强更保守
                text = aug_fn(text)

        return text

    def _random_horizontal_flip(self, image: Image.Image) -> Image.Image:
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def _random_color_jitter(self, image: Image.Image) -> Image.Image:
        factor = np.random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def _random_blur(self, image: Image.Image) -> Image.Image:
        radius = np.random.uniform(0.5, 1.5)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    def _random_crop_resize(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        crop_ratio = np.random.uniform(0.8, 0.95)
        new_w, new_h = int(w * crop_ratio), int(h * crop_ratio)
        left = np.random.randint(0, w - new_w + 1)
        top = np.random.randint(0, h - new_h + 1)
        image = image.crop((left, top, left + new_w, top + new_h))
        return image.resize((w, h), Image.BICUBIC)

    def _random_word_swap(self, text: str) -> str:
        words = text.split()
        if len(words) < 3:
            return text
        i, j = sorted(np.random.choice(len(words), 2, replace=False))
        words[i], words[j] = words[j], words[i]
        return " ".join(words)

    def _random_char_insert(self, text: str) -> str:
        # 简单重复字符增强
        if len(text) < 5:
            return text
        pos = np.random.randint(0, len(text))
        return text[:pos] + text[pos] + text[pos:]


class MultimodalPipeline:
    """端到端多模态数据处理Pipeline"""

    def __init__(
        self,
        max_text_length: int = 512,
        image_size: int = 224,
        augment: bool = True,
        augment_prob: float = 0.3,
    ):
        self.text_processor = TextPreprocessor(max_length=max_text_length)
        self.image_processor = ImagePreprocessor(image_size=image_size)
        self.augmentor = DataAugmentor(augment_prob=augment_prob) if augment else None

    def process_sample(
        self,
        text: str,
        image: Optional[Image.Image] = None,
        augment: bool = False,
    ) -> Dict[str, Any]:
        """处理单个多模态样本"""
        result = {}

        # 文本处理
        cleaned_text = self.text_processor.clean(text)
        if augment and self.augmentor:
            cleaned_text = self.augmentor.augment_text(cleaned_text)
        result["text"] = cleaned_text
        result["text_valid"] = self.text_processor.is_valid(cleaned_text)

        # 图像处理
        if image is not None:
            if augment and self.augmentor:
                image = self.augmentor.augment_image(image)
            result["pixel_values"] = self.image_processor.process(image)
            result["image_valid"] = self.image_processor.is_valid(image)
        else:
            result["pixel_values"] = torch.zeros(3, self.image_processor.image_size, self.image_processor.image_size)
            result["image_valid"] = False

        return result

    def process_batch(
        self,
        texts: List[str],
        images: Optional[List[Image.Image]] = None,
        augment: bool = False,
    ) -> Dict[str, Any]:
        """批量处理多模态样本"""
        batch_results = {
            "texts": [],
            "pixel_values": [],
            "text_valid": [],
            "image_valid": [],
        }

        for i, text in enumerate(texts):
            image = images[i] if images and i < len(images) else None
            result = self.process_sample(text, image, augment=augment)
            batch_results["texts"].append(result["text"])
            batch_results["pixel_values"].append(result["pixel_values"])
            batch_results["text_valid"].append(result["text_valid"])
            batch_results["image_valid"].append(result["image_valid"])

        batch_results["pixel_values"] = torch.stack(batch_results["pixel_values"])
        return batch_results
