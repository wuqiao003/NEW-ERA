"""
多模态数据集 - 核心数据加载与处理
支持图文对数据、偏好对数据、用户行为数据
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset, DataLoader


@dataclass
class MultimodalSample:
    """多模态样本数据结构"""
    sample_id: str
    text: str
    image_path: Optional[str] = None
    image_tensor: Optional[torch.Tensor] = None
    label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreferencePair:
    """偏好数据对结构 (用于DPO/RLHF训练)"""
    prompt: str
    chosen: str  # 优选回答
    rejected: str  # 劣选回答
    chosen_image: Optional[str] = None
    rejected_image: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultimodalDataset(Dataset):
    """
    工业级多模态数据集
    支持图文对数据加载、预处理、缓存
    """

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        image_processor=None,
        max_text_length: int = 512,
        image_size: int = 224,
        split: str = "train",
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.split = split
        self.samples: List[Dict[str, Any]] = []

        self._load_data()
        logger.info(f"📦 多模态数据集加载完成: {split}, 样本数: {len(self.samples)}")

    def _load_data(self):
        """加载数据"""
        data_file = self.data_path / f"{self.split}.json"

        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                self.samples = json.load(f)
        else:
            # 生成示例数据用于开发测试
            logger.warning(f"⚠️ 数据文件不存在: {data_file}, 生成模拟数据")
            self.samples = self._generate_mock_data()

    def _generate_mock_data(self, num_samples: int = 1000) -> List[Dict[str, Any]]:
        """生成模拟数据用于开发与测试"""
        categories = ["科技", "美食", "旅行", "时尚", "运动", "教育", "娱乐", "健康"]
        mock_texts = [
            "这款新发布的AI芯片性能提升了50%，功耗降低30%",
            "三分钟学会做正宗的红烧肉，简单美味不翻车",
            "西藏自驾游攻略：318国道沿线最美风景打卡点",
            "2024春夏流行色趋势解读，这几个颜色一定要试",
            "居家健身计划：每天30分钟，一个月见效果",
            "考研英语高频词汇记忆法，效率提升3倍",
            "年度最佳电影TOP10，你看过几部？",
            "春季养生指南：这些食物帮你提高免疫力",
        ]

        samples = []
        for i in range(num_samples):
            category = categories[i % len(categories)]
            text = mock_texts[i % len(mock_texts)]
            samples.append({
                "id": f"sample_{i:06d}",
                "text": f"[{category}] {text} (样本{i})",
                "image_path": f"images/{category}/{i:06d}.jpg",
                "category": category,
                "label": category,
                "user_feedback": {
                    "click": np.random.randint(0, 2),
                    "like": np.random.randint(0, 2),
                    "share": np.random.randint(0, 2),
                    "stay_time": round(np.random.uniform(1.0, 60.0), 2),
                },
                "content_quality_score": round(np.random.uniform(0.3, 1.0), 3),
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        processed = {"id": sample["id"]}

        # 文本编码
        if self.tokenizer is not None:
            text_encoding = self.tokenizer(
                sample["text"],
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            processed["input_ids"] = text_encoding["input_ids"].squeeze(0)
            processed["attention_mask"] = text_encoding["attention_mask"].squeeze(0)
        else:
            processed["text"] = sample["text"]

        # 图像处理
        image_path = sample.get("image_path")
        if image_path and os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            if self.image_processor is not None:
                pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
                processed["pixel_values"] = pixel_values.squeeze(0)
            else:
                processed["image"] = image
        else:
            # 生成占位图像张量
            processed["pixel_values"] = torch.randn(3, self.image_size, self.image_size)

        # 标签与元数据
        if "label" in sample:
            processed["label"] = sample["label"]
        if "user_feedback" in sample:
            processed["user_feedback"] = sample["user_feedback"]
        if "content_quality_score" in sample:
            processed["quality_score"] = sample["content_quality_score"]

        return processed


class PreferenceDataset(Dataset):
    """
    偏好数据集 (DPO/RLHF训练专用)
    数据格式: {prompt, chosen, rejected, chosen_image, rejected_image}
    """

    def __init__(
        self,
        data_path: str,
        tokenizer=None,
        image_processor=None,
        max_length: int = 512,
        split: str = "train",
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.split = split
        self.pairs: List[Dict[str, Any]] = []

        self._load_data()
        logger.info(f"📦 偏好数据集加载完成: {split}, 数据对数: {len(self.pairs)}")

    def _load_data(self):
        """加载偏好对数据"""
        data_file = self.data_path / f"preference_{self.split}.json"

        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                self.pairs = json.load(f)
        else:
            logger.warning(f"⚠️ 偏好数据不存在: {data_file}, 生成模拟偏好对")
            self.pairs = self._generate_mock_preferences()

    def _generate_mock_preferences(self, num_pairs: int = 500) -> List[Dict[str, Any]]:
        """生成模拟偏好数据对"""
        prompts = [
            "请为这张美食图片生成一段吸引人的推荐文案",
            "根据用户浏览历史，推荐相关的旅行内容",
            "为这篇科技文章生成一个引人注目的标题",
            "请描述这张风景图片的主要内容和氛围",
            "生成一段运动健身的短视频脚本",
        ]

        pairs = []
        for i in range(num_pairs):
            prompt = prompts[i % len(prompts)]
            pairs.append({
                "id": f"pref_{i:06d}",
                "prompt": f"{prompt} (场景{i})",
                "chosen": f"高质量回答: 基于多模态信息，这是一个精心优化的内容，具有高吸引力和相关性。(#{i})",
                "rejected": f"低质量回答: 普通描述，缺乏个性化和深度，用户体验一般。(#{i})",
                "chosen_score": round(np.random.uniform(0.7, 1.0), 3),
                "rejected_score": round(np.random.uniform(0.1, 0.5), 3),
            })
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        result = {"id": pair["id"]}

        if self.tokenizer is not None:
            # Prompt编码
            prompt_enc = self.tokenizer(
                pair["prompt"], max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            # Chosen编码
            chosen_enc = self.tokenizer(
                pair["chosen"], max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            # Rejected编码
            rejected_enc = self.tokenizer(
                pair["rejected"], max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            result["prompt_ids"] = prompt_enc["input_ids"].squeeze(0)
            result["prompt_mask"] = prompt_enc["attention_mask"].squeeze(0)
            result["chosen_ids"] = chosen_enc["input_ids"].squeeze(0)
            result["chosen_mask"] = chosen_enc["attention_mask"].squeeze(0)
            result["rejected_ids"] = rejected_enc["input_ids"].squeeze(0)
            result["rejected_mask"] = rejected_enc["attention_mask"].squeeze(0)
        else:
            result["prompt"] = pair["prompt"]
            result["chosen"] = pair["chosen"]
            result["rejected"] = pair["rejected"]

        return result


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """创建标准DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_multimodal_collate_fn,
    )


def _multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """多模态数据批次整理函数"""
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            collated[key] = torch.tensor(values)
        else:
            collated[key] = values

    return collated
