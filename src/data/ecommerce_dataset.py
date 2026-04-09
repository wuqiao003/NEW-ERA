"""
电商图文数据集 — 场景A落地核心
支持：
  1. 真实商品图文对数据加载 (JSON/CSV + 本地图片)
  2. 营销文案偏好对数据 (chosen vs rejected 文案)
  3. 用户行为数据 (点击/收藏/购买)
  4. 数据自动下载与预处理脚本
"""
import csv
import json
import os
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.data.preprocessing import ImagePreprocessor, TextPreprocessor


# ============ 数据结构定义 ============

@dataclass
class ProductItem:
    """商品数据结构"""
    product_id: str
    title: str
    description: str = ""
    category: str = "通用"
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    price: float = 0.0
    tags: List[str] = field(default_factory=list)
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class MarketingCopy:
    """营销文案数据结构"""
    copy_id: str
    product_id: str
    style: str  # 种草/促销/情感/专业/简约
    content: str
    quality_score: float = 0.0
    click_rate: float = 0.0
    conversion_rate: float = 0.0


@dataclass
class CopyPreferencePair:
    """文案偏好对 — 用于 DPO/RLHF 训练"""
    product_id: str
    product_title: str
    image_path: Optional[str] = None
    style: str = "通用"
    chosen_copy: str = ""
    rejected_copy: str = ""
    chosen_score: float = 0.0
    rejected_score: float = 0.0


# ============ 电商商品图文数据集 ============

class EcommerceProductDataset(Dataset):
    """
    电商商品图文数据集
    数据格式（JSON）:
    [
      {
        "product_id": "P001",
        "title": "夏季清爽防晒霜 SPF50+",
        "description": "轻薄不油腻，长效防晒12小时",
        "category": "美妆",
        "image_path": "images/products/P001.jpg",
        "price": 89.0,
        "tags": ["防晒", "夏季", "清爽"],
        "marketing_copies": {
          "种草": "姐妹们！这款防晒真的绝了...",
          "促销": "限时特惠！原价129现在只要89...",
          "情感": "夏天最怕的就是晒黑...",
          "专业": "采用物化结合防晒技术...",
          "简约": "SPF50+ PA++++，清爽不油腻"
        },
        "user_feedback": {
          "click_count": 1520,
          "favorite_count": 380,
          "purchase_count": 210,
          "avg_stay_time": 25.3
        }
      }
    ]
    """

    def __init__(
        self,
        data_path: str,
        image_dir: Optional[str] = None,
        tokenizer=None,
        image_processor=None,
        max_text_length: int = 512,
        image_size: int = 224,
        split: str = "train",
        include_copies: bool = True,
        target_style: Optional[str] = None,
    ):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir) if image_dir else self.data_path / "images"
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_length = max_text_length
        self.image_size = image_size
        self.split = split
        self.include_copies = include_copies
        self.target_style = target_style

        # 内置预处理器
        self.text_preprocessor = TextPreprocessor(max_length=max_text_length)
        self.img_preprocessor = ImagePreprocessor(image_size=image_size)

        self.products: List[Dict[str, Any]] = []
        self.samples: List[Dict[str, Any]] = []

        self._load_data()

        # 内置简易 tokenizer（当无外部 tokenizer 时用于生成 input_ids/labels）
        if self.tokenizer is None:
            self._builtin_vocab = self._build_char_vocab()

        logger.info(
            f"🛒 电商数据集加载完成: split={split}, 商品数={len(self.products)}, "
            f"训练样本数={len(self.samples)}"
        )

    def _load_data(self):
        """加载数据：优先真实数据，回退到示例数据"""
        data_file = self.data_path / f"products_{self.split}.json"
        csv_file = self.data_path / f"products_{self.split}.csv"

        if data_file.exists():
            self._load_json(data_file)
        elif csv_file.exists():
            self._load_csv(csv_file)
        else:
            logger.warning(f"⚠️ 未找到数据文件 {data_file}，生成电商示例数据")
            self._generate_ecommerce_demo_data()

        # 展开为训练样本
        self._expand_samples()

    def _load_json(self, path: Path):
        """加载 JSON 格式数据"""
        with open(path, "r", encoding="utf-8") as f:
            self.products = json.load(f)
        logger.info(f"📂 从 JSON 加载 {len(self.products)} 条商品数据")

    def _load_csv(self, path: Path):
        """加载 CSV 格式数据"""
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                product = {
                    "product_id": row.get("product_id", f"P{len(self.products):06d}"),
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "category": row.get("category", "通用"),
                    "image_path": row.get("image_path", ""),
                    "price": float(row.get("price", 0)),
                    "tags": row.get("tags", "").split(",") if row.get("tags") else [],
                }
                # 尝试解析 marketing_copies
                if "marketing_copies" in row and row["marketing_copies"]:
                    try:
                        product["marketing_copies"] = json.loads(row["marketing_copies"])
                    except json.JSONDecodeError:
                        product["marketing_copies"] = {}
                self.products.append(product)
        logger.info(f"📂 从 CSV 加载 {len(self.products)} 条商品数据")

    def _expand_samples(self):
        """将商品数据展开为训练样本（每个商品 × 每个文案风格 = 一个样本）"""
        self.samples = []
        styles = ["种草", "促销", "情感", "专业", "简约"]

        for product in self.products:
            copies = product.get("marketing_copies", {})
            feedback = product.get("user_feedback", {})

            if self.include_copies and copies:
                for style, copy_text in copies.items():
                    if self.target_style and style != self.target_style:
                        continue
                    self.samples.append({
                        "product_id": product["product_id"],
                        "title": product["title"],
                        "description": product.get("description", ""),
                        "category": product.get("category", "通用"),
                        "image_path": product.get("image_path", ""),
                        "price": product.get("price", 0),
                        "tags": product.get("tags", []),
                        "style": style,
                        "copy_text": copy_text,
                        "click_count": feedback.get("click_count", 0),
                        "favorite_count": feedback.get("favorite_count", 0),
                        "purchase_count": feedback.get("purchase_count", 0),
                        "avg_stay_time": feedback.get("avg_stay_time", 0),
                        "quality_score": self._compute_quality_score(feedback),
                    })
            else:
                # 无文案时，使用商品标题+描述作为训练文本
                self.samples.append({
                    "product_id": product["product_id"],
                    "title": product["title"],
                    "description": product.get("description", ""),
                    "category": product.get("category", "通用"),
                    "image_path": product.get("image_path", ""),
                    "price": product.get("price", 0),
                    "tags": product.get("tags", []),
                    "style": "原始",
                    "copy_text": product["title"] + " " + product.get("description", ""),
                    "quality_score": self._compute_quality_score(feedback),
                })

    @staticmethod
    def _compute_quality_score(feedback: Dict) -> float:
        """基于用户反馈计算综合质量分"""
        if not feedback:
            return 0.5
        click = feedback.get("click_count", 0)
        fav = feedback.get("favorite_count", 0)
        buy = feedback.get("purchase_count", 0)
        stay = feedback.get("avg_stay_time", 0)

        # 加权综合分（归一化到 0-1）
        score = 0.0
        if click > 0:
            score += min(fav / click, 1.0) * 0.3  # 收藏转化率
            score += min(buy / click, 1.0) * 0.4   # 购买转化率
            score += min(stay / 60.0, 1.0) * 0.3   # 停留时间（max 60s）
        return round(max(0.0, min(1.0, score)), 4)

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """加载图片，支持相对路径和绝对路径"""
        if not image_path:
            return None

        # 尝试多种路径
        candidates = [
            Path(image_path),
            self.image_dir / image_path,
            self.data_path / image_path,
        ]
        for path in candidates:
            if path.exists():
                try:
                    return Image.open(path).convert("RGB")
                except Exception as e:
                    logger.warning(f"⚠️ 图片加载失败 {path}: {e}")
                    return None
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        processed = {
            "id": sample["product_id"],
            "style": sample["style"],
            "category": sample["category"],
        }

        # 构建输入文本：商品标题 + 风格指令
        input_text = self._build_prompt(sample)
        target_text = sample["copy_text"]

        # 文本编码
        if self.tokenizer is not None:
            input_enc = self.tokenizer(
                input_text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            processed["input_ids"] = input_enc["input_ids"].squeeze(0)
            processed["attention_mask"] = input_enc["attention_mask"].squeeze(0)

            target_enc = self.tokenizer(
                target_text,
                max_length=self.max_text_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            processed["labels"] = target_enc["input_ids"].squeeze(0)
        else:
            # 无外部 tokenizer 时，使用内置字符级编码生成 input_ids / labels
            processed["input_ids"] = self._encode_text(input_text)
            processed["attention_mask"] = torch.ones_like(processed["input_ids"])
            processed["labels"] = self._encode_text(target_text)
            processed["text"] = input_text
            processed["target_text"] = target_text

        # 图像处理
        image = self._load_image(sample.get("image_path", ""))
        if image is not None:
            if self.image_processor is not None:
                pixel_values = self.image_processor(
                    images=image, return_tensors="pt"
                )["pixel_values"]
                processed["pixel_values"] = pixel_values.squeeze(0)
            else:
                processed["pixel_values"] = self.img_preprocessor.process(image)
        else:
            processed["pixel_values"] = torch.zeros(3, self.image_size, self.image_size)

        # 质量标签
        processed["quality_score"] = sample.get("quality_score", 0.5)
        processed["price"] = sample.get("price", 0.0)

        return processed

    def _build_char_vocab(self) -> Dict[str, int]:
        """构建内置字符级词表（在无外部 tokenizer 时使用）"""
        # 基础特殊 token
        vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
        idx = 4

        # 收集数据中出现的所有字符
        all_chars = set()
        for sample in self.samples:
            all_chars.update(sample.get("title", ""))
            all_chars.update(sample.get("description", ""))
            all_chars.update(sample.get("copy_text", ""))
        # 加入常用标点和数字
        for ch in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?;:，。！？；：、¥￥+-=()（）【】《》\"'\n":
            all_chars.add(ch)

        for ch in sorted(all_chars):
            if ch not in vocab:
                vocab[ch] = idx
                idx += 1

        logger.info(f"内置字符词表大小: {len(vocab)}")
        return vocab

    def _encode_text(self, text: str) -> torch.Tensor:
        """使用内置字符级词表编码文本为 token id 张量"""
        unk_id = self._builtin_vocab.get("[UNK]", 1)
        pad_id = self._builtin_vocab.get("[PAD]", 0)

        ids = [self._builtin_vocab.get(ch, unk_id) for ch in text]

        # 截断
        if len(ids) > self.max_text_length:
            ids = ids[: self.max_text_length]
        # 填充
        while len(ids) < self.max_text_length:
            ids.append(pad_id)

        return torch.tensor(ids, dtype=torch.long)

    @staticmethod
    def _build_prompt(sample: Dict) -> str:
        """构建输入提示词"""
        style_instructions = {
            "种草": "请用种草安利的语气为以下商品撰写营销文案，突出使用体验和效果：",
            "促销": "请用促销活动的语气为以下商品撰写营销文案，突出价格优惠和限时优惠：",
            "情感": "请用情感共鸣的语气为以下商品撰写营销文案，突出情感价值和生活场景：",
            "专业": "请用专业测评的语气为以下商品撰写营销文案，突出产品参数和技术优势：",
            "简约": "请用简洁明了的语气为以下商品撰写营销文案，突出核心卖点：",
        }
        instruction = style_instructions.get(
            sample["style"],
            "请为以下商品撰写营销文案："
        )

        parts = [instruction]
        parts.append(f"商品名称：{sample['title']}")
        if sample.get("description"):
            parts.append(f"商品描述：{sample['description']}")
        if sample.get("category"):
            parts.append(f"类别：{sample['category']}")
        if sample.get("tags"):
            parts.append(f"标签：{'、'.join(sample['tags'])}")
        if sample.get("price", 0) > 0:
            parts.append(f"价格：¥{sample['price']:.0f}")

        return "\n".join(parts)

    # ============ 电商示例数据生成 ============

    def _generate_ecommerce_demo_data(self, num_products: int = 200):
        """生成电商示例数据"""
        categories_data = {
            "美妆": {
                "products": [
                    ("水光精华液", "深层补水，24小时持续保湿，改善暗沉肌肤", 168.0, ["补水", "精华", "保湿"]),
                    ("哑光口红", "丝绒质地，持久不脱色，显白不挑肤色", 89.0, ["口红", "持久", "显白"]),
                    ("清爽防晒霜SPF50+", "轻薄不油腻，长效防晒12小时，适合敏感肌", 119.0, ["防晒", "清爽", "敏感肌"]),
                    ("卸妆水500ml", "温和无刺激，一擦即净，眼唇可用", 79.0, ["卸妆", "温和", "大容量"]),
                    ("气垫粉底", "自然裸妆感，遮瑕不假面，控油持妆8小时", 199.0, ["粉底", "遮瑕", "控油"]),
                ],
                "copies": {
                    "种草": "姐妹们这个{product}真的绝了！用了一周皮肤状态肉眼可见变好了～{desc}，性价比超高，闭眼入不会后悔的！",
                    "促销": "🔥限时秒杀！{product}原价¥{price_orig}，今日到手仅¥{price}！{desc}，库存不多，手慢无！",
                    "情感": "每个女孩都值得被温柔以待💕 这款{product}，{desc}，就像给肌肤穿上了一件隐形防护衣，让你自信出门每一天。",
                    "专业": "【专业评测】{product} 核心成分解析：{desc}。经过28天持续使用测试，{benefit}效果显著，综合评分4.8/5。",
                    "简约": "{product} | {desc} | ¥{price}",
                },
            },
            "数码": {
                "products": [
                    ("蓝牙耳机Pro", "主动降噪，40小时续航，Hi-Fi音质", 299.0, ["蓝牙", "降噪", "长续航"]),
                    ("机械键盘", "Cherry红轴，RGB背光，全键热插拔", 399.0, ["机械键盘", "Cherry", "RGB"]),
                    ("便携充电宝20000mAh", "65W快充，同时充3台设备，飞机可带", 159.0, ["充电宝", "快充", "大容量"]),
                    ("智能手表", "血氧检测，GPS运动追踪，7天续航", 899.0, ["智能手表", "健康", "运动"]),
                    ("Type-C扩展坞", "12合1接口，4K HDMI输出，100W PD充电", 189.0, ["扩展坞", "Type-C", "4K"]),
                ],
                "copies": {
                    "种草": "数码党必入！这个{product}用了直接回不去了😍 {desc}，做工质感一流，价格还很良心！",
                    "促销": "⚡闪购价！{product}限时直降！到手仅¥{price}！{desc}，买到就是赚到！",
                    "情感": "生活需要一点科技的温度🌟 这款{product}，{desc}，让每一天的效率和乐趣都提升一个level。",
                    "专业": "【深度体验】{product} 硬核参数：{desc}。实测{benefit}表现优异，同价位段TOP3推荐。",
                    "简约": "{product} · {desc} · ¥{price}",
                },
            },
            "食品": {
                "products": [
                    ("手工牛轧糖礼盒", "新西兰奶源，不粘牙，6种口味混合装", 59.0, ["牛轧糖", "零食", "礼盒"]),
                    ("冻干咖啡速溶", "精品阿拉比卡豆，3秒速溶，还原手冲风味", 89.0, ["咖啡", "冻干", "速溶"]),
                    ("有机坚果每日礼包", "7种坚果科学配比，独立小包装，30天量", 129.0, ["坚果", "有机", "健康"]),
                    ("红枣桂圆银耳羹", "即食免煮，低糖配方，美容养颜", 49.0, ["银耳羹", "养颜", "即食"]),
                    ("辣条大礼包", "童年味道，香辣Q弹，多种口味组合", 29.9, ["辣条", "零食", "怀旧"]),
                ],
                "copies": {
                    "种草": "吃货宝藏发现！这个{product}也太好吃了吧🤤 {desc}，一口下去幸福感爆棚，已经回购三次了！",
                    "促销": "🎉吃货福利！{product}今日特价¥{price}！{desc}，囤起来完全不心疼！",
                    "情感": "把幸福装进每一口🍬 这款{product}，{desc}，是送给自己和家人最好的小确幸。",
                    "专业": "【美食评测】{product}：{desc}。口感层次丰富，{benefit}，品质值得信赖。",
                    "简约": "{product} | {desc} | ¥{price}",
                },
            },
            "服饰": {
                "products": [
                    ("纯棉T恤", "300g重磅纯棉，不变形不起球，简约百搭", 69.0, ["T恤", "纯棉", "百搭"]),
                    ("高腰牛仔裤", "弹力面料，修身显瘦，四季可穿", 159.0, ["牛仔裤", "显瘦", "高腰"]),
                    ("轻薄羽绒服", "90%白鹅绒填充，可收纳便携，零下10℃保暖", 399.0, ["羽绒服", "轻薄", "保暖"]),
                    ("运动跑鞋", "EVA缓震鞋底，透气网面，适合日常慢跑", 259.0, ["跑鞋", "运动", "缓震"]),
                    ("真丝睡衣套装", "100%桑蚕丝，亲肤透气，四季舒适", 329.0, ["睡衣", "真丝", "舒适"]),
                ],
                "copies": {
                    "种草": "衣柜必备！这件{product}真的太好穿了👗 {desc}，上身效果贼好，闺蜜看了都要链接！",
                    "促销": "👔限时折扣！{product}到手价¥{price}！{desc}，错过等半年！",
                    "情感": "穿上喜欢的衣服，遇见更好的自己✨ 这款{product}，{desc}，让每一天都充满自信。",
                    "专业": "【面料实测】{product}：{desc}。经过水洗测试{benefit}，性价比出众。",
                    "简约": "{product} · {desc} · ¥{price}",
                },
            },
            "家居": {
                "products": [
                    ("记忆棉枕头", "慢回弹材质，人体工学设计，缓解颈椎压力", 149.0, ["枕头", "记忆棉", "护颈"]),
                    ("智能台灯", "无级调光，自然光护眼，APP智能控制", 199.0, ["台灯", "智能", "护眼"]),
                    ("香薰蜡烛礼盒", "天然大豆蜡，4种香型，燃烧时长40小时", 79.0, ["香薰", "蜡烛", "礼盒"]),
                    ("不锈钢保温杯", "316食品级钢，12小时保温，防漏设计", 89.0, ["保温杯", "不锈钢", "便携"]),
                    ("抗菌砧板", "小麦秸秆材质，抗菌率99.9%，不发霉", 39.0, ["砧板", "抗菌", "环保"]),
                ],
                "copies": {
                    "种草": "提升生活品质的好物！这个{product}用了生活幸福感直接拉满🏠 {desc}，强烈推荐！",
                    "促销": "🏡居家好价！{product}特惠¥{price}！{desc}，提升生活品质不等了！",
                    "情感": "家是最温暖的港湾🏠 这款{product}，{desc}，让家的每个角落都充满舒适与温馨。",
                    "专业": "【家居评测】{product}：{desc}。材质安全{benefit}，推荐指数⭐⭐⭐⭐⭐。",
                    "简约": "{product} | {desc} | ¥{price}",
                },
            },
        }

        self.products = []
        idx = 0
        products_per_category = max(1, num_products // len(categories_data))

        for category, cat_data in categories_data.items():
            base_products = cat_data["products"]
            copy_templates = cat_data["copies"]

            for repeat in range(max(1, products_per_category // len(base_products))):
                for name, desc, price, tags in base_products:
                    if idx >= num_products:
                        break

                    # 生成多风格文案
                    copies = {}
                    for style, template in copy_templates.items():
                        benefit_map = {
                            "美妆": "肤质改善", "数码": "性能稳定",
                            "食品": "口感评分4.5/5", "服饰": "无缩水无变形",
                            "家居": "使用寿命长",
                        }
                        price_orig = round(price * 1.3, 0)
                        copy = template.format(
                            product=name, desc=desc, price=price,
                            price_orig=price_orig,
                            benefit=benefit_map.get(category, "效果优异"),
                        )
                        copies[style] = copy

                    # 模拟用户反馈数据
                    base_click = np.random.randint(500, 5000)
                    fav_rate = np.random.uniform(0.1, 0.4)
                    buy_rate = np.random.uniform(0.03, 0.15)

                    product = {
                        "product_id": f"P{idx:06d}",
                        "title": name if repeat == 0 else f"{name} ({repeat + 1})",
                        "description": desc,
                        "category": category,
                        "image_path": f"images/products/{category}/P{idx:06d}.jpg",
                        "price": price + np.random.uniform(-price * 0.1, price * 0.1),
                        "tags": tags,
                        "marketing_copies": copies,
                        "user_feedback": {
                            "click_count": base_click,
                            "favorite_count": int(base_click * fav_rate),
                            "purchase_count": int(base_click * buy_rate),
                            "avg_stay_time": round(np.random.uniform(8.0, 45.0), 1),
                        },
                    }
                    self.products.append(product)
                    idx += 1

        logger.info(f"📦 生成 {len(self.products)} 条电商示例数据（5大类目）")


# ============ 文案偏好对数据集 ============

class CopyPreferenceDataset(Dataset):
    """
    营销文案偏好对数据集 — DPO/RLHF 训练专用
    数据格式:
    [
      {
        "product_id": "P001",
        "product_title": "防晒霜SPF50+",
        "image_path": "images/products/P001.jpg",
        "style": "种草",
        "chosen": "姐妹们这个防晒霜真的绝了！...",
        "rejected": "这款防晒霜还行，可以用...",
        "chosen_score": 0.92,
        "rejected_score": 0.45
      }
    ]
    """

    def __init__(
        self,
        data_path: str,
        image_dir: Optional[str] = None,
        tokenizer=None,
        image_processor=None,
        max_length: int = 512,
        image_size: int = 224,
        split: str = "train",
    ):
        self.data_path = Path(data_path)
        self.image_dir = Path(image_dir) if image_dir else self.data_path / "images"
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_size = image_size
        self.split = split

        self.img_preprocessor = ImagePreprocessor(image_size=image_size)
        self.pairs: List[Dict[str, Any]] = []

        self._load_data()
        logger.info(f"📋 文案偏好数据集: split={split}, 对数={len(self.pairs)}")

    def _load_data(self):
        """加载偏好对数据"""
        data_file = self.data_path / f"copy_preference_{self.split}.json"

        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                self.pairs = json.load(f)
        else:
            logger.warning(f"⚠️ 未找到偏好数据 {data_file}，生成示例偏好对")
            self._generate_demo_preferences()

    def _generate_demo_preferences(self, num_pairs: int = 500):
        """基于文案质量差异生成偏好对"""
        # 高质量文案 vs 低质量文案
        high_quality_templates = [
            "姐妹们！这个{product}真的太好用了！{desc}，用了一周效果肉眼可见，性价比超高，闭眼入不会后悔！已经安利给身边所有朋友了～",
            "🔥强烈推荐！{product}入手后幸福感直线飙升！{desc}，做工精致，用料扎实，完全超出这个价位的预期。第一次写评价就是因为它真的太值了！",
            "被{product}圈粉了！{desc}，刚开始还半信半疑，结果用了之后惊喜到不行😍 已经加购了第二件，这种品质这个价格闭眼冲就对了。",
            "分享一个最近发现的宝藏好物——{product}。{desc}，不管是自用还是送人都特别合适。包装精美，品质在线，已经成为我回购名单的常驻成员了！",
            "必须给{product}打call！{desc}。之前用过不少同类产品，这款是目前为止最满意的。细节做得很好，使用体验很棒，谁用谁知道！",
        ]

        low_quality_templates = [
            "{product}，{desc}。一般般吧。",
            "买了{product}，还行，没什么特别的。{desc}。",
            "{product}挺普通的，{desc}，没有想象中那么好。",
            "凑合能用，{product}，{desc}。不功不过。",
            "收到了{product}，{desc}。就那样吧，期望值不要太高。",
        ]

        products_info = [
            ("水光精华液", "深层补水保湿"),
            ("蓝牙耳机Pro", "40小时续航降噪"),
            ("手工牛轧糖", "不粘牙六种口味"),
            ("纯棉T恤", "300g重磅不变形"),
            ("记忆棉枕头", "护颈人体工学"),
            ("冻干咖啡", "还原手冲风味"),
            ("高腰牛仔裤", "弹力显瘦"),
            ("智能台灯", "护眼智能调光"),
            ("香薰蜡烛", "天然大豆蜡"),
            ("运动跑鞋", "EVA缓震透气"),
        ]

        styles = ["种草", "促销", "情感", "专业", "简约"]
        self.pairs = []

        for i in range(num_pairs):
            product, desc = products_info[i % len(products_info)]
            style = styles[i % len(styles)]

            chosen = high_quality_templates[i % len(high_quality_templates)].format(
                product=product, desc=desc
            )
            rejected = low_quality_templates[i % len(low_quality_templates)].format(
                product=product, desc=desc
            )

            self.pairs.append({
                "id": f"pref_{i:06d}",
                "product_id": f"P{i % 50:06d}",
                "product_title": product,
                "image_path": "",
                "style": style,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_score": round(np.random.uniform(0.75, 0.98), 3),
                "rejected_score": round(np.random.uniform(0.15, 0.45), 3),
            })

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        result = {
            "id": pair.get("id", f"pref_{idx}"),
            "style": pair.get("style", "通用"),
        }

        # 构建 prompt
        prompt = f"请为商品「{pair['product_title']}」撰写{pair.get('style', '种草')}风格的营销文案："

        if self.tokenizer is not None:
            prompt_enc = self.tokenizer(
                prompt, max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            chosen_enc = self.tokenizer(
                pair["chosen"], max_length=self.max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
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
            result["prompt"] = prompt
            result["chosen"] = pair["chosen"]
            result["rejected"] = pair["rejected"]

        # 图像（如果有）
        image_path = pair.get("image_path", "")
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                result["pixel_values"] = self.img_preprocessor.process(image)
            except Exception:
                result["pixel_values"] = torch.zeros(3, self.image_size, self.image_size)
        else:
            result["pixel_values"] = torch.zeros(3, self.image_size, self.image_size)

        result["chosen_score"] = pair.get("chosen_score", 0.8)
        result["rejected_score"] = pair.get("rejected_score", 0.3)

        return result


# ============ 用户行为数据集 ============

class UserBehaviorDataset(Dataset):
    """
    用户行为数据集 — 推荐模型训练
    数据格式:
    [
      {
        "user_id": "U001",
        "product_id": "P001",
        "action": "click",       # click / favorite / purchase / view
        "stay_time": 25.3,
        "context": {
          "hour": 14,
          "weekday": 3,
          "device": "mobile",
          "source": "search"
        }
      }
    ]
    """

    def __init__(
        self,
        data_path: str,
        num_users: int = 1000,
        num_products: int = 200,
        user_feature_dim: int = 64,
        product_feature_dim: int = 64,
        split: str = "train",
    ):
        self.data_path = Path(data_path)
        self.num_users = num_users
        self.num_products = num_products
        self.user_feature_dim = user_feature_dim
        self.product_feature_dim = product_feature_dim
        self.split = split

        self.behaviors: List[Dict[str, Any]] = []
        self.user_features: Dict[str, torch.Tensor] = {}
        self.product_features: Dict[str, torch.Tensor] = {}

        self._load_data()
        logger.info(f"👤 用户行为数据集: split={split}, 行为数={len(self.behaviors)}")

    def _load_data(self):
        data_file = self.data_path / f"user_behavior_{self.split}.json"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.behaviors = data.get("behaviors", [])
                # 用户特征和商品特征需要从特征文件加载或初始化
        else:
            self._generate_demo_behaviors()

        # 初始化特征嵌入
        self._init_features()

    def _init_features(self):
        """初始化用户和商品特征（真实场景中应从特征服务加载）"""
        torch.manual_seed(42)
        for i in range(self.num_users):
            uid = f"U{i:06d}"
            self.user_features[uid] = torch.randn(self.user_feature_dim)

        for i in range(self.num_products):
            pid = f"P{i:06d}"
            self.product_features[pid] = torch.randn(self.product_feature_dim)

    def _generate_demo_behaviors(self, num_behaviors: int = 5000):
        """生成示例用户行为数据"""
        actions = ["click", "favorite", "purchase", "view"]
        action_weights = [0.4, 0.15, 0.05, 0.4]
        sources = ["search", "recommend", "share", "homepage"]

        self.behaviors = []
        for i in range(num_behaviors):
            action = np.random.choice(actions, p=action_weights)
            user_id = f"U{np.random.randint(0, self.num_users):06d}"
            product_id = f"P{np.random.randint(0, self.num_products):06d}"

            self.behaviors.append({
                "behavior_id": f"B{i:08d}",
                "user_id": user_id,
                "product_id": product_id,
                "action": action,
                "stay_time": round(np.random.exponential(15.0), 1),
                "context": {
                    "hour": np.random.randint(0, 24),
                    "weekday": np.random.randint(0, 7),
                    "device": np.random.choice(["mobile", "pc", "tablet"], p=[0.7, 0.25, 0.05]),
                    "source": np.random.choice(sources, p=[0.3, 0.4, 0.1, 0.2]),
                },
            })

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        behavior = self.behaviors[idx]

        user_feat = self.user_features.get(
            behavior["user_id"],
            torch.zeros(self.user_feature_dim),
        )
        product_feat = self.product_features.get(
            behavior["product_id"],
            torch.zeros(self.product_feature_dim),
        )

        # 行为标签
        action_map = {"view": 0, "click": 1, "favorite": 2, "purchase": 3}
        label = action_map.get(behavior["action"], 0)

        # 隐式反馈二值标签（click/favorite/purchase=1, view=0）
        positive = 1.0 if behavior["action"] in ("click", "favorite", "purchase") else 0.0

        return {
            "user_features": user_feat,
            "product_features": product_feat,
            "action_label": label,
            "positive_label": positive,
            "stay_time": behavior.get("stay_time", 0.0),
        }


# ============ 工具函数 ============

def create_ecommerce_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """创建电商数据 DataLoader"""
    # CPU 环境下禁用 pin_memory（仅在有 CUDA 加速器时有效）
    if pin_memory and not torch.cuda.is_available():
        pin_memory = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=_ecommerce_collate_fn,
    )


def _ecommerce_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """电商数据批次整理"""
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
