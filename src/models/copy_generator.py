"""
多风格营销文案生成引擎
核心功能：
  - 基于商品图片 + 文本信息，生成 5 种风格的营销文案
  - 支持批量生成、风格混合、质量评估
  - 集成奖励模型对文案质量排序
"""
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ============ 文案风格定义 ============

COPY_STYLES = {
    "种草": {
        "description": "安利种草风格，强调使用体验和真实感受",
        "temperature": 0.85,
        "top_p": 0.92,
        "prompt_prefix": "用种草安利的语气，分享真实使用体验：",
        "characteristics": ["口语化", "情感丰富", "有使用场景", "真实可信"],
    },
    "促销": {
        "description": "促销活动风格，强调价格优势和紧迫感",
        "temperature": 0.7,
        "top_p": 0.85,
        "prompt_prefix": "用促销活动的语气，突出优惠力度：",
        "characteristics": ["价格优势", "限时紧迫", "数据量化", "行动号召"],
    },
    "情感": {
        "description": "情感共鸣风格，强调情感价值和生活品质",
        "temperature": 0.9,
        "top_p": 0.95,
        "prompt_prefix": "用温暖细腻的语气，引发情感共鸣：",
        "characteristics": ["情感共鸣", "生活场景", "价值升华", "温暖治愈"],
    },
    "专业": {
        "description": "专业测评风格，强调参数、技术和客观评价",
        "temperature": 0.6,
        "top_p": 0.8,
        "prompt_prefix": "用专业测评的语气，客观分析产品：",
        "characteristics": ["数据支撑", "客观分析", "对比评测", "专业术语"],
    },
    "简约": {
        "description": "简洁明了风格，核心卖点直达",
        "temperature": 0.5,
        "top_p": 0.7,
        "prompt_prefix": "用最简洁的方式，提炼核心卖点：",
        "characteristics": ["简洁明了", "核心卖点", "一句话概括", "信息密度高"],
    },
}


class CopyGenerationHead(nn.Module):
    """
    文案生成任务头
    输入多模态融合特征，输出文案 token 序列的 logits
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 1024,
        vocab_size: int = 32000,
        max_length: int = 256,
        num_styles: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.max_length = max_length
        self.vocab_size = vocab_size

        # 风格嵌入
        self.style_embedding = nn.Embedding(num_styles, input_dim)

        # 特征增强
        self.feature_enhancer = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # concat: feature + style
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # 输出头
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

        # 质量预测头（辅助任务）
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        features: torch.Tensor,
        style_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: [B, D] 多模态融合特征
            style_ids: [B] 风格 ID (0=种草, 1=促销, 2=情感, 3=专业, 4=简约)
        Returns:
            logits: [B, vocab_size]
            quality_scores: [B]
        """
        B = features.shape[0]

        # 默认风格
        if style_ids is None:
            style_ids = torch.zeros(B, dtype=torch.long, device=features.device)

        style_feat = self.style_embedding(style_ids)  # [B, D]

        # 拼接特征 + 风格
        combined = torch.cat([features, style_feat], dim=-1)  # [B, 2D]
        enhanced = self.feature_enhancer(combined)  # [B, hidden]

        logits = self.output_proj(enhanced)  # [B, vocab_size]
        quality = self.quality_head(enhanced).squeeze(-1)  # [B]

        return {
            "logits": logits,
            "quality_scores": quality,
            "enhanced_features": enhanced,
        }


class MarketingCopyGenerator:
    """
    营销文案生成器
    核心流程: 商品图片+文本 → 多模态特征 → 多风格文案 → 奖励排序
    """

    STYLE_TO_ID = {"种草": 0, "促销": 1, "情感": 2, "专业": 3, "简约": 4}

    def __init__(
        self,
        multimodal_model: nn.Module,
        reward_model: Optional[nn.Module] = None,
        copy_head: Optional[CopyGenerationHead] = None,
        tokenizer=None,
        device: str = "cpu",
    ):
        self.device = device
        self.multimodal_model = multimodal_model.to(device).eval()
        self.reward_model = reward_model.to(device).eval() if reward_model else None
        self.copy_head = copy_head.to(device).eval() if copy_head else None
        self.tokenizer = tokenizer

        logger.info("✍️ 营销文案生成器初始化完成")

    @torch.no_grad()
    def generate(
        self,
        product_title: str,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        styles: Optional[List[str]] = None,
        num_variants: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        """
        为一个商品生成多风格文案

        Args:
            product_title: 商品名称
            pixel_values: [1, 3, H, W] 商品图片
            input_ids: [1, L] 文本 token IDs
            attention_mask: [1, L] attention mask
            styles: 要生成的风格列表，默认全部 5 种
            num_variants: 每种风格生成的变体数
            temperature: 采样温度
            top_p: nucleus sampling

        Returns:
            {
                "product_title": str,
                "copies": [
                    {"style": str, "content": str, "quality_score": float, "reward_scores": dict}
                ],
                "best_copy": {...},
                "latency_ms": float
            }
        """
        start = time.time()
        styles = styles or list(COPY_STYLES.keys())

        # 1. 提取多模态特征
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.multimodal_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            task="generation",
        )
        features = outputs.get(
            "fused_features",
            outputs.get("text_features", outputs.get("vision_features"))
        )

        if features is None:
            logger.error("模型未返回有效特征")
            return {"product_title": product_title, "copies": [], "latency_ms": 0}

        # 2. 为每种风格生成文案
        all_copies = []
        for style in styles:
            style_config = COPY_STYLES.get(style, COPY_STYLES["种草"])
            style_id = self.STYLE_TO_ID.get(style, 0)
            style_temp = temperature or style_config["temperature"]

            for variant_idx in range(num_variants):
                copy_result = self._generate_single_copy(
                    features=features,
                    style=style,
                    style_id=style_id,
                    product_title=product_title,
                    temperature=style_temp,
                    top_p=top_p,
                )
                all_copies.append(copy_result)

        # 3. 用奖励模型评分排序
        if self.reward_model and features is not None:
            all_copies = self._rank_by_reward(all_copies, features)

        # 排序：按综合得分降序
        all_copies.sort(key=lambda x: x.get("total_score", 0), reverse=True)

        latency = (time.time() - start) * 1000

        return {
            "product_title": product_title,
            "copies": all_copies,
            "best_copy": all_copies[0] if all_copies else None,
            "latency_ms": round(latency, 2),
        }

    def _generate_single_copy(
        self,
        features: torch.Tensor,
        style: str,
        style_id: int,
        product_title: str,
        temperature: float,
        top_p: float,
    ) -> Dict[str, Any]:
        """生成单条文案"""
        # 如果有 CopyGenerationHead 和 tokenizer，走模型生成路径
        if self.copy_head is not None:
            style_tensor = torch.tensor([style_id], device=self.device)
            head_out = self.copy_head(features, style_tensor)
            logits = head_out["logits"]
            quality = head_out["quality_scores"].item()

            # 如果有 tokenizer，解码 logits
            if self.tokenizer is not None:
                # Temperature + top-p sampling
                scaled_logits = logits / max(temperature, 0.01)
                probs = F.softmax(scaled_logits, dim=-1)

                # Top-p filtering
                sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

                # Sample tokens
                sampled_idx = torch.multinomial(sorted_probs.squeeze(0), num_samples=1)
                token_id = sorted_indices.squeeze(0)[sampled_idx].item()

                # 解码（实际场景需要自回归生成完整序列）
                try:
                    content = self.tokenizer.decode([token_id], skip_special_tokens=True)
                except Exception:
                    content = ""
            else:
                quality = quality
                content = ""
        else:
            quality = 0.5
            content = ""

        # 如果模型生成的内容为空或太短，使用基于模板的文案生成
        if len(content.strip()) < 10:
            content = self._template_generate(product_title, style)

        return {
            "style": style,
            "content": content,
            "quality_score": round(quality, 4),
            "reward_scores": {},
            "total_score": quality,
        }

    def _template_generate(self, product_title: str, style: str) -> str:
        """基于模板的文案生成（模型生成不足时的后备方案）"""
        style_config = COPY_STYLES.get(style, COPY_STYLES["种草"])
        prefix = style_config["prompt_prefix"]
        return f"{prefix}\n\n关于「{product_title}」：[模型生成内容将在训练后替换此模板]"

    @torch.no_grad()
    def _rank_by_reward(
        self,
        copies: List[Dict[str, Any]],
        features: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """使用奖励模型对文案评分"""
        if self.reward_model is None:
            return copies

        # 用同一个商品特征对所有文案评分
        reward_out = self.reward_model(features)

        for copy in copies:
            copy["reward_scores"] = {
                "content_quality": float(reward_out.get("content_quality", torch.tensor(0.5)).mean()),
                "user_preference": float(reward_out.get("user_preference", torch.tensor(0.5)).mean()),
                "business_compliance": float(reward_out.get("business_compliance", torch.tensor(0.5)).mean()),
                "relevance": float(reward_out.get("relevance", torch.tensor(0.5)).mean()),
            }
            total_reward = float(reward_out.get("total_reward", torch.tensor(0.5)).mean())
            # 综合分 = 质量分 * 0.4 + 奖励分 * 0.6
            copy["total_score"] = round(
                copy["quality_score"] * 0.4 + total_reward * 0.6, 4
            )

        return copies

    @torch.no_grad()
    def batch_generate(
        self,
        products: List[Dict[str, Any]],
        styles: Optional[List[str]] = None,
        num_variants: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        批量生成文案

        Args:
            products: [{"title": str, "pixel_values": Tensor, "input_ids": Tensor, ...}]
            styles: 生成风格
            num_variants: 每个风格的变体数
        """
        results = []
        for product in products:
            result = self.generate(
                product_title=product.get("title", "商品"),
                pixel_values=product.get("pixel_values"),
                input_ids=product.get("input_ids"),
                attention_mask=product.get("attention_mask"),
                styles=styles,
                num_variants=num_variants,
            )
            results.append(result)
        return results


class CopyQualityEvaluator:
    """
    文案质量评估器
    评估维度：
      1. 内容质量（流畅性、丰富度）
      2. 风格匹配度
      3. 商品相关性
      4. 用户吸引力预测
    """

    def __init__(self, reward_model: Optional[nn.Module] = None, device: str = "cpu"):
        self.reward_model = reward_model
        self.device = device

    def evaluate_copy(
        self,
        copy_text: str,
        style: str,
        product_info: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """评估单条文案质量"""
        scores = {}

        # 1. 基础文本质量
        scores["length_score"] = self._score_length(copy_text, style)
        scores["readability_score"] = self._score_readability(copy_text)
        scores["style_match_score"] = self._score_style_match(copy_text, style)

        # 2. 综合分
        scores["overall"] = round(
            scores["length_score"] * 0.2
            + scores["readability_score"] * 0.4
            + scores["style_match_score"] * 0.4,
            4,
        )
        return scores

    @staticmethod
    def _score_length(text: str, style: str) -> float:
        """根据风格评估长度合理性"""
        length = len(text)
        ideal_ranges = {
            "种草": (80, 300),
            "促销": (50, 200),
            "情感": (100, 350),
            "专业": (100, 400),
            "简约": (15, 80),
        }
        min_len, max_len = ideal_ranges.get(style, (50, 300))
        if min_len <= length <= max_len:
            return 1.0
        elif length < min_len:
            return max(0.3, length / min_len)
        else:
            return max(0.5, 1.0 - (length - max_len) / max_len)

    @staticmethod
    def _score_readability(text: str) -> float:
        """文本可读性评分"""
        if not text.strip():
            return 0.0
        # 句子数量
        sentences = re.split(r'[。！？!?；;，,\n]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.3

        avg_sentence_len = sum(len(s) for s in sentences) / len(sentences)
        # 理想句长 10-30 字
        if 10 <= avg_sentence_len <= 30:
            score = 1.0
        elif avg_sentence_len < 10:
            score = max(0.5, avg_sentence_len / 10)
        else:
            score = max(0.4, 1.0 - (avg_sentence_len - 30) / 50)

        # 有标点加分
        punct_ratio = sum(1 for c in text if c in '，。！？、；：""''…~') / max(len(text), 1)
        if 0.03 <= punct_ratio <= 0.15:
            score = min(1.0, score + 0.1)

        return round(score, 4)

    @staticmethod
    def _score_style_match(text: str, style: str) -> float:
        """风格匹配度评分"""
        style_keywords = {
            "种草": ["推荐", "安利", "好用", "绝了", "必入", "回购", "种草", "分享"],
            "促销": ["限时", "特惠", "秒杀", "折扣", "优惠", "到手价", "手慢无", "抢"],
            "情感": ["温暖", "幸福", "美好", "值得", "生活", "每一天", "自信", "陪伴"],
            "专业": ["评测", "参数", "对比", "分析", "成分", "技术", "实测", "综合"],
            "简约": [],  # 简约风格看长度
        }

        if style == "简约":
            return 1.0 if len(text) < 80 else max(0.3, 1.0 - (len(text) - 80) / 200)

        keywords = style_keywords.get(style, [])
        if not keywords:
            return 0.5

        hits = sum(1 for kw in keywords if kw in text)
        return min(1.0, hits / max(len(keywords) * 0.3, 1))
