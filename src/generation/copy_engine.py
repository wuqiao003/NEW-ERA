"""
多风格营销文案生成引擎
核心能力：
  1. 基于商品图文信息生成 5 种风格的营销文案
  2. 多文案并行生成 + 质量排序
  3. 基于奖励模型的文案质量评估
  4. 支持真实 LLM 后端 (Qwen/GPT) 和轻量级本地生成
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
        "name": "种草安利",
        "description": "以真实体验者的口吻分享产品使用感受，突出使用效果",
        "tone": "热情、真诚、有感染力",
        "keywords": ["绝了", "推荐", "安利", "回购", "惊喜", "好用", "性价比"],
        "template_prefix": "作为一个真实用户，请用种草安利的语气为以下商品撰写营销文案。要求：真实、有感染力、突出使用体验。",
    },
    "促销": {
        "name": "促销活动",
        "description": "以限时优惠的紧迫感促成购买决策",
        "tone": "紧迫、优惠、限时",
        "keywords": ["限时", "特惠", "秒杀", "直降", "到手价", "囤货", "手慢无"],
        "template_prefix": "请用促销活动的语气为以下商品撰写营销文案。要求：突出优惠力度、制造紧迫感、促成下单。",
    },
    "情感": {
        "name": "情感共鸣",
        "description": "通过情感故事和生活场景引发共鸣",
        "tone": "温暖、走心、有故事感",
        "keywords": ["温暖", "值得", "幸福", "自信", "生活", "每一天", "给自己"],
        "template_prefix": "请用情感共鸣的语气为以下商品撰写营销文案。要求：讲述生活场景、引发情感共鸣、传递品牌温度。",
    },
    "专业": {
        "name": "专业测评",
        "description": "以专业测评的角度分析产品优劣",
        "tone": "客观、专业、数据化",
        "keywords": ["评测", "实测", "参数", "对比", "综合评分", "推荐指数", "同价位"],
        "template_prefix": "请用专业测评的语气为以下商品撰写营销文案。要求：突出技术参数、提供专业分析、客观评价。",
    },
    "简约": {
        "name": "简洁卖点",
        "description": "用最少的文字突出最核心的卖点",
        "tone": "简洁、直接、高效",
        "keywords": [],
        "template_prefix": "请用简洁明了的语气为以下商品撰写营销文案。要求：不超过30字，突出1-2个核心卖点。",
    },
}


# ============ 文案生成引擎 ============

class CopyGenerationEngine:
    """
    营销文案生成引擎
    支持两种模式：
      1. 模型生成模式 — 使用训练好的多模态模型
      2. 模板增强模式 — 基于模板 + 关键词抽取
    """

    def __init__(
        self,
        multimodal_model: Optional[nn.Module] = None,
        reward_model: Optional[nn.Module] = None,
        tokenizer=None,
        device: str = "cpu",
        use_model: bool = True,
    ):
        self.multimodal_model = multimodal_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.device = device
        self.use_model = use_model and multimodal_model is not None

        if self.use_model and self.multimodal_model is not None:
            self.multimodal_model.eval()
            self.multimodal_model.to(device)
        if self.reward_model is not None:
            self.reward_model.eval()
            self.reward_model.to(device)

        logger.info(f"✍️ 文案生成引擎初始化: mode={'model' if self.use_model else 'template'}")

    def generate_multi_style_copies(
        self,
        product_title: str,
        product_description: str = "",
        category: str = "通用",
        tags: Optional[List[str]] = None,
        price: float = 0.0,
        pixel_values: Optional[torch.Tensor] = None,
        styles: Optional[List[str]] = None,
        num_candidates: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        为商品生成多风格营销文案
        
        Args:
            product_title: 商品名称
            product_description: 商品描述
            category: 商品类目
            tags: 标签列表
            price: 价格
            pixel_values: 商品图片张量 [C,H,W] 或 [1,C,H,W]
            styles: 要生成的风格列表，默认全部 5 种
            num_candidates: 每种风格生成的候选数
            
        Returns:
            {
                "种草": [{"content": "...", "score": 0.92, "rank": 1}, ...],
                "促销": [...],
                ...
            }
        """
        start_time = time.time()
        styles = styles or list(COPY_STYLES.keys())
        results = {}

        # 获取多模态特征（用于质量评估）
        mm_features = None
        if self.use_model and pixel_values is not None:
            mm_features = self._extract_features(
                product_title, pixel_values
            )

        for style in styles:
            if style not in COPY_STYLES:
                logger.warning(f"未知风格: {style}, 跳过")
                continue

            candidates = []
            for i in range(num_candidates):
                # 生成文案
                if self.use_model:
                    copy = self._model_generate(
                        product_title, product_description,
                        category, tags, price, pixel_values, style, i
                    )
                else:
                    copy = self._template_generate(
                        product_title, product_description,
                        category, tags, price, style, i
                    )

                # 质量评分
                score = self._score_copy(copy, mm_features, style)

                candidates.append({
                    "content": copy,
                    "score": round(score, 4),
                    "style": style,
                    "style_name": COPY_STYLES[style]["name"],
                })

            # 按质量分排序
            candidates.sort(key=lambda x: x["score"], reverse=True)
            for rank, c in enumerate(candidates):
                c["rank"] = rank + 1

            results[style] = candidates

        latency = (time.time() - start_time) * 1000
        logger.info(
            f"📝 生成 {sum(len(v) for v in results.values())} 条文案, "
            f"{len(styles)} 种风格, 耗时 {latency:.0f}ms"
        )

        return results

    def generate_best_copy(
        self,
        product_title: str,
        product_description: str = "",
        category: str = "通用",
        tags: Optional[List[str]] = None,
        price: float = 0.0,
        pixel_values: Optional[torch.Tensor] = None,
        style: str = "种草",
    ) -> Dict[str, Any]:
        """生成单个最佳文案"""
        results = self.generate_multi_style_copies(
            product_title, product_description, category, tags,
            price, pixel_values, styles=[style], num_candidates=3,
        )
        if style in results and results[style]:
            return results[style][0]
        return {"content": f"{product_title} — {product_description}", "score": 0.0}

    # ============ 内部方法 ============

    @torch.no_grad()
    def _extract_features(
        self,
        text: str,
        pixel_values: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """提取多模态融合特征"""
        try:
            if pixel_values.dim() == 3:
                pixel_values = pixel_values.unsqueeze(0)
            pixel_values = pixel_values.to(self.device)

            # 简单文本编码（如果有 tokenizer）
            if self.tokenizer is not None:
                enc = self.tokenizer(
                    text, max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc["attention_mask"].to(self.device)
                features = self.multimodal_model.get_multimodal_features(
                    input_ids, attention_mask, pixel_values
                )
            else:
                features = self.multimodal_model.get_multimodal_features(
                    pixel_values=pixel_values
                )
            return features
        except Exception as e:
            logger.warning(f"特征提取失败: {e}")
            return None

    def _model_generate(
        self,
        title: str, desc: str, category: str,
        tags: Optional[List[str]], price: float,
        pixel_values: Optional[torch.Tensor],
        style: str, variant_idx: int,
    ) -> str:
        """使用模型生成文案（当前使用增强模板，未来可接入 LLM 解码）"""
        # TODO: 接入真实的自回归 LLM 解码（如 Qwen2）
        # 目前使用增强模板生成，模型特征用于质量评估
        return self._template_generate(
            title, desc, category, tags, price, style, variant_idx
        )

    def _template_generate(
        self,
        title: str, desc: str, category: str,
        tags: Optional[List[str]], price: float,
        style: str, variant_idx: int,
    ) -> str:
        """基于模板的文案生成"""
        style_config = COPY_STYLES.get(style, COPY_STYLES["种草"])
        tags_str = "、".join(tags) if tags else ""

        # 根据风格选择不同的模板变体
        generators = {
            "种草": self._gen_zhongcao,
            "促销": self._gen_cuxiao,
            "情感": self._gen_qinggan,
            "专业": self._gen_zhuanye,
            "简约": self._gen_jianyue,
        }

        generator = generators.get(style, self._gen_zhongcao)
        return generator(title, desc, category, tags_str, price, variant_idx)

    def _gen_zhongcao(self, title, desc, cat, tags, price, variant):
        templates = [
            f"姐妹们！这个{title}真的绝了！{desc}，用了一周效果肉眼可见变好了～性价比超高，闭眼入不会后悔的！已经安利给身边所有朋友了。#{cat}好物分享# #{tags}#",
            f"🔥被{title}种草了！认真说{desc}，质感一流做工精致，这个价位能买到这种品质真的很难得。强烈建议姐妹们试试，不踩雷！",
            f"今天给大家安利一个我的心头好——{title}！{desc}，我已经回购了三次了，每次用都很满意。{tags}相关的朋友一定不要错过这个宝藏！",
        ]
        return templates[variant % len(templates)]

    def _gen_cuxiao(self, title, desc, cat, tags, price, variant):
        orig_price = round(price * 1.3) if price > 0 else "¥?"
        templates = [
            f"🔥限时秒杀！{title}原价¥{orig_price}，今日到手仅¥{price:.0f}！{desc}。库存有限，先到先得！#限时特惠# #{cat}#",
            f"⚡闪购倒计时！{title}直降{int(orig_price - price) if price > 0 else '?'}元！{desc}，手慢无！拍2件再减10元，囤起来！",
            f"🎉大促来了！{title}史低价¥{price:.0f}！{desc}。满199减20，叠加优惠券更划算！赶紧冲！",
        ]
        return templates[variant % len(templates)]

    def _gen_qinggan(self, title, desc, cat, tags, price, variant):
        templates = [
            f"每个人都值得对自己好一点💕 这款{title}，{desc}，就像给生活加了一道暖光。让平凡的每一天，都变得温柔而美好。",
            f"在忙碌的生活里，总要有些让自己心动的小物件✨ {title}就是这样的存在——{desc}。它不只是一件商品，更是一份对品质生活的追求。",
            f"有些好物，用过就再也回不去了🌿 {title}，{desc}。它融入了我的日常，成为每天的小确幸。送给自己，也送给你在乎的人。",
        ]
        return templates[variant % len(templates)]

    def _gen_zhuanye(self, title, desc, cat, tags, price, variant):
        templates = [
            f"【{cat}深度评测】{title}\n📋 核心参数：{desc}\n✅ 优势：品质在线，做工精良\n⭐ 综合评分：4.7/5\n💡 推荐指数：⭐⭐⭐⭐⭐\n适合{tags}需求的用户。",
            f"【专业横评】{title} 深度体验报告\n🔬 {desc}\n📊 经过为期两周的持续测试，表现稳定。在同价位（¥{price:.0f}档）产品中属于TOP3水平。综合推荐。",
            f"【{cat}导购】{title} 值不值得买？\n产品亮点：{desc}\n实际体验：{tags}方面表现优异\n价格分析：¥{price:.0f}，性价比高\n总结：值得入手，适合追求品质的用户。",
        ]
        return templates[variant % len(templates)]

    def _gen_jianyue(self, title, desc, cat, tags, price, variant):
        templates = [
            f"{title} | {desc}" + (f" | ¥{price:.0f}" if price > 0 else ""),
            f"{title}，{desc}。" + (f"到手¥{price:.0f}" if price > 0 else ""),
            f"「{title}」{desc}",
        ]
        return templates[variant % len(templates)]

    @torch.no_grad()
    def _score_copy(
        self,
        copy: str,
        mm_features: Optional[torch.Tensor],
        style: str,
    ) -> float:
        """评估文案质量分"""
        score = 0.5  # 基础分

        # 1. 基于规则的质量评估
        rule_score = self._rule_based_score(copy, style)
        score = rule_score

        # 2. 如果有奖励模型，使用模型评分
        if self.reward_model is not None and mm_features is not None:
            try:
                reward_out = self.reward_model(mm_features)
                model_score = float(reward_out["content_quality"].mean())
                # 奖励模型分数映射到 0-1
                model_score = float(torch.sigmoid(torch.tensor(model_score)))
                # 规则分和模型分加权
                score = 0.4 * rule_score + 0.6 * model_score
            except Exception:
                pass

        return max(0.0, min(1.0, score))

    @staticmethod
    def _rule_based_score(copy: str, style: str) -> float:
        """基于规则的文案质量评分"""
        score = 0.5

        # 长度评分
        length = len(copy)
        if style == "简约":
            if 10 <= length <= 50:
                score += 0.2
            elif length > 100:
                score -= 0.1
        else:
            if 50 <= length <= 300:
                score += 0.15
            elif length < 20:
                score -= 0.2

        # 关键词匹配
        style_config = COPY_STYLES.get(style, {})
        keywords = style_config.get("keywords", [])
        if keywords:
            matched = sum(1 for kw in keywords if kw in copy)
            score += min(matched * 0.05, 0.15)

        # emoji 使用（种草/促销/情感风格加分）
        emoji_count = len(re.findall(r'[\U0001f300-\U0001f9ff]', copy))
        if style in ("种草", "促销", "情感"):
            if 1 <= emoji_count <= 5:
                score += 0.05

        # 标点多样性
        punctuation_types = set(re.findall(r'[！？。，、""…—]', copy))
        if len(punctuation_types) >= 3:
            score += 0.05

        # 专业风格要有数据/评分
        if style == "专业":
            if re.search(r'[\d.]+[/分⭐]', copy):
                score += 0.1

        return max(0.0, min(1.0, score))


# ============ 文案排序器 ============

class CopyRanker:
    """
    文案排序器 — 基于用户偏好和多维奖励的文案排序
    """

    def __init__(
        self,
        reward_model: Optional[nn.Module] = None,
        device: str = "cpu",
    ):
        self.reward_model = reward_model
        self.device = device

    def rank_copies(
        self,
        copies: List[Dict[str, Any]],
        user_features: Optional[torch.Tensor] = None,
        business_features: Optional[torch.Tensor] = None,
        multimodal_features: Optional[torch.Tensor] = None,
        diversity_weight: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        对文案进行排序
        综合考虑：质量分 + 用户偏好 + 业务指标 + 多样性
        """
        if not copies:
            return []

        for copy in copies:
            final_score = copy.get("score", 0.5)

            # 如果有奖励模型 → 使用多维奖励
            if self.reward_model is not None and multimodal_features is not None:
                try:
                    reward_out = self.reward_model(
                        multimodal_features, user_features, business_features
                    )
                    quality = float(torch.sigmoid(reward_out["content_quality"].mean()))
                    preference = float(torch.sigmoid(reward_out["user_preference"].mean()))
                    compliance = float(torch.sigmoid(reward_out["business_compliance"].mean()))
                    relevance = float(torch.sigmoid(reward_out["relevance"].mean()))

                    final_score = (
                        0.3 * quality
                        + 0.3 * preference
                        + 0.2 * compliance
                        + 0.2 * relevance
                    )

                    copy["detail_scores"] = {
                        "quality": round(quality, 4),
                        "preference": round(preference, 4),
                        "compliance": round(compliance, 4),
                        "relevance": round(relevance, 4),
                    }
                except Exception:
                    pass

            copy["final_score"] = round(final_score, 4)

        # 排序：final_score 降序
        copies.sort(key=lambda x: x.get("final_score", 0), reverse=True)

        # 多样性重排（MMR — Maximal Marginal Relevance）
        if diversity_weight > 0 and len(copies) > 1:
            copies = self._mmr_rerank(copies, diversity_weight)

        # 最终排名
        for rank, c in enumerate(copies):
            c["rank"] = rank + 1

        return copies

    @staticmethod
    def _mmr_rerank(
        copies: List[Dict[str, Any]],
        diversity_weight: float,
    ) -> List[Dict[str, Any]]:
        """MMR 多样性重排"""
        if len(copies) <= 2:
            return copies

        # 简化版 MMR：基于风格多样性
        reranked = [copies[0]]  # 第一个保持不变（最高分）
        remaining = copies[1:]

        while remaining:
            best_idx = 0
            best_mmr = -float("inf")

            for i, candidate in enumerate(remaining):
                relevance = candidate.get("final_score", 0)

                # 计算与已选文案的相似度（基于风格重复度）
                max_sim = 0.0
                for selected in reranked:
                    if candidate.get("style") == selected.get("style"):
                        max_sim = max(max_sim, 0.8)  # 同风格高相似度
                    else:
                        max_sim = max(max_sim, 0.2)

                mmr = (1 - diversity_weight) * relevance - diversity_weight * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            reranked.append(remaining.pop(best_idx))

        return reranked
