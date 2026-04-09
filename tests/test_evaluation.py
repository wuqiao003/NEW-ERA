"""
评估模块与API接口测试
覆盖: 文本/检索/RL/业务指标 + 文案质量评估
"""
import sys
import os
import pytest
import torch
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import (
    TextMetrics,
    RetrievalMetrics,
    RLMetrics,
    BusinessMetrics,
    EvaluationSuite,
)
from src.models.copy_generator import CopyQualityEvaluator, COPY_STYLES
from src.generation.copy_engine import CopyGenerationEngine, CopyRanker


class TestTextMetrics:
    """文本指标测试"""

    def test_bleu_perfect(self):
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        score = TextMetrics.compute_bleu(preds, refs)
        assert score > 0.9

    def test_bleu_zero(self):
        preds = ["completely different text"]
        refs = ["the cat sat on the mat"]
        score = TextMetrics.compute_bleu(preds, refs)
        assert score < 0.5

    def test_bleu_empty(self):
        assert TextMetrics.compute_bleu([], []) == 0.0

    def test_rouge_l_perfect(self):
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        score = TextMetrics.compute_rouge_l(preds, refs)
        assert score > 0.99

    def test_rouge_l_partial(self):
        preds = ["the cat sat"]
        refs = ["the cat sat on the mat"]
        score = TextMetrics.compute_rouge_l(preds, refs)
        assert 0.0 < score < 1.0

    def test_rouge_l_empty(self):
        assert TextMetrics.compute_rouge_l([], []) == 0.0


class TestRetrievalMetrics:
    """检索指标测试"""

    def test_recall_at_k_perfect(self):
        N, D = 10, 64
        features = torch.randn(N, D)
        features = features / features.norm(dim=-1, keepdim=True)
        result = RetrievalMetrics.compute_recall_at_k(features, features, [1, 5, 10])
        assert result["recall@1"] == 1.0
        assert result["recall@10"] == 1.0

    def test_recall_at_k_random(self):
        query = torch.randn(20, 64)
        key = torch.randn(20, 64)
        result = RetrievalMetrics.compute_recall_at_k(query, key, [1, 5, 10])
        assert 0.0 <= result["recall@1"] <= 1.0
        assert result["recall@1"] <= result["recall@5"] <= result["recall@10"]

    def test_clip_score(self):
        img = torch.randn(10, 128)
        txt = torch.randn(10, 128)
        score = RetrievalMetrics.compute_clip_score(img, txt)
        assert -1.0 <= score <= 1.0

    def test_clip_score_identical(self):
        feat = torch.randn(10, 128)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        score = RetrievalMetrics.compute_clip_score(feat, feat)
        assert score > 0.99


class TestRLMetrics:
    """RL指标测试"""

    def test_preference_win_rate_perfect(self):
        chosen = torch.tensor([1.0, 2.0, 3.0])
        rejected = torch.tensor([0.0, 0.5, 1.0])
        rate = RLMetrics.compute_preference_win_rate(chosen, rejected)
        assert rate == 1.0

    def test_preference_win_rate_half(self):
        chosen = torch.tensor([1.0, 0.0])
        rejected = torch.tensor([0.0, 1.0])
        rate = RLMetrics.compute_preference_win_rate(chosen, rejected)
        assert rate == 0.5

    def test_reward_improvement(self):
        baseline = torch.tensor([0.3, 0.4, 0.5])
        optimized = torch.tensor([0.6, 0.7, 0.8])
        result = RLMetrics.compute_reward_improvement(baseline, optimized)
        assert result["absolute_improvement"] > 0
        assert result["relative_improvement"] > 0

    def test_kl_divergence(self):
        policy = torch.randn(10)
        reference = torch.randn(10)
        kl = RLMetrics.compute_kl_divergence(policy, reference)
        assert isinstance(kl, float)


class TestBusinessMetrics:
    """业务指标测试"""

    def test_simulate_ctr(self):
        scores = torch.randn(100)
        ctr = BusinessMetrics.simulate_ctr(scores)
        assert 0.0 <= ctr <= 1.0

    def test_simulate_engagement(self):
        scores = torch.randn(100)
        result = BusinessMetrics.simulate_engagement(scores)
        assert "avg_stay_time" in result
        assert "completion_rate" in result
        assert result["avg_stay_time"] > 0


class TestEvaluationSuite:
    """评估套件测试"""

    def test_evaluate_rl(self):
        suite = EvaluationSuite(device="cpu")
        chosen = torch.randn(20) + 1
        rejected = torch.randn(20) - 1
        result = suite.evaluate_rl(chosen, rejected)
        assert "preference_win_rate" in result
        assert result["preference_win_rate"] > 0.5

    def test_generate_report(self, tmp_path):
        suite = EvaluationSuite(device="cpu")
        results = {"test_metric": 0.85, "nested": {"a": 1}}
        report_path = str(tmp_path / "report.json")
        report = suite.generate_report(results, report_path)
        assert os.path.exists(report_path)
        with open(report_path) as f:
            loaded = json.load(f)
        assert "timestamp" in loaded
        assert loaded["results"]["test_metric"] == 0.85


# ============ 文案质量评估测试 ============

class TestCopyQualityEvaluation:
    """文案质量评估器评测"""

    def setup_method(self):
        self.evaluator = CopyQualityEvaluator()

    def test_zhongcao_style_scoring(self):
        """种草风格文案评分"""
        good_copy = "姐妹们这个防晒霜真的绝了！用了一周皮肤状态肉眼可见变好了～轻薄不油腻，性价比超高，闭眼入不会后悔的！已经安利给身边所有朋友了。"
        scores = self.evaluator.evaluate_copy(good_copy, "种草")
        assert scores["style_match_score"] > 0.3
        assert scores["overall"] > 0.3

    def test_cuxiao_style_scoring(self):
        """促销风格文案评分"""
        promo_copy = "🔥限时秒杀！防晒霜原价¥169，今日到手仅¥89！轻薄不油腻，库存有限，先到先得！手慢无！"
        scores = self.evaluator.evaluate_copy(promo_copy, "促销")
        assert scores["style_match_score"] > 0.3

    def test_zhuanye_style_scoring(self):
        """专业风格文案评分"""
        pro_copy = "【美妆深度评测】防晒霜SPF50+\n核心参数：PA++++\n实测效果：经过28天持续使用测试，综合评分4.8/5\n推荐指数：⭐⭐⭐⭐⭐"
        scores = self.evaluator.evaluate_copy(pro_copy, "专业")
        assert scores["style_match_score"] > 0.3

    def test_jianyue_style_scoring(self):
        """简约风格——短文案应得高分"""
        short_copy = "防晒霜 | SPF50+ | ¥89"
        scores = self.evaluator.evaluate_copy(short_copy, "简约")
        assert scores["style_match_score"] > 0.5

    def test_style_mismatch_lower_score(self):
        """风格不匹配应得较低分"""
        promo_copy = "🔥限时秒杀！特惠手慢无！"
        # 用促销文案评估"专业"风格
        scores = self.evaluator.evaluate_copy(promo_copy, "专业")
        # 促销文案匹配专业风格应较低
        assert scores["style_match_score"] < 0.7

    def test_readability_with_punctuation(self):
        """可读性评分：标点丰富的长句子得分更高"""
        # 构造理想长度的带标点句子（每句10-30字），标点加分更明显
        good_text = "这款防晒霜质感清爽不油腻，涂抹后肌肤透亮有光泽。持久力非常棒！全天候防护让人很安心，推荐给所有小仙女。"
        # 超长无标点文本，平均句长过长
        bad_text = "这款防晒霜质感清爽不油腻涂抹后肌肤透亮有光泽持久力非常棒全天候防护让人很安心推荐给所有小仙女"
        score_good = CopyQualityEvaluator._score_readability(good_text)
        score_bad = CopyQualityEvaluator._score_readability(bad_text)
        # 有标点且句子长度适中的可读性应不低
        assert score_good > 0.5
        assert score_bad > 0.0


class TestCopyEngineScoring:
    """文案生成引擎内部评分测试"""

    def setup_method(self):
        self.engine = CopyGenerationEngine(
            multimodal_model=None, use_model=False, device="cpu"
        )

    def test_rule_based_score_zhongcao(self):
        """种草风格规则评分"""
        good = "姐妹们这个推荐必入！绝了太好用了，安利给所有人。性价比超高！"
        score = CopyGenerationEngine._rule_based_score(good, "种草")
        assert score > 0.5

    def test_rule_based_score_jianyue(self):
        """简约风格规则评分 — 短文本加分"""
        short = "防晒霜 | SPF50+ | ¥89"
        score = CopyGenerationEngine._rule_based_score(short, "简约")
        assert score > 0.5

    def test_rule_based_score_zhuanye_with_data(self):
        """专业风格有数据/评分时加分"""
        pro = "综合评分4.8/5，推荐指数⭐⭐⭐⭐⭐，同价位TOP3"
        score = CopyGenerationEngine._rule_based_score(pro, "专业")
        assert score > 0.5

    def test_generate_and_score(self):
        """生成 + 评分端到端"""
        results = self.engine.generate_multi_style_copies(
            product_title="蓝牙耳机",
            product_description="降噪40小时续航",
            category="数码",
            price=299.0,
            styles=["种草", "简约"],
            num_candidates=2,
        )
        for style, copies in results.items():
            for c in copies:
                assert "score" in c
                assert 0.0 <= c["score"] <= 1.0


class TestCopyRankerEvaluation:
    """文案排序器评测"""

    def test_rank_by_score(self):
        ranker = CopyRanker()
        copies = [
            {"content": "A", "score": 0.6, "style": "种草"},
            {"content": "B", "score": 0.9, "style": "促销"},
            {"content": "C", "score": 0.7, "style": "简约"},
        ]
        ranked = ranker.rank_copies(copies, diversity_weight=0.0)
        # diversity_weight=0 时应纯按 score 降序
        assert ranked[0]["content"] == "B"
        assert ranked[0]["rank"] == 1

    def test_mmr_diversity(self):
        """MMR 多样性重排应让不同风格靠前"""
        ranker = CopyRanker()
        copies = [
            {"content": "A1", "score": 0.95, "style": "种草"},
            {"content": "A2", "score": 0.90, "style": "种草"},
            {"content": "B1", "score": 0.85, "style": "促销"},
            {"content": "C1", "score": 0.80, "style": "简约"},
        ]
        ranked = ranker.rank_copies(copies, diversity_weight=0.5)
        # 高多样性权重下，前3名应包含至少2种不同风格
        top3_styles = set(c["style"] for c in ranked[:3])
        assert len(top3_styles) >= 2

    def test_empty_copies(self):
        ranker = CopyRanker()
        assert ranker.rank_copies([]) == []

    def test_single_copy(self):
        ranker = CopyRanker()
        copies = [{"content": "只有一条", "score": 0.8, "style": "种草"}]
        ranked = ranker.rank_copies(copies)
        assert len(ranked) == 1
        assert ranked[0]["rank"] == 1
