"""
评估模块与API接口测试
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
