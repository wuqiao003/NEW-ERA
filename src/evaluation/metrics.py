"""
多维度评估系统
技术指标 + 业务指标 + 对比实验
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class TextMetrics:
    """文本生成质量指标"""

    @staticmethod
    def compute_bleu(predictions: List[str], references: List[str], n: int = 4) -> float:
        """计算BLEU-N分数 (简化实现)"""
        if not predictions or not references:
            return 0.0

        total_score = 0.0
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            if len(pred_tokens) == 0:
                continue

            score = 0.0
            for k in range(1, n + 1):
                pred_ngrams = set()
                ref_ngrams = set()

                for i in range(len(pred_tokens) - k + 1):
                    pred_ngrams.add(tuple(pred_tokens[i:i + k]))
                for i in range(len(ref_tokens) - k + 1):
                    ref_ngrams.add(tuple(ref_tokens[i:i + k]))

                if len(pred_ngrams) > 0:
                    overlap = len(pred_ngrams & ref_ngrams)
                    precision = overlap / len(pred_ngrams)
                    score += precision

            score /= n
            total_score += score

        return total_score / len(predictions)

    @staticmethod
    def compute_rouge_l(predictions: List[str], references: List[str]) -> float:
        """计算ROUGE-L分数"""
        if not predictions or not references:
            return 0.0

        def lcs_length(s1: List[str], s2: List[str]) -> int:
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i - 1] == s2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]

        total_f = 0.0
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()

            if not pred_tokens or not ref_tokens:
                continue

            lcs = lcs_length(pred_tokens, ref_tokens)
            precision = lcs / len(pred_tokens) if pred_tokens else 0
            recall = lcs / len(ref_tokens) if ref_tokens else 0

            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            total_f += f1

        return total_f / len(predictions)


class RetrievalMetrics:
    """检索与匹配指标"""

    @staticmethod
    def compute_recall_at_k(
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        k_list: List[int] = [1, 5, 10],
    ) -> Dict[str, float]:
        """
        计算Recall@K
        query_features: [N, D] 查询特征
        key_features: [N, D] 键特征 (gt配对在对角线)
        """
        query_norm = F.normalize(query_features, dim=-1)
        key_norm = F.normalize(key_features, dim=-1)

        similarity = torch.matmul(query_norm, key_norm.T)  # [N, N]
        N = similarity.shape[0]
        gt_indices = torch.arange(N, device=similarity.device)

        results = {}
        for k in k_list:
            _, topk_indices = similarity.topk(min(k, N), dim=-1)
            correct = (topk_indices == gt_indices.unsqueeze(1)).any(dim=1)
            results[f"recall@{k}"] = correct.float().mean().item()

        return results

    @staticmethod
    def compute_clip_score(
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> float:
        """计算CLIP Score (图文相似度)"""
        img_norm = F.normalize(image_features, dim=-1)
        txt_norm = F.normalize(text_features, dim=-1)
        scores = (img_norm * txt_norm).sum(dim=-1)
        return scores.mean().item()


class RLMetrics:
    """强化学习效果指标"""

    @staticmethod
    def compute_preference_win_rate(
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> float:
        """偏好胜率: chosen reward > rejected reward 的比例"""
        return (chosen_rewards > rejected_rewards).float().mean().item()

    @staticmethod
    def compute_reward_improvement(
        baseline_rewards: torch.Tensor,
        optimized_rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """奖励提升幅度"""
        baseline_mean = baseline_rewards.mean().item()
        optimized_mean = optimized_rewards.mean().item()
        improvement = (optimized_mean - baseline_mean) / (abs(baseline_mean) + 1e-8)

        return {
            "baseline_reward": baseline_mean,
            "optimized_reward": optimized_mean,
            "absolute_improvement": optimized_mean - baseline_mean,
            "relative_improvement": improvement,
        }

    @staticmethod
    def compute_kl_divergence(
        policy_logprobs: torch.Tensor,
        reference_logprobs: torch.Tensor,
    ) -> float:
        """策略KL散度"""
        return (policy_logprobs - reference_logprobs).mean().item()


class BusinessMetrics:
    """业务指标模拟"""

    @staticmethod
    def simulate_ctr(scores: torch.Tensor, threshold: float = 0.5) -> float:
        """模拟CTR"""
        probs = torch.sigmoid(scores)
        return (probs > threshold).float().mean().item()

    @staticmethod
    def simulate_engagement(
        scores: torch.Tensor,
        base_time: float = 10.0,
    ) -> Dict[str, float]:
        """模拟用户参与度"""
        probs = torch.sigmoid(scores)
        avg_stay_time = base_time * (1 + probs.mean().item())
        completion_rate = probs.mean().item()

        return {
            "avg_stay_time": avg_stay_time,
            "completion_rate": completion_rate,
        }


class EvaluationSuite:
    """
    综合评估套件
    运行全部评估指标，输出对比实验报告
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.text_metrics = TextMetrics()
        self.retrieval_metrics = RetrievalMetrics()
        self.rl_metrics = RLMetrics()
        self.business_metrics = BusinessMetrics()

    def evaluate_multimodal(
        self,
        model: nn.Module,
        eval_dataloader,
        task: str = "matching",
    ) -> Dict[str, Any]:
        """全面评估多模态模型"""
        model.eval()
        all_text_features = []
        all_vision_features = []
        all_fused_features = []

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch.get("input_ids")
                attention_mask = batch.get("attention_mask")
                pixel_values = batch.get("pixel_values")

                if input_ids is not None:
                    input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    task=task,
                )

                if "text_features" in outputs:
                    all_text_features.append(outputs["text_features"].cpu())
                if "vision_features" in outputs:
                    all_vision_features.append(outputs["vision_features"].cpu())
                if "fused_features" in outputs:
                    all_fused_features.append(outputs["fused_features"].cpu())

        results = {}

        # 检索指标
        if all_text_features and all_vision_features:
            text_feat = torch.cat(all_text_features, dim=0)
            vision_feat = torch.cat(all_vision_features, dim=0)

            recall = self.retrieval_metrics.compute_recall_at_k(text_feat, vision_feat, [1, 5, 10])
            clip_score = self.retrieval_metrics.compute_clip_score(vision_feat, text_feat)
            results.update(recall)
            results["clip_score"] = clip_score

        return results

    def evaluate_rl(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        baseline_rewards: Optional[torch.Tensor] = None,
        optimized_rewards: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """评估强化学习效果"""
        results = {
            "preference_win_rate": self.rl_metrics.compute_preference_win_rate(
                chosen_rewards, rejected_rewards
            ),
        }

        if baseline_rewards is not None and optimized_rewards is not None:
            improvement = self.rl_metrics.compute_reward_improvement(
                baseline_rewards, optimized_rewards
            )
            results.update(improvement)

        return results

    def run_comparison_experiment(
        self,
        sft_model: nn.Module,
        rl_model: nn.Module,
        eval_dataloader,
    ) -> Dict[str, Any]:
        """
        对比实验: SFT vs SFT+RL
        验证各项指标的提升幅度
        """
        logger.info("📊 运行对比实验: SFT vs SFT+RL")

        sft_results = self.evaluate_multimodal(sft_model, eval_dataloader)
        rl_results = self.evaluate_multimodal(rl_model, eval_dataloader)

        comparison = {}
        for key in set(sft_results.keys()) | set(rl_results.keys()):
            sft_val = sft_results.get(key, 0)
            rl_val = rl_results.get(key, 0)
            if isinstance(sft_val, (int, float)) and isinstance(rl_val, (int, float)):
                improvement = (rl_val - sft_val) / (abs(sft_val) + 1e-8)
                comparison[key] = {
                    "sft": sft_val,
                    "sft_rl": rl_val,
                    "improvement": f"{improvement * 100:.2f}%",
                }

        return comparison

    def generate_report(self, results: Dict[str, Any], output_path: str = "outputs/eval_report.json"):
        """生成评估报告"""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": results,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"📄 评估报告已保存: {output_path}")
        return report
