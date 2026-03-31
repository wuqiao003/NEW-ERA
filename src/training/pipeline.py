"""
端到端训练Pipeline
统一编排: 数据加载 -> SFT微调 -> 奖励模型训练 -> DPO偏好对齐 -> PPO决策优化
"""
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from loguru import logger

from src.data.dataset import MultimodalDataset, PreferenceDataset, create_dataloader
from src.models.multimodal_model import MultimodalBaseModel
from src.rl.reward_model import MultimodalRewardModel, RewardModelTrainer
from src.rl.dpo_trainer import DPOTrainer
from src.rl.ppo_trainer import PPOTrainer
from src.training.sft_trainer import SFTTrainer
from src.utils.config import ConfigManager, setup_seed, setup_logging


class TrainingPipeline:
    """
    全流程训练Pipeline
    Stage 1: SFT监督微调
    Stage 2: 奖励模型训练
    Stage 3: DPO偏好对齐
    Stage 4: PPO决策优化
    """

    def __init__(self, config_path: str = "configs/base_config.yaml"):
        self.config = ConfigManager.load(config_path)
        self.cfg = self.config.config

        # 全局设置
        setup_seed(self.cfg.get("project", {}).get("seed", 42))
        setup_logging("INFO")

        self.device = self.cfg.get("project", {}).get("device", "cuda")
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logger.warning("⚠️ CUDA不可用，回退到CPU")

        self.output_dir = Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🔧 训练Pipeline初始化: device={self.device}")

    def run_full_pipeline(self) -> Dict[str, Any]:
        """运行完整训练流程"""
        results = {}

        logger.info("=" * 60)
        logger.info("🚀 开始全流程训练Pipeline")
        logger.info("=" * 60)

        # Stage 1: SFT
        logger.info("\n📋 Stage 1/4: SFT 监督微调")
        sft_result = self.stage_sft()
        results["sft"] = sft_result

        # Stage 2: 奖励模型
        logger.info("\n📋 Stage 2/4: 奖励模型训练")
        rm_result = self.stage_reward_model()
        results["reward_model"] = rm_result

        # Stage 3: DPO
        logger.info("\n📋 Stage 3/4: DPO 偏好对齐")
        dpo_result = self.stage_dpo()
        results["dpo"] = dpo_result

        # Stage 4: PPO
        logger.info("\n📋 Stage 4/4: PPO 决策优化")
        ppo_result = self.stage_ppo()
        results["ppo"] = ppo_result

        logger.info("=" * 60)
        logger.info("✅ 全流程训练完成!")
        logger.info("=" * 60)

        return results

    def stage_sft(self) -> Dict[str, Any]:
        """Stage 1: SFT监督微调"""
        data_cfg = self.cfg.get("data", {})
        sft_cfg = self.cfg.get("sft", {})

        # 构建模型
        model = MultimodalBaseModel(self.cfg)

        # 加载数据
        train_dataset = MultimodalDataset(
            data_path=data_cfg.get("train_data_path", "data/processed/train"),
            max_text_length=data_cfg.get("max_text_length", 512),
            image_size=data_cfg.get("image_size", 224),
            split="train",
        )
        eval_dataset = MultimodalDataset(
            data_path=data_cfg.get("val_data_path", "data/processed/val"),
            max_text_length=data_cfg.get("max_text_length", 512),
            image_size=data_cfg.get("image_size", 224),
            split="val",
        )

        train_loader = create_dataloader(
            train_dataset,
            batch_size=data_cfg.get("batch_size", 16),
            shuffle=True,
            num_workers=0,
        )
        eval_loader = create_dataloader(
            eval_dataset,
            batch_size=data_cfg.get("batch_size", 16),
            shuffle=False,
            num_workers=0,
        )

        # 训练
        trainer = SFTTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            learning_rate=sft_cfg.get("learning_rate", 2e-4),
            num_epochs=sft_cfg.get("num_epochs", 3),
            gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 4),
            max_grad_norm=sft_cfg.get("max_grad_norm", 1.0),
            output_dir="outputs/sft",
            device=self.device,
        )

        result = trainer.train()
        self._sft_model = model  # 保留用于后续阶段
        return result

    def stage_reward_model(self) -> Dict[str, Any]:
        """Stage 2: 奖励模型训练"""
        rm_cfg = self.cfg.get("reward_model", {})

        # 构建奖励模型
        reward_model = MultimodalRewardModel(
            multimodal_dim=self.cfg["model"]["projection"]["shared_dim"],
            hidden_size=rm_cfg.get("hidden_size", 1024),
            num_reward_heads=rm_cfg.get("num_reward_heads", 4),
            reward_weights=rm_cfg.get("reward_weights"),
        )

        # 训练器
        rm_trainer = RewardModelTrainer(
            model=reward_model,
            learning_rate=rm_cfg.get("learning_rate", 1e-5),
            device=self.device,
        )

        # 模拟训练 (实际生产中接入真实偏好数据)
        logger.info("🏋️ 奖励模型训练...")
        metrics_history = []
        num_steps = 100

        for step in range(num_steps):
            # 模拟偏好对特征
            chosen = torch.randn(rm_cfg.get("batch_size", 8), 512)
            rejected = torch.randn(rm_cfg.get("batch_size", 8), 512) * 0.8

            metrics = rm_trainer.train_step(chosen, rejected)
            metrics_history.append(metrics)

            if (step + 1) % 20 == 0:
                logger.info(
                    f"  RM Step {step+1}/{num_steps}: "
                    f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}"
                )

        self._reward_model = reward_model
        return {"history": metrics_history, "final_metrics": metrics_history[-1]}

    def stage_dpo(self) -> Dict[str, Any]:
        """Stage 3: DPO偏好对齐"""
        dpo_cfg = self.cfg.get("dpo", {})

        model = getattr(self, "_sft_model", None)
        if model is None:
            model = MultimodalBaseModel(self.cfg)

        # DPO训练器
        dpo_trainer = DPOTrainer(
            policy_model=model,
            beta=dpo_cfg.get("beta", 0.1),
            loss_type=dpo_cfg.get("loss_type", "sigmoid"),
            learning_rate=dpo_cfg.get("learning_rate", 5e-6),
            gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 8),
            device=self.device,
        )

        # 模拟DPO训练
        logger.info("🎯 DPO偏好对齐训练...")
        metrics_history = []
        num_steps = 50

        for step in range(num_steps):
            batch_size = dpo_cfg.get("batch_size", 4)
            seq_len = 64

            chosen_ids = torch.randint(0, 32000, (batch_size, seq_len))
            chosen_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            rejected_ids = torch.randint(0, 32000, (batch_size, seq_len))
            rejected_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            pixel_values = torch.randn(batch_size, 3, 224, 224)

            metrics = dpo_trainer.train_step(
                chosen_ids, chosen_mask, rejected_ids, rejected_mask,
                pixel_values, pixel_values,
            )
            metrics_history.append(metrics)

            if (step + 1) % 10 == 0:
                logger.info(
                    f"  DPO Step {step+1}/{num_steps}: "
                    f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}"
                )

        dpo_trainer.save_checkpoint(str(self.output_dir / "dpo" / "checkpoint.pt"))
        self._dpo_model = model
        return {"history": metrics_history, "final_metrics": metrics_history[-1]}

    def stage_ppo(self) -> Dict[str, Any]:
        """Stage 4: PPO决策优化"""
        ppo_cfg = self.cfg.get("ppo", {})

        model = getattr(self, "_dpo_model", None) or getattr(self, "_sft_model", None)
        if model is None:
            model = MultimodalBaseModel(self.cfg)

        reward_model = getattr(self, "_reward_model", None)
        if reward_model is None:
            reward_model = MultimodalRewardModel(
                multimodal_dim=self.cfg["model"]["projection"]["shared_dim"]
            )

        # PPO训练器
        ppo_trainer = PPOTrainer(
            policy_model=model,
            reward_model=reward_model,
            state_dim=self.cfg["model"]["projection"]["shared_dim"],
            learning_rate=ppo_cfg.get("learning_rate", 1e-6),
            clip_range=ppo_cfg.get("clip_range", 0.2),
            gamma=ppo_cfg.get("gamma", 0.99),
            lam=ppo_cfg.get("lam", 0.95),
            ppo_epochs=ppo_cfg.get("ppo_epochs", 4),
            target_kl=ppo_cfg.get("target_kl", 0.01),
            device=self.device,
        )

        # 模拟PPO训练
        logger.info("🎮 PPO决策优化训练...")
        metrics_history = []
        num_steps = 30

        for step in range(num_steps):
            batch_size = ppo_cfg.get("batch_size", 4)
            seq_len = 64

            input_ids = torch.randint(0, 32000, (batch_size, seq_len))
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            pixel_values = torch.randn(batch_size, 3, 224, 224)

            # 采集经验
            experience = ppo_trainer.collect_experience(
                input_ids, attention_mask, pixel_values
            )

            # 训练
            metrics = ppo_trainer.train_step(experience)
            metrics_history.append(metrics)

            if (step + 1) % 5 == 0:
                logger.info(
                    f"  PPO Step {step+1}/{num_steps}: "
                    f"policy_loss={metrics['policy_loss']:.4f}, "
                    f"value_loss={metrics['value_loss']:.4f}, "
                    f"kl={metrics['kl_div']:.4f}"
                )

        ppo_trainer.save_checkpoint(str(self.output_dir / "ppo" / "checkpoint.pt"))
        return {"history": metrics_history, "final_metrics": metrics_history[-1]}


def main():
    """主入口"""
    pipeline = TrainingPipeline("configs/base_config.yaml")
    results = pipeline.run_full_pipeline()
    logger.info(f"📊 训练结果摘要: {list(results.keys())}")
    return results


if __name__ == "__main__":
    main()
