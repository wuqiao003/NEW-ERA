"""
SFT监督微调训练器
支持QLoRA、梯度累积、混合精度、分布式训练
"""
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader


class SFTTrainer:
    """
    工业级SFT微调训练器
    支持: QLoRA低秩适配 / 梯度累积 / 混合精度 / 早停
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_ratio: float = 0.1,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        save_steps: int = 500,
        eval_steps: int = 200,
        logging_steps: int = 50,
        output_dir: str = "outputs/sft",
        device: str = "cuda",
        use_amp: bool = True,
        task: str = "matching",
        early_stopping_patience: int = 5,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.output_dir = Path(output_dir)
        self.device = device
        self.use_amp = use_amp
        self.task = task
        self.early_stopping_patience = early_stopping_patience

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 优化器
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # 学习率调度
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = self._create_scheduler(warmup_steps, total_steps)

        # 混合精度
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device == "cuda")

        # 训练状态
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self.patience_counter = 0

        param_count = sum(p.numel() for p in trainable_params)
        total_count = sum(p.numel() for p in model.parameters())
        logger.info(
            f"🏋️ SFT训练器初始化: 可训练参数={param_count:,} / 总参数={total_count:,} "
            f"({100*param_count/total_count:.2f}%)"
        )

    def _create_scheduler(self, warmup_steps: int, total_steps: int):
        """创建带warmup的余弦退火调度器"""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> Dict[str, Any]:
        """完整训练循环"""
        logger.info(f"🚀 开始SFT训练: epochs={self.num_epochs}, task={self.task}")
        train_history = []
        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_metrics = self._train_epoch(epoch)
            train_history.append(epoch_metrics)

            # Epoch级评估
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                logger.info(
                    f"📊 Epoch {epoch+1}/{self.num_epochs} 评估: "
                    f"loss={eval_metrics['eval_loss']:.4f}"
                )

                # 早停检查
                if eval_metrics["eval_loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["eval_loss"]
                    self.patience_counter = 0
                    self._save_checkpoint("best_model")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_patience:
                        logger.info(f"⏹️ 早停触发: patience={self.early_stopping_patience}")
                        break

        total_time = time.time() - start_time
        logger.info(f"✅ SFT训练完成: 总耗时={total_time:.1f}s, 最佳eval_loss={self.best_eval_loss:.4f}")

        return {
            "train_history": train_history,
            "best_eval_loss": self.best_eval_loss,
            "total_time": total_time,
            "total_steps": self.global_step,
        }

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """单个Epoch训练"""
        self.model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(self.train_dataloader):
            loss = self._train_step(batch)
            epoch_loss += loss
            epoch_steps += 1

            # 日志
            if self.global_step % self.logging_steps == 0 and self.global_step > 0:
                avg_loss = epoch_loss / epoch_steps
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  Step {self.global_step}: loss={avg_loss:.4f}, lr={lr:.2e}"
                )

            # Step级评估
            if (self.eval_dataloader is not None
                    and self.global_step % self.eval_steps == 0
                    and self.global_step > 0):
                eval_metrics = self.evaluate()
                logger.info(f"  📊 Step {self.global_step} eval: loss={eval_metrics['eval_loss']:.4f}")
                self.model.train()

            # 保存检查点
            if self.global_step % self.save_steps == 0 and self.global_step > 0:
                self._save_checkpoint(f"checkpoint-{self.global_step}")

        return {
            "epoch": epoch + 1,
            "avg_loss": epoch_loss / max(epoch_steps, 1),
            "steps": epoch_steps,
        }

    def _train_step(self, batch: Dict[str, Any]) -> float:
        """单步训练"""
        # 准备输入
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        pixel_values = batch.get("pixel_values")
        labels = batch.get("labels")

        if input_ids is not None:
            input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        # 前向传播 (混合精度)
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                task=self.task,
                labels=labels,
            )
            loss = outputs.get("loss")
            if loss is None:
                logger.warning("⚠️ 模型未返回loss，跳过本步")
                self.global_step += 1
                return 0.0
            loss = loss / self.gradient_accumulation_steps

        # 反向传播
        self.scaler.scale(loss).backward()

        self.global_step += 1

        # 梯度累积后更新
        if self.global_step % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()

        return loss.item() * self.gradient_accumulation_steps

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """评估"""
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        for batch in self.eval_dataloader:
            input_ids = batch.get("input_ids")
            attention_mask = batch.get("attention_mask")
            pixel_values = batch.get("pixel_values")
            labels = batch.get("labels")

            if input_ids is not None:
                input_ids = input_ids.to(self.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.device)
            if labels is not None:
                labels = labels.to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                task=self.task,
                labels=labels,
            )
            loss = outputs.get("loss", torch.tensor(0.0))
            total_loss += loss.item()
            total_steps += 1

        return {"eval_loss": total_loss / max(total_steps, 1)}

    def _save_checkpoint(self, name: str):
        """保存检查点"""
        ckpt_dir = self.output_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
        }, ckpt_dir / "checkpoint.pt")

        logger.info(f"💾 检查点已保存: {ckpt_dir}")
