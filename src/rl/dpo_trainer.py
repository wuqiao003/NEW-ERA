"""
DPO (Direct Preference Optimization) 直接偏好优化
工业级实现，支持多模态场景
核心优势：规避RLHF复杂流程，无需单独奖励模型即可完成偏好对齐
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


class DPOLoss(nn.Module):
    """
    DPO损失函数
    支持多种变体: sigmoid (原版DPO), hinge, ipo
    L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]
    """

    def __init__(
        self,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        label_smoothing: float = 0.0,
        reference_free: bool = False,
    ):
        super().__init__()
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing
        self.reference_free = reference_free

        logger.info(f"📐 DPO损失: beta={beta}, type={loss_type}, smooth={label_smoothing}")

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: Optional[torch.Tensor] = None,
        reference_rejected_logps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算DPO损失
        Args:
            policy_chosen_logps: 策略模型对chosen的对数概率 [B]
            policy_rejected_logps: 策略模型对rejected的对数概率 [B]
            reference_chosen_logps: 参考模型对chosen的对数概率 [B]
            reference_rejected_logps: 参考模型对rejected的对数概率 [B]
        """
        if self.reference_free:
            # Reference-free DPO
            chosen_logratios = policy_chosen_logps
            rejected_logratios = policy_rejected_logps
        else:
            if reference_chosen_logps is None or reference_rejected_logps is None:
                raise ValueError("非reference_free模式需要提供参考模型log概率")
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # 核心: π(y_w)/π_ref(y_w) vs π(y_l)/π_ref(y_l)
        logits = self.beta * (chosen_logratios - rejected_logratios)

        # 计算损失
        if self.loss_type == "sigmoid":
            # 原版DPO
            if self.label_smoothing > 0:
                # 带标签平滑
                losses = (
                    -self.label_smoothing * F.logsigmoid(-logits)
                    - (1 - self.label_smoothing) * F.logsigmoid(logits)
                )
            else:
                losses = -F.logsigmoid(logits)

        elif self.loss_type == "hinge":
            # Hinge DPO
            losses = torch.relu(1 - logits)

        elif self.loss_type == "ipo":
            # IPO (Identity Preference Optimization)
            losses = (logits - 1 / (2 * self.beta)) ** 2

        else:
            raise ValueError(f"不支持的DPO损失类型: {self.loss_type}")

        loss = losses.mean()

        # 详细统计
        metrics = {
            "loss": loss.detach(),
            "chosen_rewards": (self.beta * chosen_logratios).detach().mean(),
            "rejected_rewards": (self.beta * rejected_logratios).detach().mean(),
            "reward_margin": (self.beta * (chosen_logratios - rejected_logratios)).detach().mean(),
            "accuracy": (chosen_logratios > rejected_logratios).float().mean().detach(),
            "logits_mean": logits.detach().mean(),
        }

        return loss, metrics


class DPOTrainer:
    """
    工业级DPO训练器
    支持多模态输入、QLoRA、梯度累积、混合精度
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        beta: float = 0.1,
        loss_type: str = "sigmoid",
        label_smoothing: float = 0.0,
        learning_rate: float = 5e-6,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 8,
        warmup_ratio: float = 0.1,
        device: str = "cuda",
        use_amp: bool = True,
    ):
        self.policy_model = policy_model.to(device)
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        self.global_step = 0

        # 参考模型 (冻结)
        if reference_model is not None:
            self.reference_model = reference_model.to(device)
        else:
            # 深拷贝策略模型作为参考
            self.reference_model = copy.deepcopy(policy_model).to(device)
        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

        # DPO损失
        self.dpo_loss = DPOLoss(
            beta=beta,
            loss_type=loss_type,
            label_smoothing=label_smoothing,
        )

        # 优化器
        self.optimizer = torch.optim.AdamW(
            [p for p in policy_model.parameters() if p.requires_grad],
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # 混合精度
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        logger.info(
            f"🎯 DPO训练器初始化: beta={beta}, lr={learning_rate}, "
            f"grad_accum={gradient_accumulation_steps}"
        )

    def _compute_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算模型输出的对数概率"""
        with torch.amp.autocast("cuda", enabled=self.use_amp):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                task="generation",
            )

        logits = outputs.get("logits")
        if logits is None:
            # 如果模型没有直接的logits输出，使用特征空间的相似度
            features = outputs.get("fused_features", outputs.get("text_features"))
            if features is not None:
                # 简化的log概率计算
                logps = -torch.norm(features, dim=-1)
                return logps
            return torch.zeros(input_ids.shape[0], device=self.device)

        # 标准log概率计算
        logps = self._get_batch_logps(logits, labels, attention_mask)
        return logps

    def _get_batch_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """从logits计算批量对数概率"""
        if logits.dim() == 2:
            # [B, V] -> 扩展为序列
            log_probs = F.log_softmax(logits, dim=-1)
            # 取对角线概率作为近似
            batch_logps = log_probs.mean(dim=-1)
            return batch_logps

        # [B, L, V] 标准序列输出
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        mask = attention_mask[:, 1:].clone()

        log_probs = F.log_softmax(logits, dim=-1)

        # Gather对应label的log prob
        per_token_logps = torch.gather(
            log_probs, 2, labels.unsqueeze(2)
        ).squeeze(2)

        # 按mask取平均
        batch_logps = (per_token_logps * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return batch_logps

    def train_step(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
        chosen_pixel_values: Optional[torch.Tensor] = None,
        rejected_pixel_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """DPO单步训练"""
        self.policy_model.train()

        # 移到设备
        chosen_input_ids = chosen_input_ids.to(self.device)
        chosen_attention_mask = chosen_attention_mask.to(self.device)
        rejected_input_ids = rejected_input_ids.to(self.device)
        rejected_attention_mask = rejected_attention_mask.to(self.device)

        if chosen_pixel_values is not None:
            chosen_pixel_values = chosen_pixel_values.to(self.device)
        if rejected_pixel_values is not None:
            rejected_pixel_values = rejected_pixel_values.to(self.device)

        # 策略模型的log概率
        policy_chosen_logps = self._compute_logps(
            self.policy_model, chosen_input_ids, chosen_attention_mask,
            chosen_input_ids, chosen_pixel_values,
        )
        policy_rejected_logps = self._compute_logps(
            self.policy_model, rejected_input_ids, rejected_attention_mask,
            rejected_input_ids, rejected_pixel_values,
        )

        # 参考模型的log概率 (no grad)
        with torch.no_grad():
            ref_chosen_logps = self._compute_logps(
                self.reference_model, chosen_input_ids, chosen_attention_mask,
                chosen_input_ids, chosen_pixel_values,
            )
            ref_rejected_logps = self._compute_logps(
                self.reference_model, rejected_input_ids, rejected_attention_mask,
                rejected_input_ids, rejected_pixel_values,
            )

        # DPO损失
        loss, metrics = self.dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
        )

        # 梯度累积
        scaled_loss = loss / self.gradient_accumulation_steps
        self.scaler.scale(scaled_loss).backward()

        self.global_step += 1

        if self.global_step % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(), self.max_grad_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}

    @torch.no_grad()
    def evaluate(self, eval_dataloader) -> Dict[str, float]:
        """评估DPO模型"""
        self.policy_model.eval()
        total_metrics = {}
        num_batches = 0

        for batch in eval_dataloader:
            chosen_logps = self._compute_logps(
                self.policy_model,
                batch["chosen_ids"].to(self.device),
                batch["chosen_mask"].to(self.device),
                batch["chosen_ids"].to(self.device),
            )
            rejected_logps = self._compute_logps(
                self.policy_model,
                batch["rejected_ids"].to(self.device),
                batch["rejected_mask"].to(self.device),
                batch["rejected_ids"].to(self.device),
            )
            ref_chosen_logps = self._compute_logps(
                self.reference_model,
                batch["chosen_ids"].to(self.device),
                batch["chosen_mask"].to(self.device),
                batch["chosen_ids"].to(self.device),
            )
            ref_rejected_logps = self._compute_logps(
                self.reference_model,
                batch["rejected_ids"].to(self.device),
                batch["rejected_mask"].to(self.device),
                batch["rejected_ids"].to(self.device),
            )

            _, metrics = self.dpo_loss(
                chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps
            )

            for k, v in metrics.items():
                val = v.item() if torch.is_tensor(v) else v
                total_metrics[k] = total_metrics.get(k, 0) + val
            num_batches += 1

        return {f"eval_{k}": v / max(num_batches, 1) for k, v in total_metrics.items()}

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            "policy_model": self.policy_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)
        logger.info(f"💾 DPO检查点已保存: {path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_model.load_state_dict(checkpoint["policy_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        logger.info(f"📂 DPO检查点已加载: {path}, step={self.global_step}")
