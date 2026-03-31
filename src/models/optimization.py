"""
模型轻量化与优化
量化压缩、模型蒸馏、推理加速
"""
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from loguru import logger


class ModelQuantizer:
    """模型量化压缩"""

    @staticmethod
    def dynamic_quantize(model: nn.Module) -> nn.Module:
        """PyTorch动态量化 (INT8)"""
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8,
        )
        logger.info("✅ 动态INT8量化完成")
        return quantized

    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """获取模型大小统计"""
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        total_mb = (param_size + buffer_size) / (1024 * 1024)
        param_count = sum(p.numel() for p in model.parameters())

        return {
            "param_count": param_count,
            "param_size_mb": param_size / (1024 * 1024),
            "buffer_size_mb": buffer_size / (1024 * 1024),
            "total_size_mb": total_mb,
        }

    @staticmethod
    def benchmark_inference(
        model: nn.Module,
        input_tensors: Dict[str, torch.Tensor],
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """推理性能基准测试"""
        model.eval()
        device = next(model.parameters()).device

        # 移动输入到设备
        inputs = {k: v.to(device) for k, v in input_tensors.items()}

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model(**inputs, task="matching")

        # 同步CUDA
        if device.type == "cuda":
            torch.cuda.synchronize()

        # 计时
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                model(**inputs, task="matching")
                if device.type == "cuda":
                    torch.cuda.synchronize()
                latencies.append((time.perf_counter() - start) * 1000)

        return {
            "mean_latency_ms": sum(latencies) / len(latencies),
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "throughput_qps": 1000 / (sum(latencies) / len(latencies)),
        }


class ModelDistiller:
    """模型蒸馏"""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.5,
        learning_rate: float = 1e-4,
        device: str = "cuda",
    ):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.temperature = temperature
        self.alpha = alpha
        self.device = device

        for p in self.teacher.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=learning_rate
        )

        logger.info(
            f"🧪 蒸馏器初始化: T={temperature}, alpha={alpha}"
        )

    def distill_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """单步蒸馏"""
        self.student.train()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)

        # Teacher前向 (无梯度)
        with torch.no_grad():
            teacher_out = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                task="matching",
            )

        # Student前向
        student_out = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            task="matching",
        )

        # 蒸馏损失: 特征空间MSE + 任务损失
        distill_loss = torch.tensor(0.0, device=self.device)

        # 特征对齐损失
        teacher_feat = teacher_out.get("fused_features")
        student_feat = student_out.get("fused_features")
        if teacher_feat is not None and student_feat is not None:
            if teacher_feat.shape != student_feat.shape:
                # 投影对齐
                proj = nn.Linear(student_feat.shape[-1], teacher_feat.shape[-1]).to(self.device)
                student_feat = proj(student_feat)
            distill_loss = nn.functional.mse_loss(student_feat, teacher_feat)

        # 任务损失 (如果有标签)
        task_loss = student_out.get("loss", torch.tensor(0.0, device=self.device))

        # 总损失
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * task_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "distill_loss": distill_loss.item(),
            "task_loss": task_loss.item() if torch.is_tensor(task_loss) else task_loss,
        }


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_ids_shape: tuple = (1, 128),
    image_shape: tuple = (1, 3, 224, 224),
):
    """导出ONNX模型"""
    model.eval()
    device = next(model.parameters()).device

    dummy_input_ids = torch.randint(0, 32000, input_ids_shape).to(device)
    dummy_mask = torch.ones(input_ids_shape, dtype=torch.long).to(device)
    dummy_pixels = torch.randn(image_shape).to(device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_mask, dummy_pixels),
            str(output_path),
            input_names=["input_ids", "attention_mask", "pixel_values"],
            output_names=["fused_features"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "pixel_values": {0: "batch"},
            },
            opset_version=14,
        )
        logger.info(f"✅ ONNX模型导出: {output_path}")
    except Exception as e:
        logger.error(f"❌ ONNX导出失败: {e}")
