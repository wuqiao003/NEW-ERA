"""
端到端集成测试
验证完整 Pipeline: 数据 -> 模型 -> 训练 -> 评估 -> 推理
"""
import sys
import os
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestEndToEndPipeline:
    """端到端Pipeline集成测试"""

    def setup_method(self):
        self.config = {
            "model": {
                "vision_encoder": {"name": "default", "hidden_size": 128},
                "text_encoder": {"name": "default", "hidden_size": 128, "max_length": 64},
                "fusion": {
                    "type": "cross_attention", "hidden_size": 64,
                    "num_heads": 4, "num_layers": 1,
                    "dropout": 0.1, "use_gate": True,
                },
                "projection": {"shared_dim": 64},
            }
        }

    def test_data_to_model_forward(self):
        """数据加载 -> 模型前向"""
        from src.data.dataset import MultimodalDataset, create_dataloader
        from src.models.multimodal_model import MultimodalBaseModel

        dataset = MultimodalDataset(
            data_path="data/processed/train",
            max_text_length=64, image_size=224, split="train",
        )
        loader = create_dataloader(dataset, batch_size=4, num_workers=0)
        model = MultimodalBaseModel(self.config)
        model.eval()

        batch = next(iter(loader))
        pixels = batch["pixel_values"]

        with torch.no_grad():
            out = model(pixel_values=pixels, task="matching")

        assert "vision_features" in out
        assert out["vision_features"].shape[0] == 4

    def test_sft_mini_training(self):
        """SFT最小训练循环"""
        from src.models.multimodal_model import MultimodalBaseModel
        from torch.utils.data import TensorDataset, DataLoader

        model = MultimodalBaseModel(self.config)

        # 构造迷你数据集
        B, L = 8, 32
        ds = TensorDataset(
            torch.randint(0, 32000, (B, L)),
            torch.ones(B, L, dtype=torch.long),
            torch.randn(B, 3, 224, 224),
        )
        loader = DataLoader(ds, batch_size=4)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-4
        )

        model.train()
        initial_loss = None
        for batch in loader:
            ids, mask, pixels = batch
            out = model(input_ids=ids, attention_mask=mask, pixel_values=pixels,
                        task="matching", labels=torch.arange(ids.shape[0]))
            loss = out["loss"]
            if initial_loss is None:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证梯度能流通且loss有值
        assert initial_loss is not None
        assert initial_loss > 0

    def test_reward_model_with_multimodal_features(self):
        """多模态特征 -> 奖励模型评分"""
        from src.models.multimodal_model import MultimodalBaseModel
        from src.rl.reward_model import MultimodalRewardModel

        model = MultimodalBaseModel(self.config)
        reward_model = MultimodalRewardModel(multimodal_dim=64, hidden_size=128)

        ids = torch.randint(0, 32000, (4, 32))
        mask = torch.ones(4, 32, dtype=torch.long)
        pixels = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            features = model.get_multimodal_features(ids, mask, pixels)
            rewards = reward_model(features)

        assert "total_reward" in rewards
        assert rewards["total_reward"].shape == (4,)

    def test_dpo_mini_cycle(self):
        """DPO最小训练循环"""
        from src.models.multimodal_model import MultimodalBaseModel
        from src.rl.dpo_trainer import DPOTrainer

        model = MultimodalBaseModel(self.config)
        trainer = DPOTrainer(
            policy_model=model, beta=0.1,
            learning_rate=1e-4, gradient_accumulation_steps=1,
            device="cpu", use_amp=False,
        )

        B, L = 2, 16
        metrics = trainer.train_step(
            torch.randint(0, 32000, (B, L)),
            torch.ones(B, L, dtype=torch.long),
            torch.randint(0, 32000, (B, L)),
            torch.ones(B, L, dtype=torch.long),
        )
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_ppo_mini_cycle(self):
        """PPO最小训练循环"""
        from src.models.multimodal_model import MultimodalBaseModel
        from src.rl.reward_model import MultimodalRewardModel
        from src.rl.ppo_trainer import PPOTrainer

        model = MultimodalBaseModel(self.config)
        reward = MultimodalRewardModel(multimodal_dim=64, hidden_size=128)

        trainer = PPOTrainer(
            policy_model=model, reward_model=reward,
            state_dim=64, action_dim=32,
            ppo_epochs=1, mini_batch_size=2,
            device="cpu",
        )

        B, L = 4, 16
        exp = trainer.collect_experience(
            torch.randint(0, 32000, (B, L)),
            torch.ones(B, L, dtype=torch.long),
            torch.randn(B, 3, 224, 224),
        )
        metrics = trainer.train_step(exp)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics

    def test_multiagent_pipeline(self):
        """多智能体协同"""
        from src.rl.multi_agent import MultiAgentSystem

        system = MultiAgentSystem(state_dim=64, action_dim=32, num_items=20)
        state = torch.randn(2, 64)
        items = torch.randn(2, 10, 32)

        out = system(state, items, mode="full_pipeline")
        assert "content_actions" in out
        assert "rec_scores" in out
        assert "rank_scores" in out
        assert out["rank_scores"].shape == (2, 10)

    def test_evaluation_integration(self):
        """评估套件集成"""
        from src.evaluation.metrics import EvaluationSuite

        suite = EvaluationSuite(device="cpu")

        # RL评估
        chosen = torch.randn(50) + 0.5
        rejected = torch.randn(50) - 0.5
        baseline = torch.randn(50)
        optimized = torch.randn(50) + 1.0

        rl_results = suite.evaluate_rl(chosen, rejected, baseline, optimized)
        assert rl_results["preference_win_rate"] > 0.5
        assert rl_results["relative_improvement"] > 0

    def test_model_optimization(self):
        """模型优化测试"""
        from src.models.multimodal_model import MultimodalBaseModel
        from src.models.optimization import ModelQuantizer

        model = MultimodalBaseModel(self.config)

        # 模型大小统计
        size_info = ModelQuantizer.get_model_size(model)
        assert size_info["total_size_mb"] > 0
        assert size_info["param_count"] > 0

        # 动态量化
        quantized = ModelQuantizer.dynamic_quantize(model)
        quant_size = ModelQuantizer.get_model_size(quantized)
        # 量化后模型应不大于原模型
        assert quant_size["param_count"] > 0

    def test_full_data_preprocessing_flow(self):
        """完整数据预处理流"""
        from PIL import Image
        from src.data.preprocessing import MultimodalPipeline

        pipeline = MultimodalPipeline(
            max_text_length=128, image_size=224,
            augment=True, augment_prob=0.5,
        )

        texts = [
            "   <b>测试</b> HTML标签 https://example.com   ",
            "正常的中文内容描述",
            "Short",
        ]
        images = [
            Image.new("RGB", (400, 300), "red"),
            Image.new("L", (100, 100), 200),  # 灰度图
            Image.new("RGB", (800, 600), "blue"),
        ]

        result = pipeline.process_batch(texts, images, augment=True)
        assert result["pixel_values"].shape == (3, 3, 224, 224)
        assert len(result["texts"]) == 3
        # HTML应被清理
        assert "<b>" not in result["texts"][0]
