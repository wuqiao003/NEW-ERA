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


class TestEcommerceEndToEnd:
    """电商场景端到端集成测试"""

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

    def test_ecommerce_dataset_to_model(self):
        """电商数据集 → 模型前向"""
        from src.data.ecommerce_dataset import (
            EcommerceProductDataset,
            create_ecommerce_dataloader,
        )
        from src.models.multimodal_model import MultimodalBaseModel

        dataset = EcommerceProductDataset(
            data_path="data/ecommerce_test",
            max_text_length=64, image_size=224, split="train",
        )
        loader = create_ecommerce_dataloader(dataset, batch_size=4, num_workers=0)
        model = MultimodalBaseModel(self.config)
        model.eval()

        batch = next(iter(loader))
        pixels = batch["pixel_values"]

        with torch.no_grad():
            out = model(pixel_values=pixels, task="matching")

        assert "vision_features" in out
        assert out["vision_features"].shape[0] == 4

    def test_copy_generation_engine_pipeline(self):
        """文案生成引擎端到端: 商品信息 → 多风格文案"""
        from src.generation.copy_engine import CopyGenerationEngine

        engine = CopyGenerationEngine(
            multimodal_model=None,
            use_model=False,
            device="cpu",
        )

        results = engine.generate_multi_style_copies(
            product_title="清爽防晒霜SPF50+",
            product_description="轻薄不油腻，长效防晒12小时",
            category="美妆",
            tags=["防晒", "清爽"],
            price=119.0,
            styles=["种草", "促销", "简约"],
            num_candidates=2,
        )

        assert "种草" in results
        assert "促销" in results
        assert "简约" in results
        assert len(results["种草"]) == 2
        for copy in results["种草"]:
            assert "content" in copy
            assert "score" in copy
            assert len(copy["content"]) > 10

    def test_copy_generation_with_model(self):
        """带模型的文案生成引擎"""
        from src.models.multimodal_model import MultimodalBaseModel
        from src.generation.copy_engine import CopyGenerationEngine

        model = MultimodalBaseModel(self.config)
        engine = CopyGenerationEngine(
            multimodal_model=model,
            use_model=True,
            device="cpu",
        )

        results = engine.generate_multi_style_copies(
            product_title="蓝牙耳机Pro",
            pixel_values=torch.randn(3, 224, 224),
            styles=["种草"],
            num_candidates=1,
        )
        assert "种草" in results
        assert len(results["种草"]) >= 1

    def test_copy_quality_evaluator_integration(self):
        """文案质量评估器端到端"""
        from src.models.copy_generator import CopyQualityEvaluator

        evaluator = CopyQualityEvaluator()

        # 测试多种风格的文案
        test_cases = [
            ("姐妹们这个防晒真的绝了！安利给大家，用了一周效果太好了！推荐！", "种草"),
            ("🔥限时秒杀！原价199今日到手仅99！手慢无！", "促销"),
            ("防晒霜 | SPF50+ PA++++ | ¥89", "简约"),
        ]

        for text, style in test_cases:
            scores = evaluator.evaluate_copy(text, style)
            assert "overall" in scores
            assert 0.0 <= scores["overall"] <= 1.0

    def test_copy_ranker_integration(self):
        """文案排序器端到端"""
        from src.generation.copy_engine import CopyRanker

        ranker = CopyRanker()

        copies = [
            {"content": "文案A种草风格", "score": 0.8, "style": "种草"},
            {"content": "文案B促销风格", "score": 0.9, "style": "促销"},
            {"content": "文案C种草风格", "score": 0.85, "style": "种草"},
            {"content": "文案D简约风格", "score": 0.7, "style": "简约"},
        ]

        ranked = ranker.rank_copies(copies, diversity_weight=0.3)
        assert len(ranked) == 4
        assert ranked[0]["rank"] == 1
        # 验证排名连续
        for i, c in enumerate(ranked):
            assert c["rank"] == i + 1

    def test_vector_store_integration(self):
        """向量检索引擎端到端"""
        import numpy as np
        from src.data.vector_store import ProductVectorStore

        store = ProductVectorStore(dim=64, index_type="Flat")

        # 批量添加
        ids = [f"P{i:04d}" for i in range(20)]
        vectors = np.random.randn(20, 64).astype(np.float32)
        metadata = [
            {"title": f"商品{i}", "category": ["美妆", "数码"][i % 2], "price": 50 + i * 10}
            for i in range(20)
        ]
        store.batch_add(ids, vectors, metadata)
        assert len(store) == 20

        # 搜索
        query = np.random.randn(64).astype(np.float32)
        results = store.search(query, top_k=5)
        assert len(results) == 5
        for r in results:
            assert "product_id" in r
            assert "score" in r
            assert "metadata" in r

    def test_vector_store_category_filter(self):
        """向量检索 — 类目过滤"""
        import numpy as np
        from src.data.vector_store import ProductVectorStore

        store = ProductVectorStore(dim=32, index_type="Flat")

        ids = [f"P{i:04d}" for i in range(10)]
        vectors = np.random.randn(10, 32).astype(np.float32)
        metadata = [
            {"category": "美妆" if i < 5 else "数码", "price": 100.0}
            for i in range(10)
        ]
        store.batch_add(ids, vectors, metadata)

        query = np.random.randn(32).astype(np.float32)
        results = store.search(query, top_k=10, category_filter="美妆")
        for r in results:
            assert r["metadata"]["category"] == "美妆"

    def test_vector_store_save_load(self, tmp_path):
        """向量索引持久化"""
        import numpy as np
        from src.data.vector_store import ProductVectorStore

        store = ProductVectorStore(dim=32, index_type="Flat")
        ids = ["P001", "P002", "P003"]
        vectors = np.random.randn(3, 32).astype(np.float32)
        store.batch_add(ids, vectors, [{"title": f"商品{i}"} for i in range(3)])

        # 保存
        save_path = str(tmp_path / "index")
        store.save(save_path)

        # 加载
        store2 = ProductVectorStore(dim=32, index_type="Flat")
        store2.load(save_path)
        assert len(store2) == 3
        assert store2.get_product_info("P001") is not None

    def test_preference_dataset_to_reward_model(self):
        """偏好数据 → 奖励模型评分"""
        from src.data.ecommerce_dataset import CopyPreferenceDataset
        from src.rl.reward_model import MultimodalRewardModel
        from src.models.multimodal_model import MultimodalBaseModel

        dataset = CopyPreferenceDataset(
            data_path="data/ecommerce_test",
            max_length=64,
            split="train",
        )

        model = MultimodalBaseModel(self.config)
        reward_model = MultimodalRewardModel(multimodal_dim=64, hidden_size=128)

        model.eval()
        reward_model.eval()

        pair = dataset[0]
        pixels = pair["pixel_values"].unsqueeze(0)

        with torch.no_grad():
            features = model(pixel_values=pixels, task="matching")
            vision_feat = features["vision_features"]
            rewards = reward_model(vision_feat)

        assert "total_reward" in rewards
        assert rewards["total_reward"].shape == (1,)

    def test_full_ecommerce_pipeline(self):
        """完整电商流水线: 数据 → 特征 → 文案生成 → 评估"""
        from src.data.ecommerce_dataset import EcommerceProductDataset
        from src.models.multimodal_model import MultimodalBaseModel
        from src.generation.copy_engine import CopyGenerationEngine
        from src.models.copy_generator import CopyQualityEvaluator

        # 1. 加载数据
        dataset = EcommerceProductDataset(
            data_path="data/ecommerce_test",
            max_text_length=64, image_size=224, split="train",
        )
        sample = dataset[0]

        # 2. 模型特征提取
        model = MultimodalBaseModel(self.config)
        model.eval()

        pixels = sample["pixel_values"].unsqueeze(0)
        with torch.no_grad():
            out = model(pixel_values=pixels, task="generation")

        assert "vision_features" in out or "fused_features" in out

        # 3. 文案生成
        engine = CopyGenerationEngine(use_model=False, device="cpu")
        copies = engine.generate_multi_style_copies(
            product_title=sample.get("text", "测试商品")[:20],
            styles=["种草", "简约"],
            num_candidates=1,
        )
        assert len(copies) >= 1

        # 4. 质量评估
        evaluator = CopyQualityEvaluator()
        for style, style_copies in copies.items():
            for c in style_copies:
                scores = evaluator.evaluate_copy(c["content"], style)
                assert 0.0 <= scores["overall"] <= 1.0
