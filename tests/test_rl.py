"""
强化学习模块单元测试
覆盖: RewardModel、DPO、PPO、MultiAgent
"""
import sys
import os
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rl.reward_model import (
    MultiDimensionalRewardHead,
    MultimodalRewardModel,
    RewardModelTrainer,
)
from src.rl.dpo_trainer import DPOLoss, DPOTrainer
from src.rl.ppo_trainer import (
    ValueNetwork,
    PolicyNetwork,
    PPOTrainer,
    PPOExperience,
)
from src.rl.multi_agent import (
    ContentAgent,
    RecommendationAgent,
    RankingAgent,
    MultiAgentSystem,
    AgentCommunication,
)


# ============ 奖励模型测试 ============

class TestMultiDimensionalRewardHead:
    """多维奖励头测试"""

    def setup_method(self):
        self.head = MultiDimensionalRewardHead(input_dim=256, num_heads=4, hidden_dim=128)

    def test_output_keys(self):
        features = torch.randn(4, 256)
        scores = self.head(features)
        assert "content_quality" in scores
        assert "user_preference" in scores
        assert "business_compliance" in scores
        assert "relevance" in scores

    def test_output_shapes(self):
        features = torch.randn(4, 256)
        scores = self.head(features)
        for key, val in scores.items():
            assert val.shape == (4,), f"{key} shape mismatch"


class TestMultimodalRewardModel:
    """多模态奖励模型测试"""

    def setup_method(self):
        self.model = MultimodalRewardModel(
            multimodal_dim=128,
            user_feature_dim=32,
            business_feature_dim=16,
            hidden_size=256,
        )

    def test_forward_all_features(self):
        mm = torch.randn(4, 128)
        user = torch.randn(4, 32)
        biz = torch.randn(4, 16)
        out = self.model(mm, user, biz)
        assert "total_reward" in out
        assert "content_quality" in out
        assert "reward_weights" in out
        assert out["total_reward"].shape == (4,)

    def test_forward_only_multimodal(self):
        mm = torch.randn(4, 128)
        out = self.model(mm)
        assert out["total_reward"].shape == (4,)

    def test_preference_loss(self):
        chosen = torch.randn(4, 128)
        rejected = torch.randn(4, 128) * 0.5
        result = self.model.compute_preference_loss(chosen, rejected)
        assert "loss" in result
        assert "accuracy" in result
        assert result["loss"].requires_grad

    def test_reward_weights_sum_to_one(self):
        mm = torch.randn(2, 128)
        out = self.model(mm)
        weights = out["reward_weights"]
        assert abs(weights.sum().item() - 1.0) < 1e-5


class TestRewardModelTrainer:
    """奖励模型训练器测试"""

    def test_train_step(self):
        model = MultimodalRewardModel(multimodal_dim=64, user_feature_dim=32, business_feature_dim=16, hidden_size=128)
        trainer = RewardModelTrainer(model, learning_rate=1e-4, device="cpu")

        chosen = torch.randn(4, 64)
        rejected = torch.randn(4, 64) * 0.5
        metrics = trainer.train_step(chosen, rejected)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["loss"], float)


# ============ DPO测试 ============

class TestDPOLoss:
    """DPO损失测试"""

    def test_sigmoid_loss(self):
        loss_fn = DPOLoss(beta=0.1, loss_type="sigmoid")
        chosen = torch.randn(4)
        rejected = torch.randn(4) - 1
        ref_chosen = torch.randn(4)
        ref_rejected = torch.randn(4) - 1

        loss, metrics = loss_fn(chosen, rejected, ref_chosen, ref_rejected)
        assert loss.requires_grad
        assert "accuracy" in metrics

    def test_hinge_loss(self):
        loss_fn = DPOLoss(beta=0.1, loss_type="hinge")
        chosen = torch.randn(4)
        rejected = torch.randn(4)
        ref_chosen = torch.randn(4)
        ref_rejected = torch.randn(4)

        loss, metrics = loss_fn(chosen, rejected, ref_chosen, ref_rejected)
        assert loss >= 0

    def test_ipo_loss(self):
        loss_fn = DPOLoss(beta=0.1, loss_type="ipo")
        chosen = torch.randn(4)
        rejected = torch.randn(4)
        ref_chosen = torch.randn(4)
        ref_rejected = torch.randn(4)

        loss, metrics = loss_fn(chosen, rejected, ref_chosen, ref_rejected)
        assert loss >= 0

    def test_reference_free(self):
        loss_fn = DPOLoss(beta=0.1, reference_free=True)
        chosen = torch.randn(4)
        rejected = torch.randn(4)
        loss, metrics = loss_fn(chosen, rejected)
        assert loss.requires_grad

    def test_label_smoothing(self):
        loss_fn = DPOLoss(beta=0.1, label_smoothing=0.1)
        chosen = torch.randn(4)
        rejected = torch.randn(4)
        ref_chosen = torch.randn(4)
        ref_rejected = torch.randn(4)
        loss, _ = loss_fn(chosen, rejected, ref_chosen, ref_rejected)
        assert loss.requires_grad


class TestDPOTrainer:
    """DPO训练器测试"""

    def setup_method(self):
        from src.models.multimodal_model import MultimodalBaseModel
        config = {
            "model": {
                "vision_encoder": {"name": "default", "hidden_size": 128},
                "text_encoder": {"name": "default", "hidden_size": 128, "max_length": 64},
                "fusion": {"type": "cross_attention", "hidden_size": 64, "num_heads": 4, "num_layers": 1, "dropout": 0.1, "use_gate": False},
                "projection": {"shared_dim": 64},
            }
        }
        self.model = MultimodalBaseModel(config)
        self.trainer = DPOTrainer(
            policy_model=self.model,
            beta=0.1, learning_rate=1e-4,
            gradient_accumulation_steps=1,
            device="cpu", use_amp=False,
        )

    def test_train_step(self):
        B, L = 2, 32
        metrics = self.trainer.train_step(
            chosen_input_ids=torch.randint(0, 32000, (B, L)),
            chosen_attention_mask=torch.ones(B, L, dtype=torch.long),
            rejected_input_ids=torch.randint(0, 32000, (B, L)),
            rejected_attention_mask=torch.ones(B, L, dtype=torch.long),
            chosen_pixel_values=torch.randn(B, 3, 224, 224),
            rejected_pixel_values=torch.randn(B, 3, 224, 224),
        )
        assert "loss" in metrics
        assert "accuracy" in metrics


# ============ PPO测试 ============

class TestValueNetwork:
    """价值网络测试"""

    def test_forward(self):
        net = ValueNetwork(input_dim=128, hidden_dim=256)
        states = torch.randn(4, 128)
        values = net(states)
        assert values.shape == (4,)

    def test_gradient(self):
        net = ValueNetwork(input_dim=128)
        states = torch.randn(4, 128)
        values = net(states)
        values.sum().backward()
        assert all(p.grad is not None for p in net.parameters())


class TestPolicyNetwork:
    """策略网络测试"""

    def test_discrete_forward(self):
        net = PolicyNetwork(input_dim=128, action_dim=64, continuous=False)
        states = torch.randn(4, 128)
        out = net(states)
        assert "logits" in out
        assert out["logits"].shape == (4, 64)

    def test_continuous_forward(self):
        net = PolicyNetwork(input_dim=128, action_dim=64, continuous=True)
        states = torch.randn(4, 128)
        out = net(states)
        assert "mean" in out
        assert "std" in out

    def test_get_action_discrete(self):
        net = PolicyNetwork(input_dim=128, action_dim=64, continuous=False)
        states = torch.randn(4, 128)
        actions, log_probs = net.get_action(states)
        assert actions.shape == (4,)
        assert log_probs.shape == (4,)

    def test_get_action_continuous(self):
        net = PolicyNetwork(input_dim=128, action_dim=64, continuous=True)
        states = torch.randn(4, 128)
        actions, log_probs = net.get_action(states)
        assert actions.shape == (4, 64)
        assert log_probs.shape == (4,)

    def test_evaluate_actions_discrete(self):
        net = PolicyNetwork(input_dim=128, action_dim=64, continuous=False)
        states = torch.randn(4, 128)
        actions = torch.randint(0, 64, (4,))
        log_probs, entropy = net.evaluate_actions(states, actions)
        assert log_probs.shape == (4,)
        assert entropy.shape == (4,)

    def test_deterministic_mode(self):
        net = PolicyNetwork(input_dim=128, action_dim=64, continuous=False)
        states = torch.randn(2, 128)
        a1, _ = net.get_action(states, deterministic=True)
        a2, _ = net.get_action(states, deterministic=True)
        assert torch.equal(a1, a2)


class TestPPOTrainer:
    """PPO训练器测试"""

    def setup_method(self):
        from src.models.multimodal_model import MultimodalBaseModel
        from src.rl.reward_model import MultimodalRewardModel

        config = {
            "model": {
                "vision_encoder": {"name": "default", "hidden_size": 128},
                "text_encoder": {"name": "default", "hidden_size": 128, "max_length": 64},
                "fusion": {"type": "gated", "hidden_size": 64, "num_heads": 4, "num_layers": 1, "dropout": 0.1, "use_gate": False},
                "projection": {"shared_dim": 64},
            }
        }
        model = MultimodalBaseModel(config)
        reward_model = MultimodalRewardModel(multimodal_dim=64, hidden_size=128)

        self.trainer = PPOTrainer(
            policy_model=model,
            reward_model=reward_model,
            state_dim=64, action_dim=32,
            ppo_epochs=2, mini_batch_size=2,
            device="cpu",
        )

    def test_compute_gae(self):
        rewards = torch.randn(8)
        values = torch.randn(8)
        advantages, returns = self.trainer.compute_gae(rewards, values)
        assert advantages.shape == (8,)
        assert returns.shape == (8,)

    def test_collect_and_train(self):
        B, L = 4, 32
        ids = torch.randint(0, 32000, (B, L))
        mask = torch.ones(B, L, dtype=torch.long)
        pixels = torch.randn(B, 3, 224, 224)

        exp = self.trainer.collect_experience(ids, mask, pixels)
        assert exp.states is not None
        assert exp.rewards is not None

        metrics = self.trainer.train_step(exp)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "kl_div" in metrics


# ============ 多智能体测试 ============

class TestContentAgent:
    def test_forward(self):
        agent = ContentAgent(state_dim=128, action_dim=64)
        state = torch.randn(4, 128)
        out = agent(state)
        assert out["action_logits"].shape == (4, 64)
        assert out["value"].shape == (4,)

    def test_get_action(self):
        agent = ContentAgent(state_dim=128, action_dim=64)
        state = torch.randn(4, 128)
        action, log_prob, value = agent.get_action(state)
        assert action.shape == (4,)


class TestRecommendationAgent:
    def test_forward(self):
        agent = RecommendationAgent(state_dim=128, num_items=50)
        state = torch.randn(4, 128)
        out = agent(state)
        assert out["scores"].shape == (4, 50)

    def test_get_ranking(self):
        agent = RecommendationAgent(state_dim=128, num_items=50)
        state = torch.randn(4, 128)
        top = agent.get_ranking(state, top_k=5)
        assert top.shape == (4, 5)


class TestRankingAgent:
    def test_forward(self):
        agent = RankingAgent(state_dim=128, item_dim=64)
        user = torch.randn(4, 128)
        items = torch.randn(4, 10, 64)
        scores = agent(user, items)
        assert scores.shape == (4, 10)


class TestMultiAgentSystem:
    def setup_method(self):
        self.system = MultiAgentSystem(
            state_dim=128, action_dim=64,
            num_items=50, hidden_dim=256,
        )

    def test_full_pipeline(self):
        state = torch.randn(2, 128)
        items = torch.randn(2, 10, 64)
        out = self.system(state, items, mode="full_pipeline")
        assert "content_actions" in out
        assert "rec_scores" in out
        assert "rank_scores" in out

    def test_generate_only(self):
        state = torch.randn(2, 128)
        out = self.system(state, mode="generate_only")
        assert "content_actions" in out
        assert "rec_scores" not in out

    def test_recommend_only(self):
        state = torch.randn(2, 128)
        out = self.system(state, mode="recommend_only")
        assert "rec_scores" in out
        assert "content_actions" not in out


class TestAgentCommunication:
    def test_communication(self):
        comm = AgentCommunication(state_dim=128, hidden_dim=256)
        receiver = torch.randn(4, 128)
        sender = torch.randn(4, 128)
        result = comm(receiver, sender)
        assert result.shape == (4, 128)
