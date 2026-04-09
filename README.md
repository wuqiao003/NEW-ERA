# MMRL Content System

**基于多模态决策强化学习的智能内容生成与推荐系统**

[![Python](https://img.shields.io/badge/Python-≥3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-122%20passed-brightgreen.svg)](#测试)

---

## 目录

- [项目简介](#项目简介)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [核心模块详解](#核心模块详解)
  - [多模态模型 (src/models)](#1-多模态模型-srcmodels)
  - [数据处理 (src/data)](#2-数据处理-srcdata)
  - [强化学习 (src/rl)](#3-强化学习-srcrl)
  - [训练管线 (src/training)](#4-训练管线-srctraining)
  - [评估系统 (src/evaluation)](#5-评估系统-srcevaluation)
  - [API 服务 (src/api)](#6-api-服务-srcapi)
  - [工具模块 (src/utils)](#7-工具模块-srcutils)
- [配置系统](#配置系统)
- [快速开始](#快速开始)
- [CLI 命令参考](#cli-命令参考)
- [API 接口文档](#api-接口文档)
- [训练流程](#训练流程)
- [模型优化与部署](#模型优化与部署)
- [测试](#测试)
- [Docker 部署](#docker-部署)
- [技术栈](#技术栈)
- [许可证](#许可证)

---

## 项目简介

MMRL Content System 是一个**工业级多模态强化学习系统**，覆盖从数据预处理、多模态模型训练、强化学习对齐到 API 部署的完整链路。系统采用 **SFT → RM → DPO → PPO** 四阶段训练范式，结合多智能体协同决策，实现高质量内容生成、个性化推荐与跨模态检索。

### 核心能力

- 🎯 **多模态融合**：CLIP-ViT 视觉编码 + Qwen2 文本编码 + 三种跨模态融合策略
- 🧠 **四阶段 RLHF**：SFT 监督微调 → 奖励模型训练 → DPO 偏好对齐 → PPO 策略优化
- 🤖 **多智能体协同**：内容生成 → 推荐排序 → 精排打分 三级联动
- 📊 **多维评估**：文本质量 / 检索精度 / RL 效果 / 业务指标全方位评估
- 🚀 **生产就绪**：FastAPI 服务 + Docker 容器化 + 模型量化/蒸馏/ONNX 导出

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLI 入口 (run.py / Click)                       │
├────────────┬────────────┬─────────────────┬─────────────────────────────┤
│   train    │   serve    │    evaluate     │           test              │
├────────────┴────────────┴─────────────────┴─────────────────────────────┤
│                                                                         │
│  ┌──────────────────── Training Pipeline (4 Stages) ─────────────────┐ │
│  │ Stage 1: SFT → Stage 2: Reward Model → Stage 3: DPO → Stage 4: PPO│ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────── MultimodalBaseModel ──────────────────────────────┐ │
│  │  VisionEncoder ──┐                                                │ │
│  │  (CLIP-ViT)      ├── CrossModalFusion ──── Task Heads             │ │
│  │  TextEncoder  ───┘   (Attention/Gate/MLP)  ├─ Matching (对比学习)  │ │
│  │  (Qwen2 + QLoRA)                          ├─ Generation (内容生成)│ │
│  │                                            └─ Recommendation (推荐)│ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────── RL Modules ───────────────────────────────────────┐ │
│  │ RewardModel (4-Head)  │  DPOTrainer  │  PPOTrainer (Actor-Critic) │ │
│  │ MultiAgentSystem: ContentAgent → RecommendAgent → RankingAgent    │ │
│  │                   └── AgentCommunication (门控消息传递) ──┘        │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────── API Layer (FastAPI) ──────────────────────────────┐ │
│  │ POST /api/v1/generate   │ POST /api/v1/recommend                  │ │
│  │ POST /api/v1/search     │ GET /health                             │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────── Evaluation Suite ─────────────────────────────────┐ │
│  │ BLEU / ROUGE-L │ Recall@K / CLIP Score │ Win Rate / KL Divergence │ │
│  │ CTR Simulation │ Engagement Metrics    │ SFT vs RL 对比报告       │ │
│  └───────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
mmrl-content-system/
├── configs/                          # 配置文件
│   ├── base_config.yaml              # 基础配置 (模型/训练/推理/评估)
│   └── deepspeed_config.json         # DeepSpeed 分布式训练配置
│
├── src/                              # 核心源代码
│   ├── __init__.py                   # 包入口 (版本 1.0.0)
│   │
│   ├── models/                       # 多模态模型模块
│   │   ├── vision_encoder.py         # 视觉编码器
│   │   ├── text_encoder.py           # 文本编码器
│   │   ├── fusion.py                 # 跨模态融合
│   │   ├── multimodal_model.py       # 多模态基座模型
│   │   └── optimization.py           # 量化 / 蒸馏 / ONNX 导出
│   │
│   ├── data/                         # 数据处理模块
│   │   ├── dataset.py                # 数据集定义与加载
│   │   └── preprocessing.py          # 文本/图像预处理与增强
│   │
│   ├── rl/                           # 强化学习模块
│   │   ├── reward_model.py           # 多维奖励模型
│   │   ├── dpo_trainer.py            # DPO 偏好优化
│   │   ├── ppo_trainer.py            # PPO 策略优化
│   │   └── multi_agent.py            # 多智能体协同系统
│   │
│   ├── training/                     # 训练模块
│   │   ├── pipeline.py               # 端到端训练编排
│   │   └── sft_trainer.py            # SFT 监督微调
│   │
│   ├── evaluation/                   # 评估模块
│   │   └── metrics.py                # 综合评估指标
│   │
│   ├── api/                          # API 服务
│   │   └── server.py                 # FastAPI 服务端
│   │
│   └── utils/                        # 工具
│       └── config.py                 # 配置管理 / 种子 / 日志
│
├── tests/                            # 测试套件 (122 tests)
│   ├── test_models.py                # 模型测试 (10 classes)
│   ├── test_rl.py                    # 强化学习测试 (12 classes)
│   ├── test_data.py                  # 数据测试 (7 classes)
│   ├── test_evaluation.py            # 评估测试 (5 classes)
│   ├── test_integration.py           # 集成测试 (10 methods)
│   ├── test_api.py                   # API 异步测试 (4 tests)
│   └── test_utils.py                 # 工具测试 (6 methods)
│
├── pyproject.toml                    # 构建配置与元数据
├── requirements.txt                  # Python 依赖
├── Dockerfile                        # Docker 容器配置
└── run.py                            # CLI 主入口
```

---

## 核心模块详解

### 1. 多模态模型 (`src/models`)

#### 视觉编码器 — `vision_encoder.py`

| 类 | 说明 |
|---|---|
| **`VisionEncoder`** | 工业级视觉编码器。优先加载 CLIP-ViT-L/14 预训练权重，回退使用 `LightweightViT`。支持全部冻结 / 仅冻结前 N 层的灵活策略。包含投影层将视觉特征映射到共享嵌入空间 |
| **`LightweightViT`** | 轻量级 Vision Transformer，用于开发测试。实现 Patch Embedding + CLS Token + 可学习位置编码 + Transformer Encoder |

**核心参数**:
- 默认模型：`openai/clip-vit-large-patch14`
- 隐藏维度：1024
- 图像尺寸：224 × 224，patch_size = 16
- 输出：`features` (CLS 特征) / `projected` (投影后) / `patch_features` (所有 patch)

#### 文本编码器 — `text_encoder.py`

| 类 | 说明 |
|---|---|
| **`TextEncoder`** | 工业级文本编码器。支持 Qwen2-7B-Instruct 等主流 LLM。内置 4-bit QLoRA 量化加载 (BitsAndBytes NF4)。支持 LoRA 微调 (r=16, alpha=32)。对 token 特征做均值池化后投影到共享空间 |
| **`LightweightTextEncoder`** | 轻量级文本编码器，用于开发测试。词嵌入 + 位置嵌入 + Transformer Encoder |

**QLoRA 配置**:
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
```

#### 跨模态融合 — `fusion.py`

系统提供三种融合策略，通过配置灵活切换：

| 融合方式 | 类名 | 原理 |
|----------|------|------|
| **交叉注意力** | `CrossAttentionFusion` | 双向 Text↔Vision 交叉注意力，捕获细粒度跨模态语义对齐 |
| **门控融合** | `GatedFusion` | Sigmoid 门控自适应学习模态融合权重，解决模态信息失衡 |
| **MLP 融合** | `MLPFusion` | 特征拼接 + 三层 MLP，简单高效的基线方案 |

`MultimodalFusionModule` 作为总调度器，根据 `config.fusion.type` 选择策略，并可叠加额外门控增强。

#### 多模态基座模型 — `multimodal_model.py`

**`MultimodalBaseModel`** 是系统的核心，整合了视觉编码器、文本编码器、融合模块和三个任务头：

```python
class MultimodalBaseModel(nn.Module):
    # 支持三种任务模式
    def forward(self, input_ids, attention_mask, pixel_values, task="matching"):
        # task = "matching"       → CLIP 风格对比学习
        # task = "generation"     → 内容生成 logits
        # task = "recommendation" → 推荐打分
```

| 任务头 | 类名 | 输出 |
|--------|------|------|
| **图文匹配** | `MatchingHead` | 温度缩放的余弦相似度矩阵，用于对比学习 |
| **内容生成** | `ContentGenerationHead` | 生成 logits (vocab_size 维度) |
| **推荐排序** | `RecommendationHead` | 单值排序分数 |

**对比学习损失**：
```python
def compute_contrastive_loss(self, text_proj, vision_proj, temperature=0.07):
    # 双向 InfoNCE: L = (L_t2v + L_v2t) / 2
```

#### 模型优化 — `optimization.py`

| 组件 | 功能 |
|------|------|
| **`ModelQuantizer`** | INT8 动态量化 (`torch.quantization.quantize_dynamic`)；模型大小统计（参数量 / MB）；推理基准测试（P50/P95/P99 延迟、QPS） |
| **`ModelDistiller`** | 知识蒸馏：Teacher→Student 特征对齐 (MSE) + 任务损失加权混合；AdamW 优化器 |
| **`export_onnx()`** | ONNX 导出，支持动态 batch_size 和 seq_len 维度 |

---

### 2. 数据处理 (`src/data`)

#### 数据集 — `dataset.py`

| 类 | 说明 |
|---|---|
| **`MultimodalSample`** | 数据结构 (dataclass)：`sample_id`, `text`, `image_path`, `image_tensor`, `label`, `metadata` |
| **`PreferencePair`** | 偏好数据对 (DPO/RLHF)：`prompt`, `chosen`, `rejected`, 可选图像 |
| **`MultimodalDataset`** | 图文对数据集。支持 JSON 加载；开发模式自动生成 1000 条模拟数据 (8 个品类)；集成 tokenizer 和 image_processor |
| **`PreferenceDataset`** | DPO/RLHF 偏好对数据集。开发模式自动生成 500 组偏好对 |
| **`create_dataloader()`** | DataLoader 工厂，自定义 `collate_fn` 处理异构批次，支持 `pin_memory` |

**模拟数据类别**：护肤品、彩妆、数码、服装、美食、家居、运动、图书

#### 预处理 — `preprocessing.py`

| 组件 | 功能 |
|------|------|
| **`TextPreprocessor`** | 文本清洗管线：去除 URL → 去除 HTML 标签 → 清除控制字符 → 压缩空白 → 截断。质量校验：最小长度 + 非纯标点 |
| **`ImagePreprocessor`** | 图像标准化：RGB 转换 → Resize → ToTensor → CLIP 标准化 (mean=[0.48145466, 0.4578275, 0.40821073])。最小尺寸校验 |
| **`DataAugmentor`** | 数据增强：**图像** - 随机水平翻转 / 色彩抖动 / 高斯模糊 / 随机裁剪；**文本** - 随机词交换 / 随机字符插入 |
| **`MultimodalPipeline`** | 端到端管线：组合清洗→增强→标准化，支持单样本和批量处理 |

---

### 3. 强化学习 (`src/rl`)

#### 多维奖励模型 — `reward_model.py`

系统采用**四维度奖励评分**机制，全面衡量内容质量：

| 奖励维度 | 权重 (默认) | 说明 |
|----------|-------------|------|
| `content_quality` | 0.30 | 内容质量分数 |
| `user_preference` | 0.30 | 用户偏好匹配度 |
| `business_compliance` | 0.20 | 业务合规性 |
| `relevance` | 0.20 | 内容相关性 |

```python
class MultimodalRewardModel(nn.Module):
    # 输入: 多模态特征 + 用户特征(可选) + 业务特征(可选)
    # 输出: 各维度分数 + Softmax 加权后的 total_reward
    
    def compute_preference_loss(self, chosen_features, rejected_features):
        # Bradley-Terry 偏好损失: -log(σ(r_chosen - r_rejected))
```

**`RewardModelTrainer`**：AdamW 优化器 + CosineAnnealingLR 调度 + 梯度裁剪

#### DPO 偏好优化 — `dpo_trainer.py`

| 组件 | 说明 |
|------|------|
| **`DPOLoss`** | 支持三种损失变体：`sigmoid` (原版 DPO)、`hinge` (铰链损失)、`ipo` (Identity Preference Optimization)。支持标签平滑和 reference-free 模式 |
| **`DPOTrainer`** | 工业级训练器：自动深拷贝策略模型作为参考模型 (冻结)；梯度累积；AMP 混合精度；检查点保存/加载 |

**DPO 核心公式** (Sigmoid 变体):
```
L_DPO = -log σ(β · (log π_θ(y_w|x) - log π_ref(y_w|x) - log π_θ(y_l|x) + log π_ref(y_l|x)))
```

#### PPO 策略优化 — `ppo_trainer.py`

| 组件 | 说明 |
|------|------|
| **`ValueNetwork`** | Critic：3 层 MLP + LayerNorm + Dropout，估计状态价值 V(s) |
| **`PolicyNetwork`** | Actor：支持**离散**动作空间 (Categorical) 和**连续**动作空间 (Normal)。提供 `get_action()` 和 `evaluate_actions()` |
| **`PPOExperience`** | 经验数据结构：states, actions, log_probs, rewards, values, advantages, returns |
| **`PPOTrainer`** | 工业级 PPO 训练器 |

**PPO 核心特性**：
- **GAE 优势估计**：`γ = 0.99`, `λ = 0.95`
- **Clipped 策略损失**：`clip_range = 0.2`
- **Clipped 值函数损失**：防止价值网络过拟合
- **熵正则化**：`entropy_coef = 0.01`，鼓励探索
- **自适应 KL 系数**：`target_kl = 0.01`，KL 过大时自动调整或提前停止
- **Mini-batch 多 epoch 更新**：每轮收集经验后进行 `ppo_epochs` 次更新
- **经验缓冲池**：`deque(maxlen=10000)`

#### 多智能体系统 — `multi_agent.py`

```
┌─────────────┐    通信     ┌──────────────────┐           ┌──────────────┐
│ ContentAgent │ ─────────→ │ RecommendAgent    │ ────────→ │ RankingAgent │
│ (内容生成)   │  门控消息   │ (推荐排序)        │  候选物品  │ (精排打分)   │
└─────────────┘            └──────────────────┘           └──────────────┘
```

| 智能体 | 架构 | 功能 |
|--------|------|------|
| **`ContentAgent`** | Actor-Critic (MLP) | 输入用户状态 → 输出内容动作 logits + 价值估计 |
| **`RecommendationAgent`** | Scorer + Value | 输入增强状态 → 输出全部候选分数 + top-k 排序 |
| **`RankingAgent`** | Cross-Scorer | 输入用户状态 × 候选物品 → 输出精排分数 |
| **`AgentCommunication`** | 门控消息传递 | 延迟初始化投影层 + Sigmoid 门控 + 残差连接 |

**`MultiAgentSystem`** 支持三种运行模式：
- `full_pipeline`：生成 → 推荐 → 精排 全链路
- `generate_only`：仅内容生成
- `recommend_only`：仅推荐排序

---

### 4. 训练管线 (`src/training`)

#### 端到端训练 — `pipeline.py`

**`TrainingPipeline`** 编排四个训练阶段：

```
Stage 1: SFT 监督微调
    ↓  (得到微调后的基座模型)
Stage 2: Reward Model 训练
    ↓  (得到多维奖励模型)
Stage 3: DPO 偏好对齐
    ↓  (策略与人类偏好对齐)
Stage 4: PPO 决策优化
    ↓  (策略进一步优化)
输出 → outputs/ 目录
```

每个阶段可独立运行，也可通过 `run_all()` 全流程串行执行。

#### SFT 微调 — `sft_trainer.py`

**`SFTTrainer`** 核心特性：

| 特性 | 说明 |
|------|------|
| **QLoRA** | 低秩适配 (r=16, alpha=32)，大幅降低训练显存 |
| **梯度累积** | 配置 `gradient_accumulation_steps` 模拟大 batch |
| **AMP 混合精度** | 自动混合精度训练 (bf16/fp16) |
| **学习率调度** | Warmup + 余弦退火 (CosineAnnealingLR) |
| **早停** | 验证指标连续 N 步不改善时停止 |
| **检查点** | 保存/加载模型、优化器、调度器完整状态 |

---

### 5. 评估系统 (`src/evaluation`)

#### 综合评估 — `metrics.py`

| 评估类别 | 类名 | 指标 |
|----------|------|------|
| **文本质量** | `TextMetrics` | BLEU (N-gram)、ROUGE-L (LCS F1) |
| **检索精度** | `RetrievalMetrics` | Recall@K (K=1,5,10)、CLIP Score (图文余弦相似度) |
| **RL 效果** | `RLMetrics` | 偏好胜率 (Win Rate)、奖励提升比 (Reward Improvement)、KL 散度 |
| **业务指标** | `BusinessMetrics` | CTR 模拟、用户参与度 (停留时间 / 完成率) |
| **综合套件** | `EvaluationSuite` | 整合全部指标，支持 JSON 报告导出、SFT vs RL 对比实验 |

---

### 6. API 服务 (`src/api`)

#### FastAPI 服务 — `server.py`

基于 FastAPI 构建的生产级推理服务，使用单例模式管理模型生命周期：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 (版本 / 模型状态 / 设备 / 运行时间) |
| `/api/v1/generate` | POST | 内容生成 (prompt + category → content + quality_score) |
| `/api/v1/recommend` | POST | 个性化推荐 (user_id + num_items → items list) |
| `/api/v1/search` | POST | 跨模态检索 (query_text/image → results with scores) |

**Pydantic 数据模型**：严格的请求/响应校验，包含字段约束与默认值。

**`ModelService`** 单例：
- 延迟加载多模态模型和奖励模型
- 自动设备检测 (CUDA → CPU 回退)
- 线程安全的推理方法

---

### 7. 工具模块 (`src/utils`)

#### 配置管理 — `config.py`

| 组件 | 说明 |
|------|------|
| **`ConfigManager`** | 单例模式 YAML 配置管理器。支持深度合并覆盖；点号嵌套键访问 (如 `get("model.fusion.type")`) |
| **`setup_seed()`** | 全局随机种子：`random` / `numpy` / `torch` / `torch.cuda` deterministic |
| **`setup_logging()`** | Loguru 日志系统：彩色控制台输出 + 可选文件滚转 (100MB/文件, 保留 30 天) |

---

## 配置系统

### 基础配置 (`configs/base_config.yaml`)

```yaml
project:
  name: mmrl-content-system
  version: "1.0.0"
  seed: 42
  device: cuda
  mixed_precision: bf16

data:
  max_text_length: 512
  image_size: 224
  batch_size: 16

model:
  vision_encoder:
    name: openai/clip-vit-large-patch14    # CLIP-ViT 预训练
    hidden_size: 1024
    freeze: all                            # 冻结视觉编码器

  text_encoder:
    name: Qwen/Qwen2-7B-Instruct          # Qwen2 大语言模型
    hidden_size: 4096
    max_length: 512

  fusion:
    type: cross_attention                  # 交叉注意力融合
    hidden_size: 1024
    num_heads: 16
    num_layers: 4
    use_gate: true                         # 叠加门控增强

sft:
  lora_r: 16                              # LoRA 秩
  lora_alpha: 32
  use_qlora: true                         # 启用 4-bit QLoRA
  learning_rate: 2.0e-4
  num_epochs: 3

reward_model:
  num_reward_heads: 4                      # 四维奖励
  learning_rate: 1.0e-5

dpo:
  beta: 0.1                               # DPO 温度系数
  loss_type: sigmoid
  learning_rate: 5.0e-6

ppo:
  learning_rate: 1.0e-6
  clip_range: 0.2                          # PPO 裁剪范围
  gamma: 0.99                             # 折扣因子
  lam: 0.95                               # GAE lambda
  target_kl: 0.01                         # KL 散度目标
```

### DeepSpeed 配置 (`configs/deepspeed_config.json`)

- **混合精度**：bf16
- **优化器**：AdamW (lr=2e-4, betas=[0.9, 0.999])
- **调度器**：WarmupDecayLR (100 warmup / 10000 total steps)
- **ZeRO-2**：参数分片 + CPU Offload + 通信重叠
- **Batch**：micro_batch=4/GPU, gradient_accumulation=4

---

## 快速开始

### 环境要求

- Python ≥ 3.10
- PyTorch ≥ 2.1
- CUDA 11.8+ (GPU 训练) 或 CPU

### 安装

```bash
# 克隆项目
git clone <repo_url>
cd mmrl-content-system

# 安装依赖
pip install -r requirements.txt

# 验证依赖
python check_deps.py
```

### 运行测试

```bash
python run.py test
# 或直接使用 pytest
python -m pytest tests/ -v
```

### 快速启动服务

```bash
python run.py serve --host 0.0.0.0 --port 8000
```

---

## CLI 命令参考

项目提供基于 Click 的 CLI，包含 4 个子命令：

### 训练

```bash
# 全流程训练 (SFT → RM → DPO → PPO)
python run.py train --config configs/base_config.yaml --stage all

# 仅运行 SFT 阶段
python run.py train --config configs/base_config.yaml --stage sft

# 仅运行奖励模型训练
python run.py train --stage rm

# 仅运行 DPO 偏好对齐
python run.py train --stage dpo

# 仅运行 PPO 策略优化
python run.py train --stage ppo
```

### 部署服务

```bash
python run.py serve --host 0.0.0.0 --port 8000 --workers 4
```

### 评估

```bash
python run.py evaluate --config configs/base_config.yaml --output evaluation_report.json
```

### 测试

```bash
python run.py test
```

---

## API 接口文档

### 健康检查

```http
GET /health
```

**响应示例**:
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "model_loaded": true,
    "device": "cuda",
    "uptime_seconds": 3600.5
}
```

### 内容生成

```http
POST /api/v1/generate
Content-Type: application/json
```

**请求体**:
```json
{
    "prompt": "推荐一款适合夏天的防晒霜",
    "category": "美妆",
    "max_length": 256,
    "temperature": 0.7,
    "top_p": 0.9
}
```

**响应**:
```json
{
    "request_id": "uuid-string",
    "content": "生成的内容文本...",
    "quality_score": 0.85,
    "latency_ms": 120.5
}
```

### 个性化推荐

```http
POST /api/v1/recommend
Content-Type: application/json
```

**请求体**:
```json
{
    "user_id": "user_001",
    "num_items": 5,
    "category": "美妆"
}
```

**响应**:
```json
{
    "user_id": "user_001",
    "items": [
        {"item_id": "item_1", "score": 0.95, "category": "美妆"},
        ...
    ],
    "latency_ms": 45.2
}
```

### 跨模态检索

```http
POST /api/v1/search
Content-Type: application/json
```

**请求体**:
```json
{
    "query_text": "美丽的日落风景",
    "top_k": 5,
    "search_type": "text2image"
}
```

**响应**:
```json
{
    "results": [
        {"id": "img_001", "score": 0.92, "type": "image"},
        ...
    ],
    "latency_ms": 78.3
}
```

---

## 训练流程

### 完整四阶段训练

```
输入数据
  │
  ▼
┌──────────────────────────────────────────────────────┐
│ Stage 1: SFT (监督微调)                               │
│ ─────────────────────                                 │
│ • 加载预训练多模态模型                                  │
│ • QLoRA 低秩适配 (r=16, alpha=32, 4-bit)              │
│ • 梯度累积 + AMP 混合精度                              │
│ • 余弦退火学习率 + 早停                                │
│ • 输出: 微调后的基座模型                               │
├──────────────────────────────────────────────────────┤
│ Stage 2: Reward Model (奖励模型训练)                   │
│ ─────────────────────                                 │
│ • 四维度奖励头: 质量/偏好/合规/相关性                    │
│ • 偏好对数据训练 (chosen vs rejected)                   │
│ • Bradley-Terry 偏好损失                               │
│ • 输出: 多维奖励模型                                   │
├──────────────────────────────────────────────────────┤
│ Stage 3: DPO (直接偏好优化)                            │
│ ─────────────────────                                 │
│ • 自动创建冻结的参考模型                               │
│ • Sigmoid / Hinge / IPO 三种损失可选                   │
│ • 支持标签平滑与 reference-free 模式                    │
│ • 输出: 偏好对齐后的策略模型                            │
├──────────────────────────────────────────────────────┤
│ Stage 4: PPO (近端策略优化)                            │
│ ─────────────────────                                 │
│ • Actor-Critic 架构                                   │
│ • GAE 优势估计 (γ=0.99, λ=0.95)                      │
│ • Clipped 策略/价值损失 + 熵正则化                     │
│ • 自适应 KL 系数控制                                   │
│ • 输出: 最终优化的策略模型                             │
└──────────────────────────────────────────────────────┘
  │
  ▼
模型评估 & 部署
```

---

## 模型优化与部署

### INT8 动态量化

```python
from src.models.optimization import ModelQuantizer

quantizer = ModelQuantizer()
quantized_model = quantizer.dynamic_quantize(model)

# 查看压缩效果
original_size = quantizer.get_model_size(model)
compressed_size = quantizer.get_model_size(quantized_model)

# 推理基准测试 (P50/P95/P99/QPS)
benchmark = quantizer.benchmark_inference(model, input_data, num_runs=100)
```

### 知识蒸馏

```python
from src.models.optimization import ModelDistiller

distiller = ModelDistiller(
    teacher_model=large_model,
    student_model=small_model,
    temperature=4.0,
    alpha=0.7,            # 蒸馏损失权重
)
distiller.train(train_loader, num_epochs=10)
```

### ONNX 导出

```python
from src.models.optimization import export_onnx

export_onnx(
    model=model,
    save_path="model.onnx",
    input_ids=sample_ids,
    attention_mask=sample_mask,
    pixel_values=sample_pixels,
)
```

---

## 测试

项目包含 **122 个测试用例**，覆盖所有核心模块：

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行特定模块
python -m pytest tests/test_models.py -v      # 模型测试
python -m pytest tests/test_rl.py -v           # 强化学习测试
python -m pytest tests/test_data.py -v         # 数据处理测试
python -m pytest tests/test_evaluation.py -v   # 评估测试
python -m pytest tests/test_integration.py -v  # 集成测试
python -m pytest tests/test_api.py -v          # API 测试
python -m pytest tests/test_utils.py -v        # 工具测试
```

### 测试覆盖分布

| 模块 | 测试文件 | 测试类/方法数 | 覆盖内容 |
|------|----------|---------------|----------|
| 模型 | `test_models.py` | 10 类 | ViT / TextEncoder / 三种融合 / 任务头 / 基座模型 |
| RL | `test_rl.py` | 12 类 | 奖励模型 / DPO / PPO / 多智能体 / 通信 |
| 数据 | `test_data.py` | 7 类 | 文本/图像预处理 / 增强 / 数据集 / DataLoader |
| 评估 | `test_evaluation.py` | 5 类 | 文本 / 检索 / RL / 业务指标 / 综合套件 |
| 集成 | `test_integration.py` | 10 方法 | 端到端流程 / SFT / DPO / PPO / 量化 |
| API | `test_api.py` | 4 个异步测试 | 健康检查 / 生成 / 推荐 / 检索 |
| 工具 | `test_utils.py` | 6 方法 | 配置加载 / 合并 / 嵌套访问 |

---

## Docker 部署

### 构建镜像

```bash
docker build -t mmrl-content-system:latest .
```

### 运行容器

```bash
docker run -d \
    --name mmrl-api \
    -p 8000:8000 \
    -e CONFIG_PATH=configs/base_config.yaml \
    --gpus all \
    mmrl-content-system:latest
```

### Dockerfile 特性

- **基础镜像**：`python:3.10-slim`
- **健康检查**：`curl /health`，30s 间隔，5s 超时，3 次重试
- **进程管理**：uvicorn 双 Worker
- **环境变量**：`PYTHONUNBUFFERED=1`，`CONFIG_PATH` 可配置

---

## 技术栈

| 类别 | 技术 |
|------|------|
| **深度学习框架** | PyTorch ≥ 2.1, torchvision, torchaudio |
| **Transformers 生态** | HuggingFace Transformers ≥ 4.40, Accelerate, PEFT, BitsAndBytes |
| **强化学习** | TRL ≥ 0.8, Gymnasium, Stable-Baselines3 |
| **视觉模型** | OpenCLIP, timm |
| **分布式训练** | DeepSpeed (ZeRO-2) |
| **API 框架** | FastAPI, Uvicorn, Pydantic |
| **数据处理** | NumPy, Pandas, scikit-learn, Pillow |
| **监控日志** | TensorBoard, Weights & Biases, Prometheus, Loguru |
| **模型优化** | ONNX Runtime, Optimum |
| **测试** | pytest, pytest-asyncio, httpx |
| **CLI** | Click, Rich |
| **配置管理** | PyYAML, OmegaConf |
| **代码质量** | Ruff (line-length=120) |

---

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 开源。
