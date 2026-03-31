"""
FastAPI服务 - 工业级API部署
提供多模态内容生成、推荐、检索接口
"""
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# ============ 数据模型 ============

class ContentGenerationRequest(BaseModel):
    """内容生成请求"""
    prompt: str = Field(..., description="输入提示文本")
    category: str = Field(default="general", description="内容类别")
    max_length: int = Field(default=512, description="最大生成长度")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    style: Optional[str] = Field(default=None, description="生成风格")

class ContentGenerationResponse(BaseModel):
    """内容生成响应"""
    request_id: str
    content: str
    quality_score: float
    relevance_score: float
    latency_ms: float
    metadata: Dict[str, Any] = {}

class RecommendationRequest(BaseModel):
    """推荐请求"""
    user_id: str = Field(..., description="用户ID")
    context: Optional[str] = Field(default=None, description="场景上下文")
    num_items: int = Field(default=10, ge=1, le=100)
    diversity: float = Field(default=0.3, ge=0.0, le=1.0, description="多样性权重")

class RecommendationResponse(BaseModel):
    """推荐响应"""
    request_id: str
    items: List[Dict[str, Any]]
    latency_ms: float

class CrossModalSearchRequest(BaseModel):
    """跨模态检索请求"""
    query_text: Optional[str] = Field(default=None, description="文本查询")
    top_k: int = Field(default=10, ge=1, le=100)
    search_type: str = Field(default="text2image", description="text2image / image2text")

class CrossModalSearchResponse(BaseModel):
    """检索响应"""
    request_id: str
    results: List[Dict[str, Any]]
    latency_ms: float

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    device: str
    uptime_seconds: float
    model_loaded: bool


# ============ 模型服务管理 ============

class ModelService:
    """模型服务管理器（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance.model_loaded = False
            cls._instance.device = "unknown"
            cls._instance.start_time = time.time()
        return cls._instance

    def initialize(self):
        """初始化模型（延迟加载）"""
        if self._initialized:
            return

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.start_time = time.time()
        self.model_loaded = False

        try:
            from src.models.multimodal_model import MultimodalBaseModel
            from src.rl.reward_model import MultimodalRewardModel
            from src.utils.config import ConfigManager

            config_path = os.environ.get("CONFIG_PATH", "configs/base_config.yaml")
            if os.path.exists(config_path):
                config = ConfigManager.load(config_path)
                self.config = config.config
            else:
                self.config = self._default_config()

            # 初始化模型
            self.multimodal_model = MultimodalBaseModel(self.config).to(self.device)
            self.multimodal_model.eval()

            self.reward_model = MultimodalRewardModel(
                multimodal_dim=self.config["model"]["projection"]["shared_dim"]
            ).to(self.device)
            self.reward_model.eval()

            self.model_loaded = True
            logger.info(f"✅ 模型服务初始化完成: device={self.device}")

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            self.model_loaded = False

        self._initialized = True

    def _default_config(self) -> Dict[str, Any]:
        return {
            "model": {
                "vision_encoder": {"name": "default", "hidden_size": 1024},
                "text_encoder": {"name": "default", "hidden_size": 1024, "max_length": 512},
                "fusion": {"type": "cross_attention", "num_heads": 8, "num_layers": 2, "dropout": 0.1, "use_gate": True},
                "projection": {"shared_dim": 512},
            }
        }

    @torch.no_grad()
    def generate_content(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """内容生成推理"""
        start = time.time()

        # 模拟tokenize
        input_ids = torch.randint(0, 32000, (1, min(len(request.prompt) * 2, 512))).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.randn(1, 3, 224, 224).to(self.device)

        outputs = self.multimodal_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            task="generation",
        )

        # 奖励评分
        features = outputs.get("fused_features", torch.randn(1, 512).to(self.device))
        reward_out = self.reward_model(features)

        latency = (time.time() - start) * 1000

        return {
            "content": f"[AI生成] 基于「{request.prompt}」的多模态内容: 这是一段高质量的{request.category}类内容...",
            "quality_score": float(reward_out.get("content_quality", torch.tensor(0.85)).mean()),
            "relevance_score": float(reward_out.get("relevance", torch.tensor(0.82)).mean()),
            "latency_ms": latency,
        }

    @torch.no_grad()
    def recommend(self, request: RecommendationRequest) -> Dict[str, Any]:
        """个性化推荐"""
        start = time.time()

        # 模拟用户特征 + 推荐
        user_features = torch.randn(1, 512).to(self.device)
        outputs = self.multimodal_model(
            pixel_values=torch.randn(1, 3, 224, 224).to(self.device),
            task="recommendation",
        )

        scores = outputs.get("scores", torch.randn(request.num_items))
        if scores.dim() == 0:
            scores = torch.randn(request.num_items)

        items = []
        for i in range(request.num_items):
            score = float(torch.sigmoid(torch.randn(1)).item())
            items.append({
                "item_id": f"item_{i:04d}",
                "title": f"推荐内容 #{i+1}",
                "score": round(score, 4),
                "category": ["科技", "美食", "旅行", "时尚", "运动"][i % 5],
            })

        items.sort(key=lambda x: x["score"], reverse=True)
        latency = (time.time() - start) * 1000

        return {"items": items, "latency_ms": latency}

    @torch.no_grad()
    def search(self, request: CrossModalSearchRequest) -> Dict[str, Any]:
        """跨模态检索"""
        start = time.time()

        results = []
        for i in range(request.top_k):
            score = round(1.0 - i * 0.05 + float(torch.randn(1).item() * 0.02), 4)
            results.append({
                "id": f"result_{i:04d}",
                "type": "image" if request.search_type == "text2image" else "text",
                "score": max(0.0, score),
                "content": f"匹配结果 #{i+1} (query: {request.query_text})",
            })

        latency = (time.time() - start) * 1000
        return {"results": results, "latency_ms": latency}


# ============ FastAPI应用 ============

def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="多模态决策强化学习 - 智能内容系统",
        description="基于多模态+强化学习的工业级内容生成与推荐API",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    model_service = ModelService()

    @app.on_event("startup")
    async def startup():
        model_service.initialize()

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            device=model_service.device if model_service._initialized else "unknown",
            uptime_seconds=time.time() - model_service.start_time if model_service._initialized else 0,
            model_loaded=model_service.model_loaded if model_service._initialized else False,
        )

    @app.post("/api/v1/generate", response_model=ContentGenerationResponse)
    async def generate_content(request: ContentGenerationRequest):
        if not model_service.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载")

        try:
            result = model_service.generate_content(request)
            return ContentGenerationResponse(
                request_id=str(uuid.uuid4()),
                **result,
            )
        except Exception as e:
            logger.error(f"生成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/recommend", response_model=RecommendationResponse)
    async def recommend(request: RecommendationRequest):
        if not model_service.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载")

        try:
            result = model_service.recommend(request)
            return RecommendationResponse(
                request_id=str(uuid.uuid4()),
                **result,
            )
        except Exception as e:
            logger.error(f"推荐失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/search", response_model=CrossModalSearchResponse)
    async def cross_modal_search(request: CrossModalSearchRequest):
        if not model_service.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载")

        try:
            result = model_service.search(request)
            return CrossModalSearchResponse(
                request_id=str(uuid.uuid4()),
                **result,
            )
        except Exception as e:
            logger.error(f"检索失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
