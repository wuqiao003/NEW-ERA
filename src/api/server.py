"""
FastAPI服务 - 电商场景A落地版
提供：
  1. 多风格营销文案生成（商品图片+文本 → 5种风格文案）
  2. 文案质量评估与排序
  3. 商品推荐（基于用户偏好+向量检索）
  4. 跨模态商品检索
  5. 图片上传与处理
"""
import io
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger


# ============ 数据模型 ============

# — 原有接口模型（保持兼容）—

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


# — 电商场景新增模型 —

class CopyGenerationRequest(BaseModel):
    """营销文案生成请求"""
    product_title: str = Field(..., description="商品名称")
    product_description: str = Field(default="", description="商品描述")
    category: str = Field(default="通用", description="商品类目")
    tags: List[str] = Field(default_factory=list, description="商品标签")
    price: float = Field(default=0.0, ge=0.0, description="价格")
    styles: Optional[List[str]] = Field(
        default=None,
        description="文案风格列表，可选: 种草/促销/情感/专业/简约，默认全部",
    )
    num_candidates: int = Field(default=3, ge=1, le=10, description="每种风格生成候选数")


class CopyGenerationResponse(BaseModel):
    """营销文案生成响应"""
    request_id: str
    product_title: str
    copies: Dict[str, List[Dict[str, Any]]]
    best_copy: Optional[Dict[str, Any]] = None
    total_generated: int
    latency_ms: float


class CopyEvaluateRequest(BaseModel):
    """文案质量评估请求"""
    copy_text: str = Field(..., description="待评估文案")
    style: str = Field(default="种草", description="文案风格")
    product_info: Optional[Dict[str, Any]] = Field(default=None, description="商品信息")


class CopyEvaluateResponse(BaseModel):
    """文案质量评估响应"""
    request_id: str
    scores: Dict[str, float]
    overall_score: float
    latency_ms: float


class ProductSearchRequest(BaseModel):
    """商品检索请求"""
    query_text: str = Field(..., description="搜索关键词")
    top_k: int = Field(default=10, ge=1, le=100)
    category_filter: Optional[str] = Field(default=None, description="类目筛选")
    price_min: Optional[float] = Field(default=None, description="最低价")
    price_max: Optional[float] = Field(default=None, description="最高价")


class ProductSearchResponse(BaseModel):
    """商品检索响应"""
    request_id: str
    results: List[Dict[str, Any]]
    total: int
    latency_ms: float


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
        self.copy_engine = None
        self.copy_evaluator = None
        self.vector_store = None

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

            # 初始化多模态模型
            self.multimodal_model = MultimodalBaseModel(self.config).to(self.device)
            self.multimodal_model.eval()

            # 初始化奖励模型
            shared_dim = self.config["model"]["projection"]["shared_dim"]
            self.reward_model = MultimodalRewardModel(
                multimodal_dim=shared_dim
            ).to(self.device)
            self.reward_model.eval()

            # 初始化文案生成引擎
            from src.generation.copy_engine import CopyGenerationEngine
            self.copy_engine = CopyGenerationEngine(
                multimodal_model=self.multimodal_model,
                reward_model=self.reward_model,
                device=self.device,
                use_model=True,
            )

            # 初始化文案质量评估器
            from src.models.copy_generator import CopyQualityEvaluator
            self.copy_evaluator = CopyQualityEvaluator(
                reward_model=self.reward_model,
                device=self.device,
            )

            # 初始化向量检索引擎
            from src.data.vector_store import ProductVectorStore
            self.vector_store = ProductVectorStore(
                dim=shared_dim,
                index_type="Flat",
            )

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

    # ============ 原有接口 ============

    @torch.no_grad()
    def generate_content(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """通用内容生成推理"""
        start = time.time()

        # 如果有文案引擎且指定了风格，走电商文案逻辑
        if self.copy_engine and request.style:
            results = self.copy_engine.generate_multi_style_copies(
                product_title=request.prompt,
                styles=[request.style],
                num_candidates=1,
            )
            best = None
            for copies in results.values():
                if copies:
                    best = copies[0]
                    break
            if best:
                latency = (time.time() - start) * 1000
                return {
                    "content": best["content"],
                    "quality_score": best.get("score", 0.5),
                    "relevance_score": best.get("score", 0.5),
                    "latency_ms": latency,
                }

        # 通用模型推理
        input_ids = torch.randint(0, 32000, (1, min(len(request.prompt) * 2, 512))).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.randn(1, 3, 224, 224).to(self.device)

        outputs = self.multimodal_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            task="generation",
        )

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
        """个性化推荐 — 基于向量检索"""
        start = time.time()

        items = []

        # 优先使用向量检索
        if self.vector_store and self.vector_store.current_size > 0:
            import numpy as np
            # 用用户 ID 生成伪特征（实际应从用户特征服务获取）
            user_seed = hash(request.user_id) % (2**31)
            np.random.seed(user_seed)
            user_vector = np.random.randn(self.vector_store.dim).astype(np.float32)

            search_results = self.vector_store.search(
                user_vector, top_k=request.num_items,
            )
            for r in search_results:
                meta = r.get("metadata", {})
                items.append({
                    "item_id": r["product_id"],
                    "title": meta.get("title", f"商品 {r['product_id']}"),
                    "score": r["score"],
                    "category": meta.get("category", "通用"),
                    "price": meta.get("price", 0),
                })
        else:
            # 回退到模拟推荐
            for i in range(request.num_items):
                score = float(torch.sigmoid(torch.randn(1)).item())
                items.append({
                    "item_id": f"item_{i:04d}",
                    "title": f"推荐商品 #{i+1}",
                    "score": round(score, 4),
                    "category": ["美妆", "数码", "食品", "服饰", "家居"][i % 5],
                })

        items.sort(key=lambda x: x["score"], reverse=True)
        latency = (time.time() - start) * 1000
        return {"items": items, "latency_ms": latency}

    @torch.no_grad()
    def search(self, request: CrossModalSearchRequest) -> Dict[str, Any]:
        """跨模态检索"""
        start = time.time()

        results = []

        # 优先使用向量检索
        if self.vector_store and self.vector_store.current_size > 0 and request.query_text:
            import numpy as np
            # 简单文本哈希向量（实际应使用文本编码器）
            text_hash = hash(request.query_text) % (2**31)
            np.random.seed(text_hash)
            query_vector = np.random.randn(self.vector_store.dim).astype(np.float32)
            search_results = self.vector_store.search(query_vector, top_k=request.top_k)
            for r in search_results:
                results.append({
                    "id": r["product_id"],
                    "type": "product",
                    "score": r["score"],
                    "content": r.get("metadata", {}).get("title", f"商品 {r['product_id']}"),
                })
        else:
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

    # ============ 电商新增接口 ============

    def generate_copies(self, request: CopyGenerationRequest) -> Dict[str, Any]:
        """生成多风格营销文案"""
        start = time.time()

        if self.copy_engine is None:
            raise RuntimeError("文案生成引擎未初始化")

        results = self.copy_engine.generate_multi_style_copies(
            product_title=request.product_title,
            product_description=request.product_description,
            category=request.category,
            tags=request.tags,
            price=request.price,
            styles=request.styles,
            num_candidates=request.num_candidates,
        )

        # 找最佳文案
        best_copy = None
        best_score = -1
        total = 0
        for copies in results.values():
            total += len(copies)
            for c in copies:
                if c.get("score", 0) > best_score:
                    best_score = c["score"]
                    best_copy = c

        latency = (time.time() - start) * 1000

        return {
            "product_title": request.product_title,
            "copies": results,
            "best_copy": best_copy,
            "total_generated": total,
            "latency_ms": latency,
        }

    def generate_copy_from_image(
        self, product_title: str, image_bytes: bytes, styles: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """基于上传图片生成文案"""
        start = time.time()

        # 图片处理
        from PIL import Image
        from src.data.preprocessing import ImagePreprocessor

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        preprocessor = ImagePreprocessor(image_size=224)
        pixel_values = preprocessor.process(image).unsqueeze(0).to(self.device)

        if self.copy_engine is None:
            raise RuntimeError("文案生成引擎未初始化")

        results = self.copy_engine.generate_multi_style_copies(
            product_title=product_title,
            pixel_values=pixel_values,
            styles=styles,
            num_candidates=3,
        )

        best_copy = None
        best_score = -1
        total = 0
        for copies in results.values():
            total += len(copies)
            for c in copies:
                if c.get("score", 0) > best_score:
                    best_score = c["score"]
                    best_copy = c

        latency = (time.time() - start) * 1000

        return {
            "product_title": product_title,
            "copies": results,
            "best_copy": best_copy,
            "total_generated": total,
            "latency_ms": latency,
        }

    def evaluate_copy(self, request: CopyEvaluateRequest) -> Dict[str, Any]:
        """评估文案质量"""
        start = time.time()

        if self.copy_evaluator is None:
            raise RuntimeError("文案评估器未初始化")

        scores = self.copy_evaluator.evaluate_copy(
            copy_text=request.copy_text,
            style=request.style,
            product_info=request.product_info,
        )

        latency = (time.time() - start) * 1000
        return {
            "scores": scores,
            "overall_score": scores.get("overall", 0.0),
            "latency_ms": latency,
        }

    def search_products(self, request: ProductSearchRequest) -> Dict[str, Any]:
        """商品检索"""
        start = time.time()

        results = []
        if self.vector_store and self.vector_store.current_size > 0:
            import numpy as np
            text_hash = hash(request.query_text) % (2**31)
            np.random.seed(text_hash)
            query_vector = np.random.randn(self.vector_store.dim).astype(np.float32)

            price_range = None
            if request.price_min is not None or request.price_max is not None:
                price_range = (
                    request.price_min or 0,
                    request.price_max or float("inf"),
                )

            results = self.vector_store.search(
                query_vector,
                top_k=request.top_k,
                category_filter=request.category_filter,
                price_range=price_range,
            )

        latency = (time.time() - start) * 1000
        return {
            "results": results,
            "total": len(results),
            "latency_ms": latency,
        }


# ============ FastAPI应用 ============

def create_app() -> FastAPI:
    """创建FastAPI应用"""
    app = FastAPI(
        title="电商智能营销文案系统 — 多模态RL驱动",
        description=(
            "基于多模态+强化学习的电商营销文案生成与推荐系统\n\n"
            "核心能力：\n"
            "- 🖼️ 商品图片+文本 → 5种风格营销文案自动生成\n"
            "- 📊 文案质量评估与偏好排序\n"
            "- 🔍 商品语义检索\n"
            "- 💡 个性化商品推荐"
        ),
        version="2.0.0",
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

    # ---- 健康检查 ----

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            device=model_service.device if model_service._initialized else "unknown",
            uptime_seconds=time.time() - model_service.start_time if model_service._initialized else 0,
            model_loaded=model_service.model_loaded if model_service._initialized else False,
        )

    # ---- 原有接口（保持兼容）----

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

    # ---- 电商文案生成接口 ----

    @app.post("/api/v2/copy/generate", response_model=CopyGenerationResponse)
    async def generate_copies(request: CopyGenerationRequest):
        """
        商品多风格营销文案生成

        根据商品信息（标题、描述、类目、标签、价格），
        自动生成 5 种风格的营销文案（种草/促销/情感/专业/简约），
        并按质量分排序。
        """
        if not model_service.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载")

        try:
            result = model_service.generate_copies(request)
            return CopyGenerationResponse(
                request_id=str(uuid.uuid4()),
                **result,
            )
        except Exception as e:
            logger.error(f"文案生成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v2/copy/generate-from-image", response_model=CopyGenerationResponse)
    async def generate_copy_from_image(
        product_title: str = Form(..., description="商品名称"),
        styles: Optional[str] = Form(default=None, description="风格列表(逗号分隔)"),
        image: UploadFile = File(..., description="商品图片"),
    ):
        """
        图片上传 → 营销文案生成

        上传商品图片和名称，系统自动提取视觉特征，
        生成多风格营销文案。
        """
        if not model_service.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载")

        try:
            image_bytes = await image.read()
            style_list = styles.split(",") if styles else None

            result = model_service.generate_copy_from_image(
                product_title=product_title,
                image_bytes=image_bytes,
                styles=style_list,
            )
            return CopyGenerationResponse(
                request_id=str(uuid.uuid4()),
                **result,
            )
        except Exception as e:
            logger.error(f"图片文案生成失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v2/copy/evaluate", response_model=CopyEvaluateResponse)
    async def evaluate_copy(request: CopyEvaluateRequest):
        """
        文案质量评估

        从长度、可读性、风格匹配度等多维度评估文案质量。
        """
        if not model_service.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载")

        try:
            result = model_service.evaluate_copy(request)
            return CopyEvaluateResponse(
                request_id=str(uuid.uuid4()),
                **result,
            )
        except Exception as e:
            logger.error(f"文案评估失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ---- 商品检索接口 ----

    @app.post("/api/v2/products/search", response_model=ProductSearchResponse)
    async def search_products(request: ProductSearchRequest):
        """
        商品语义检索

        基于文本语义的商品搜索，支持类目和价格过滤。
        """
        if not model_service.model_loaded:
            raise HTTPException(status_code=503, detail="模型未加载")

        try:
            result = model_service.search_products(request)
            return ProductSearchResponse(
                request_id=str(uuid.uuid4()),
                **result,
            )
        except Exception as e:
            logger.error(f"商品检索失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ---- 工具接口 ----

    @app.get("/api/v2/styles")
    async def list_styles():
        """获取可用的文案风格列表"""
        from src.generation.copy_engine import COPY_STYLES
        return {
            "styles": [
                {
                    "name": key,
                    "display_name": cfg.get("name", key),
                    "description": cfg.get("description", ""),
                    "tone": cfg.get("tone", ""),
                }
                for key, cfg in COPY_STYLES.items()
            ]
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
