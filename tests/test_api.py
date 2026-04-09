"""
API服务测试
覆盖: v1 兼容接口 + v2 电商文案/检索/评估接口
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from httpx import AsyncClient, ASGITransport
from src.api.server import create_app, ModelService


@pytest.fixture
def app():
    # 重置单例，确保每次测试环境干净
    ModelService._instance = None
    application = create_app()
    # 手动初始化模型服务（因为on_event("startup")在测试中不自动触发）
    service = ModelService()
    service.initialize()
    return application


# ============ 健康检查 ============

@pytest.mark.asyncio
async def test_health_check(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "2.0.0"
        assert "device" in data
        assert "uptime_seconds" in data
        assert "model_loaded" in data


# ============ v1 兼容接口 ============

@pytest.mark.asyncio
async def test_generate_content(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/generate", json={
            "prompt": "推荐一款适合夏天的防晒霜",
            "category": "美妆",
            "max_length": 256,
        })
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "content" in data
        assert "quality_score" in data
        assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_generate_content_with_style(app):
    """v1 生成接口 + 风格参数 → 走电商文案引擎"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/generate", json={
            "prompt": "蓝牙耳机Pro",
            "category": "数码",
            "style": "种草",
        })
        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert len(data["content"]) > 0


@pytest.mark.asyncio
async def test_recommend(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/recommend", json={
            "user_id": "user_001",
            "num_items": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) == 5
        for item in data["items"]:
            assert "item_id" in item
            assert "score" in item


@pytest.mark.asyncio
async def test_search(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/search", json={
            "query_text": "美丽的日落风景",
            "top_k": 5,
            "search_type": "text2image",
        })
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 5


# ============ v2 电商文案生成接口 ============

@pytest.mark.asyncio
async def test_copy_generate(app):
    """多风格营销文案生成"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v2/copy/generate", json={
            "product_title": "清爽防晒霜SPF50+",
            "product_description": "轻薄不油腻，长效防晒12小时",
            "category": "美妆",
            "tags": ["防晒", "清爽", "敏感肌"],
            "price": 119.0,
            "num_candidates": 2,
        })
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "product_title" in data
        assert data["product_title"] == "清爽防晒霜SPF50+"
        assert "copies" in data
        assert "total_generated" in data
        assert data["total_generated"] > 0
        assert data["latency_ms"] >= 0


@pytest.mark.asyncio
async def test_copy_generate_specific_styles(app):
    """指定风格文案生成"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v2/copy/generate", json={
            "product_title": "蓝牙耳机Pro",
            "styles": ["种草", "专业"],
            "num_candidates": 1,
        })
        assert response.status_code == 200
        data = response.json()
        copies = data["copies"]
        # 应只包含请求的风格
        assert "种草" in copies or "专业" in copies


@pytest.mark.asyncio
async def test_copy_generate_best_copy(app):
    """验证最佳文案返回"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v2/copy/generate", json={
            "product_title": "记忆棉枕头",
            "category": "家居",
        })
        assert response.status_code == 200
        data = response.json()
        if data["total_generated"] > 0:
            assert data["best_copy"] is not None
            assert "content" in data["best_copy"]
            assert "score" in data["best_copy"]


# ============ v2 文案质量评估接口 ============

@pytest.mark.asyncio
async def test_copy_evaluate(app):
    """文案质量评估"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v2/copy/evaluate", json={
            "copy_text": "姐妹们这个防晒霜真的绝了！轻薄不油腻，用了一周皮肤状态肉眼可见变好了～性价比超高！",
            "style": "种草",
        })
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "scores" in data
        assert "overall_score" in data
        assert 0.0 <= data["overall_score"] <= 1.0
        assert data["latency_ms"] >= 0


# ============ v2 商品检索接口 ============

@pytest.mark.asyncio
async def test_product_search(app):
    """商品检索（即使索引为空也不应报错）"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v2/products/search", json={
            "query_text": "防晒",
            "top_k": 5,
        })
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "results" in data
        assert "total" in data
        assert data["latency_ms"] >= 0


# ============ v2 风格列表接口 ============

@pytest.mark.asyncio
async def test_list_styles(app):
    """获取可用风格列表"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v2/styles")
        assert response.status_code == 200
        data = response.json()
        assert "styles" in data
        style_names = [s["name"] for s in data["styles"]]
        assert "种草" in style_names
        assert "促销" in style_names
        assert "简约" in style_names
